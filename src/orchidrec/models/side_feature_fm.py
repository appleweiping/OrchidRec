"""Bounded pairwise factorization machine over training-only user/item features."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping
from typing import Any, Self, cast

from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, InteractionDataset, stable_id_key, validate_entity_id
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.features import (
    PAD_INDEX,
    EncodedFeatureDataset,
    EncodedFeatureRow,
    FeatureDataset,
    FeatureKind,
    FeatureSource,
    FittedFeaturePipeline,
)
from orchidrec.models.base import BaseRecommender, make_envelope, parse_envelope

MAX_USERS = 2_048
MAX_ITEMS = 2_048
MAX_INTERACTIONS = 100_000
MAX_DIMENSIONS = 8_192
MAX_ACTIVE = 256
MAX_WORK = 100_000_000
MAX_FACTORS = 64
MAX_EPOCHS = 100
MAX_ABS_WEIGHT = 1_000_000.0

Sparse = tuple[tuple[int, float], ...]


def _integer(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValidationError(f"{name} must be an integer between 1 and {maximum}")
    return value


def _number(value: object, name: str, *, positive: bool) -> float:
    if type(value) not in {int, float}:
        raise ValidationError(f"{name} must be a finite bounded number")
    number = safe_float(cast(int | float, value))
    if not math.isfinite(number) or number > 1.0 or (number <= 0 if positive else number < 0):
        raise ValidationError(f"{name} must be a finite bounded number in [0, 1]")
    return number


def _feature_digest(encoded: EncodedFeatureDataset) -> str:
    payload = json.dumps(
        encoded.to_state(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bounded_identifier(value: object) -> EntityId:
    identifier = validate_entity_id(value)
    if isinstance(identifier, int):
        if abs(identifier).bit_length() > 512:
            raise ValidationError("FM identifier exceeds 512 integer bits")
    else:
        try:
            size = len(identifier.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as error:
            raise ValidationError("FM identifier must be valid UTF-8") from error
        if size > 2048:
            raise ValidationError("FM identifier exceeds 2048 UTF-8 bytes")
    return identifier


class SideFeatureFM(BaseRecommender):
    """Implicit BPR FM with sparse user/item ID and typed side-feature coordinates.

    The user/item feature pipeline is fitted only on the supplied training entities.
    Interaction-context fields are unsupported because Interaction has no event ID.
    """

    model_type = "side_feature_fm"

    def __init__(
        self,
        *,
        factors: int = 8,
        epochs: int = 12,
        learning_rate: float = 0.02,
        regularization: float = 0.001,
        seed: int = 42,
        max_work_units: int = MAX_WORK,
    ) -> None:
        super().__init__()
        self.factors = _integer(factors, "factors", MAX_FACTORS)
        self.epochs = _integer(epochs, "epochs", MAX_EPOCHS)
        self.learning_rate = _number(learning_rate, "learning_rate", positive=True)
        self.regularization = _number(regularization, "regularization", positive=False)
        if type(seed) is not int or not -(1 << 63) <= seed < (1 << 63):
            raise ValidationError("seed must be a signed 64-bit integer")
        self.seed = seed
        self.max_work_units = _integer(max_work_units, "max_work_units", MAX_WORK)
        self._pipeline: FittedFeaturePipeline | None = None
        self._encoded: EncodedFeatureDataset | None = None
        self._user_vectors: dict[EntityId, Sparse] = {}
        self._item_vectors: dict[EntityId, Sparse] = {}
        self._dimensions = 0
        self._linear: list[float] = []
        self._latent: list[list[float]] = []
        self._updates = 0

    def fit(self, dataset: InteractionDataset, features: FeatureDataset | None = None) -> Self:
        """Fit a deterministic BPR objective on positives and sampled unseen items."""

        if type(dataset) is not InteractionDataset or not dataset:
            raise ValidationError("dataset must be a non-empty InteractionDataset")
        if type(features) is not FeatureDataset:
            raise ValidationError("features must be a FeatureDataset")
        if len(dataset) > MAX_INTERACTIONS:
            raise ValidationError("interaction limit exceeded")
        if len(dataset.user_ids) > MAX_USERS or len(dataset.item_ids) > MAX_ITEMS:
            raise ValidationError("user or item limit exceeded")
        for identifier in (*dataset.user_ids, *dataset.item_ids):
            _bounded_identifier(identifier)
        if len(features) > MAX_USERS + MAX_ITEMS:
            raise ValidationError("side-feature row limit exceeded")
        for source in (FeatureSource.USER, FeatureSource.ITEM):
            width = 1 + sum(
                spec.sequence_length or 1 for spec in features.schema.for_source(source)
            )
            if width > MAX_ACTIVE // 2:
                raise ValidationError("side-feature active-coordinate limit exceeded")
        if any(spec.source is FeatureSource.INTERACTION for spec in features.schema):
            raise ValidationError("interaction features require event IDs and are not supported")
        expected = {
            (FeatureSource.USER, key)
            for key in dataset.user_ids
            if features.schema.for_source(FeatureSource.USER)
        } | {
            (FeatureSource.ITEM, key)
            for key in dataset.item_ids
            if features.schema.for_source(FeatureSource.ITEM)
        }
        actual = {(row.source, row.key) for row in features}
        if actual != expected:
            raise ValidationError("feature rows must exactly match training user/item entities")
        pipeline = FittedFeaturePipeline.fit(features)
        encoded = pipeline.transform(features)
        self._fitted = False
        self._pipeline = pipeline
        self._encoded = encoded
        self._prepare_vectors(dataset.user_ids, dataset.item_ids, encoded, pipeline)
        positives = {(event.user_id, event.item_id) for event in dataset}
        positive_by_user: dict[EntityId, set[EntityId]] = {
            user_id: set() for user_id in dataset.user_ids
        }
        for user, item in positives:
            positive_by_user[user].add(item)
        negative_users = sum(
            len(seen) < len(dataset.item_ids) for seen in positive_by_user.values()
        )
        if negative_users == 0:
            raise ValidationError("BPR requires at least one unobserved item for a training user")
        pair_work = 2 * MAX_ACTIVE * self.factors
        if len(positives) * self.epochs * pair_work > self.max_work_units:
            raise ValidationError("training exceeds max_work_units")
        super().fit(dataset)
        try:
            rng = random.Random(self.seed)
            scale = 0.05 / math.sqrt(self.factors)
            self._linear = [0.0] * self._dimensions
            self._latent = [
                [rng.uniform(-scale, scale) for _ in range(self.factors)]
                for _ in range(self._dimensions)
            ]
            ordered = sorted(
                positives, key=lambda pair: (stable_id_key(pair[0]), stable_id_key(pair[1]))
            )
            negative = {
                user: tuple(item for item in self._catalog if item not in self._seen[user])
                for user in self._seen
            }
            self._updates = 0
            for _ in range(self.epochs):
                pairs = ordered.copy()
                rng.shuffle(pairs)
                for user, positive in pairs:
                    available = negative[user]
                    if available:
                        self._update_pair(user, positive, available[rng.randrange(len(available))])
                        self._updates += 1
            if not self._updates:
                raise ValidationError("no BPR updates were possible")
        except BaseException:
            self._fitted = False
            raise
        self._fitted = True
        return self

    def _fit_model(self, dataset: InteractionDataset) -> None:
        del dataset  # The overridden fit trains after the shared catalog/seen setup.

    def _prepare_vectors(
        self,
        users: tuple[EntityId, ...],
        items: tuple[EntityId, ...],
        encoded: EncodedFeatureDataset,
        pipeline: FittedFeaturePipeline,
    ) -> None:
        for source in (FeatureSource.USER, FeatureSource.ITEM):
            width = 1 + sum(
                spec.sequence_length or 1 for spec in pipeline.schema.for_source(source)
            )
            if width > MAX_ACTIVE // 2:
                raise ValidationError("side-feature active-coordinate limit exceeded")
        indices: dict[tuple[str, str, int], int] = {}
        for key in users:
            indices[("user-id", json.dumps(key, ensure_ascii=False), 0)] = len(indices)
        for key in items:
            indices[("item-id", json.dumps(key, ensure_ascii=False), 0)] = len(indices)
        for spec in pipeline.schema:
            if spec.kind.is_token:
                for token_index in range(1, len(pipeline.token_vocabularies[spec.name]) + 2):
                    indices[("token", spec.name, token_index)] = len(indices)
            elif spec.kind is FeatureKind.FLOAT:
                indices[("float", spec.name, 0)] = len(indices)
            else:
                if spec.sequence_length is None:
                    raise ValidationError("sequence feature has no fixed width")
                for position in range(spec.sequence_length):
                    indices[("float-position", spec.name, position)] = len(indices)
        if len(indices) > MAX_DIMENSIONS:
            raise ValidationError("side-feature dimension limit exceeded")
        self._dimensions = len(indices)
        rows = {(row.source, row.key): row for row in encoded}

        def vector(source: FeatureSource, key: EntityId) -> Sparse:
            id_kind = "user-id" if source is FeatureSource.USER else "item-id"
            accumulator = {indices[(id_kind, json.dumps(key, ensure_ascii=False), 0)]: 1.0}
            row = rows.get((source, key))
            if row is not None:
                self._append_row(accumulator, row, pipeline, indices)
            if len(accumulator) > MAX_ACTIVE // 2:
                raise ValidationError("entity feature vector exceeds the active-coordinate limit")
            return tuple(sorted(accumulator.items()))

        self._user_vectors = {key: vector(FeatureSource.USER, key) for key in users}
        self._item_vectors = {key: vector(FeatureSource.ITEM, key) for key in items}

    @staticmethod
    def _append_row(
        values: dict[int, float],
        row: EncodedFeatureRow,
        pipeline: FittedFeaturePipeline,
        indices: Mapping[tuple[str, str, int], int],
    ) -> None:
        for spec in pipeline.schema.for_source(row.source):
            raw = row.values[spec.name]
            if spec.kind is FeatureKind.TOKEN:
                if type(raw) is not int:
                    raise ValidationError("encoded token must be an integer")
                if raw == PAD_INDEX:
                    raise ValidationError("scalar token cannot use the padding index")
                index = indices.get(("token", spec.name, raw))
                if index is None:
                    raise ValidationError("encoded token index is outside the fitted vocabulary")
                values[index] = 1.0
            elif spec.kind is FeatureKind.FLOAT:
                if type(raw) not in {int, float}:
                    raise ValidationError("encoded numeric feature must be a number")
                number = float(cast(int | float, raw))
                if number:
                    values[indices[("float", spec.name, 0)]] = number
            elif spec.kind is FeatureKind.TOKEN_SEQUENCE:
                if not isinstance(raw, tuple):
                    raise ValidationError("encoded token sequence must be a tuple")
                length = row.sequence_lengths[spec.name]
                if any(value == PAD_INDEX for value in raw[:length]) or any(
                    value != PAD_INDEX for value in raw[length:]
                ):
                    raise ValidationError("token sequence padding does not match its length")
                active = [value for value in raw if value != PAD_INDEX]
                for token_index in active:
                    if type(token_index) is not int:
                        raise ValidationError("encoded token index must be an integer")
                    index = indices.get(("token", spec.name, token_index))
                    if index is None:
                        raise ValidationError(
                            "encoded token index is outside the fitted vocabulary"
                        )
                    values[index] = values.get(index, 0.0) + 1.0 / len(active)
            else:
                if not isinstance(raw, tuple):
                    raise ValidationError("encoded numeric sequence must be a tuple")
                length = row.sequence_lengths[spec.name]
                if any(value != 0.0 for value in raw[length:]):
                    raise ValidationError("numeric sequence padding does not match its length")
                for position in range(length):
                    value = raw[position]
                    if type(value) not in {int, float}:
                        raise ValidationError("encoded numeric sequence must contain numbers")
                    if value:
                        values[indices[("float-position", spec.name, position)]] = float(value)
        if any(not math.isfinite(value) or abs(value) > 1_000 for value in values.values()):
            raise ValidationError("encoded feature magnitude exceeds the FM limit")

    def _pair_vector(self, user: EntityId, item: EntityId) -> Sparse:
        return self._user_vectors[user] + self._item_vectors[item]

    def _score_sparse(self, vector: Sparse) -> float:
        score = math.fsum(self._linear[index] * value for index, value in vector)
        for factor in range(self.factors):
            first = math.fsum(self._latent[index][factor] * value for index, value in vector)
            squares = math.fsum(
                (self._latent[index][factor] * value) ** 2 for index, value in vector
            )
            score += 0.5 * (first * first - squares)
        return score

    def _update_pair(self, user: EntityId, positive: EntityId, negative: EntityId) -> None:
        left = dict(self._pair_vector(user, positive))
        right = dict(self._pair_vector(user, negative))
        positive_score = self._score_sparse(tuple(left.items()))
        negative_score = self._score_sparse(tuple(right.items()))
        difference = positive_score - negative_score
        gradient = (
            math.exp(-difference) / (1.0 + math.exp(-difference))
            if difference >= 0
            else 1.0 / (1.0 + math.exp(difference))
        )
        sums_left = [
            math.fsum(self._latent[index][factor] * value for index, value in left.items())
            for factor in range(self.factors)
        ]
        sums_right = [
            math.fsum(self._latent[index][factor] * value for index, value in right.items())
            for factor in range(self.factors)
        ]
        for index in sorted(left.keys() | right.keys()):
            x_left = left.get(index, 0.0)
            x_right = right.get(index, 0.0)
            old_linear = self._linear[index]
            self._linear[index] += self.learning_rate * (
                gradient * (x_left - x_right) - self.regularization * old_linear
            )
            for factor in range(self.factors):
                old = self._latent[index][factor]
                derivative = x_left * (sums_left[factor] - old * x_left) - x_right * (
                    sums_right[factor] - old * x_right
                )
                self._latent[index][factor] += self.learning_rate * (
                    gradient * derivative - self.regularization * old
                )
            if (
                not math.isfinite(self._linear[index])
                or abs(self._linear[index]) > MAX_ABS_WEIGHT
                or any(
                    not math.isfinite(value) or abs(value) > MAX_ABS_WEIGHT
                    for value in self._latent[index]
                )
            ):
                raise ValidationError(
                    "FM training diverged; reduce learning_rate or feature magnitude"
                )

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        if user_id not in self._user_vectors:
            return self._popularity_fallback(item_id)
        return self._score_sparse(self._pair_vector(user_id, item_id))

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        if self._pipeline is None or self._encoded is None:
            raise SerializationError("fitted FM is missing feature state")
        return make_envelope(
            self.model_type,
            {
                "factors": self.factors,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "seed": self.seed,
                "max_work_units": self.max_work_units,
            },
            self._base_state(),
            {
                "pipeline": self._pipeline.to_state(),
                "encoded": self._encoded.to_state(),
                "encoded_sha256": _feature_digest(self._encoded),
                "linear": self._linear.copy(),
                "latent": [row.copy() for row in self._latent],
                "updates": self._updates,
            },
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        expected_parameters = {
            "factors",
            "epochs",
            "learning_rate",
            "regularization",
            "seed",
            "max_work_units",
        }
        if set(parameters) != expected_parameters:
            raise SerializationError("FM parameters are malformed")
        if set(model) != {"pipeline", "encoded", "encoded_sha256", "linear", "latent", "updates"}:
            raise SerializationError("FM model fields are malformed")
        try:
            instance = cls(**parameters)
            if type(base.get("catalog")) is not list or len(base["catalog"]) > MAX_ITEMS:
                raise SerializationError("FM catalog limit exceeded")
            if type(base.get("users")) is not list or len(base["users"]) > MAX_USERS:
                raise SerializationError("FM user limit exceeded")
            for raw in base["catalog"]:
                _bounded_identifier(raw)
            for entry in base["users"]:
                if not isinstance(entry, Mapping) or "user_id" not in entry:
                    raise SerializationError("FM user state is malformed")
                _bounded_identifier(entry["user_id"])
            encoded_state = model["encoded"]
            if type(encoded_state) is not dict or type(encoded_state.get("rows")) is not list:
                raise SerializationError("FM encoded feature rows are malformed")
            if len(encoded_state["rows"]) > MAX_USERS + MAX_ITEMS:
                raise SerializationError("FM encoded feature row limit exceeded")
            instance._restore_base_state(base)
            if sum(len(seen) for seen in instance._seen.values()) > MAX_INTERACTIONS:
                raise SerializationError("FM interaction limit exceeded")
            pipeline = FittedFeaturePipeline.from_state(model["pipeline"])
            encoded = EncodedFeatureDataset.from_state(model["encoded"], limits=pipeline.limits)
            if (
                encoded.schema != pipeline.schema
                or encoded.pipeline_sha256 != pipeline.state_sha256
            ):
                raise SerializationError("FM encoded features do not match their pipeline")
            if pipeline.training_rows != len(encoded):
                raise SerializationError("FM training row count does not match encoded features")
            if model["encoded_sha256"] != _feature_digest(encoded):
                raise SerializationError("FM encoded feature digest mismatch")
            actual = {(row.source, row.key) for row in encoded}
            expected = {
                (FeatureSource.USER, key)
                for key in instance._seen
                if pipeline.schema.for_source(FeatureSource.USER)
            } | {
                (FeatureSource.ITEM, key)
                for key in instance._catalog
                if pipeline.schema.for_source(FeatureSource.ITEM)
            }
            if actual != expected or pipeline.schema.for_source(FeatureSource.INTERACTION):
                raise SerializationError("FM feature join is malformed")
            instance._prepare_vectors(
                tuple(sorted(instance._seen, key=stable_id_key)),
                instance._catalog,
                encoded,
                pipeline,
            )
            linear = model["linear"]
            latent = model["latent"]
            if type(linear) is not list or len(linear) != instance._dimensions:
                raise SerializationError("FM linear weight length is malformed")
            if type(latent) is not list or len(latent) != instance._dimensions:
                raise SerializationError("FM latent weight row count is malformed")
            checked_linear: list[float] = []
            checked_latent: list[list[float]] = []
            for raw, row in zip(linear, latent, strict=True):
                if (
                    type(raw) not in {int, float}
                    or not math.isfinite(safe_float(raw))
                    or abs(safe_float(raw)) > MAX_ABS_WEIGHT
                ):
                    raise SerializationError("FM linear weight is invalid")
                if type(row) is not list or len(row) != instance.factors:
                    raise SerializationError("FM latent weight width is malformed")
                if any(
                    type(value) not in {int, float}
                    or not math.isfinite(safe_float(value))
                    or abs(safe_float(value)) > MAX_ABS_WEIGHT
                    for value in row
                ):
                    raise SerializationError("FM latent weight is invalid")
                checked_linear.append(safe_float(raw))
                checked_latent.append([safe_float(value) for value in row])
            updates = model["updates"]
            expected_updates = instance.epochs * sum(
                len(seen) for seen in instance._seen.values() if len(seen) < len(instance._catalog)
            )
            if type(updates) is not int or updates != expected_updates or updates < 1:
                raise SerializationError("FM update count is malformed")
            all_positive = sum(len(seen) for seen in instance._seen.values())
            if (
                all_positive * instance.epochs * 2 * MAX_ACTIVE * instance.factors
                > instance.max_work_units
            ):
                raise SerializationError("FM training work bound is malformed")
            instance._pipeline = pipeline
            instance._encoded = encoded
            instance._linear = checked_linear
            instance._latent = checked_latent
            instance._updates = updates
            return instance
        except ValidationError as error:
            raise SerializationError(f"invalid FM state: {error}") from error
