"""Bounded, deterministic nonnegative SLIM elastic-net for implicit feedback."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Self, cast

from orchidrec._json import MAX_JSON_INTEGER_DIGITS
from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, InteractionDataset, stable_id_key, validate_entity_id
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.models.base import BaseRecommender, make_envelope, parse_envelope
from orchidrec.models.io import MAX_MODEL_FILE_BYTES

MAX_SLIM_ITEMS = 256
MAX_SLIM_USERS = 32_768
MAX_SLIM_INTERACTIONS = 2_000_000
MAX_SLIM_SWEEPS = 1_000
MAX_SLIM_WORK_UNITS = 1_000_000_000
MAX_SLIM_IDENTIFIER_BYTES = 16_384
MAX_SLIM_TOTAL_IDENTIFIER_BYTES = 32 * 1024 * 1024


def _positive_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValidationError(f"{name} must be an integer between 1 and {maximum}")
    return value


def _penalty(value: object, name: str, *, positive: bool) -> float:
    if type(value) not in {int, float}:
        raise ValidationError(f"{name} must be a finite bounded number")
    number = safe_float(cast(int | float, value))
    if not math.isfinite(number) or number > 1e12 or (number < 1e-8 if positive else number < 0):
        raise ValidationError(f"{name} must be a finite bounded number")
    return number


def _identifier_bytes(value: object) -> int:
    try:
        identifier = validate_entity_id(value)
    except ValidationError as error:
        raise SerializationError(str(error)) from error
    if type(identifier) is int:
        try:
            digits = len(str(abs(identifier)))
        except ValueError as error:
            raise SerializationError("integer identifier exceeds JSON digit limit") from error
        if digits > MAX_JSON_INTEGER_DIGITS:
            raise SerializationError("integer identifier exceeds JSON digit limit")
        return max(1, (abs(identifier).bit_length() + 7) // 8)
    try:
        size = len(cast(str, identifier).encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise SerializationError("identifier is not valid Unicode scalar text") from error
    if size > MAX_SLIM_IDENTIFIER_BYTES:
        raise SerializationError("identifier exceeds UTF-8 byte limit")
    return size


def _work_bound(seen: Mapping[EntityId, frozenset[EntityId]], items: int, sweeps: int) -> int:
    return sum(len(profile) ** 2 for profile in seen.values()) + 2 * items**3 * sweeps


def _preflight_base(
    base: Mapping[str, Any],
    *,
    max_items: int,
    max_interactions: int,
    max_work: int,
    max_sweeps: int,
) -> tuple[int, int]:
    """Reject oversized/corrupt raw state before generic restoration copies it."""

    if set(base) != {"catalog", "popularity", "users"}:
        raise SerializationError("SLIM base state has invalid fields")
    catalog, popularity, users = base["catalog"], base["popularity"], base["users"]
    if type(catalog) is not list or type(popularity) is not list or type(users) is not list:
        raise SerializationError("SLIM base arrays are malformed")
    items = len(catalog)
    if not 1 <= items <= max_items or len(popularity) != items:
        raise SerializationError("SLIM catalog size or popularity array is malformed")
    if not 1 <= len(users) <= MAX_SLIM_USERS:
        raise SerializationError("SLIM user limit exceeded")
    work = 2 * items**3 * max_sweeps
    if work > max_work:
        raise SerializationError("SLIM work limit exceeded")
    identifier_bytes = 0
    catalog_ids: set[EntityId] = set()
    for raw in catalog:
        identifier_bytes += _identifier_bytes(raw)
        catalog_ids.add(validate_entity_id(raw))
    if len(catalog_ids) != items:
        raise SerializationError("SLIM catalog contains duplicate IDs")
    interactions = 0
    for entry in users:
        if not isinstance(entry, Mapping) or set(entry) != {"user_id", "seen"}:
            raise SerializationError("SLIM user state is malformed")
        seen = entry["seen"]
        if type(seen) is not list or not 1 <= len(seen) <= items:
            raise SerializationError("SLIM user history is malformed")
        identifier_bytes += _identifier_bytes(entry["user_id"])
        interactions += len(seen)
        work += len(seen) ** 2
        if interactions > max_interactions or work > max_work:
            raise SerializationError("SLIM interaction or work limit exceeded")
        history: set[EntityId] = set()
        for raw in seen:
            identifier_bytes += _identifier_bytes(raw)
            identifier = validate_entity_id(raw)
            if identifier not in catalog_ids or identifier in history:
                raise SerializationError("SLIM user history contains an invalid item")
            history.add(identifier)
        if identifier_bytes > MAX_SLIM_TOTAL_IDENTIFIER_BYTES:
            raise SerializationError("SLIM aggregate identifier byte limit exceeded")
    return interactions, work


def _check_file_size(state: Mapping[str, Any]) -> None:
    encoder = json.JSONEncoder(indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    total = 1
    try:
        for fragment in encoder.iterencode(state):
            total += len(fragment.encode("utf-8", errors="strict"))
            if total > MAX_MODEL_FILE_BYTES:
                raise ValidationError("SLIM state exceeds model-file byte limit")
    except (TypeError, UnicodeError) as error:
        raise ValidationError("SLIM state is not strict JSON") from error


def _gram_matrix(
    seen: Mapping[EntityId, frozenset[EntityId]], catalog: Sequence[EntityId]
) -> list[list[float]]:
    indices = {item: index for index, item in enumerate(catalog)}
    gram = [[0.0] * len(catalog) for _ in catalog]
    for user in sorted(seen, key=stable_id_key):
        profile = sorted(indices[item] for item in seen[user])
        for source in profile:
            for target in profile:
                gram[source][target] += 1.0
    return gram


def _solve_columns(
    gram: Sequence[Sequence[float]],
    *,
    l1: float,
    l2: float,
    max_sweeps: int,
    tolerance: float,
) -> tuple[list[list[float]], list[int], float]:
    """Cyclic exact coordinate updates with checked nonnegative KKT residuals."""

    items = len(gram)
    weights = [[0.0] * items for _ in range(items)]
    sweeps: list[int] = []
    largest_kkt = 0.0
    for target in range(items):
        column = [0.0] * items
        # residual[i] = X_i.T (X_target - X @ column)
        residual = [gram[source][target] for source in range(items)]
        for sweep in range(1, max_sweeps + 1):
            for source in range(items):
                if source == target:
                    continue
                old = column[source]
                numerator = residual[source] + gram[source][source] * old - l1
                updated = max(0.0, numerator / (gram[source][source] + l2))
                if not math.isfinite(updated):
                    raise ValidationError("SLIM coordinate update is non-finite")
                delta = updated - old
                if delta:
                    column[source] = updated
                    for row in range(items):
                        residual[row] -= gram[row][source] * delta
            # Recompute against the original Gram matrix before deciding
            # convergence; incremental updates can accumulate cancellation.
            for source in range(items):
                residual[source] = gram[source][target] - math.fsum(
                    gram[source][predictor] * column[predictor]
                    for predictor in range(items)
                    if predictor != target
                )
            kkt = 0.0
            for source in range(items):
                if source == target:
                    continue
                gradient = l1 + l2 * column[source] - residual[source]
                if not math.isfinite(gradient):
                    raise ValidationError("SLIM KKT gradient is non-finite")
                violation = abs(gradient) if column[source] > 0 else max(0.0, -gradient)
                kkt = max(kkt, violation / max(1.0, gram[source][target]))
            if kkt <= tolerance:
                sweeps.append(sweep)
                largest_kkt = max(largest_kkt, kkt)
                break
        else:
            raise ValidationError(
                f"SLIM target {target} did not satisfy KKT tolerance in {max_sweeps} sweeps"
            )
        for source in range(items):
            weights[source][target] = column[source]
    return weights, sweeps, largest_kkt


class SLIMElastic(BaseRecommender):
    """Nonnegative, zero-diagonal sparse linear recommender on binary histories.

    Each target column solves 1/2 ||X_j - X w_j||² + l1 ||w_j||_1
    + l2/2 ||w_j||² with w_j >= 0 and w_j[j] = 0. Event values and
    duplicates do not alter X; they only affect unknown-user popularity.
    """

    model_type = "slim_elastic"

    def __init__(
        self,
        *,
        l1: float = 0.1,
        l2: float = 0.1,
        max_sweeps: int = 100,
        tolerance: float = 1e-7,
        max_items: int = 96,
        max_interactions: int = MAX_SLIM_INTERACTIONS,
        max_work_units: int = 200_000_000,
    ) -> None:
        super().__init__()
        self.l1 = _penalty(l1, "l1", positive=False)
        self.l2 = _penalty(l2, "l2", positive=True)
        self.max_sweeps = _positive_int(max_sweeps, "max_sweeps", MAX_SLIM_SWEEPS)
        self.tolerance = _penalty(tolerance, "tolerance", positive=True)
        if self.tolerance > 0.1:
            raise ValidationError("tolerance must not exceed 0.1")
        self.max_items = _positive_int(max_items, "max_items", MAX_SLIM_ITEMS)
        self.max_interactions = _positive_int(
            max_interactions, "max_interactions", MAX_SLIM_INTERACTIONS
        )
        self.max_work_units = _positive_int(max_work_units, "max_work_units", MAX_SLIM_WORK_UNITS)
        self._weights: list[list[float]] = []
        self._indices: dict[EntityId, int] = {}
        self._sweeps: list[int] = []
        self._largest_kkt = 0.0
        self._training_interactions = 0
        self._work_units = 0

    @property
    def largest_kkt(self) -> float:
        self._require_fitted()
        return self._largest_kkt

    @property
    def work_units(self) -> int:
        self._require_fitted()
        return self._work_units

    def _parameters(self) -> dict[str, float | int]:
        return {
            "l1": self.l1,
            "l2": self.l2,
            "max_sweeps": self.max_sweeps,
            "tolerance": self.tolerance,
            "max_items": self.max_items,
            "max_interactions": self.max_interactions,
            "max_work_units": self.max_work_units,
        }

    def _model_state(self) -> dict[str, Any]:
        return {
            "weights": [row.copy() for row in self._weights],
            "sweeps": self._sweeps.copy(),
            "largest_kkt": self._largest_kkt,
            "training_interactions": self._training_interactions,
            "work_units": self._work_units,
        }

    def fit(self, dataset: InteractionDataset) -> Self:
        """Reject oversized raw input before the shared history/popularity pass."""

        if not isinstance(dataset, InteractionDataset):
            raise ValidationError("dataset must be an InteractionDataset")
        if len(dataset) > self.max_interactions:
            raise ValidationError("SLIM raw interaction limit exceeded")
        return super().fit(dataset)

    def _fit_model(self, dataset: InteractionDataset) -> None:
        items = len(self._catalog)
        if items > self.max_items:
            raise ValidationError("SLIM catalog item limit exceeded")
        work = _work_bound(self._seen, items, self.max_sweeps)
        if work > self.max_work_units:
            raise ValidationError("SLIM work unit limit exceeded")
        base = self._snapshot_base_state()
        try:
            unique, checked_work = _preflight_base(
                base,
                max_items=self.max_items,
                max_interactions=self.max_interactions,
                max_work=self.max_work_units,
                max_sweeps=self.max_sweeps,
            )
        except SerializationError as error:
            raise ValidationError(f"SLIM training state cannot be persisted: {error}") from error
        if unique > len(dataset) or checked_work != work:
            raise ValidationError("SLIM training state is inconsistent")
        weights, sweeps, kkt = _solve_columns(
            _gram_matrix(self._seen, self._catalog),
            l1=self.l1,
            l2=self.l2,
            max_sweeps=self.max_sweeps,
            tolerance=self.tolerance,
        )
        self._weights = weights
        self._indices = {item: index for index, item in enumerate(self._catalog)}
        self._sweeps = sweeps
        self._largest_kkt = kkt
        self._training_interactions = len(dataset)
        self._work_units = work
        _check_file_size(
            make_envelope(self.model_type, self._parameters(), base, self._model_state())
        )

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        history = self._seen.get(user_id)
        if history is None:
            return self._popularity_fallback(item_id)
        target = self._indices[item_id]
        return math.fsum(self._weights[self._indices[item]][target] for item in history)

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        return make_envelope(
            self.model_type, self._parameters(), self._base_state(), self._model_state()
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        if set(parameters) != {
            "l1",
            "l2",
            "max_sweeps",
            "tolerance",
            "max_items",
            "max_interactions",
            "max_work_units",
        }:
            raise SerializationError("SLIM parameters are malformed")
        try:
            instance = cls(**dict(parameters))
        except (TypeError, ValidationError) as error:
            raise SerializationError(f"invalid SLIM parameters: {error}") from error
        if set(model) != {
            "weights",
            "sweeps",
            "largest_kkt",
            "training_interactions",
            "work_units",
        }:
            raise SerializationError("SLIM model state is malformed")
        unique, work = _preflight_base(
            base,
            max_items=instance.max_items,
            max_interactions=instance.max_interactions,
            max_work=instance.max_work_units,
            max_sweeps=instance.max_sweeps,
        )
        training_interactions = model["training_interactions"]
        if (
            type(training_interactions) is not int
            or not unique <= training_interactions <= instance.max_interactions
            or type(model["work_units"]) is not int
            or model["work_units"] != work
        ):
            raise SerializationError("SLIM interaction or work count is malformed")
        weights, sweeps = model["weights"], model["sweeps"]
        items = len(base["catalog"])
        if type(weights) is not list or len(weights) != items:
            raise SerializationError("SLIM weight rows are malformed")
        for row_index, row in enumerate(weights):
            if type(row) is not list or len(row) != items:
                raise SerializationError("SLIM weight columns are malformed")
            for column_index, raw in enumerate(row):
                if type(raw) not in {int, float}:
                    raise SerializationError("SLIM weights must be finite nonnegative numbers")
                value = safe_float(raw)
                if (
                    not math.isfinite(value)
                    or value < 0
                    or (row_index == column_index and value != 0)
                ):
                    raise SerializationError(
                        "SLIM weights must be finite, nonnegative, zero-diagonal"
                    )
        if (
            type(sweeps) is not list
            or len(sweeps) != items
            or any(
                type(value) is not int or not 1 <= value <= instance.max_sweeps for value in sweeps
            )
        ):
            raise SerializationError("SLIM sweep diagnostics are malformed")
        raw_kkt = model["largest_kkt"]
        if type(raw_kkt) not in {int, float} or not math.isfinite(safe_float(raw_kkt)):
            raise SerializationError("SLIM KKT diagnostic is malformed")
        if not 0 <= safe_float(raw_kkt) <= instance.tolerance:
            raise SerializationError("SLIM KKT diagnostic is outside tolerance")
        instance._restore_base_state(base)
        try:
            canonical, canonical_sweeps, canonical_kkt = _solve_columns(
                _gram_matrix(instance._seen, instance._catalog),
                l1=instance.l1,
                l2=instance.l2,
                max_sweeps=instance.max_sweeps,
                tolerance=instance.tolerance,
            )
        except ValidationError as error:
            raise SerializationError(f"invalid canonical SLIM state: {error}") from error
        if weights != canonical or sweeps != canonical_sweeps or raw_kkt != canonical_kkt:
            raise SerializationError("SLIM model state does not match canonical base state")
        instance._weights = canonical
        instance._indices = {item: index for index, item in enumerate(instance._catalog)}
        instance._sweeps = canonical_sweeps
        instance._largest_kkt = canonical_kkt
        instance._training_interactions = training_interactions
        instance._work_units = work
        return instance
