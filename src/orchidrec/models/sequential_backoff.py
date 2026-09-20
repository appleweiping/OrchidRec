"""Bounded second-order transition ranking with support-weighted backoff."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Self, TypeVar

from orchidrec._numeric import safe_float
from orchidrec.data import (
    EntityId,
    Interaction,
    InteractionDataset,
    stable_id_key,
    validate_entity_id,
)
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.models.base import BaseRecommender, make_envelope, parse_envelope

MAX_INTERACTIONS = 50_000
MAX_CATALOG = 5_000
MAX_USERS = 50_000
MAX_TOTAL_WEIGHT = 1e15
Context = tuple[EntityId, EntityId]
_ContextKey = TypeVar("_ContextKey", EntityId, Context)


def _bounded_number(value: object, label: str, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be a finite number in [0, {upper}]")
    numeric = safe_float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= upper:
        raise ValidationError(f"{label} must be a finite number in [0, {upper}]")
    return numeric


def _add_count(
    matrix: dict[_ContextKey, dict[EntityId, float]],
    context: _ContextKey,
    target: EntityId,
    weight: float,
) -> None:
    row = matrix.setdefault(context, {})
    updated = row.get(target, 0.0) + weight
    if not math.isfinite(updated) or updated > MAX_TOTAL_WEIGHT:
        raise ValidationError("sequential transition count exceeded finite bound")
    row[target] = updated


def _read_row(raw: object, catalog: set[EntityId], label: str) -> dict[EntityId, float]:
    if not isinstance(raw, list) or len(raw) > len(catalog):
        raise SerializationError(f"{label} must be a bounded array")
    row: dict[EntityId, float] = {}
    for entry in raw:
        if not isinstance(entry, Mapping) or set(entry) != {"item_id", "weight"}:
            raise SerializationError(f"{label} entry is malformed")
        try:
            item_id = validate_entity_id(entry["item_id"], f"{label} item ID")
            weight = _bounded_number(entry["weight"], f"{label} weight", MAX_TOTAL_WEIGHT)
        except ValidationError as error:
            raise SerializationError(str(error)) from error
        if item_id not in catalog or item_id in row or weight <= 0:
            raise SerializationError(f"{label} target or weight is invalid")
        row[item_id] = weight
    if list(row) != sorted(row, key=stable_id_key):
        raise SerializationError(f"{label} targets are not stably ordered")
    try:
        total = math.fsum(row.values())
    except (OverflowError, ValueError) as error:
        raise SerializationError(f"{label} total is not finite") from error
    if total > MAX_TOTAL_WEIGHT:
        raise SerializationError(f"{label} total exceeds bound")
    return row


def _row_state(row: Mapping[EntityId, float]) -> list[dict[str, EntityId | float]]:
    return [
        {"item_id": target, "weight": weight}
        for target, weight in sorted(row.items(), key=lambda pair: stable_id_key(pair[0]))
    ]


class SequentialBackoff(BaseRecommender):
    """Interpolate second-order and first-order transitions with popularity fallback.

    A context with support ``c`` receives second-order weight ``c/(c+alpha)``;
    missing contexts back off entirely to the first-order distribution. Every
    event is ordered by timestamp and then its original input ordinal.
    """

    model_type = "sequential_backoff"

    def __init__(
        self,
        *,
        weighted: bool = True,
        backoff_strength: float = 1.0,
        popularity_mix: float = 0.05,
        max_interactions: int = MAX_INTERACTIONS,
    ) -> None:
        super().__init__()
        if type(weighted) is not bool:
            raise ValidationError("weighted must be a boolean")
        if type(max_interactions) is not int or not 1 <= max_interactions <= MAX_INTERACTIONS:
            raise ValidationError(f"max_interactions must be in [1, {MAX_INTERACTIONS}]")
        self.weighted = weighted
        self.backoff_strength = _bounded_number(backoff_strength, "backoff_strength", 1e6)
        self.popularity_mix = _bounded_number(popularity_mix, "popularity_mix", 1.0)
        self.max_interactions = max_interactions
        self._first: dict[EntityId, dict[EntityId, float]] = {}
        self._second: dict[Context, dict[EntityId, float]] = {}
        self._last_two: dict[EntityId, tuple[EntityId, ...]] = {}
        self._first_totals: dict[EntityId, float] = {}
        self._second_totals: dict[Context, float] = {}
        self._max_popularity = 0.0

    def fit(self, dataset: InteractionDataset) -> Self:
        """Reject oversized and non-temporal input before building common state."""

        if not isinstance(dataset, InteractionDataset):
            raise ValidationError("dataset must be an InteractionDataset")
        if len(dataset) > self.max_interactions:
            raise ValidationError("SequentialBackoff interaction limit exceeded")
        total_weight = 0.0
        for event in dataset:
            if event.timestamp is None:
                raise ValidationError("SequentialBackoff requires a timestamp on every interaction")
            if event.value > MAX_TOTAL_WEIGHT:
                raise ValidationError("SequentialBackoff event value exceeded finite bound")
            total_weight += event.value
            if not math.isfinite(total_weight) or total_weight > MAX_TOTAL_WEIGHT:
                raise ValidationError("SequentialBackoff total event weight exceeded finite bound")
        return super().fit(dataset)

    def _fit_model(self, dataset: InteractionDataset) -> None:
        if len(self._catalog) > MAX_CATALOG or len(self._seen) > MAX_USERS:
            raise ValidationError("SequentialBackoff catalog or user limit exceeded")
        grouped: dict[EntityId, list[tuple[int, Interaction]]] = defaultdict(list)
        for ordinal, event in enumerate(dataset):
            grouped[event.user_id].append((ordinal, event))
        first: dict[EntityId, dict[EntityId, float]] = {}
        second: dict[Context, dict[EntityId, float]] = {}
        last_two: dict[EntityId, tuple[EntityId, ...]] = {}
        for user_id in sorted(grouped, key=stable_id_key):
            ordered = [
                event
                for _, event in sorted(grouped[user_id], key=lambda row: (row[1].timestamp, row[0]))
            ]
            last_two[user_id] = tuple(event.item_id for event in ordered[-2:])
            for position in range(1, len(ordered)):
                target = ordered[position]
                weight = target.value if self.weighted else 1.0
                _add_count(first, ordered[position - 1].item_id, target.item_id, weight)
                if position >= 2:
                    context = (ordered[position - 2].item_id, ordered[position - 1].item_id)
                    _add_count(second, context, target.item_id, weight)
        self._first = first
        self._second = second
        self._last_two = last_two
        self._first_totals = {source: math.fsum(row.values()) for source, row in first.items()}
        self._second_totals = {context: math.fsum(row.values()) for context, row in second.items()}
        self._max_popularity = max(self._popularity.values(), default=0.0)

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        fallback = self._popularity.get(item_id, 0.0) / self._max_popularity
        history = self._last_two.get(user_id, ())
        if not history:
            return fallback
        first = self._first.get(history[-1], {})
        first_total = self._first_totals.get(history[-1], 0.0)
        first_probability = first.get(item_id, 0.0) / first_total if first_total else fallback
        second = self._second.get((history[-2], history[-1]), {}) if len(history) == 2 else {}
        support = (
            self._second_totals.get((history[-2], history[-1]), 0.0) if len(history) == 2 else 0.0
        )
        if support:
            second_probability = second.get(item_id, 0.0) / support
            confidence = support / (support + self.backoff_strength)
            sequence = confidence * second_probability + (1 - confidence) * first_probability
        else:
            sequence = first_probability
        return (1 - self.popularity_mix) * sequence + self.popularity_mix * fallback

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        return make_envelope(
            self.model_type,
            {
                "weighted": self.weighted,
                "backoff_strength": self.backoff_strength,
                "popularity_mix": self.popularity_mix,
                "max_interactions": self.max_interactions,
            },
            self._base_state(),
            {
                "first_order": [
                    _row_state(self._first.get(source, {})) for source in self._catalog
                ],
                "second_order": [
                    {
                        "previous_item_id": previous,
                        "latest_item_id": latest,
                        "targets": _row_state(row),
                    }
                    for (previous, latest), row in sorted(
                        self._second.items(),
                        key=lambda entry: (stable_id_key(entry[0][0]), stable_id_key(entry[0][1])),
                    )
                ],
                "last_two": [
                    {"user_id": user, "items": list(self._last_two[user])}
                    for user in sorted(self._last_two, key=stable_id_key)
                ],
            },
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        if set(parameters) != {
            "weighted",
            "backoff_strength",
            "popularity_mix",
            "max_interactions",
        }:
            raise SerializationError("SequentialBackoff parameters are malformed")
        try:
            instance = cls(**parameters)
        except (TypeError, ValidationError) as error:
            raise SerializationError(f"invalid SequentialBackoff parameters: {error}") from error
        instance._restore_base_state(base)
        if not instance._catalog or not instance._seen:
            raise SerializationError("SequentialBackoff state must contain fitted items and users")
        if len(instance._catalog) > MAX_CATALOG or len(instance._seen) > MAX_USERS:
            raise SerializationError("SequentialBackoff catalog or users exceed limits")
        try:
            popularity_total = math.fsum(instance._popularity.values())
        except (OverflowError, ValueError) as error:
            raise SerializationError("SequentialBackoff popularity total is not finite") from error
        if popularity_total > MAX_TOTAL_WEIGHT:
            raise SerializationError("SequentialBackoff popularity total exceeds finite bound")
        if set(model) != {"first_order", "second_order", "last_two"}:
            raise SerializationError("SequentialBackoff model state is malformed")
        raw_first = model["first_order"]
        raw_second = model["second_order"]
        raw_history = model["last_two"]
        if (
            not isinstance(raw_first, list)
            or len(raw_first) != len(instance._catalog)
            or not isinstance(raw_second, list)
            or len(raw_second) > instance.max_interactions
            or not isinstance(raw_history, list)
            or len(raw_history) != len(instance._seen)
        ):
            raise SerializationError("SequentialBackoff state arrays are malformed")
        catalog = set(instance._catalog)
        first: dict[EntityId, dict[EntityId, float]] = {}
        first_cells = 0
        for source, raw_row in zip(instance._catalog, raw_first, strict=True):
            row = _read_row(raw_row, catalog, "first-order")
            first_cells += len(row)
            if first_cells > instance.max_interactions:
                raise SerializationError("first-order cells exceed interaction limit")
            if row:
                first[source] = row
        second: dict[Context, dict[EntityId, float]] = {}
        second_cells = 0
        for entry in raw_second:
            if not isinstance(entry, Mapping) or set(entry) != {
                "previous_item_id",
                "latest_item_id",
                "targets",
            }:
                raise SerializationError("second-order context is malformed")
            try:
                previous = validate_entity_id(entry["previous_item_id"], "previous_item_id")
                latest = validate_entity_id(entry["latest_item_id"], "latest_item_id")
            except ValidationError as error:
                raise SerializationError(str(error)) from error
            context = (previous, latest)
            if previous not in catalog or latest not in catalog or context in second:
                raise SerializationError("second-order context is invalid")
            row = _read_row(entry["targets"], catalog, "second-order")
            if not row:
                raise SerializationError("second-order context must have targets")
            second_cells += len(row)
            if second_cells > instance.max_interactions:
                raise SerializationError("second-order cells exceed interaction limit")
            second[context] = row
        if list(second) != sorted(
            second, key=lambda pair: (stable_id_key(pair[0]), stable_id_key(pair[1]))
        ):
            raise SerializationError("second-order contexts are not stably ordered")
        second_by_latest: dict[EntityId, dict[EntityId, float]] = {}
        try:
            for (_, latest), row in second.items():
                for target, weight in row.items():
                    _add_count(second_by_latest, latest, target, weight)
        except ValidationError as error:
            raise SerializationError("second-order aggregate exceeds finite bound") from error
        for latest, row in second_by_latest.items():
            for target, weight in row.items():
                first_weight = first.get(latest, {}).get(target, 0.0)
                if weight > first_weight and not math.isclose(weight, first_weight, rel_tol=1e-12):
                    raise SerializationError("second-order count exceeds first-order transition")
        history: dict[EntityId, tuple[EntityId, ...]] = {}
        for entry in raw_history:
            if not isinstance(entry, Mapping) or set(entry) != {"user_id", "items"}:
                raise SerializationError("last-two user entry is malformed")
            try:
                user = validate_entity_id(entry["user_id"], "user_id")
            except ValidationError as error:
                raise SerializationError(str(error)) from error
            items = entry["items"]
            if not isinstance(items, list) or not 1 <= len(items) <= 2:
                raise SerializationError("last-two items must have length one or two")
            try:
                validated = tuple(validate_entity_id(item, "last-two item") for item in items)
            except ValidationError as error:
                raise SerializationError(str(error)) from error
            if (
                user in history
                or user not in instance._seen
                or any(item not in instance._seen[user] for item in validated)
            ):
                raise SerializationError("last-two user history is invalid")
            history[user] = validated
        if list(history) != sorted(instance._seen, key=stable_id_key):
            raise SerializationError("last-two users are not stably ordered")
        instance._first = first
        instance._second = second
        instance._last_two = history
        instance._first_totals = {source: math.fsum(row.values()) for source, row in first.items()}
        instance._second_totals = {
            context: math.fsum(row.values()) for context, row in second.items()
        }
        instance._max_popularity = max(instance._popularity.values(), default=0.0)
        return instance
