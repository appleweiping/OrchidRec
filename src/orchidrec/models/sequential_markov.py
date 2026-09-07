"""A deterministic first-order sequential recommender."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from itertools import pairwise
from typing import Any, Self

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


class SequentialMarkov(BaseRecommender):
    """Rank from the latest item using fitted first-order transition probabilities.

    Events are ordered per user by timestamp and then by their declared input
    order.  The latter is an explicit deterministic tie rule.  Every event must
    have a timestamp; silently treating an unordered implicit-feedback table as
    a sequence would create a model with invented chronology.
    """

    model_type = "sequential_markov"

    def __init__(self, *, weighted: bool = True, popularity_mix: float = 0.05) -> None:
        super().__init__()
        if not isinstance(weighted, bool):
            raise ValidationError("weighted must be a boolean")
        if isinstance(popularity_mix, bool) or not isinstance(popularity_mix, (int, float)):
            raise ValidationError("popularity_mix must be a finite number between 0 and 1")
        numeric_mix = safe_float(popularity_mix)
        if not math.isfinite(numeric_mix) or not 0.0 <= numeric_mix <= 1.0:
            raise ValidationError("popularity_mix must be a finite number between 0 and 1")
        self.weighted = weighted
        self.popularity_mix = numeric_mix
        self._transitions: dict[EntityId, dict[EntityId, float]] = {}
        self._last_items: dict[EntityId, EntityId] = {}

    def _fit_model(self, dataset: InteractionDataset) -> None:
        grouped: dict[EntityId, list[tuple[int, Interaction]]] = defaultdict(list)
        for ordinal, event in enumerate(dataset):
            if event.timestamp is None:
                raise ValidationError("SequentialMarkov requires a timestamp on every interaction")
            grouped[event.user_id].append((ordinal, event))
        transitions: dict[EntityId, dict[EntityId, float]] = defaultdict(lambda: defaultdict(float))
        last_items: dict[EntityId, EntityId] = {}
        for user_id, entries in grouped.items():
            ordered = [
                event for _, event in sorted(entries, key=lambda pair: (pair[1].timestamp, pair[0]))
            ]
            last_items[user_id] = ordered[-1].item_id
            for source, target in pairwise(ordered):
                increment = target.value if self.weighted else 1.0
                updated = transitions[source.item_id][target.item_id] + increment
                if not math.isfinite(updated):
                    raise ValidationError("SequentialMarkov transition counts overflowed")
                transitions[source.item_id][target.item_id] = updated
        self._transitions = {source: dict(row) for source, row in transitions.items()}
        self._last_items = last_items

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        fallback = self._popularity_fallback(item_id)
        source = self._last_items.get(user_id)
        row = self._transitions.get(source, {}) if source is not None else {}
        total = math.fsum(row.values())
        if total <= 0.0:
            return fallback
        transition_probability = row.get(item_id, 0.0) / total
        return (1.0 - self.popularity_mix) * transition_probability + self.popularity_mix * fallback

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        return make_envelope(
            self.model_type,
            {"weighted": self.weighted, "popularity_mix": self.popularity_mix},
            self._base_state(),
            {
                "transitions": [
                    [
                        {"item_id": target, "weight": weight}
                        for target, weight in sorted(
                            self._transitions.get(source, {}).items(),
                            key=lambda pair: stable_id_key(pair[0]),
                        )
                    ]
                    for source in self._catalog
                ],
                "last_items": [
                    {"user_id": user_id, "item_id": self._last_items[user_id]}
                    for user_id in sorted(self._last_items, key=stable_id_key)
                ],
            },
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        if set(parameters) != {"weighted", "popularity_mix"}:
            raise SerializationError("SequentialMarkov parameters are malformed")
        try:
            instance = cls(
                weighted=parameters["weighted"], popularity_mix=parameters["popularity_mix"]
            )
        except ValidationError as exc:
            raise SerializationError(f"invalid SequentialMarkov parameters: {exc}") from exc
        instance._restore_base_state(base)
        if set(model) != {"transitions", "last_items"}:
            raise SerializationError("SequentialMarkov model state is malformed")
        raw_transitions = model["transitions"]
        raw_last = model["last_items"]
        if (
            not isinstance(raw_transitions, list)
            or len(raw_transitions) != len(instance._catalog)
            or not isinstance(raw_last, list)
        ):
            raise SerializationError("SequentialMarkov state arrays are malformed")
        catalog = set(instance._catalog)
        transitions: dict[EntityId, dict[EntityId, float]] = {}
        for source, row in zip(instance._catalog, raw_transitions, strict=True):
            if not isinstance(row, list):
                raise SerializationError("SequentialMarkov transition row must be a list")
            restored_row: dict[EntityId, float] = {}
            for entry in row:
                if not isinstance(entry, Mapping) or set(entry) != {"item_id", "weight"}:
                    raise SerializationError("SequentialMarkov transition entry is malformed")
                try:
                    target = validate_entity_id(entry["item_id"], "transition item ID")
                except ValidationError as exc:
                    raise SerializationError(str(exc)) from exc
                raw_weight = entry["weight"]
                if target not in catalog or target in restored_row:
                    raise SerializationError(
                        "SequentialMarkov transition references an invalid item"
                    )
                if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                    raise SerializationError("SequentialMarkov transition weight must be positive")
                weight = safe_float(raw_weight)
                if not math.isfinite(weight) or weight <= 0.0:
                    raise SerializationError("SequentialMarkov transition weight must be positive")
                restored_row[target] = weight
            if list(restored_row) != sorted(restored_row, key=stable_id_key):
                raise SerializationError("SequentialMarkov transition row is not stably ordered")
            if restored_row:
                transitions[source] = restored_row
        last_items: dict[EntityId, EntityId] = {}
        for entry in raw_last:
            if not isinstance(entry, Mapping) or set(entry) != {"user_id", "item_id"}:
                raise SerializationError("SequentialMarkov last-item entry is malformed")
            try:
                user_id = validate_entity_id(entry["user_id"], "user_id")
                item_id = validate_entity_id(entry["item_id"], "last item ID")
            except ValidationError as exc:
                raise SerializationError(str(exc)) from exc
            if user_id in last_items or user_id not in instance._seen or item_id not in catalog:
                raise SerializationError("SequentialMarkov last-item entry is invalid")
            if item_id not in instance._seen[user_id]:
                raise SerializationError("SequentialMarkov last item is absent from user history")
            last_items[user_id] = item_id
        if list(last_items) != sorted(instance._seen, key=stable_id_key) or set(last_items) != set(
            instance._seen
        ):
            raise SerializationError("SequentialMarkov users do not match base state")
        instance._transitions = transitions
        instance._last_items = last_items
        return instance
