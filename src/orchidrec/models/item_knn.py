"""Item-item collaborative filtering implemented with the standard library."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Self

from orchidrec._numeric import safe_float
from orchidrec.data import (
    EntityId,
    InteractionDataset,
    stable_id_key,
    validate_entity_id,
)
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.models.base import BaseRecommender, make_envelope, parse_envelope


class ItemKNN(BaseRecommender):
    """Cosine item-neighborhood recommender over weighted user histories."""

    model_type = "item_knn"

    def __init__(self, *, neighbors: int = 40, shrinkage: float = 10.0) -> None:
        super().__init__()
        if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors <= 0:
            raise ValidationError("neighbors must be a positive integer")
        if isinstance(shrinkage, bool) or not isinstance(shrinkage, (int, float)):
            raise ValidationError("shrinkage must be a finite non-negative number")
        numeric_shrinkage = safe_float(shrinkage)
        if not math.isfinite(numeric_shrinkage) or numeric_shrinkage < 0:
            raise ValidationError("shrinkage must be a finite non-negative number")
        self.neighbors = neighbors
        self.shrinkage = numeric_shrinkage
        self._similarities: dict[EntityId, dict[EntityId, float]] = {}
        self._user_values: dict[EntityId, dict[EntityId, float]] = {}

    def _fit_model(self, dataset: InteractionDataset) -> None:
        user_values: dict[EntityId, dict[EntityId, float]] = defaultdict(lambda: defaultdict(float))
        for event in dataset:
            user_values[event.user_id][event.item_id] += event.value
        norms_squared: dict[EntityId, float] = defaultdict(float)
        dots: dict[tuple[EntityId, EntityId], float] = defaultdict(float)
        for values in user_values.values():
            ordered = sorted(values, key=stable_id_key)
            for item_id in ordered:
                updated_norm = norms_squared[item_id] + values[item_id] * values[item_id]
                if not math.isfinite(updated_norm):
                    raise ValidationError(
                        "ItemKNN numeric overflow while computing item norms"
                    )
                norms_squared[item_id] = updated_norm
            for left_index, left_id in enumerate(ordered):
                for right_id in ordered[left_index + 1 :]:
                    pair = (left_id, right_id)
                    updated_dot = dots[pair] + values[left_id] * values[right_id]
                    if not math.isfinite(updated_dot):
                        raise ValidationError(
                            "ItemKNN numeric overflow while computing item similarities"
                        )
                    dots[pair] = updated_dot
        similarities: dict[EntityId, dict[EntityId, float]] = {item_id: {} for item_id in self._catalog}
        for (left_id, right_id), dot in dots.items():
            denominator = (
                math.sqrt(norms_squared[left_id]) * math.sqrt(norms_squared[right_id])
                + self.shrinkage
            )
            if not math.isfinite(denominator):
                raise ValidationError(
                    "ItemKNN numeric overflow in the similarity denominator"
                )
            similarity = dot / denominator if denominator > 0 else 0.0
            if similarity > 0:
                similarities[left_id][right_id] = similarity
                similarities[right_id][left_id] = similarity
        for item_id, neighbors in similarities.items():
            ranked = sorted(neighbors.items(), key=lambda pair: (-pair[1], stable_id_key(pair[0])))
            similarities[item_id] = dict(ranked[: self.neighbors])
        self._similarities = similarities
        self._user_values = {user_id: dict(values) for user_id, values in user_values.items()}

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        history = self._user_values.get(user_id)
        fallback = self._popularity_fallback(item_id)
        if not history:
            return fallback
        similarities = self._similarities.get(item_id, {})
        weighted_sum = 0.0
        matched = False
        for history_item, value in history.items():
            similarity = similarities.get(history_item, 0.0)
            weighted_sum += similarity * value
            matched = matched or similarity != 0.0
        if not matched:
            return fallback * 1e-9
        return weighted_sum + fallback * 1e-9

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        users = sorted(self._user_values, key=stable_id_key)
        return make_envelope(
            self.model_type,
            {"neighbors": self.neighbors, "shrinkage": self.shrinkage},
            self._base_state(),
            {
                "similarities": [
                    [
                        {"item_id": neighbor_id, "score": score}
                        for neighbor_id, score in sorted(
                            self._similarities.get(item_id, {}).items(),
                            key=lambda pair: (-pair[1], stable_id_key(pair[0])),
                        )
                    ]
                    for item_id in self._catalog
                ],
                "user_values": [
                    {
                        "user_id": user_id,
                        "values": [
                            {"item_id": item_id, "value": self._user_values[user_id][item_id]}
                            for item_id in sorted(self._user_values[user_id], key=stable_id_key)
                        ],
                    }
                    for user_id in users
                ],
            },
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        if set(parameters) != {"neighbors", "shrinkage"}:
            raise SerializationError("ItemKNN parameters are malformed")
        try:
            instance = cls(neighbors=parameters["neighbors"], shrinkage=parameters["shrinkage"])
        except ValidationError as exc:
            raise SerializationError(f"invalid ItemKNN parameters: {exc}") from exc
        instance._restore_base_state(base)
        if set(model) != {"similarities", "user_values"}:
            raise SerializationError("ItemKNN model state is malformed")
        rows = model["similarities"]
        user_rows = model["user_values"]
        if not isinstance(rows, list) or len(rows) != len(instance._catalog) or not isinstance(user_rows, list):
            raise SerializationError("ItemKNN state arrays are malformed")
        catalog = set(instance._catalog)
        similarities: dict[EntityId, dict[EntityId, float]] = {}
        for item_id, row in zip(instance._catalog, rows, strict=True):
            if not isinstance(row, list) or len(row) > instance.neighbors:
                raise SerializationError("ItemKNN similarity row is malformed")
            restored_row: dict[EntityId, float] = {}
            previous_key: tuple[float, tuple[int, int | str]] | None = None
            for entry in row:
                if not isinstance(entry, Mapping) or set(entry) != {"item_id", "score"}:
                    raise SerializationError("ItemKNN similarity entry is malformed")
                try:
                    neighbor_id = validate_entity_id(entry["item_id"], "neighbor item ID")
                except ValidationError as exc:
                    raise SerializationError(str(exc)) from exc
                raw_score = entry["score"]
                if neighbor_id not in catalog or neighbor_id == item_id or neighbor_id in restored_row:
                    raise SerializationError("ItemKNN similarity references an invalid neighbor")
                if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                    raise SerializationError("ItemKNN similarity must be a finite positive number")
                score = safe_float(raw_score)
                if not math.isfinite(score) or score <= 0 or score > 1.0 + 1e-12:
                    raise SerializationError(
                        "ItemKNN similarity must be finite and in the interval (0, 1]"
                    )
                ordering = (-score, stable_id_key(neighbor_id))
                if previous_key is not None and ordering < previous_key:
                    raise SerializationError("ItemKNN neighbors are not deterministically ordered")
                previous_key = ordering
                restored_row[neighbor_id] = score
            similarities[item_id] = restored_row
        user_values: dict[EntityId, dict[EntityId, float]] = {}
        for row in user_rows:
            if not isinstance(row, Mapping) or set(row) != {"user_id", "values"} or not isinstance(row["values"], list):
                raise SerializationError("ItemKNN user-value row is malformed")
            try:
                user_id = validate_entity_id(row["user_id"], "user_id")
            except ValidationError as exc:
                raise SerializationError(str(exc)) from exc
            if user_id in user_values or user_id not in instance._seen:
                raise SerializationError("ItemKNN user-value row references an invalid user")
            values: dict[EntityId, float] = {}
            for entry in row["values"]:
                if not isinstance(entry, Mapping) or set(entry) != {"item_id", "value"}:
                    raise SerializationError("ItemKNN user value is malformed")
                try:
                    history_item = validate_entity_id(entry["item_id"], "history item ID")
                except ValidationError as exc:
                    raise SerializationError(str(exc)) from exc
                raw_value = entry["value"]
                if history_item not in catalog or history_item in values:
                    raise SerializationError("ItemKNN user value references an invalid item")
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    raise SerializationError("ItemKNN history values must be finite positive numbers")
                value = safe_float(raw_value)
                if not math.isfinite(value) or value <= 0:
                    raise SerializationError("ItemKNN history values must be finite positive numbers")
                values[history_item] = value
            if set(values) != set(instance._seen[user_id]):
                raise SerializationError("ItemKNN history values do not match seen-item state")
            if list(values) != sorted(values, key=stable_id_key):
                raise SerializationError("ItemKNN history values are not in stable order")
            user_values[user_id] = values
        if set(user_values) != set(instance._seen) or list(user_values) != sorted(user_values, key=stable_id_key):
            raise SerializationError("ItemKNN user-value state does not match fitted users")
        instance._similarities = similarities
        instance._user_values = user_values
        return instance
