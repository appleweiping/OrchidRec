"""User-user collaborative filtering with deterministic cosine neighborhoods."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Self

from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, InteractionDataset, stable_id_key, validate_entity_id
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.models.base import BaseRecommender, make_envelope, parse_envelope


class UserKNN(BaseRecommender):
    """Cosine user-neighborhood recommender over positive interaction values.

    Similarities are computed from the complete training vectors.  Prediction
    for an item is the similarity-weighted mean of the nearest users that
    interacted with that item.  The tiny popularity term only gives a stable
    order when no neighbor contributes; it cannot change a non-zero neighbor
    prediction.
    """

    model_type = "user_knn"

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
        self._user_values: dict[EntityId, dict[EntityId, float]] = {}
        self._neighbor_weights: dict[EntityId, tuple[tuple[EntityId, float], ...]] = {}

    def _fit_model(self, dataset: InteractionDataset) -> None:
        values: dict[EntityId, dict[EntityId, float]] = defaultdict(lambda: defaultdict(float))
        for event in dataset:
            updated = values[event.user_id][event.item_id] + event.value
            if not math.isfinite(updated):
                raise ValidationError("UserKNN numeric overflow while aggregating interactions")
            values[event.user_id][event.item_id] = updated
        self._user_values = {user_id: dict(row) for user_id, row in values.items()}
        self._rebuild_neighbors()

    def _rebuild_neighbors(self) -> None:
        norms_squared: dict[EntityId, float] = {}
        for user_id, row in self._user_values.items():
            norm = math.fsum(value * value for value in row.values())
            if not math.isfinite(norm):
                raise ValidationError("UserKNN numeric overflow while computing user norms")
            norms_squared[user_id] = norm

        by_item: dict[EntityId, list[tuple[EntityId, float]]] = defaultdict(list)
        for user_id, row in self._user_values.items():
            for item_id, value in row.items():
                by_item[item_id].append((user_id, value))
        dots: dict[tuple[EntityId, EntityId], float] = defaultdict(float)
        for item_users in by_item.values():
            ordered = sorted(item_users, key=lambda pair: stable_id_key(pair[0]))
            for left_index, (left_id, left_value) in enumerate(ordered):
                for right_id, right_value in ordered[left_index + 1 :]:
                    pair = (left_id, right_id)
                    updated = dots[pair] + left_value * right_value
                    if not math.isfinite(updated):
                        raise ValidationError(
                            "UserKNN numeric overflow while computing similarities"
                        )
                    dots[pair] = updated

        rows: dict[EntityId, list[tuple[EntityId, float]]] = {
            user_id: [] for user_id in self._user_values
        }
        for (left_id, right_id), dot in dots.items():
            denominator = (
                math.sqrt(norms_squared[left_id]) * math.sqrt(norms_squared[right_id])
                + self.shrinkage
            )
            if not math.isfinite(denominator):
                raise ValidationError("UserKNN numeric overflow in similarity denominator")
            similarity = dot / denominator if denominator > 0.0 else 0.0
            if similarity > 0.0:
                rows[left_id].append((right_id, similarity))
                rows[right_id].append((left_id, similarity))
        self._neighbor_weights = {
            user_id: tuple(
                sorted(row, key=lambda pair: (-pair[1], stable_id_key(pair[0])))[: self.neighbors]
            )
            for user_id, row in rows.items()
        }

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        neighbors = self._neighbor_weights.get(user_id)
        fallback = self._popularity_fallback(item_id)
        if not neighbors:
            return fallback
        contributors = [
            (neighbor_id, similarity)
            for neighbor_id, similarity in neighbors
            if item_id in self._user_values[neighbor_id]
        ]
        denominator = math.fsum(similarity for _, similarity in contributors)
        if denominator <= 0.0:
            return fallback * 1e-12
        prediction = (
            math.fsum(
                similarity * self._user_values[neighbor_id][item_id]
                for neighbor_id, similarity in contributors
            )
            / denominator
        )
        return prediction + fallback * 1e-12

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        return make_envelope(
            self.model_type,
            {"neighbors": self.neighbors, "shrinkage": self.shrinkage},
            self._base_state(),
            {
                "user_values": [
                    {
                        "user_id": user_id,
                        "values": [
                            {"item_id": item_id, "value": self._user_values[user_id][item_id]}
                            for item_id in sorted(self._user_values[user_id], key=stable_id_key)
                        ],
                    }
                    for user_id in sorted(self._user_values, key=stable_id_key)
                ]
            },
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        if set(parameters) != {"neighbors", "shrinkage"}:
            raise SerializationError("UserKNN parameters are malformed")
        try:
            instance = cls(neighbors=parameters["neighbors"], shrinkage=parameters["shrinkage"])
        except ValidationError as exc:
            raise SerializationError(f"invalid UserKNN parameters: {exc}") from exc
        instance._restore_base_state(base)
        if set(model) != {"user_values"} or not isinstance(model["user_values"], list):
            raise SerializationError("UserKNN model state is malformed")
        catalog = set(instance._catalog)
        restored: dict[EntityId, dict[EntityId, float]] = {}
        for row in model["user_values"]:
            if (
                not isinstance(row, Mapping)
                or set(row) != {"user_id", "values"}
                or not isinstance(row["values"], list)
            ):
                raise SerializationError("UserKNN user-value row is malformed")
            try:
                user_id = validate_entity_id(row["user_id"], "user_id")
            except ValidationError as exc:
                raise SerializationError(str(exc)) from exc
            if user_id in restored or user_id not in instance._seen:
                raise SerializationError("UserKNN state references an invalid user")
            values: dict[EntityId, float] = {}
            for entry in row["values"]:
                if not isinstance(entry, Mapping) or set(entry) != {"item_id", "value"}:
                    raise SerializationError("UserKNN history entry is malformed")
                try:
                    item_id = validate_entity_id(entry["item_id"], "history item ID")
                except ValidationError as exc:
                    raise SerializationError(str(exc)) from exc
                raw_value = entry["value"]
                if item_id not in catalog or item_id in values:
                    raise SerializationError("UserKNN history references an invalid item")
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    raise SerializationError(
                        "UserKNN history values must be positive finite numbers"
                    )
                value = safe_float(raw_value)
                if not math.isfinite(value) or value <= 0.0:
                    raise SerializationError(
                        "UserKNN history values must be positive finite numbers"
                    )
                values[item_id] = value
            if set(values) != set(instance._seen[user_id]):
                raise SerializationError("UserKNN history does not match base seen-item state")
            if list(values) != sorted(values, key=stable_id_key):
                raise SerializationError("UserKNN history is not in stable item order")
            restored[user_id] = values
        if list(restored) != sorted(instance._seen, key=stable_id_key) or set(restored) != set(
            instance._seen
        ):
            raise SerializationError("UserKNN users do not match base state")
        instance._user_values = restored
        try:
            instance._rebuild_neighbors()
        except ValidationError as exc:
            raise SerializationError(f"invalid UserKNN numeric state: {exc}") from exc
        return instance
