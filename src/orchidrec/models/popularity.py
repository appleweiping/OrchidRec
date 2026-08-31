"""Weighted or event-count popularity baseline."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Self

from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, InteractionDataset
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.models.base import BaseRecommender, make_envelope, parse_envelope


class Popularity(BaseRecommender):
    """Rank items by their aggregate training interaction strength."""

    model_type = "popularity"

    def __init__(self, *, weighted: bool = True) -> None:
        super().__init__()
        if not isinstance(weighted, bool):
            raise ValidationError("weighted must be a boolean")
        self.weighted = weighted
        self._scores: dict[EntityId, float] = {}

    def _fit_model(self, dataset: InteractionDataset) -> None:
        scores: dict[EntityId, float] = defaultdict(float)
        for event in dataset:
            scores[event.item_id] += event.value if self.weighted else 1.0
        self._scores = dict(scores)

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        del user_id
        return self._scores.get(item_id, 0.0)

    def to_state(self) -> dict[str, Any]:
        return make_envelope(
            self.model_type,
            {"weighted": self.weighted},
            self._base_state(),
            {"scores": [self._scores.get(item_id, 0.0) for item_id in self._catalog]},
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        if set(parameters) != {"weighted"} or not isinstance(parameters["weighted"], bool):
            raise SerializationError("popularity parameters are malformed")
        if set(model) != {"scores"} or not isinstance(model["scores"], list):
            raise SerializationError("popularity score state is malformed")
        instance = cls(weighted=parameters["weighted"])
        instance._restore_base_state(base)
        if len(model["scores"]) != len(instance._catalog):
            raise SerializationError("popularity score length does not match catalog")
        scores: dict[EntityId, float] = {}
        for item_id, raw_score in zip(instance._catalog, model["scores"], strict=True):
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise SerializationError("popularity scores must be finite non-negative numbers")
            score = safe_float(raw_score)
            if not math.isfinite(score) or score <= 0:
                raise SerializationError("popularity scores must be finite positive numbers")
            scores[item_id] = score
        instance._scores = scores
        return instance
