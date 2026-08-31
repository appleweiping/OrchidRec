"""Shared interfaces and deterministic ranking behavior for recommenders."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Self

from orchidrec._numeric import safe_float
from orchidrec.data import (
    EntityId,
    InteractionDataset,
    StableIdMap,
    stable_id_key,
    validate_entity_id,
)
from orchidrec.errors import NotFittedError, SerializationError, ValidationError

MODEL_FORMAT = "orchidrec.model"
MODEL_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A ranked item and its model score."""

    item_id: EntityId
    score: float
    rank: int

    def to_dict(self) -> dict[str, EntityId | float | int]:
        return asdict(self)


class BaseRecommender(ABC):
    """Base class that owns catalog, seen-item filtering, and top-k ranking."""

    model_type: ClassVar[str]

    def __init__(self) -> None:
        self._fitted = False
        self._catalog: tuple[EntityId, ...] = ()
        self._seen: dict[EntityId, frozenset[EntityId]] = {}
        self._popularity: dict[EntityId, float] = {}

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def catalog(self) -> tuple[EntityId, ...]:
        self._require_fitted()
        return self._catalog

    def fit(self, dataset: InteractionDataset) -> Self:
        """Fit the common state and model-specific parameters."""

        if not isinstance(dataset, InteractionDataset):
            raise ValidationError("dataset must be an InteractionDataset")
        if not dataset:
            raise ValidationError("cannot fit a recommender on an empty dataset")
        self._fitted = False
        self._catalog = dataset.item_ids
        self._seen = {
            user_id: frozenset(event.item_id for event in events)
            for user_id, events in dataset.by_user().items()
        }
        self._popularity = dataset.item_counts(weighted=True)
        self._fit_model(dataset)
        self._fitted = True
        return self

    @abstractmethod
    def _fit_model(self, dataset: InteractionDataset) -> None:
        """Populate model-specific fitted state."""

    @abstractmethod
    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        """Score one catalog item for one validated user ID."""

    @abstractmethod
    def to_state(self) -> dict[str, Any]:
        """Return a complete JSON-compatible model state."""

    @classmethod
    @abstractmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        """Restore a model from validated serialized state."""

    def score_items(
        self,
        user_id: EntityId,
        candidates: Iterable[EntityId] | None = None,
    ) -> dict[EntityId, float]:
        """Score catalog candidates in deterministic ID order."""

        self._require_fitted()
        validated_user = validate_entity_id(user_id, "user_id")
        candidate_ids = self._validate_candidates(candidates)
        scores: dict[EntityId, float] = {}
        for item_id in candidate_ids:
            score = safe_float(self._score(validated_user, item_id))
            if not math.isfinite(score):
                raise ValidationError(f"model produced a non-finite score for item {item_id!r}")
            scores[item_id] = score
        return scores

    def recommend(
        self,
        user_id: EntityId,
        k: int = 10,
        *,
        exclude_seen: bool = True,
        candidates: Iterable[EntityId] | None = None,
    ) -> list[Recommendation]:
        """Return stable score-descending top-k recommendations."""

        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValidationError("k must be a positive integer")
        if not isinstance(exclude_seen, bool):
            raise ValidationError("exclude_seen must be a boolean")
        validated_user = validate_entity_id(user_id, "user_id")
        scores = self.score_items(validated_user, candidates)
        seen = self._seen.get(validated_user, frozenset()) if exclude_seen else frozenset()
        ranked = sorted(
            ((item_id, score) for item_id, score in scores.items() if item_id not in seen),
            key=lambda pair: (-pair[1], stable_id_key(pair[0])),
        )[:k]
        return [
            Recommendation(item_id=item_id, score=score, rank=rank)
            for rank, (item_id, score) in enumerate(ranked, start=1)
        ]

    def seen_items(self, user_id: EntityId) -> frozenset[EntityId]:
        """Return the fitted user's seen-item set."""

        self._require_fitted()
        validated_user = validate_entity_id(user_id, "user_id")
        return self._seen.get(validated_user, frozenset())

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(f"{type(self).__name__} has not been fitted")

    def _validate_candidates(self, candidates: Iterable[EntityId] | None) -> tuple[EntityId, ...]:
        if candidates is None:
            return self._catalog
        if isinstance(candidates, (str, bytes)):
            raise ValidationError("candidates must be an iterable of item IDs, not a string")
        try:
            materialized = tuple(validate_entity_id(value, "candidate item ID") for value in candidates)
        except TypeError as exc:
            raise ValidationError("candidates must be an iterable of item IDs") from exc
        if len(set(materialized)) != len(materialized):
            raise ValidationError("candidates must not contain duplicates")
        catalog = set(self._catalog)
        unknown = set(materialized) - catalog
        if unknown:
            first = sorted(unknown, key=stable_id_key)[0]
            raise ValidationError(f"candidate item {first!r} is not in the fitted catalog")
        return tuple(sorted(materialized, key=stable_id_key))

    def _popularity_fallback(self, item_id: EntityId) -> float:
        maximum = max(self._popularity.values(), default=0.0)
        return self._popularity.get(item_id, 0.0) / maximum if maximum > 0 else 0.0

    def _base_state(self) -> dict[str, Any]:
        self._require_fitted()
        user_ids = sorted(self._seen, key=stable_id_key)
        return {
            "catalog": list(self._catalog),
            "popularity": [self._popularity.get(item_id, 0.0) for item_id in self._catalog],
            "users": [
                {
                    "user_id": user_id,
                    "seen": sorted(self._seen[user_id], key=stable_id_key),
                }
                for user_id in user_ids
            ],
        }

    def _restore_base_state(self, state: object) -> None:
        if not isinstance(state, Mapping) or set(state) != {"catalog", "popularity", "users"}:
            raise SerializationError("base model state has invalid fields")
        catalog_raw = state["catalog"]
        popularity_raw = state["popularity"]
        users_raw = state["users"]
        if not isinstance(catalog_raw, list) or not isinstance(popularity_raw, list) or not isinstance(users_raw, list):
            raise SerializationError("base model arrays are malformed")
        try:
            catalog_map = StableIdMap.from_state({"ids": catalog_raw})
        except SerializationError as exc:
            raise SerializationError(f"invalid catalog: {exc}") from exc
        if len(popularity_raw) != len(catalog_map):
            raise SerializationError("popularity length does not match catalog")
        popularity: dict[EntityId, float] = {}
        for item_id, raw_score in zip(catalog_map.ids, popularity_raw, strict=True):
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise SerializationError("popularity values must be finite non-negative numbers")
            score = safe_float(raw_score)
            if not math.isfinite(score) or score <= 0:
                raise SerializationError("popularity values must be finite positive numbers")
            popularity[item_id] = score
        seen: dict[EntityId, frozenset[EntityId]] = {}
        for entry in users_raw:
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"user_id", "seen"}
                or not isinstance(entry["seen"], list)
            ):
                raise SerializationError("user seen-item state is malformed")
            try:
                user_id = validate_entity_id(entry["user_id"], "user_id")
                seen_items = tuple(validate_entity_id(value, "seen item ID") for value in entry["seen"])
            except ValidationError as exc:
                raise SerializationError(str(exc)) from exc
            if user_id in seen:
                raise SerializationError("duplicate user in seen-item state")
            if not seen_items:
                raise SerializationError("user seen-item state must not be empty")
            if len(set(seen_items)) != len(seen_items) or list(sorted(seen_items, key=stable_id_key)) != entry["seen"]:
                raise SerializationError("seen-item IDs must be unique and stably sorted")
            if not set(seen_items).issubset(set(catalog_map.ids)):
                raise SerializationError("seen-item state references an unknown catalog item")
            seen[user_id] = frozenset(seen_items)
        if list(sorted(seen, key=stable_id_key)) != [entry["user_id"] for entry in users_raw]:
            raise SerializationError("users must be in stable ID order")
        self._catalog = catalog_map.ids
        self._popularity = popularity
        self._seen = seen
        self._fitted = True


def make_envelope(
    model_type: str,
    parameters: Mapping[str, Any],
    base: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a versioned model serialization envelope."""

    return {
        "format": MODEL_FORMAT,
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": model_type,
        "parameters": dict(parameters),
        "base": dict(base),
        "model": dict(model),
    }


def parse_envelope(
    state: Mapping[str, Any], expected_type: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate common envelope fields and return its three payloads."""

    required = {"format", "schema_version", "model_type", "parameters", "base", "model"}
    if not isinstance(state, Mapping) or set(state) != required:
        raise SerializationError("model state has missing or unknown top-level fields")
    if (
        not isinstance(state["format"], str)
        or state["format"] != MODEL_FORMAT
        or isinstance(state["schema_version"], bool)
        or not isinstance(state["schema_version"], int)
        or state["schema_version"] != MODEL_SCHEMA_VERSION
    ):
        raise SerializationError("unsupported OrchidRec model format or schema version")
    if state["model_type"] != expected_type:
        raise SerializationError(f"expected model type {expected_type!r}, got {state['model_type']!r}")
    parameters, base, model = state["parameters"], state["base"], state["model"]
    if not isinstance(parameters, Mapping) or not isinstance(base, Mapping) or not isinstance(model, Mapping):
        raise SerializationError("parameters, base, and model state must be objects")
    return parameters, base, model
