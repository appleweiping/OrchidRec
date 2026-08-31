"""Ranking metrics for offline recommendation experiments.

The functions in this module intentionally operate on plain mappings and
sequences.  They are therefore useful outside the experiment runner and do
not require a fitted OrchidRec model.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, log2

from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, stable_id_key, validate_entity_id
from orchidrec.errors import ValidationError


@dataclass(frozen=True, slots=True)
class MetricReport:
    """A complete top-k evaluation summary."""

    precision: float
    recall: float
    ndcg: float
    mrr: float
    coverage: float
    novelty: float
    users: int
    k: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValidationError("k must be a positive integer")


def _evaluation_users(relevant: Mapping[EntityId, Collection[EntityId]]) -> tuple[EntityId, ...]:
    if not isinstance(relevant, Mapping):
        raise ValidationError("relevant must be a mapping")
    users: list[EntityId] = []
    for user_id, items in relevant.items():
        validated_user = validate_entity_id(user_id, "relevant user ID")
        _validated_items(items, f"relevant items for user {validated_user!r}")
        if items:
            users.append(validated_user)
    if not users:
        raise ValidationError("relevant must contain at least one user with a relevant item")
    return tuple(users)


def _validated_items(values: object, name: str) -> tuple[EntityId, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise ValidationError(f"{name} must be a collection of item IDs")
    return tuple(validate_entity_id(value, f"{name} item") for value in values)


def _ranking_items(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    user_id: EntityId,
) -> tuple[EntityId, ...]:
    if not isinstance(recommendations, Mapping):
        raise ValidationError("recommendations must be a mapping")
    validate_entity_id(user_id, "recommendation user ID")
    raw_items = recommendations.get(user_id, ())
    if isinstance(raw_items, (str, bytes)) or not isinstance(raw_items, Sequence):
        raise ValidationError(f"recommendations for user {user_id!r} must be a sequence")
    items = tuple(
        validate_entity_id(item_id, f"recommended item for user {user_id!r}")
        for item_id in raw_items
    )
    if len(set(items)) != len(items):
        raise ValidationError(f"recommendations for user {user_id!r} contain duplicate items")
    return items


def _validate_recommendations(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
) -> None:
    if not isinstance(recommendations, Mapping):
        raise ValidationError("recommendations must be a mapping")
    for user_id in recommendations:
        validate_entity_id(user_id, "recommendation user ID")
        _ranking_items(recommendations, user_id)


def _top_items(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    user_id: EntityId,
    k: int,
) -> tuple[EntityId, ...]:
    return _ranking_items(recommendations, user_id)[:k]


def _catalog_items(catalog: Collection[EntityId]) -> set[EntityId]:
    values = _validated_items(catalog, "catalog")
    if not values:
        raise ValidationError("catalog must not be empty")
    if len(set(values)) != len(values):
        raise ValidationError("catalog must not contain duplicate items")
    return set(values)


def precision_at_k(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    relevant: Mapping[EntityId, Collection[EntityId]],
    k: int,
) -> float:
    """Compute macro-averaged precision@k.

    The denominator is always ``k``.  A model returning fewer than ``k``
    items is therefore not rewarded for an incomplete recommendation list.
    """

    _validate_k(k)
    _validate_recommendations(recommendations)
    users = _evaluation_users(relevant)
    total = 0.0
    for user_id in users:
        relevant_items = set(relevant[user_id])
        hits = sum(item_id in relevant_items for item_id in _top_items(recommendations, user_id, k))
        total += hits / k
    return total / len(users)


def recall_at_k(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    relevant: Mapping[EntityId, Collection[EntityId]],
    k: int,
) -> float:
    """Compute macro-averaged recall@k."""

    _validate_k(k)
    _validate_recommendations(recommendations)
    users = _evaluation_users(relevant)
    total = 0.0
    for user_id in users:
        relevant_items = set(relevant[user_id])
        hits = sum(item_id in relevant_items for item_id in _top_items(recommendations, user_id, k))
        total += hits / len(relevant_items)
    return total / len(users)


def ndcg_at_k(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    relevant: Mapping[EntityId, Collection[EntityId]],
    k: int,
) -> float:
    """Compute macro-averaged binary NDCG@k."""

    _validate_k(k)
    _validate_recommendations(recommendations)
    users = _evaluation_users(relevant)
    total = 0.0
    for user_id in users:
        relevant_items = set(relevant[user_id])
        ranked = _top_items(recommendations, user_id, k)
        dcg = sum(1.0 / log2(rank + 2) for rank, item_id in enumerate(ranked) if item_id in relevant_items)
        ideal_length = min(k, len(relevant_items))
        ideal = sum(1.0 / log2(rank + 2) for rank in range(ideal_length))
        total += dcg / ideal
    return total / len(users)


def mrr_at_k(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    relevant: Mapping[EntityId, Collection[EntityId]],
    k: int,
) -> float:
    """Compute macro-averaged reciprocal rank at ``k``."""

    _validate_k(k)
    _validate_recommendations(recommendations)
    users = _evaluation_users(relevant)
    total = 0.0
    for user_id in users:
        relevant_items = set(relevant[user_id])
        for rank, item_id in enumerate(_top_items(recommendations, user_id, k), start=1):
            if item_id in relevant_items:
                total += 1.0 / rank
                break
    return total / len(users)


def catalog_coverage(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    catalog: Collection[EntityId],
    k: int,
) -> float:
    """Return the fraction of catalog items appearing in top-k lists."""

    _validate_k(k)
    _validate_recommendations(recommendations)
    catalog_items = _catalog_items(catalog)
    exposed: set[EntityId] = set()
    for user_id in recommendations:
        ranking = _ranking_items(recommendations, user_id)
        unknown = set(ranking) - catalog_items
        if unknown:
            item_id = sorted(unknown, key=stable_id_key)[0]
            raise ValidationError(f"recommended item {item_id!r} is not in the catalog")
        for item_id in ranking[:k]:
            exposed.add(item_id)
    return len(exposed) / len(catalog_items)


def novelty_at_k(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    item_counts: Mapping[EntityId, int | float],
    catalog: Collection[EntityId],
    k: int,
) -> float:
    """Compute mean self-information of recommended items.

    Laplace smoothing gives catalog items unseen in training a finite score:
    ``-log2((count + 1) / (total + catalog_size))``.
    """

    _validate_k(k)
    _validate_recommendations(recommendations)
    catalog_items = _catalog_items(catalog)
    if not isinstance(item_counts, Mapping):
        raise ValidationError("item_counts must be a mapping")
    for item_id in item_counts:
        validate_entity_id(item_id, "item_counts item ID")
    unknown_counts = set(item_counts) - catalog_items
    if unknown_counts:
        raise ValidationError("item_counts contains items outside the catalog")
    for item_id, count in item_counts.items():
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or not isfinite(safe_float(count))
            or count < 0
        ):
            raise ValidationError(f"count for item {item_id!r} must be finite and non-negative")
    total_count = sum(safe_float(count) for count in item_counts.values())
    if not isfinite(total_count):
        raise ValidationError("the aggregate item count must be finite")
    denominator = total_count + len(catalog_items)
    if not isfinite(denominator):
        raise ValidationError("the smoothed item-count denominator must be finite")
    values: list[float] = []
    for user_id in recommendations:
        ranking = _ranking_items(recommendations, user_id)
        unknown = set(ranking) - catalog_items
        if unknown:
            item_id = sorted(unknown, key=stable_id_key)[0]
            raise ValidationError(f"recommended item {item_id!r} is not in the catalog")
        for item_id in ranking[:k]:
            probability = (safe_float(item_counts.get(item_id, 0)) + 1.0) / denominator
            values.append(-log2(probability))
    return sum(values) / len(values) if values else 0.0


def evaluate_ranking(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    relevant: Mapping[EntityId, Collection[EntityId]],
    catalog: Collection[EntityId],
    item_counts: Mapping[EntityId, int | float],
    k: int,
) -> MetricReport:
    """Evaluate recommendations with all metrics shipped by OrchidRec."""

    _validate_k(k)
    users = _evaluation_users(relevant)
    return MetricReport(
        precision=precision_at_k(recommendations, relevant, k),
        recall=recall_at_k(recommendations, relevant, k),
        ndcg=ndcg_at_k(recommendations, relevant, k),
        mrr=mrr_at_k(recommendations, relevant, k),
        coverage=catalog_coverage(recommendations, catalog, k),
        novelty=novelty_at_k(recommendations, item_counts, catalog, k),
        users=len(users),
        k=k,
    )
