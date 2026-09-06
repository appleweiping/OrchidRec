"""Exposure-corrected ranking metrics.

:mod:`orchidrec.metrics` counts a hit whenever a recommended item appears in the
holdout. That treats the holdout as a random sample of relevance, which it is
not: it is what a previous system chose to show. The estimators here weight each
observed interaction by the inverse of its propensity, so an item that was rarely
exposed counts for more when it does appear.

Two things about the form of these estimators matter more than the weighting
itself.

They are normalized over the whole population, not per user. A per-user ratio
looks like the natural analogue of the macro-averaged metrics next door, but it
cannot correct anything: for a user with a single observed interaction the ratio
is ``w / w`` on a hit and ``0 / w`` on a miss, so the weight cancels exactly.
Strong exposure bias is precisely the regime where most users have one observed
interaction, so the per-user form fails hardest where it is needed most. A
population-wide ratio keeps the weights, and its bias falls as the total number
of observations grows rather than as the smallest user grows.

They therefore estimate a *micro*-averaged quantity -- an average over
interactions -- while :func:`~orchidrec.metrics.recall_at_k` and its neighbours
average over users. The two are different numbers even with no exposure
correction at all, so comparing them directly says nothing. Evaluate under
:func:`~orchidrec.propensity.uniform_exposure` to obtain the uncorrected
micro-averaged baseline: every weight is then 1 and each estimator reduces
exactly to its unweighted micro form, which is the like-for-like comparison.

Every report carries the effective sample size of the weights it used. Inverse
weighting concentrates an estimate on rarely-observed items, and an estimate
resting on a few heavily weighted observations is not more trustworthy than the
biased one it replaced -- it is differently untrustworthy. The number says which
situation you are in.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import log2
from typing import Any

from orchidrec.data import EntityId
from orchidrec.errors import ValidationError
from orchidrec.metrics import (
    _evaluation_users,
    _top_items,
    _validate_k,
    _validate_recommendations,
)
from orchidrec.propensity import ExposureModel

_WeightedItems = tuple[tuple[EntityId, float], ...]


@dataclass(frozen=True, slots=True)
class UnbiasedMetricReport:
    """An exposure-corrected top-k summary and the evidence behind it."""

    recall: float
    ndcg: float
    users: int
    k: int
    observed_interactions: int
    clipped_interactions: int
    effective_sample_size: float
    exposure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def _weighted(
    relevant: Mapping[EntityId, Collection[EntityId]],
    users: Sequence[EntityId],
    exposure: ExposureModel,
) -> dict[EntityId, _WeightedItems]:
    """Attach an inverse-propensity weight to every observed interaction."""

    if not isinstance(exposure, ExposureModel):
        raise ValidationError("exposure must be an ExposureModel")
    return {
        user_id: tuple((item_id, exposure.weight(item_id)) for item_id in relevant[user_id])
        for user_id in users
    }


def _effective_sample_size(weighted: Mapping[EntityId, _WeightedItems]) -> float:
    """Return Kish effective sample size as a fraction of the observations.

    ``(sum w)^2 / sum w^2`` is the number of equally weighted observations that
    would carry the same information. Dividing by the actual count puts it on a
    0-to-1 scale, where 1 means the weights are uniform and a small value means a
    few observations dominate the estimate.
    """

    total = 0.0
    squares = 0.0
    count = 0
    for pairs in weighted.values():
        for _item_id, weight in pairs:
            total += weight
            squares += weight * weight
            count += 1
    if count == 0 or squares <= 0.0:  # pragma: no cover - callers guarantee observations
        return 0.0
    return (total * total / squares) / count


def _clipped_interactions(
    weighted: Mapping[EntityId, _WeightedItems], exposure: ExposureModel
) -> int:
    """Count observations whose item sat on the propensity floor.

    Their weight is a policy decision rather than a measurement, so a report that
    leans on many of them is leaning on the floor.
    """

    if exposure.minimum >= 1.0:
        return 0
    floor_weight = 1.0 / exposure.minimum
    return sum(
        1 for pairs in weighted.values() for _item_id, weight in pairs if weight >= floor_weight
    )


def ips_recall_at_k(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    relevant: Mapping[EntityId, Collection[EntityId]],
    exposure: ExposureModel,
    k: int,
) -> float:
    """Compute self-normalized inverse-propensity recall@k over all interactions.

    The weighted hits of every user are divided by the weight of every observed
    interaction, so a rarely-exposed item raises the denominator for the whole
    population rather than only for the user who happened to see it.
    """

    _validate_k(k)
    _validate_recommendations(recommendations)
    users = _evaluation_users(relevant)
    weighted = _weighted(relevant, users, exposure)
    hits = 0.0
    total = 0.0
    for user_id in users:
        ranked = set(_top_items(recommendations, user_id, k))
        for item_id, weight in weighted[user_id]:
            total += weight
            if item_id in ranked:
                hits += weight
    return hits / total


def ips_ndcg_at_k(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    relevant: Mapping[EntityId, Collection[EntityId]],
    exposure: ExposureModel,
    k: int,
) -> float:
    """Compute self-normalized inverse-propensity NDCG@k over all interactions.

    Each user contributes the discounted gain of the weights it actually found
    and the ideal gain it could have found, and the two are summed across users
    before dividing. A user whose heaviest observations are all ranked first
    therefore contributes its full ideal, exactly as in the unweighted metric.
    """

    _validate_k(k)
    _validate_recommendations(recommendations)
    users = _evaluation_users(relevant)
    weighted = _weighted(relevant, users, exposure)
    gain = 0.0
    ideal = 0.0
    for user_id in users:
        pairs = weighted[user_id]
        lookup = dict(pairs)
        ranked = _top_items(recommendations, user_id, k)
        gain += sum(
            lookup[item_id] / log2(rank + 2)
            for rank, item_id in enumerate(ranked)
            if item_id in lookup
        )
        best = sorted((weight for _item_id, weight in pairs), reverse=True)[:k]
        ideal += sum(weight / log2(rank + 2) for rank, weight in enumerate(best))
    return gain / ideal


def evaluate_unbiased_ranking(
    recommendations: Mapping[EntityId, Sequence[EntityId]],
    relevant: Mapping[EntityId, Collection[EntityId]],
    exposure: ExposureModel,
    k: int,
) -> UnbiasedMetricReport:
    """Evaluate recommendations under an explicit exposure model.

    Coverage and novelty are absent by design: neither estimates user relevance,
    so inverse-propensity weighting has nothing to correct in them. Read those
    from :func:`~orchidrec.metrics.evaluate_ranking` as before.
    """

    _validate_k(k)
    _validate_recommendations(recommendations)
    users = _evaluation_users(relevant)
    weighted = _weighted(relevant, users, exposure)
    return UnbiasedMetricReport(
        recall=ips_recall_at_k(recommendations, relevant, exposure, k),
        ndcg=ips_ndcg_at_k(recommendations, relevant, exposure, k),
        users=len(users),
        k=k,
        observed_interactions=sum(len(pairs) for pairs in weighted.values()),
        clipped_interactions=_clipped_interactions(weighted, exposure),
        effective_sample_size=_effective_sample_size(weighted),
        exposure=exposure.to_dict(),
    )


__all__ = [
    "UnbiasedMetricReport",
    "evaluate_unbiased_ranking",
    "ips_ndcg_at_k",
    "ips_recall_at_k",
]
