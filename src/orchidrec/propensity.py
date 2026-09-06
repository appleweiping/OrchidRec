"""Exposure models for unbiased offline evaluation.

An offline holdout is not a random sample of what a user finds relevant. A user
can only interact with what a previous system showed them, and previous systems
show popular items far more often. Counting hits against that holdout therefore
rewards a recommender for agreeing with whatever produced the log, and a
popularity ranker scores best of all.

The standard correction weights each observed interaction by the inverse of its
propensity -- the probability that a relevant interaction would have been
observed at all -- so that rarely-exposed items count for more. This module
supplies those propensities. It does not estimate them from anything hidden: the
model is named, its parameters are recorded, and a caller who knows the logging
policy can supply exact propensities instead.

The popularity model follows Yang et al., *Unbiased Offline Recommender
Evaluation for Missing-Not-At-Random Implicit Feedback* (RecSys 2018), which
takes ``p_i`` proportional to ``n_i ** ((eta + 1) / 2)`` for item popularity
``n_i``. :data:`DEFAULT_EXPONENT` is that expression at the published
``eta = 0.5``.

Inverse weighting removes bias only to the extent the propensity model is right.
A wrong model does not fail loudly; it moves the estimate somewhere else. Every
estimator built on this module therefore reports its effective sample size and
how much mass was clipped, so an estimate resting on a handful of heavily
weighted observations is visible as such rather than quoted as a number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, stable_id_key, validate_entity_id
from orchidrec.errors import ValidationError

# The published ``eta = 0.5`` under the exponent ``(eta + 1) / 2``.
DEFAULT_EXPONENT = 0.75

# Propensities below this floor are raised to it. Unclipped inverse weights are
# unbounded, so one barely-exposed item can dominate an estimate; clipping trades
# a bounded bias for a bounded variance, and the trade is reported rather than
# hidden.
DEFAULT_MINIMUM_PROPENSITY = 0.01

# A propensity model over more items than this is refused rather than built, so a
# malformed count table cannot allocate without bound.
MAX_PROPENSITY_ITEMS = 5_000_000


@dataclass(frozen=True, slots=True)
class ExposureModel:
    """Per-item probability that a relevant interaction would be observed.

    Propensities are normalized so the most exposed item has propensity 1. Only
    ratios matter to a self-normalized estimator, and fixing the maximum keeps
    every inverse weight at or above 1, which makes the numbers readable.
    """

    propensities: Mapping[EntityId, float]
    exponent: float
    minimum: float
    clipped_items: int
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.propensities, Mapping):
            raise ValidationError("propensities must be a mapping")
        if not self.propensities:
            raise ValidationError("propensities must contain at least one item")
        if len(self.propensities) > MAX_PROPENSITY_ITEMS:
            raise ValidationError(f"propensities must not exceed {MAX_PROPENSITY_ITEMS} items")
        cleaned: dict[EntityId, float] = {}
        for item_id, value in self.propensities.items():
            validated = validate_entity_id(item_id, "propensity item ID")
            number = safe_float(value) if isinstance(value, (int, float)) else float("nan")
            if isinstance(value, bool) or not isfinite(number) or not 0.0 < number <= 1.0:
                raise ValidationError(
                    f"propensity for item {validated!r} must be a real number in (0, 1]"
                )
            cleaned[validated] = number
        ordered = dict(sorted(cleaned.items(), key=lambda pair: stable_id_key(pair[0])))
        object.__setattr__(self, "propensities", MappingProxyType(ordered))
        for name in ("exponent", "minimum"):
            raw = getattr(self, name)
            number = safe_float(raw) if isinstance(raw, (int, float)) else float("nan")
            if isinstance(raw, bool) or not isfinite(number):
                raise ValidationError(f"{name} must be a finite number")
            object.__setattr__(self, name, number)
        if not 0.0 <= self.exponent <= 1.0:
            raise ValidationError("exponent must be a real number between 0 and 1")
        if not 0.0 < self.minimum <= 1.0:
            raise ValidationError("minimum must be a real number in (0, 1]")
        if isinstance(self.clipped_items, bool) or not isinstance(self.clipped_items, int):
            raise ValidationError("clipped_items must be an integer")
        if not 0 <= self.clipped_items <= len(self.propensities):
            raise ValidationError("clipped_items must not exceed the number of items")
        if not isinstance(self.source, str) or not self.source:
            raise ValidationError("source must be a non-empty string")

    def weight(self, item_id: EntityId) -> float:
        """Return the inverse-propensity weight of one observed item.

        An item absent from the model was never in the count table it was built
        from. Assuming a propensity for it would invent exposure evidence, so it
        is refused instead.
        """

        validated = validate_entity_id(item_id, "item ID")
        try:
            propensity = self.propensities[validated]
        except KeyError:
            raise ValidationError(
                f"item {validated!r} has no propensity; "
                f"it was absent from the exposure model"
            ) from None
        return 1.0 / propensity

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description without the per-item table."""

        return {
            "source": self.source,
            "exponent": self.exponent,
            "minimum": self.minimum,
            "items": len(self.propensities),
            "clipped_items": self.clipped_items,
        }


def popularity_exposure(
    item_counts: Mapping[EntityId, int | float],
    *,
    exponent: float = DEFAULT_EXPONENT,
    minimum: float = DEFAULT_MINIMUM_PROPENSITY,
) -> ExposureModel:
    """Estimate propensities from observed item popularity.

    ``exponent`` controls how strongly popularity is believed to drive exposure.
    At ``0`` every item is equally likely to be observed and the correction
    becomes the identity, which is the honest way to say "no exposure model"; at
    ``1`` propensity is taken to be proportional to popularity.

    An item with a zero count has never been observed, so popularity says nothing
    about its exposure. It takes the floor rather than a propensity of zero,
    which would be an infinite weight.
    """

    if not isinstance(item_counts, Mapping):
        raise ValidationError("item_counts must be a mapping")
    if not item_counts:
        raise ValidationError("item_counts must contain at least one item")
    if len(item_counts) > MAX_PROPENSITY_ITEMS:
        raise ValidationError(f"item_counts must not exceed {MAX_PROPENSITY_ITEMS} items")
    exponent_value = safe_float(exponent) if isinstance(exponent, (int, float)) else float("nan")
    if isinstance(exponent, bool) or not isfinite(exponent_value):
        raise ValidationError("exponent must be a real number between 0 and 1")
    if not 0.0 <= exponent_value <= 1.0:
        raise ValidationError("exponent must be a real number between 0 and 1")
    minimum_value = safe_float(minimum) if isinstance(minimum, (int, float)) else float("nan")
    if isinstance(minimum, bool) or not isfinite(minimum_value):
        raise ValidationError("minimum must be a real number in (0, 1]")
    if not 0.0 < minimum_value <= 1.0:
        raise ValidationError("minimum must be a real number in (0, 1]")

    counts: dict[EntityId, float] = {}
    for item_id, value in item_counts.items():
        validated = validate_entity_id(item_id, "item count ID")
        number = safe_float(value) if isinstance(value, (int, float)) else float("nan")
        if isinstance(value, bool) or not isfinite(number) or number < 0.0:
            raise ValidationError(
                f"count for item {validated!r} must be a non-negative real number"
            )
        counts[validated] = number

    largest = max(counts.values())
    if largest <= 0.0:
        raise ValidationError("item_counts must contain at least one positive count")

    propensities: dict[EntityId, float] = {}
    clipped = 0
    for item_id, count in counts.items():
        raw = (count / largest) ** exponent_value if count > 0.0 else 0.0
        if raw < minimum_value:
            clipped += 1
            raw = minimum_value
        propensities[item_id] = min(raw, 1.0)
    return ExposureModel(
        propensities=propensities,
        exponent=exponent_value,
        minimum=minimum_value,
        clipped_items=clipped,
        source="popularity",
    )


def uniform_exposure(item_counts: Mapping[EntityId, int | float]) -> ExposureModel:
    """Build the exposure model that applies no correction.

    Every item is equally likely to be observed, so every inverse weight is 1 and
    the corrected estimators reduce exactly to the uncorrected ones. It exists so
    that "we are not modelling exposure" can be stated in the same vocabulary as
    any other assumption, and so that the reduction can be tested.
    """

    model = popularity_exposure(item_counts, exponent=0.0, minimum=1.0)
    return ExposureModel(
        propensities=model.propensities,
        exponent=0.0,
        minimum=1.0,
        clipped_items=0,
        source="uniform",
    )


__all__ = [
    "DEFAULT_EXPONENT",
    "DEFAULT_MINIMUM_PROPENSITY",
    "MAX_PROPENSITY_ITEMS",
    "ExposureModel",
    "popularity_exposure",
    "uniform_exposure",
]
