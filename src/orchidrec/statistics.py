"""Deterministic, dependency-free bootstrap statistics."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from orchidrec._numeric import safe_float
from orchidrec.errors import ValidationError


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A percentile bootstrap confidence interval around an estimate."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "samples": self.samples,
        }


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Bootstrap comparison where positive differences favor ``right``."""

    left_mean: float
    right_mean: float
    difference: BootstrapInterval
    probability_right_better: float
    two_sided_p_value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "left_mean": self.left_mean,
            "right_mean": self.right_mean,
            "right_minus_left": self.difference.to_dict(),
            "probability_right_better": self.probability_right_better,
            "two_sided_p_value": self.two_sided_p_value,
        }


def _validated_values(values: Iterable[int | float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{name} must be an iterable of finite numbers")
    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise ValidationError(f"{name} must be an iterable of finite numbers") from exc
    if not materialized:
        raise ValidationError(f"{name} must not be empty")
    result: list[float] = []
    for value in materialized:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must contain only finite numbers")
        numeric = safe_float(value)
        if not math.isfinite(numeric):
            raise ValidationError(f"{name} must contain only finite numbers")
        result.append(numeric)
    return tuple(result)


def _validated_options(samples: int, confidence: int | float, seed: int) -> float:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValidationError("samples must be a positive integer")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValidationError("confidence must be a finite number between 0 and 1")
    numeric_confidence = safe_float(confidence)
    if not math.isfinite(numeric_confidence) or not 0.0 < numeric_confidence < 1.0:
        raise ValidationError("confidence must be a finite number between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError("seed must be an integer")
    return numeric_confidence


def _mean(values: Sequence[float]) -> float:
    return math.fsum(value / len(values) for value in values)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def interval_from_draws(
    estimate: int | float,
    draws: Iterable[int | float],
    *,
    confidence: int | float,
) -> BootstrapInterval:
    """Build a percentile interval from already generated bootstrap draws."""

    values = _validated_values(draws, "draws")
    numeric_confidence = _validated_options(len(values), confidence, 0)
    if isinstance(estimate, bool) or not isinstance(estimate, (int, float)):
        raise ValidationError("estimate must be a finite number")
    numeric_estimate = safe_float(estimate)
    if not math.isfinite(numeric_estimate):
        raise ValidationError("estimate must be a finite number")
    ordered = sorted(values)
    alpha = (1.0 - numeric_confidence) / 2.0
    return BootstrapInterval(
        estimate=numeric_estimate,
        lower=_quantile(ordered, alpha),
        upper=_quantile(ordered, 1.0 - alpha),
        confidence=numeric_confidence,
        samples=len(values),
    )


def bootstrap_mean(
    values: Iterable[int | float],
    *,
    samples: int = 1_000,
    confidence: int | float = 0.95,
    seed: int = 42,
) -> BootstrapInterval:
    """Estimate a mean interval by seeded sampling with replacement."""

    observations = _validated_values(values, "values")
    numeric_confidence = _validated_options(samples, confidence, seed)
    generator = random.Random(seed)
    count = len(observations)
    draws = [
        _mean(tuple(observations[generator.randrange(count)] for _ in range(count)))
        for _ in range(samples)
    ]
    return interval_from_draws(_mean(observations), draws, confidence=numeric_confidence)


def paired_bootstrap_mean(
    left: Iterable[int | float],
    right: Iterable[int | float],
    *,
    samples: int = 1_000,
    confidence: int | float = 0.95,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Compare paired means using shared seeded bootstrap resamples."""

    left_values = _validated_values(left, "left")
    right_values = _validated_values(right, "right")
    if len(left_values) != len(right_values):
        raise ValidationError("left and right must have the same number of observations")
    numeric_confidence = _validated_options(samples, confidence, seed)
    generator = random.Random(seed)
    count = len(left_values)
    differences: list[float] = []
    for _ in range(samples):
        indices = tuple(generator.randrange(count) for _ in range(count))
        left_mean = _mean(tuple(left_values[index] for index in indices))
        right_mean = _mean(tuple(right_values[index] for index in indices))
        differences.append(right_mean - left_mean)
    positive = sum(value > 0.0 for value in differences)
    zero = sum(value == 0.0 for value in differences)
    nonpositive = len(differences) - positive
    nonnegative = positive + zero
    p_value = min(
        1.0,
        2.0
        * min(
            (nonpositive + 1.0) / (samples + 1.0),
            (nonnegative + 1.0) / (samples + 1.0),
        ),
    )
    left_mean = _mean(left_values)
    right_mean = _mean(right_values)
    return PairedBootstrapResult(
        left_mean=left_mean,
        right_mean=right_mean,
        difference=interval_from_draws(
            right_mean - left_mean,
            differences,
            confidence=numeric_confidence,
        ),
        probability_right_better=(positive + 0.5 * zero) / samples,
        two_sided_p_value=p_value,
    )


def paired_comparison_from_draws(
    left_estimate: int | float,
    right_estimate: int | float,
    left_draws: Iterable[int | float],
    right_draws: Iterable[int | float],
    *,
    confidence: int | float,
) -> PairedBootstrapResult:
    """Compare aligned bootstrap distributions produced from shared resamples."""

    left_values = _validated_values(left_draws, "left_draws")
    right_values = _validated_values(right_draws, "right_draws")
    if len(left_values) != len(right_values):
        raise ValidationError("left_draws and right_draws must have the same length")
    if isinstance(left_estimate, bool) or not isinstance(left_estimate, (int, float)):
        raise ValidationError("left_estimate must be a finite number")
    if isinstance(right_estimate, bool) or not isinstance(right_estimate, (int, float)):
        raise ValidationError("right_estimate must be a finite number")
    left_mean = safe_float(left_estimate)
    right_mean = safe_float(right_estimate)
    if not math.isfinite(left_mean) or not math.isfinite(right_mean):
        raise ValidationError("estimates must be finite numbers")
    differences = tuple(
        right - left for left, right in zip(left_values, right_values, strict=True)
    )
    positive = sum(value > 0.0 for value in differences)
    zero = sum(value == 0.0 for value in differences)
    samples = len(differences)
    nonpositive = samples - positive
    nonnegative = positive + zero
    p_value = min(
        1.0,
        2.0
        * min(
            (nonpositive + 1.0) / (samples + 1.0),
            (nonnegative + 1.0) / (samples + 1.0),
        ),
    )
    return PairedBootstrapResult(
        left_mean=left_mean,
        right_mean=right_mean,
        difference=interval_from_draws(
            right_mean - left_mean,
            differences,
            confidence=confidence,
        ),
        probability_right_better=(positive + 0.5 * zero) / samples,
        two_sided_p_value=p_value,
    )
