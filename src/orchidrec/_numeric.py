"""Small numeric helpers for strict public-boundary validation."""

from __future__ import annotations

import math


def safe_float(value: int | float) -> float:
    """Convert a real number without leaking ``OverflowError`` for huge integers."""

    try:
        return float(value)
    except OverflowError:
        return math.inf if value >= 0 else -math.inf
