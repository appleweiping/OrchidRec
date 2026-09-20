"""Bounded, train-only negative sampling for pairwise recommenders.

This is an original local policy, not RecBole's sampler implementation.  The
default uniform code paths in older models intentionally remain unchanged.
"""

from __future__ import annotations

import bisect
import math
import random
from collections import Counter
from collections.abc import Mapping

from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, InteractionDataset
from orchidrec.errors import ValidationError

MAX_POP_TRAIN_INTERACTIONS = 20_000
MAX_POP_USERS = 512
MAX_POP_ITEMS = 512
MAX_POOL_CHECKS = 262_144
MAX_DRAWS = 1_000_000


def validate_training_sampling(strategy: str, alpha: float) -> tuple[str, float]:
    """Validate a policy without allocating pools or consuming randomness."""
    if type(strategy) is not str or strategy not in {"uniform", "popularity"}:
        raise ValidationError("negative_strategy must be uniform or popularity")
    if type(alpha) not in (int, float):
        raise ValidationError("popularity_alpha must be finite and in (0, 2]")
    value = safe_float(alpha)
    if not math.isfinite(value) or not 0 < value <= 2:
        raise ValidationError("popularity_alpha must be finite and in (0, 2]")
    if strategy == "uniform" and value != 1.0:
        raise ValidationError("popularity_alpha applies only to popularity sampling")
    return strategy, value


class TrainingNegativeSampler:
    """Select unseen training-catalog negatives with replacement.

    Each user has a stable catalog-ordered pool.  Popularity weights are
    training event counts raised to ``alpha``; validation/test data are absent
    from this constructor and never influence the pools.
    """

    def __init__(
        self,
        dataset: InteractionDataset,
        *,
        seen: Mapping[EntityId, frozenset[EntityId]],
        catalog: tuple[EntityId, ...],
        strategy: str,
        alpha: float,
        epochs: int,
        draws_per_positive: int,
    ) -> None:
        self.strategy, self.alpha = validate_training_sampling(strategy, alpha)
        if self.strategy != "popularity":
            raise ValidationError("legacy uniform sampling does not need a training sampler")
        if type(dataset) is not InteractionDataset or not dataset:
            raise ValidationError("popularity sampler requires nonempty training interactions")
        if (
            not isinstance(seen, Mapping)
            or type(catalog) is not tuple
            or any(type(items) is not frozenset for items in seen.values())
        ):
            raise ValidationError("popularity sampler requires frozen training pools")
        if (
            type(epochs) is not int
            or epochs <= 0
            or type(draws_per_positive) is not int
            or draws_per_positive <= 0
        ):
            raise ValidationError("popularity sampler requires positive bounded draw parameters")
        if (
            len(dataset) > MAX_POP_TRAIN_INTERACTIONS
            or len(seen) > MAX_POP_USERS
            or len(catalog) > MAX_POP_ITEMS
            or len(seen) * len(catalog) > MAX_POOL_CHECKS
            or sum(map(len, seen.values())) * epochs * draws_per_positive > MAX_DRAWS
        ):
            raise ValidationError("popularity sampling resource limit exceeded")
        expected_seen = {
            user: frozenset(event.item_id for event in events)
            for user, events in dataset.by_user().items()
        }
        if tuple(catalog) != dataset.item_ids or dict(seen) != expected_seen:
            raise ValidationError("popularity sampler pools must match training interactions")
        counts = Counter(event.item_id for event in dataset)
        if set(counts) != set(catalog):
            raise ValidationError("popularity sampler catalog differs from training items")
        self._pools: dict[EntityId, tuple[EntityId, ...]] = {}
        self._cumulative: dict[EntityId, tuple[float, ...]] = {}
        for user, observed in seen.items():
            pool = tuple(item for item in catalog if item not in observed)
            cumulative: list[float] = []
            total = 0.0
            for item in pool:
                total += counts[item] ** self.alpha
                if not math.isfinite(total) or (cumulative and total <= cumulative[-1]):
                    raise ValidationError("popularity sampling weights lost precision")
                cumulative.append(total)
            self._pools[user] = pool
            self._cumulative[user] = tuple(cumulative)

    def sample(self, user: EntityId, rng: random.Random) -> EntityId:
        """Draw one negative from this user's nonempty pool."""
        if user not in self._pools:
            raise ValidationError("user is outside the training sampler")
        pool = self._pools[user]
        if not pool:
            raise ValidationError("user has no unseen training-catalog item")
        cumulative = self._cumulative[user]
        index = bisect.bisect_right(cumulative, rng.random() * cumulative[-1])
        return pool[min(index, len(pool) - 1)]

    def pool(self, user: EntityId) -> tuple[EntityId, ...]:
        return self._pools[user]
