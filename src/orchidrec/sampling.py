"""Bounded, reproducible candidate sampling for implicit-feedback evaluation.

Sampled ranking is a different estimand from full-catalog ranking.  This
module keeps the candidate plan explicit so that model comparisons use the
same positives and negatives, and so that training positives can never be
silently treated as negatives.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from orchidrec.data import EntityId, stable_id_key, validate_entity_id
from orchidrec.errors import ConfigurationError, ValidationError

MAX_CATALOG_ITEMS = 100_000
MAX_EVALUATION_USERS = 100_000
MAX_CANDIDATE_PAIRS = 5_000_000
MAX_NEGATIVES_PER_USER = 10_000
MAX_SAMPLING_WORK_UNITS = 20_000_000
MAX_TRAIN_ITEM_COUNT = 1_000_000_000
MAX_ID_UTF8_BYTES = 4096
MAX_ID_BITS = 4096


def _checked_id(value: object, name: str) -> EntityId:
    entity_id = validate_entity_id(value, name)
    if isinstance(entity_id, int):
        if entity_id.bit_length() > MAX_ID_BITS:
            raise ValidationError(f"{name} exceeds {MAX_ID_BITS} bits")
    else:
        try:
            encoded = entity_id.encode("utf-8")
        except UnicodeError as exc:
            raise ValidationError(f"{name} must be valid UTF-8") from exc
        if len(encoded) > MAX_ID_UTF8_BYTES:
            raise ValidationError(f"{name} exceeds {MAX_ID_UTF8_BYTES} UTF-8 bytes")
    return entity_id


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Opt-in, finite candidate universe for each evaluated user."""

    strategy: str
    negatives: int

    def __post_init__(self) -> None:
        if type(self.strategy) is not str or self.strategy not in SAMPLER_REGISTRY:
            raise ConfigurationError("sampling.strategy must be uniform or popularity")
        if type(self.negatives) is not int or not 1 <= self.negatives <= MAX_NEGATIVES_PER_USER:
            raise ConfigurationError(
                f"sampling.negatives must be between 1 and {MAX_NEGATIVES_PER_USER}"
            )

    def to_dict(self) -> dict[str, str | int]:
        return {"strategy": self.strategy, "negatives": self.negatives}


def parse_sampling(
    value: object, *, location: str = "evaluation.sampling"
) -> SamplingConfig | None:
    """Parse the optional strict JSON sampling configuration."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ConfigurationError(f"{location} must be an object or null")
    extra = set(value) - {"strategy", "negatives"}
    if extra or set(value) != {"strategy", "negatives"}:
        raise ConfigurationError(f"{location} requires only strategy and negatives")
    try:
        return SamplingConfig(value["strategy"], value["negatives"])
    except ConfigurationError as exc:
        raise ConfigurationError(f"invalid {location}: {exc}") from exc


def _bounded_ids(
    values: Iterable[EntityId], limit: int, name: str, *, unique: bool = False
) -> frozenset[EntityId]:
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{name} must be an iterable of item IDs")
    try:
        materialized = [_checked_id(value, name) for value in itertools.islice(values, limit + 1)]
    except TypeError as exc:
        raise ValidationError(f"{name} must be an iterable of item IDs") from exc
    if len(materialized) > limit:
        raise ValidationError(f"{name} exceeds {limit} items")
    frozen = frozenset(materialized)
    if unique and len(frozen) != len(materialized):
        raise ValidationError(f"{name} must not contain duplicate item IDs")
    return frozen


def _uniform(
    pool: tuple[EntityId, ...], count: int, rng: random.Random, _weights: Mapping[EntityId, int]
) -> tuple[EntityId, ...]:
    return tuple(sorted(rng.sample(pool, count), key=stable_id_key))


def _popularity(
    pool: tuple[EntityId, ...], count: int, rng: random.Random, weights: Mapping[EntityId, int]
) -> tuple[EntityId, ...]:
    # Exponential race is exact weighted sampling without replacement.  The
    # weights come exclusively from the training split, never holdout counts.
    keys = ((-math.log1p(-rng.random()) / weights[item], item) for item in pool)
    selected = sorted(keys, key=lambda row: (row[0], stable_id_key(row[1])))[:count]
    return tuple(sorted((item for _, item in selected), key=stable_id_key))


Sampler = Callable[
    [tuple[EntityId, ...], int, random.Random, Mapping[EntityId, int]], tuple[EntityId, ...]
]
SAMPLER_REGISTRY: Mapping[str, Sampler] = MappingProxyType(
    {"uniform": _uniform, "popularity": _popularity}
)


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    """Shared user-specific candidates and their auditable identity."""

    sampling: SamplingConfig
    candidates: Mapping[EntityId, tuple[EntityId, ...]]
    fingerprint: str
    candidate_pairs: int
    positive_pairs: int
    negative_pairs: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", MappingProxyType(dict(self.candidates)))


def sample_candidates(
    config: SamplingConfig,
    *,
    seed: int,
    catalog: Iterable[EntityId],
    train_counts: Mapping[EntityId, int],
    relevant: Mapping[EntityId, Collection[EntityId]],
    seen: Mapping[EntityId, Collection[EntityId]],
) -> CandidatePlan:
    """Include every held-out positive and sample unseen, non-positive items.

    Calls for the same user are independent of user order.  If fewer than the
    requested negatives exist, all eligible negatives are used.  A user with
    no negatives remains evaluable on positives only, never via replacement.
    """

    if not isinstance(config, SamplingConfig):
        raise ValidationError("config must be a SamplingConfig")
    if type(seed) is not int or not -(2**63) <= seed < 2**63:
        raise ValidationError("seed must be a signed 64-bit integer")
    catalog_set = _bounded_ids(catalog, MAX_CATALOG_ITEMS, "catalog", unique=True)
    if not catalog_set:
        raise ValidationError("catalog must not be empty")
    ordered_catalog = tuple(sorted(catalog_set, key=stable_id_key))
    if (
        not isinstance(train_counts, Mapping)
        or not isinstance(relevant, Mapping)
        or not isinstance(seen, Mapping)
    ):
        raise ValidationError("train_counts, relevant, and seen must be mappings")
    if not relevant:
        raise ValidationError("relevant must contain at least one evaluated user")
    if len(relevant) > MAX_EVALUATION_USERS:
        raise ValidationError("too many evaluated users")
    if len(seen) != len(relevant):
        raise ValidationError("seen must contain exactly the evaluated relevant users")
    if len(relevant) * len(catalog_set) > MAX_SAMPLING_WORK_UNITS:
        raise ValidationError(f"sampling work exceeds {MAX_SAMPLING_WORK_UNITS} user-item checks")
    counts: dict[EntityId, int] = {}
    if len(train_counts) != len(catalog_set):
        raise ValidationError("training counts must cover the catalog exactly")
    for item, count in train_counts.items():
        item_id = _checked_id(item, "training count item")
        if (
            item_id not in catalog_set
            or type(count) is not int
            or not 0 < count <= MAX_TRAIN_ITEM_COUNT
        ):
            raise ValidationError("training counts must be positive integers for catalog items")
        counts[item_id] = count
    if len(counts) != len(catalog_set):
        raise ValidationError("training counts must cover the catalog exactly")
    relevant_users = {_checked_id(user, "relevant user") for user in relevant}
    seen_users = {_checked_id(user, "seen user") for user in seen}
    if (
        len(relevant_users) != len(relevant)
        or len(seen_users) != len(seen)
        or relevant_users != seen_users
    ):
        raise ValidationError("seen must contain exactly the evaluated relevant users")
    rows: dict[EntityId, tuple[EntityId, ...]] = {}
    digest = hashlib.sha256()
    candidate_pairs = positive_pairs = negative_pairs = 0
    for raw_user in sorted(relevant, key=stable_id_key):
        user = _checked_id(raw_user, "relevant user")
        positives = _bounded_ids(relevant[raw_user], MAX_CATALOG_ITEMS, "relevant")
        seen_items = _bounded_ids(seen[user], MAX_CATALOG_ITEMS, "seen")
        if not positives or not positives <= catalog_set or not seen_items <= catalog_set:
            raise ValidationError("relevant must be nonempty and all IDs must be in catalog")
        if positives & seen_items:
            raise ValidationError("training positives overlap held-out positives")
        pool = tuple(
            item for item in ordered_catalog if item not in positives and item not in seen_items
        )
        count = min(config.negatives, len(pool))
        identity = json.dumps([seed, user], ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        rng = random.Random(int.from_bytes(hashlib.sha256(identity).digest(), "big"))
        negatives = SAMPLER_REGISTRY[config.strategy](pool, count, rng, counts)
        selected = tuple(sorted(positives | set(negatives), key=stable_id_key))
        candidate_pairs += len(selected)
        positive_pairs += len(positives)
        negative_pairs += len(negatives)
        if candidate_pairs > MAX_CANDIDATE_PAIRS:
            raise ValidationError(f"candidate pairs exceed {MAX_CANDIDATE_PAIRS}")
        rows[user] = selected
        digest.update(
            json.dumps([user, selected], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return CandidatePlan(
        config, rows, digest.hexdigest(), candidate_pairs, positive_pairs, negative_pairs
    )
