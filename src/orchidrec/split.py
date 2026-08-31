"""Deterministic strategies for offline train/test splitting."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, InteractionDataset, stable_id_key
from orchidrec.errors import SplitError


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Train and test partitions produced by a split strategy."""

    train: InteractionDataset
    test: InteractionDataset

    def __post_init__(self) -> None:
        if not isinstance(self.train, InteractionDataset) or not isinstance(self.test, InteractionDataset):
            raise SplitError("train and test must be InteractionDataset instances")
        train_pairs = {(event.user_id, event.item_id) for event in self.train}
        test_pairs = {(event.user_id, event.item_id) for event in self.test}
        if train_pairs & test_pairs:
            raise SplitError("a user-item pair cannot appear in both train and test")

    @property
    def size(self) -> int:
        return len(self.train) + len(self.test)


def _test_size(length: int, test_ratio: float) -> int:
    if length < 2:
        raise SplitError("at least two interactions are required for a ratio split")
    if isinstance(test_ratio, bool) or not isinstance(test_ratio, (int, float)):
        raise SplitError("test_ratio must be a finite number between 0 and 1")
    ratio = safe_float(test_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise SplitError("test_ratio must be a finite number between 0 and 1")
    rounded = int(length * ratio + 0.5)
    return max(1, min(length - 1, rounded))


def random_split(dataset: InteractionDataset, test_ratio: float = 0.2, seed: int = 42) -> SplitResult:
    """Select seeded user-item groups and preserve event order in each partition."""

    if not isinstance(dataset, InteractionDataset):
        raise SplitError("dataset must be an InteractionDataset")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitError("seed must be an integer")
    n_test = _test_size(len(dataset), test_ratio)
    indices_by_pair: dict[tuple[EntityId, EntityId], list[int]] = defaultdict(list)
    for index, event in enumerate(dataset):
        indices_by_pair[(event.user_id, event.item_id)].append(index)
    pairs = list(indices_by_pair)
    if len(pairs) < 2:
        raise SplitError("a ratio split requires at least two distinct user-item pairs")
    random.Random(seed).shuffle(pairs)
    cumulative = 0
    prefix_sizes: list[int] = []
    for pair in pairs[:-1]:
        cumulative += len(indices_by_pair[pair])
        prefix_sizes.append(cumulative)
    prefix_length = min(
        range(1, len(pairs)),
        key=lambda length: (abs(prefix_sizes[length - 1] - n_test), length),
    )
    selected_pairs = set(pairs[:prefix_length])
    test_indices = frozenset(
        index
        for pair in selected_pairs
        for index in indices_by_pair[pair]
    )
    train = InteractionDataset(event for index, event in enumerate(dataset) if index not in test_indices)
    test = InteractionDataset(event for index, event in enumerate(dataset) if index in test_indices)
    return SplitResult(train=train, test=test)


def temporal_split(dataset: InteractionDataset, test_ratio: float = 0.2) -> SplitResult:
    """Put the newest pair-disjoint suffix of interactions into test."""

    if not isinstance(dataset, InteractionDataset):
        raise SplitError("dataset must be an InteractionDataset")
    n_test = _test_size(len(dataset), test_ratio)
    missing = [index for index, event in enumerate(dataset) if event.timestamp is None]
    if missing:
        raise SplitError("temporal_split requires a timestamp on every interaction")
    ordered = sorted(range(len(dataset)), key=lambda index: (dataset[index].timestamp, index))
    positions_by_pair: dict[tuple[EntityId, EntityId], list[int]] = defaultdict(list)
    for position, index in enumerate(ordered):
        event = dataset[index]
        positions_by_pair[(event.user_id, event.item_id)].append(position)
    crossing_delta = [0] * (len(ordered) + 1)
    for positions in positions_by_pair.values():
        if positions[0] < positions[-1]:
            crossing_delta[positions[0] + 1] += 1
            crossing_delta[positions[-1] + 1] -= 1
    valid_boundaries: list[int] = []
    crossing_pairs = 0
    for boundary in range(1, len(ordered)):
        crossing_pairs += crossing_delta[boundary]
        if crossing_pairs == 0:
            valid_boundaries.append(boundary)
    if not valid_boundaries:
        raise SplitError(
            "no temporal boundary can keep repeated user-item pairs in one partition"
        )
    boundary = min(
        valid_boundaries,
        key=lambda candidate: (abs((len(ordered) - candidate) - n_test), candidate),
    )
    train = InteractionDataset(dataset[index] for index in ordered[:boundary])
    test = InteractionDataset(dataset[index] for index in ordered[boundary:])
    return SplitResult(train=train, test=test)


def leave_one_out(dataset: InteractionDataset) -> SplitResult:
    """Hold out all events for each multi-item user's latest item.

    A user's timestamps are used only when all of that user's events have a
    timestamp.  Otherwise original input order provides the deterministic
    fallback.  Users with only one distinct item remain entirely in training.
    """

    if not isinstance(dataset, InteractionDataset):
        raise SplitError("dataset must be an InteractionDataset")
    if not dataset:
        raise SplitError("leave_one_out requires at least one interaction")
    indices_by_user: dict[EntityId, list[int]] = defaultdict(list)
    for index, event in enumerate(dataset):
        indices_by_user[event.user_id].append(index)
    test_indices: set[int] = set()
    for user_id in sorted(indices_by_user, key=stable_id_key):
        indices = indices_by_user[user_id]
        distinct_items = {dataset[index].item_id for index in indices}
        if len(distinct_items) < 2:
            continue
        if all(dataset[index].timestamp is not None for index in indices):
            held_out = max(indices, key=lambda index: (dataset[index].timestamp, index))
        else:
            held_out = indices[-1]
        held_out_item = dataset[held_out].item_id
        test_indices.update(
            index for index in indices if dataset[index].item_id == held_out_item
        )
    train = InteractionDataset(event for index, event in enumerate(dataset) if index not in test_indices)
    test = InteractionDataset(event for index, event in enumerate(dataset) if index in test_indices)
    return SplitResult(train=train, test=test)
