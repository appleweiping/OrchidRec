from __future__ import annotations

import random
import unittest

from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import SplitError
from orchidrec.split import SplitResult, leave_one_out, random_split, temporal_split


def sample_dataset(size: int = 12) -> InteractionDataset:
    return InteractionDataset(
        Interaction(user_id=f"u{index % 3}", item_id=f"i{index}", timestamp=float(index))
        for index in range(size)
    )


def pairs(dataset: InteractionDataset) -> set[tuple[object, object]]:
    return {(event.user_id, event.item_id) for event in dataset}


def repeated_pair_dataset() -> InteractionDataset:
    return InteractionDataset(
        [
            Interaction("u", "a", timestamp=1),
            Interaction("u", "b", timestamp=2),
            Interaction("u", "a", timestamp=3),
            Interaction("v", "c", timestamp=4),
        ]
    )


class RandomSplitTests(unittest.TestCase):
    def test_split_result_rejects_pair_overlap(self) -> None:
        with self.assertRaises(SplitError):
            SplitResult(
                train=InteractionDataset([Interaction("u", "a")]),
                test=InteractionDataset([Interaction("u", "a")]),
            )

    def test_random_split_is_deterministic(self) -> None:
        dataset = sample_dataset()
        self.assertEqual(random_split(dataset, 0.25, 7), random_split(dataset, 0.25, 7))

    def test_random_split_does_not_change_global_random_state(self) -> None:
        random.seed(991)
        state = random.getstate()
        random_split(sample_dataset(), 0.25, 7)
        self.assertEqual(random.getstate(), state)

    def test_different_seeds_change_partition(self) -> None:
        dataset = sample_dataset()
        self.assertNotEqual(random_split(dataset, 0.25, 1).test, random_split(dataset, 0.25, 2).test)

    def test_split_preserves_every_record_once(self) -> None:
        dataset = sample_dataset()
        result = random_split(dataset, 0.25, 4)
        self.assertEqual(result.size, len(dataset))
        self.assertEqual(set(result.train) | set(result.test), set(dataset))
        self.assertFalse(set(result.train) & set(result.test))

    def test_repeated_user_item_pair_never_crosses_partitions(self) -> None:
        result = random_split(repeated_pair_dataset(), 0.5, 1)
        self.assertTrue(pairs(result.train).isdisjoint(pairs(result.test)))

    def test_small_nonzero_ratio_still_holds_out_one(self) -> None:
        result = random_split(sample_dataset(5), 0.01, 1)
        self.assertEqual(len(result.test), 1)
        self.assertEqual(len(result.train), 4)

    def test_invalid_ratios_are_rejected(self) -> None:
        for ratio in (0.0, 1.0, -0.5, 2.0, float("nan"), True, 10**400):
            with self.subTest(ratio=ratio), self.assertRaises(SplitError):
                random_split(sample_dataset(), ratio, 1)  # type: ignore[arg-type]

    def test_invalid_seed_is_rejected(self) -> None:
        with self.assertRaises(SplitError):
            random_split(sample_dataset(), 0.2, True)

    def test_ratio_split_requires_two_events(self) -> None:
        with self.assertRaises(SplitError):
            random_split(sample_dataset(1), 0.2, 1)


class TemporalSplitTests(unittest.TestCase):
    def test_newest_events_are_test_data(self) -> None:
        result = temporal_split(sample_dataset(10), 0.3)
        self.assertEqual([event.timestamp for event in result.test], [7.0, 8.0, 9.0])

    def test_input_order_does_not_override_timestamps(self) -> None:
        dataset = InteractionDataset(
            [Interaction("u", "new", timestamp=10), Interaction("u", "old", timestamp=1)]
        )
        result = temporal_split(dataset, 0.5)
        self.assertEqual(result.test[0].item_id, "new")

    def test_equal_timestamps_use_input_order(self) -> None:
        dataset = InteractionDataset(
            [Interaction("u", "first", timestamp=1), Interaction("u", "second", timestamp=1)]
        )
        self.assertEqual(temporal_split(dataset, 0.5).test[0].item_id, "second")

    def test_missing_timestamp_is_rejected(self) -> None:
        dataset = InteractionDataset([Interaction("u", "a", timestamp=1), Interaction("u", "b")])
        with self.assertRaises(SplitError):
            temporal_split(dataset, 0.5)

    def test_repeated_user_item_pair_never_crosses_temporal_boundary(self) -> None:
        dataset = InteractionDataset(
            [
                Interaction("u", "a", timestamp=1),
                Interaction("u", "a", timestamp=2),
                Interaction("u", "b", timestamp=3),
                Interaction("v", "c", timestamp=4),
            ]
        )
        result = temporal_split(dataset, 0.5)
        self.assertTrue(pairs(result.train).isdisjoint(pairs(result.test)))
        self.assertLessEqual(result.train[-1].timestamp, result.test[0].timestamp)  # type: ignore[operator]

    def test_interleaved_pairs_without_safe_temporal_boundary_are_rejected(self) -> None:
        dataset = InteractionDataset(
            [
                Interaction("u", "a", timestamp=1),
                Interaction("u", "b", timestamp=2),
                Interaction("u", "a", timestamp=3),
                Interaction("u", "b", timestamp=4),
            ]
        )
        with self.assertRaises(SplitError):
            temporal_split(dataset, 0.5)


class LeaveOneOutTests(unittest.TestCase):
    def test_latest_event_per_user_is_held_out(self) -> None:
        dataset = InteractionDataset(
            [
                Interaction("u1", "a", timestamp=2),
                Interaction("u1", "b", timestamp=5),
                Interaction("u2", "c", timestamp=7),
                Interaction("u2", "d", timestamp=3),
            ]
        )
        result = leave_one_out(dataset)
        self.assertEqual({event.item_id for event in result.test}, {"b", "c"})

    def test_missing_timestamp_falls_back_to_input_order(self) -> None:
        dataset = InteractionDataset(
            [Interaction("u", "timestamped", timestamp=99), Interaction("u", "last", timestamp=None)]
        )
        self.assertEqual(leave_one_out(dataset).test[0].item_id, "last")

    def test_single_event_user_remains_in_training(self) -> None:
        dataset = InteractionDataset(
            [
                Interaction("single", "a", timestamp=1),
                Interaction("other", "b", timestamp=1),
                Interaction("other", "c", timestamp=2),
            ]
        )
        result = leave_one_out(dataset)
        self.assertIn(dataset[0], result.train)
        self.assertNotIn(dataset[0], result.test)

    def test_leave_one_out_is_deterministic(self) -> None:
        dataset = sample_dataset()
        self.assertEqual(leave_one_out(dataset), leave_one_out(dataset))

    def test_leave_one_out_preserves_size(self) -> None:
        dataset = sample_dataset()
        self.assertEqual(leave_one_out(dataset).size, len(dataset))

    def test_leave_one_out_holds_all_events_for_latest_item(self) -> None:
        result = leave_one_out(repeated_pair_dataset())
        self.assertTrue(pairs(result.train).isdisjoint(pairs(result.test)))
        self.assertEqual(
            [event.item_id for event in result.test if event.user_id == "u"],
            ["a", "a"],
        )

    def test_user_with_repeats_of_only_one_item_stays_in_training(self) -> None:
        dataset = InteractionDataset(
            [Interaction("u", "a", timestamp=1), Interaction("u", "a", timestamp=2)]
        )
        result = leave_one_out(dataset)
        self.assertEqual(result.train, dataset)
        self.assertFalse(result.test)

    def test_empty_dataset_is_rejected(self) -> None:
        with self.assertRaises(SplitError):
            leave_one_out(InteractionDataset())


if __name__ == "__main__":
    unittest.main()
