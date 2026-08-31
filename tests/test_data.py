from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from orchidrec.data import Interaction, InteractionDataset, StableIdMap
from orchidrec.errors import SerializationError, ValidationError


class InteractionTests(unittest.TestCase):
    def test_numeric_values_are_normalized_to_float(self) -> None:
        event = Interaction(user_id=1, item_id="book", value=2, timestamp=10)
        self.assertEqual(event.value, 2.0)
        self.assertEqual(event.timestamp, 10.0)

    def test_integer_and_string_ids_are_distinct(self) -> None:
        first = Interaction(user_id=1, item_id="1")
        second = Interaction(user_id="1", item_id=1)
        self.assertNotEqual(first.user_id, second.user_id)

    def test_boolean_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Interaction(user_id=True, item_id="x")

    def test_empty_string_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Interaction(user_id="", item_id="x")

    def test_nonpositive_or_nonfinite_value_is_rejected(self) -> None:
        for value in (0, -1, math.inf, math.nan, True, 10**400):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Interaction(user_id="u", item_id="i", value=value)  # type: ignore[arg-type]

    def test_nonfinite_timestamp_is_rejected(self) -> None:
        for timestamp in (math.inf, 10**400):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValidationError):
                Interaction(user_id="u", item_id="i", timestamp=timestamp)

    def test_from_record_supplies_defaults(self) -> None:
        event = Interaction.from_record({"user_id": "u", "item_id": "i"})
        self.assertEqual(event.value, 1.0)
        self.assertIsNone(event.timestamp)

    def test_from_record_rejects_missing_fields(self) -> None:
        with self.assertRaises(ValidationError):
            Interaction.from_record({"user_id": "u"})

    def test_from_record_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            Interaction.from_record({"user_id": "u", "item_id": "i", "typo": 1})

    def test_from_record_rejects_non_string_field_names(self) -> None:
        with self.assertRaises(ValidationError):
            Interaction.from_record({"user_id": "u", "item_id": "i", 3: "bad"})  # type: ignore[dict-item]


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = InteractionDataset(
            [
                Interaction("u2", "b", 2.0, 2.0),
                Interaction("u1", "a", 1.0, 1.0),
                Interaction("u1", "b", 3.0, 3.0),
            ]
        )

    def test_dataset_rejects_non_interaction_elements(self) -> None:
        with self.assertRaises(ValidationError):
            InteractionDataset([{"user_id": "u"}])  # type: ignore[list-item]

    def test_dataset_rejects_string_iterable_even_when_empty(self) -> None:
        with self.assertRaises(ValidationError):
            InteractionDataset("")  # type: ignore[arg-type]

    def test_ids_are_stably_sorted(self) -> None:
        self.assertEqual(self.dataset.user_ids, ("u1", "u2"))
        self.assertEqual(self.dataset.item_ids, ("a", "b"))

    def test_by_user_preserves_event_order(self) -> None:
        groups = self.dataset.by_user()
        self.assertEqual([event.item_id for event in groups["u1"]], ["a", "b"])

    def test_seen_items_for_unknown_user_is_empty(self) -> None:
        self.assertEqual(self.dataset.seen_items("missing"), frozenset())

    def test_item_counts_can_be_weighted(self) -> None:
        self.assertEqual(self.dataset.item_counts(), {"a": 1.0, "b": 2.0})
        self.assertEqual(self.dataset.item_counts(weighted=True), {"a": 1.0, "b": 5.0})

    def test_weighted_item_count_overflow_is_rejected(self) -> None:
        dataset = InteractionDataset(
            [Interaction("u1", "a", 1e308), Interaction("u2", "a", 1e308)]
        )
        with self.assertRaises(ValidationError):
            dataset.item_counts(weighted=True)

    def test_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "events.json"
            self.dataset.save_json(path)
            restored = InteractionDataset.load_json(path)
        self.assertEqual(restored, self.dataset)

    def test_malformed_json_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(SerializationError):
                InteractionDataset.load_json(path)

    def test_duplicate_interaction_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                '[{"user_id":"u1","user_id":"u2","item_id":"a"}]',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SerializationError, "duplicate JSON object key"):
                InteractionDataset.load_json(path)

    def test_non_array_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"interactions": []}), encoding="utf-8")
            with self.assertRaises(SerializationError):
                InteractionDataset.load_json(path)

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaises(SerializationError):
            InteractionDataset.load_json("definitely-not-an-orchidrec-file.json")


class StableIdMapTests(unittest.TestCase):
    def test_mapping_is_independent_of_input_order(self) -> None:
        first = StableIdMap.from_values(["b", "a", 2, 1])
        second = StableIdMap.from_values([1, "a", 2, "b"])
        self.assertEqual(first, second)
        self.assertEqual(first.ids, (1, 2, "a", "b"))

    def test_duplicate_ids_are_collapsed(self) -> None:
        mapping = StableIdMap.from_values(["a", "a", "b"])
        self.assertEqual(mapping.ids, ("a", "b"))

    def test_direct_constructor_requires_canonical_ids(self) -> None:
        for ids in (("b", "a"), ("a", "a")):
            with self.subTest(ids=ids), self.assertRaises(ValidationError):
                StableIdMap(ids)

    def test_id_values_reject_a_bare_string_iterable(self) -> None:
        with self.assertRaises(ValidationError):
            StableIdMap.from_values("ab")

    def test_membership_does_not_conflate_boolean_and_integer_ids(self) -> None:
        mapping = StableIdMap.from_values([1])
        self.assertIn(1, mapping)
        self.assertNotIn(True, mapping)

    def test_transform_and_inverse_transform(self) -> None:
        mapping = StableIdMap.from_values(["b", "a"])
        encoded = mapping.transform(["b", "a"])
        self.assertEqual(encoded, (1, 0))
        self.assertEqual(mapping.inverse_transform(encoded), ("b", "a"))

    def test_unknown_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            StableIdMap.from_values(["a"]).index_of("missing")

    def test_invalid_indices_are_rejected(self) -> None:
        mapping = StableIdMap.from_values(["a"])
        for index in (-1, 1, True):
            with self.subTest(index=index), self.assertRaises(ValidationError):
                mapping.id_at(index)  # type: ignore[arg-type]

    def test_state_round_trip(self) -> None:
        mapping = StableIdMap.from_values([3, 1, "x"])
        self.assertEqual(StableIdMap.from_state(mapping.to_state()), mapping)

    def test_unsorted_state_is_rejected(self) -> None:
        with self.assertRaises(SerializationError):
            StableIdMap.from_state({"ids": ["z", "a"]})

    def test_duplicate_state_is_rejected(self) -> None:
        with self.assertRaises(SerializationError):
            StableIdMap.from_state({"ids": ["a", "a"]})


if __name__ == "__main__":
    unittest.main()
