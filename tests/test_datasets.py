from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orchidrec.data import Interaction, InteractionDataset
from orchidrec.datasets import (
    interaction_fingerprint,
    load_dataset,
    load_json_dataset,
    load_movielens,
)
from orchidrec.errors import DatasetError

FIXTURES = Path(__file__).parent / "fixtures"


class DatasetAdapterTests(unittest.TestCase):
    def test_movielens_100k_directory_adapter_is_content_addressed(self) -> None:
        first = load_movielens(
            FIXTURES / "movielens-100k", format="movielens-100k", minimum_rating=4
        )
        second = load_dataset(
            FIXTURES / "movielens-100k" / "u.data",
            format="movielens-100k",
            minimum_rating=4.0,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.summary.source_rows, 18)
        self.assertEqual(first.summary.retained_interactions, 17)
        self.assertEqual(first.summary.dropped_interactions, 1)
        self.assertEqual(first.summary.users, 6)
        self.assertEqual(first.summary.items, 5)
        self.assertEqual(first.summary.rating_min, 2.0)
        self.assertEqual(first.summary.rating_max, 5.0)
        self.assertEqual(first.summary.timestamp_min, 1_000_000_001.0)
        self.assertEqual(first.summary.timestamp_max, 1_000_000_053.0)
        self.assertEqual(len(first.summary.source_sha256), 64)
        self.assertEqual(first.summary.interactions_sha256, interaction_fingerprint(first.dataset))
        self.assertEqual(first.summary.to_dict()["schema_version"], 1)

    def test_movielens_1m_adapter_uses_double_colon_layout(self) -> None:
        loaded = load_movielens(
            FIXTURES / "movielens-1m", format="movielens-1m", minimum_rating=None
        )
        self.assertEqual(len(loaded.dataset), 5)
        self.assertEqual(loaded.summary.dropped_interactions, 1)
        self.assertEqual(loaded.summary.minimum_rating, 4.0)
        self.assertTrue(all(event.value == 1.0 for event in loaded.dataset))

    def test_json_adapter_summarizes_missing_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interactions.json"
            expected = InteractionDataset(
                [Interaction("u", "a", 2.0), Interaction("v", "b", 3.0)]
            )
            expected.save_json(path)
            loaded = load_json_dataset(path)
            through_dispatch = load_dataset(path, format="orchidrec-json")
        self.assertEqual(loaded, through_dispatch)
        self.assertEqual(loaded.dataset, expected)
        self.assertIsNone(loaded.summary.timestamp_min)
        self.assertIsNone(loaded.summary.timestamp_max)
        self.assertEqual(loaded.summary.rating_min, 2.0)
        self.assertEqual(loaded.summary.rating_max, 3.0)

    def test_json_directory_uses_interactions_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            InteractionDataset([Interaction(1, 2)]).save_json(
                Path(directory) / "interactions.json"
            )
            loaded = load_dataset(directory, format="orchidrec-json")
        self.assertEqual(len(loaded.dataset), 1)

    def test_json_rejects_threshold_and_empty_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interactions.json"
            path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "only valid"):
                load_dataset(path, format="orchidrec-json", minimum_rating=4)
            with self.assertRaisesRegex(DatasetError, "no rows"):
                load_json_dataset(path)

    def test_interaction_fingerprint_is_order_sensitive_and_validated(self) -> None:
        forward = InteractionDataset([Interaction(1, 1), Interaction(2, 2)])
        reverse = InteractionDataset(reversed(forward.interactions))
        self.assertNotEqual(interaction_fingerprint(forward), interaction_fingerprint(reverse))
        with self.assertRaisesRegex(DatasetError, "InteractionDataset"):
            interaction_fingerprint([])  # type: ignore[arg-type]

    def test_missing_empty_and_unsupported_sources_fail_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(DatasetError, "does not exist"):
                load_movielens(missing, format="movielens-100k")
            empty = Path(directory) / "empty.data"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(DatasetError, "empty"):
                load_movielens(empty, format="movielens-100k")
            with self.assertRaisesRegex(DatasetError, "unsupported"):
                load_dataset(empty, format="unknown")  # type: ignore[arg-type]
            with self.assertRaisesRegex(DatasetError, "format must"):
                load_movielens(empty, format="bad")  # type: ignore[arg-type]

    def test_threshold_validation_and_all_dropped(self) -> None:
        path = FIXTURES / "movielens-1m" / "ratings.dat"
        for value in (True, "4", float("nan"), float("inf"), 0, 6, 10**400):
            with self.subTest(value=value), self.assertRaises(DatasetError):
                load_movielens(path, format="movielens-1m", minimum_rating=value)  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as directory:
            low = Path(directory) / "ratings.dat"
            low.write_text("1::1::1::1\n", encoding="ascii")
            with self.assertRaisesRegex(DatasetError, "removed every"):
                load_movielens(low, format="movielens-1m", minimum_rating=5)

    def test_strict_movielens_parse_errors_include_line(self) -> None:
        malformed = {
            "non-ascii": "1::1::5::é\n".encode(),
            "blank": b"1::1::5::1\n\n2::2::5::2\n",
            "fields": b"1::1::5\n",
            "signed": b"-1::1::5::1\n",
            "zero": b"0::1::5::1\n",
            "rating": b"1::1::6::1\n",
            "duplicate": b"1::1::5::1\n1::1::4::2\n",
            "huge-id": f"{'9' * 5_000}::1::5::1\n".encode(),
            "huge-timestamp": f"1::1::5::{'9' * 400}\n".encode(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.dat"
            for name, payload in malformed.items():
                with self.subTest(name=name):
                    path.write_bytes(payload)
                    with self.assertRaises(DatasetError):
                        load_movielens(path, format="movielens-1m")

    def test_invalid_json_is_wrapped_as_dataset_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interactions.json"
            path.write_text('{"not":"an array"}', encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "array"):
                load_json_dataset(path)


if __name__ == "__main__":
    unittest.main()
