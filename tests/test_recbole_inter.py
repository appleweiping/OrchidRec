from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from orchidrec.benchmark_config import benchmark_config_from_dict
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.datasets import (
    MAX_RECBOLE_INTER_BYTES,
    MAX_RECBOLE_INTER_LINE_BYTES,
    MAX_RECBOLE_INTER_ROWS,
    interaction_fingerprint,
    load_dataset,
    load_recbole_inter,
)
from orchidrec.errors import DatasetError
from orchidrec.split import temporal_split

FIXTURE = Path(__file__).parent / "fixtures" / "recbole-inter" / "synthetic.inter"
HEADER = b"user_id:token\titem_id:token\trating:float\ttimestamp:float\n"


class RecBoleInteractionAdapterTests(unittest.TestCase):
    def test_synthetic_fixture_has_independent_oracle_and_stable_hashes(self) -> None:
        oracle = InteractionDataset(
            [
                Interaction("001", "A", 1.0, 100.0),
                Interaction("001", "B", 1.0, 110.0),
                Interaction("1", "A", 1.0, 130.0),
                Interaction("1", "B", 1.0, 140.0),
                Interaction("1", "C", 1.0, 150.0),
            ]
        )
        direct = load_recbole_inter(FIXTURE, minimum_rating=3.5)
        directory = load_dataset(FIXTURE.parent, format="recbole-inter", minimum_rating=3.5)
        alias = load_dataset(
            FIXTURE.parent / "." / FIXTURE.name, format="recbole-inter", minimum_rating=3.5
        )
        self.assertEqual(direct, directory)
        self.assertEqual(direct, alias)
        self.assertEqual(direct.dataset, oracle)
        self.assertEqual(direct.dataset.user_ids, ("001", "1"))
        self.assertEqual(
            direct.summary.source_sha256, hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        )
        self.assertEqual(direct.summary.interactions_sha256, interaction_fingerprint(oracle))
        self.assertEqual(direct.summary.source_name, f"sha256-{direct.summary.source_sha256}.inter")
        self.assertEqual(direct.summary.source_rows, 6)
        self.assertEqual(direct.summary.retained_interactions, 5)
        self.assertEqual(direct.summary.dropped_interactions, 1)
        self.assertEqual(direct.summary.rating_min, 1.0)
        self.assertEqual(direct.summary.rating_max, 5.0)
        self.assertEqual(direct.summary.timestamp_min, 100.0)
        self.assertEqual(direct.summary.timestamp_max, 150.0)
        self.assertEqual(direct.summary.minimum_rating, 3.5)
        self.assertEqual(direct.summary.to_dict()["format"], "recbole-inter")

    def test_independent_movielens_layout_yields_equivalent_normalized_events(self) -> None:
        # Two layouts are independently specified here; expected values are not
        # computed from either adapter's output.
        inter = HEADER + b"1\t10\t5\t100\n1\t11\t4\t110\n2\t10\t5\t120\n2\t11\t4\t130\n"
        movie = b"1\t10\t5\t100\n1\t11\t4\t110\n2\t10\t5\t120\n2\t11\t4\t130\n"
        expected = [
            {"user_id": "1", "item_id": "10", "value": 1.0, "timestamp": 100.0},
            {"user_id": "1", "item_id": "11", "value": 1.0, "timestamp": 110.0},
            {"user_id": "2", "item_id": "10", "value": 1.0, "timestamp": 120.0},
            {"user_id": "2", "item_id": "11", "value": 1.0, "timestamp": 130.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "events.inter").write_bytes(inter)
            (root / "u.data").write_bytes(movie)
            recbole = load_dataset(root / "events.inter", format="recbole-inter", minimum_rating=4)
            movielens = load_dataset(root / "u.data", format="movielens-100k", minimum_rating=4)
        self.assertEqual(recbole.dataset.to_records(), expected)
        self.assertEqual(
            [
                {**row, "user_id": str(row["user_id"]), "item_id": str(row["item_id"])}
                for row in movielens.dataset.to_records()
            ],
            expected,
        )
        self.assertEqual(recbole.summary.source_rows, movielens.summary.source_rows)

    def test_unrated_and_unstamped_events_use_unit_implicit_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.inter"
            path.write_bytes(b"item_id:token\tuser_id:token\nA\t001\nB\t1")
            loaded = load_recbole_inter(path)
            self.assertEqual(
                loaded.dataset.to_records(),
                [
                    {"user_id": "001", "item_id": "A", "value": 1.0, "timestamp": None},
                    {"user_id": "1", "item_id": "B", "value": 1.0, "timestamp": None},
                ],
            )
            self.assertEqual(loaded.summary.rating_min, 1.0)
            self.assertEqual(loaded.summary.rating_max, 1.0)
            self.assertIsNone(loaded.summary.minimum_rating)
            self.assertIsNone(loaded.summary.timestamp_min)
            self.assertIsNone(loaded.summary.timestamp_max)
            with self.assertRaisesRegex(DatasetError, "requires a rating"):
                load_recbole_inter(path, minimum_rating=0)

    def test_arbitrary_rating_scale_and_threshold_extremes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.inter"
            path.write_bytes(
                b"user_id:token\titem_id:token\trating:float\na\tx\t-100\na\ty\t0\na\tz\t100\n"
            )
            all_rows = load_recbole_inter(path, minimum_rating=-100)
            self.assertEqual(len(all_rows.dataset), 3)
            at_high = load_recbole_inter(path, minimum_rating=100)
            self.assertEqual(at_high.dataset.to_records()[0]["item_id"], "z")
            self.assertEqual(at_high.summary.dropped_interactions, 2)
            with self.assertRaisesRegex(DatasetError, "removed every"):
                load_recbole_inter(path, minimum_rating=101)
            with self.assertRaisesRegex(DatasetError, "requires explicit"):
                load_recbole_inter(path)
            for invalid in (True, "4", float("nan"), float("inf"), 10**400):
                with self.subTest(invalid=invalid), self.assertRaises(DatasetError):
                    load_recbole_inter(path, minimum_rating=invalid)  # type: ignore[arg-type]

    def test_header_errors_reject_duplicate_incompatible_and_unknown_fields(self) -> None:
        cases = (
            (b"user_id:token\tuser_id:token\titem_id:token\n", "duplicate"),
            (b"user_id:float\titem_id:token\n", "unsupported"),
            (b"user_id:token\titem_id:float\n", "unsupported"),
            (b"user_id:token\titem_id:token\treview:token_seq\n", "unsupported"),
            (b"user_id:token\titem_id:token\tvalue:float\n", "unsupported"),
            (b"user_id:token\n", "required"),
            (b"item_id:token\n", "required"),
            (b"user_id:token\titem_id:token\t\n", "unsupported"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.inter"
            for header, message in cases:
                with self.subTest(header=header):
                    path.write_bytes(header + b"a\tb\n")
                    with self.assertRaisesRegex(DatasetError, message):
                        load_recbole_inter(path)

    def test_malformed_rows_have_physical_line_numbers(self) -> None:
        cases = (
            (b"u\ti\t4\t1\n\n", "line 3: blank"),
            (b"u\ti\t4\t1\nu\n", "line 3: expected"),
            (b"\ti\t4\t1\n", "line 2: user"),
            (b"u\t \t4\t1\n", "line 2: user"),
            (b"u\ti\t4\t1\nu\ti\t4\t2\n", "line 3: duplicate"),
            (b"u\ti\tNaN\t1\n", "line 2: rating"),
            (b"u\ti\tinf\t1\n", "line 2: rating"),
            (b"u\ti\t4\t-inf\n", "line 2: timestamp"),
            (b"u\ti\t4\t-1\n", "line 2: timestamp"),
            (b"u\ti\t\t1\n", "line 2: rating"),
            (b"u\ti\t4\t\n", "line 2: timestamp"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.inter"
            for rows, message in cases:
                with self.subTest(rows=rows):
                    path.write_bytes(HEADER + rows)
                    with self.assertRaisesRegex(DatasetError, message):
                        load_recbole_inter(path, minimum_rating=4)

    def test_utf8_crlf_empty_file_empty_data_and_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "events.inter"
            path.write_bytes(b"user_id:token\titem_id:token\r\n\xc3\xa9\ti\r\n")
            loaded = load_recbole_inter(path)
            self.assertEqual(loaded.dataset[0].user_id, "é")
            self.assertEqual(loaded.summary.source_bytes, len(path.read_bytes()))
            path.write_bytes(b"user_id:token\titem_id:token\n\xff\ti\n")
            with self.assertRaisesRegex(DatasetError, "UTF-8"):
                load_recbole_inter(path)
            path.write_bytes(b"")
            with self.assertRaisesRegex(DatasetError, "empty"):
                load_recbole_inter(path)
            path.write_bytes(b"user_id:token\titem_id:token\n")
            with self.assertRaisesRegex(DatasetError, "no data rows"):
                load_recbole_inter(path)
            wrong = root / "events.txt"
            wrong.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(DatasetError, "suffix"):
                load_recbole_inter(wrong)

    def test_bare_carriage_returns_fail_in_header_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.inter"
            for contents, message in (
                (b"user_id:token\titem_id:token\ru\ti\r", "line 1: bare carriage"),
                (b"user_id:token\titem_id:token\nu\ri\n", "line 2: bare carriage"),
                (b"user_id:token\titem_id:token\nu\ti\r\r\n", "line 2: bare carriage"),
                (b"user_id:token\titem_id:token\nu\ti\r", "line 2: bare carriage"),
            ):
                with self.subTest(contents=contents):
                    path.write_bytes(contents)
                    with self.assertRaisesRegex(DatasetError, message):
                        load_recbole_inter(path)

    def test_content_addressed_source_name_is_stable_under_renamed_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.inter"
            alias = root / "renamed.inter"
            copied = root / "copied.inter"
            first.write_bytes(FIXTURE.read_bytes())
            alias.hardlink_to(first)
            copied.write_bytes(first.read_bytes())
            expected = load_recbole_inter(first, minimum_rating=3.5)
            self.assertEqual(load_recbole_inter(alias, minimum_rating=3.5), expected)
            self.assertEqual(load_recbole_inter(copied, minimum_rating=3.5), expected)
            self.assertEqual(
                expected.summary, load_recbole_inter(FIXTURE, minimum_rating=3.5).summary
            )

    def test_directory_selection_requires_exactly_one_inter_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(DatasetError, "exactly one"):
                load_recbole_inter(root)
            (root / "not-a-file.inter").mkdir()
            with self.assertRaisesRegex(DatasetError, "exactly one"):
                load_recbole_inter(root)
            (root / "a.inter").write_bytes(b"user_id:token\titem_id:token\nu\ti\n")
            self.assertEqual(len(load_recbole_inter(root).dataset), 1)
            (root / "b.inter").write_bytes(b"user_id:token\titem_id:token\nu\ti\n")
            with self.assertRaisesRegex(DatasetError, "exactly one"):
                load_recbole_inter(root)
            with self.assertRaisesRegex(DatasetError, "does not exist"):
                load_recbole_inter(root / "missing.inter")

    def test_source_limits_precede_decoding_and_row_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.inter"
            path.write_bytes(b"x" * (MAX_RECBOLE_INTER_BYTES + 1))
            with self.assertRaisesRegex(DatasetError, "exceeds.*bytes"):
                load_recbole_inter(path)
            path.write_bytes(b"x" * (MAX_RECBOLE_INTER_LINE_BYTES + 1))
            with self.assertRaisesRegex(DatasetError, "line 1:.*exceeds"):
                load_recbole_inter(path)
            path.write_bytes(
                b"user_id:token\titem_id:token\n" + b"u\ti\n" * (MAX_RECBOLE_INTER_ROWS + 1)
            )
            with self.assertRaisesRegex(DatasetError, "data rows"):
                load_recbole_inter(path)

    def test_benchmark_config_and_temporal_split_preserve_float_timestamps(self) -> None:
        payload: dict[str, object] = {
            "schema_version": 1,
            "data": {"path": str(FIXTURE), "format": "recbole-inter", "minimum_rating": 3.5},
            "split": {"method": "temporal", "test_ratio": 0.4},
            "seed": 8,
        }
        config = benchmark_config_from_dict(payload)
        self.assertEqual(config.data.minimum_rating, 3.5)
        self.assertEqual(config.data.format, "recbole-inter")
        self.assertEqual(benchmark_config_from_dict(config.to_dict()), config)
        loaded = load_dataset(
            config.data.path, format=config.data.format, minimum_rating=config.data.minimum_rating
        )
        split = temporal_split(loaded.dataset, test_ratio=0.4)
        for user_id in loaded.dataset.user_ids:
            train_times = [event.timestamp for event in split.train if event.user_id == user_id]
            test_times = [event.timestamp for event in split.test if event.user_id == user_id]
            if train_times and test_times:
                self.assertLess(max(train_times), min(test_times))
        no_threshold = benchmark_config_from_dict(
            {"schema_version": 1, "data": {"path": "events.inter", "format": "recbole-inter"}}
        )
        self.assertIsNone(no_threshold.data.minimum_rating)
        wide = benchmark_config_from_dict(
            {
                "schema_version": 1,
                "data": {"path": "events.inter", "format": "recbole-inter", "minimum_rating": -10},
            }
        )
        self.assertEqual(wide.data.minimum_rating, -10.0)


if __name__ == "__main__":
    unittest.main()
