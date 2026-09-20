"""Independent synthetic/adversarial oracles for local side-feature import."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchidrec.cli import main
from orchidrec.errors import DatasetError, SerializationError, ValidationError
from orchidrec.features import FeatureKind, FeatureSource, feature_dataset_sha256
from orchidrec.recbole_side import (
    LoadedRecBoleSideFeatures,
    RecBoleSideLimits,
    import_recbole_side_features,
    load_recbole_side_features,
    save_recbole_side_features,
)

USER = (
    b"user_id:token\tage:float\tgroup:token\tinterests:token_seq\tweights:float_seq\n"
    b"001\t2.5\ta\tx y\t1 -2.5\n"
    b"1\t0\tb\t\t\n"
)
ITEM = b"item_id:token\tprice:float\tcategory:token\n02\t3.25\tc\n"


def _tables(root: Path, user: bytes = USER, item: bytes = ITEM) -> tuple[Path, Path]:
    user_path = root / "tiny.user"
    item_path = root / "tiny.item"
    user_path.write_bytes(user)
    item_path.write_bytes(item)
    return user_path, item_path


class RecBoleSideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_independent_oracle_and_roundtrip(self) -> None:
        user_path, item_path = _tables(self.root)
        loaded = import_recbole_side_features(user_path=user_path, item_path=item_path)
        self.assertEqual(
            [(row.source, row.key) for row in loaded.dataset],
            [(FeatureSource.ITEM, "02"), (FeatureSource.USER, "001"), (FeatureSource.USER, "1")],
        )
        self.assertEqual(
            {feature.name: feature.kind for feature in loaded.dataset.schema},
            {
                "item.category": FeatureKind.TOKEN,
                "item.price": FeatureKind.FLOAT,
                "user.age": FeatureKind.FLOAT,
                "user.group": FeatureKind.TOKEN,
                "user.interests": FeatureKind.TOKEN_SEQUENCE,
                "user.weights": FeatureKind.FLOAT_SEQUENCE,
            },
        )
        widths = {feature.name: feature.sequence_length for feature in loaded.dataset.schema}
        self.assertEqual(widths["user.interests"], 2)
        self.assertEqual(widths["user.weights"], 2)
        self.assertEqual(
            dict(loaded.dataset[1].values),
            {
                "user.age": 2.5,
                "user.group": "a",
                "user.interests": ("x", "y"),
                "user.weights": (1.0, -2.5),
            },
        )
        self.assertEqual(loaded.dataset[2].values["user.interests"], ())
        self.assertEqual(loaded.sources[0].sha256, hashlib.sha256(USER).hexdigest())
        self.assertEqual(loaded.sources[1].sha256, hashlib.sha256(ITEM).hexdigest())
        output = self.root / "features.json"
        save_recbole_side_features(loaded, output)
        self.assertTrue(output.read_bytes().endswith(b"\n"))
        restored = load_recbole_side_features(output)
        self.assertEqual(restored.to_state(), loaded.to_state())
        self.assertEqual(
            restored.to_state()["dataset_sha256"], feature_dataset_sha256(loaded.dataset)
        )

    def test_order_and_newline_provenance(self) -> None:
        user_path, _ = _tables(self.root)
        first = import_recbole_side_features(user_path=user_path)
        user_path.write_bytes(USER.replace(b"\n", b"\r\n"))
        second = import_recbole_side_features(user_path=user_path)
        self.assertEqual(first.to_state()["dataset_sha256"], second.to_state()["dataset_sha256"])
        self.assertNotEqual(first.sources[0].sha256, second.sources[0].sha256)
        user_path.write_bytes(
            USER.split(b"\n", 1)[0] + b"\n1\t0\tb\t\t\n001\t2.5\ta\tx y\t1 -2.5\n"
        )
        reordered = import_recbole_side_features(user_path=user_path)
        self.assertEqual(reordered.to_state()["dataset_sha256"], first.to_state()["dataset_sha256"])

    def test_rejects_ambiguous_tables(self) -> None:
        bad_tables = (
            b"user_id:token\tage:float\n001\tnan\n",
            b"user_id:token\tage:float\n001\t1e999\n",
            b"user_id:token\tage:float\n001\t1_0\n",
            b"user_id:token\tname:token_seq\n001\tx  y\n",
            b"user_id:token\tname:token_seq\n001\t x\n",
            b"user_id:token\tname:token\n001\tbad value\n",
            b"user_id:token\tname:token\n001\ta\n001\tb\n",
            b"user_id:token\tname:token\n001\ta\r002\tb\n",
            b"user_id:token\tname:token\n001\t\xff\n",
            b"user_id:token\tname:token\n001\ta\n\n",
            b"user_id:token\tname:token\n001\n",
            b"user_id:token\tname:token\tname:float\n001\ta\t1\n",
            b"user_id:float\tname:token\n001\ta\n",
            b"user_id:token\tname:unknown\n001\ta\n",
            b"item_id:token\tname:token\n001\ta\n",
        )
        path = self.root / "bad.user"
        for table in bad_tables:
            with self.subTest(table=table):
                path.write_bytes(table)
                with self.assertRaises(DatasetError):
                    import_recbole_side_features(user_path=path)

    def test_resource_limits(self) -> None:
        user_path, _ = _tables(self.root)
        for override in (
            {"max_file_bytes": len(USER) - 1},
            {"max_rows": 1},
            {"max_fields": 1},
            {"max_sequence_values": 1},
            {"max_total_values": 1},
            {"max_line_bytes": 10},
        ):
            with self.subTest(override=override), self.assertRaises(DatasetError):
                import_recbole_side_features(
                    user_path=user_path, limits=RecBoleSideLimits(**override)
                )
        loaded = import_recbole_side_features(
            user_path=user_path, limits=RecBoleSideLimits(max_output_bytes=1)
        )
        output = self.root / "bounded.json"
        with self.assertRaisesRegex(SerializationError, "max_output_bytes"):
            save_recbole_side_features(loaded, output)
        self.assertFalse(output.exists())
        with self.assertRaises(ValidationError):
            RecBoleSideLimits(max_rows=0)

    def test_field_name_and_expansion_bound_before_state_materialization(self) -> None:
        path = self.root / "wide.user"
        path.write_bytes(b"user_id:token\t" + b"x" * 65 + b":token\n1\tv\n")
        with self.assertRaisesRegex(DatasetError, "header"):
            import_recbole_side_features(user_path=path)

        name = b"x" * 64
        path.write_bytes(
            b"user_id:token\t"
            + name
            + b":token\n"
            + b"".join(str(index).encode() + b"\tv\n" for index in range(100))
        )
        loaded = import_recbole_side_features(
            user_path=path, limits=RecBoleSideLimits(max_output_bytes=4_096)
        )
        output = self.root / "expanded.json"
        with (
            patch.object(
                LoadedRecBoleSideFeatures,
                "to_state",
                side_effect=AssertionError("whole state must not be materialized"),
            ),
            self.assertRaisesRegex(SerializationError, "max_output_bytes"),
        ):
            save_recbole_side_features(loaded, output)
        self.assertFalse(output.exists())

    def test_no_overwrite_and_cleanup_on_link_failure(self) -> None:
        user_path, _ = _tables(self.root)
        loaded = import_recbole_side_features(user_path=user_path)
        output = self.root / "existing.json"
        output.write_bytes(b"untouched")
        with self.assertRaisesRegex(ValidationError, "exists"):
            save_recbole_side_features(loaded, output)
        self.assertEqual(output.read_bytes(), b"untouched")
        self.assertFalse(list(self.root.glob(".existing.json.*.tmp")))
        with (
            patch("orchidrec.recbole_side.os.link", side_effect=OSError("injected link failure")),
            self.assertRaisesRegex(SerializationError, "link failure"),
        ):
            save_recbole_side_features(loaded, self.root / "new.json")
        self.assertFalse((self.root / "new.json").exists())
        self.assertFalse(list(self.root.glob(".new.json.*.tmp")))

    def test_short_write_cleans_staged_file(self) -> None:
        user_path, _ = _tables(self.root)
        loaded = import_recbole_side_features(user_path=user_path)
        original = tempfile.NamedTemporaryFile

        class ShortWriter:
            def __init__(self, **kwargs: object) -> None:
                self.stream = original(**kwargs)
                self.name = self.stream.name

            def __enter__(self) -> ShortWriter:
                return self

            def __exit__(self, *args: object) -> None:
                self.stream.close()

            def write(self, data: bytes) -> int:
                return self.stream.write(data[:1])

        output = self.root / "short.json"
        with (
            patch("orchidrec.recbole_side.tempfile.NamedTemporaryFile", ShortWriter),
            self.assertRaisesRegex(SerializationError, "short write"),
        ):
            save_recbole_side_features(loaded, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".short.json.*.tmp")))

    def test_checksum_duplicate_keys_and_provenance(self) -> None:
        user_path, _ = _tables(self.root)
        output = self.root / "features.json"
        save_recbole_side_features(import_recbole_side_features(user_path=user_path), output)
        raw = output.read_bytes()
        output.write_bytes(raw.replace(b'"dataset_sha256":"', b'"dataset_sha256":"0', 1))
        with self.assertRaisesRegex(SerializationError, "checksum"):
            load_recbole_side_features(output)
        output.write_bytes(raw.replace(b'"format":', b'"format":"duplicate","format":', 1))
        with self.assertRaisesRegex(SerializationError, "duplicate"):
            load_recbole_side_features(output)
        state = json.loads(raw)
        state["sources"][0]["rows"] = 999
        state["state_sha256"] = hashlib.sha256(
            json.dumps(
                {key: value for key, value in state.items() if key != "state_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        output.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaisesRegex(SerializationError, "row counts"):
            load_recbole_side_features(output)

    def test_cli_import_fit_transform_and_alias(self) -> None:
        user_path, item_path = _tables(self.root)
        artifact = self.root / "features.json"
        pipeline = self.root / "pipeline.json"
        encoded = self.root / "encoded.json"
        self.assertEqual(
            main(
                [
                    "import-recbole-features",
                    "--user",
                    str(user_path),
                    "--item",
                    str(item_path),
                    "--output",
                    str(artifact),
                ]
            ),
            0,
        )
        self.assertEqual(
            main(
                [
                    "fit-features",
                    "--input",
                    str(artifact),
                    "--input-format",
                    "recbole-side",
                    "--output",
                    str(pipeline),
                ]
            ),
            0,
        )
        self.assertEqual(
            main(
                [
                    "transform-features",
                    "--input",
                    str(artifact),
                    "--input-format",
                    "recbole-side",
                    "--pipeline",
                    str(pipeline),
                    "--output",
                    str(encoded),
                ]
            ),
            0,
        )
        self.assertTrue(encoded.exists())
        self.assertEqual(
            main(["import-recbole-features", "--user", str(user_path), "--output", str(user_path)]),
            2,
        )
        self.assertEqual(
            main(["import-recbole-features", "--user", str(user_path), "--output", str(artifact)]),
            2,
        )

    def test_training_schema_reference_and_heldout_transform(self) -> None:
        train_dir = self.root / "train"
        held_dir = self.root / "held"
        train_dir.mkdir()
        held_dir.mkdir()
        train_user, train_item = _tables(train_dir)
        training = import_recbole_side_features(user_path=train_user, item_path=train_item)
        train_artifact = self.root / "train.json"
        save_recbole_side_features(training, train_artifact)
        held_user, held_item = _tables(
            held_dir,
            b"user_id:token\tage:float\tgroup:token\tinterests:token_seq\tweights:float_seq\n"
            b"new\t4\tunseen\tunknown\t2\n",
            b"item_id:token\tprice:float\tcategory:token\nnew-item\t8\tnew-category\n",
        )
        held = import_recbole_side_features(
            user_path=held_user, item_path=held_item, schema_from=training
        )
        self.assertEqual(held.dataset.schema.to_state(), training.dataset.schema.to_state())
        self.assertEqual(held.schema_reference_sha256, training.to_state()["state_sha256"])
        held_artifact = self.root / "held.json"
        self.assertEqual(
            main(
                [
                    "import-recbole-features",
                    "--user",
                    str(held_user),
                    "--item",
                    str(held_item),
                    "--schema-from",
                    str(train_artifact),
                    "--output",
                    str(held_artifact),
                ]
            ),
            0,
        )
        self.assertEqual(load_recbole_side_features(held_artifact).to_state(), held.to_state())
        pipeline = self.root / "pipeline.json"
        encoded = self.root / "held-encoded.json"
        self.assertEqual(
            main(
                [
                    "fit-features",
                    "--input",
                    str(train_artifact),
                    "--input-format",
                    "recbole-side",
                    "--output",
                    str(pipeline),
                ]
            ),
            0,
        )
        self.assertEqual(
            main(
                [
                    "transform-features",
                    "--pipeline",
                    str(pipeline),
                    "--input",
                    str(held_artifact),
                    "--input-format",
                    "recbole-side",
                    "--output",
                    str(encoded),
                ]
            ),
            0,
        )
        self.assertEqual(len(json.loads(encoded.read_text(encoding="utf-8"))["rows"]), 2)
        held_user.write_bytes(
            b"user_id:token\tage:float\tgroup:token\tinterests:token_seq\tweights:float_seq\n"
            b"new\t4\tunseen\ta b c\t2\n"
        )
        with self.assertRaisesRegex(DatasetError, "trained sequence width"):
            import_recbole_side_features(
                user_path=held_user, item_path=held_item, schema_from=training
            )
        held_user.write_bytes(
            b"user_id:token\tage:float\tgroup:token\tinterests:token_seq\tWRONG:float_seq\n"
            b"new\t4\tunseen\ta\t2\n"
        )
        with self.assertRaisesRegex(DatasetError, "fields and kinds"):
            import_recbole_side_features(
                user_path=held_user, item_path=held_item, schema_from=training
            )


if __name__ == "__main__":
    unittest.main()
