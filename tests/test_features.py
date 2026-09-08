from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from orchidrec.cli import main
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.features import (
    PAD_INDEX,
    UNKNOWN_INDEX,
    EncodedFeatureDataset,
    FeatureDataset,
    FeatureKind,
    FeatureLimits,
    FeatureRow,
    FeatureSchema,
    FeatureSource,
    FeatureSpec,
    FittedFeaturePipeline,
    SequenceKeep,
    feature_dataset_sha256,
    load_encoded_features,
    load_feature_dataset,
    save_encoded_features,
    save_feature_dataset,
)


def schema() -> FeatureSchema:
    return FeatureSchema(
        (
            FeatureSpec("country", FeatureKind.TOKEN, FeatureSource.USER),
            FeatureSpec("age", FeatureKind.FLOAT, FeatureSource.USER),
            FeatureSpec(
                "genres",
                FeatureKind.TOKEN_SEQUENCE,
                FeatureSource.ITEM,
                sequence_length=3,
                keep=SequenceKeep.HEAD,
            ),
            FeatureSpec(
                "history",
                FeatureKind.FLOAT_SEQUENCE,
                FeatureSource.INTERACTION,
                sequence_length=2,
                keep=SequenceKeep.TAIL,
            ),
        )
    )


def training() -> FeatureDataset:
    return FeatureDataset(
        schema(),
        (
            FeatureRow("user", "u2", {"country": "US", "age": 40}),
            FeatureRow("item", "i1", {"genres": ["drama", "comedy"]}),
            FeatureRow("interaction", "r1", {"history": [1, 3, 5]}),
            FeatureRow("user", "u1", {"country": "CA", "age": 20}),
        ),
    )


class SwitchingTuple(tuple[object, ...]):
    alternate = False

    def __iter__(self):  # type: ignore[no-untyped-def]
        selected = ("forged",) if self.alternate else tuple.__iter__(self)
        return iter(selected)


class LyingString(str):
    def __len__(self) -> int:
        return 0


class FeatureSchemaTests(unittest.TestCase):
    def test_schema_distinguishes_source_kind_and_sequence_policy(self) -> None:
        current = schema()
        self.assertEqual(len(current), 4)
        self.assertEqual(
            [feature.name for feature in current.for_source("user")],
            ["country", "age"],
        )
        self.assertTrue(current[2].kind.is_sequence)
        self.assertTrue(current[2].kind.is_token)

    def test_invalid_feature_contracts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "sequence keep"):
            FeatureSpec("tags", FeatureKind.TOKEN_SEQUENCE, FeatureSource.ITEM, 3)
        with self.assertRaisesRegex(ValidationError, "scalar features"):
            FeatureSpec("age", FeatureKind.FLOAT, FeatureSource.USER, 3, SequenceKeep.HEAD)
        with self.assertRaisesRegex(ValidationError, "globally unique"):
            FeatureSchema(
                (
                    FeatureSpec("same", FeatureKind.TOKEN, FeatureSource.USER),
                    FeatureSpec("same", FeatureKind.FLOAT, FeatureSource.ITEM),
                )
            )
        with self.assertRaisesRegex(ValidationError, "must not contain control"):
            FeatureSpec("bad\nname", FeatureKind.FLOAT, FeatureSource.USER)
        with self.assertRaisesRegex(ValidationError, "positive integer"):
            FeatureLimits(max_rows=True)  # type: ignore[arg-type]

    def test_rows_and_schema_are_deep_snapshots(self) -> None:
        topics: list[object] = ["drama", "comedy"]
        raw: dict[str, object] = {"genres": topics}
        row = FeatureRow("item", LyingString("i1"), raw)
        dataset = FeatureDataset(
            FeatureSchema(
                (
                    FeatureSpec(
                        "genres",
                        FeatureKind.TOKEN_SEQUENCE,
                        FeatureSource.ITEM,
                        3,
                        SequenceKeep.HEAD,
                    ),
                )
            ),
            (row,),
        )
        topics.append("forged")
        raw.clear()
        self.assertEqual(dataset[0].values["genres"], ("drama", "comedy"))
        self.assertIs(type(dataset[0].key), str)
        with self.assertRaises(TypeError):
            dataset[0].values["genres"] = ()  # type: ignore[index]

    def test_tuple_subclass_iteration_cannot_change_a_row_after_construction(self) -> None:
        values = SwitchingTuple(("drama", "comedy"))
        row = FeatureRow("item", "i", {"genres": values})
        values.alternate = True
        self.assertEqual(row.values["genres"], ("drama", "comedy"))
        self.assertIs(type(row.values["genres"]), tuple)

    def test_dataset_canonicalizes_order_and_validates_row_contracts(self) -> None:
        first = training()
        second = FeatureDataset(first.schema, reversed(first.rows))
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(feature_dataset_sha256(first), feature_dataset_sha256(second))
        with self.assertRaisesRegex(ValidationError, "missing"):
            FeatureDataset(schema(), (FeatureRow("user", "u", {"country": "US"}),))
        with self.assertRaisesRegex(ValidationError, "unique"):
            FeatureDataset(
                FeatureSchema((FeatureSpec("x", FeatureKind.FLOAT, FeatureSource.USER),)),
                (
                    FeatureRow("user", "u", {"x": 1}),
                    FeatureRow("user", "u", {"x": 2}),
                ),
            )

    def test_row_iterable_limit_stops_at_max_plus_one(self) -> None:
        consumed = 0

        def rows():
            nonlocal consumed
            for index in range(10):
                consumed += 1
                yield FeatureRow("user", index, {"x": index})

        with self.assertRaisesRegex(ValidationError, "limit of 2"):
            FeatureDataset(
                FeatureSchema((FeatureSpec("x", FeatureKind.FLOAT, FeatureSource.USER),)),
                rows(),
                limits=FeatureLimits(max_rows=2),
            )
        self.assertEqual(consumed, 3)


class FittedFeaturePipelineTests(unittest.TestCase):
    def test_fit_has_manual_vocabulary_and_numeric_oracle(self) -> None:
        pipeline = FittedFeaturePipeline.fit(training())
        self.assertEqual(pipeline.token_vocabularies["country"], ("CA", "US"))
        self.assertEqual(pipeline.token_vocabularies["genres"], ("comedy", "drama"))
        self.assertEqual(pipeline.numeric_statistics["age"].mean, 30.0)
        self.assertEqual(pipeline.numeric_statistics["age"].scale, 10.0)
        history = pipeline.numeric_statistics["history"]
        self.assertEqual(history.count, 3)
        self.assertAlmostEqual(history.mean, 3.0)
        self.assertAlmostEqual(history.scale, math.sqrt(8 / 3))
        self.assertEqual(pipeline.token_at("country", 2), "CA")
        self.assertEqual(pipeline.token_at("country", PAD_INDEX), None)
        self.assertEqual(pipeline.token_at("country", UNKNOWN_INDEX), None)

    def test_transform_is_train_only_unknown_safe_and_fixed_width(self) -> None:
        pipeline = FittedFeaturePipeline.fit(training())
        validation = FeatureDataset(
            schema(),
            (
                FeatureRow("user", "u3", {"country": "DE", "age": 50}),
                FeatureRow("item", "i2", {"genres": ["news", "drama", "x", "y"]}),
                FeatureRow("interaction", "r2", {"history": [7, 9, 11]}),
            ),
        )
        encoded = pipeline.transform(validation)
        by_identity = {(row.source, row.key): row for row in encoded}
        user = by_identity[(FeatureSource.USER, "u3")]
        self.assertEqual(user.values, {"country": UNKNOWN_INDEX, "age": 2.0})
        item = by_identity[(FeatureSource.ITEM, "i2")]
        self.assertEqual(item.values["genres"], (UNKNOWN_INDEX, 3, UNKNOWN_INDEX))
        self.assertEqual(item.sequence_lengths["genres"], 3)
        interaction = by_identity[(FeatureSource.INTERACTION, "r2")]
        expected = ((9 - 3) / math.sqrt(8 / 3), (11 - 3) / math.sqrt(8 / 3))
        self.assertAlmostEqual(interaction.values["history"][0], expected[0])  # type: ignore[index]
        self.assertAlmostEqual(interaction.values["history"][1], expected[1])  # type: ignore[index]
        self.assertEqual(interaction.sequence_lengths["history"], 2)
        self.assertEqual(encoded.pipeline_sha256, pipeline.state_sha256)

    def test_short_sequences_are_padded_and_keep_real_length(self) -> None:
        pipeline = FittedFeaturePipeline.fit(training())
        validation = FeatureDataset(
            schema(),
            (
                FeatureRow("item", "i2", {"genres": ["drama"]}),
                FeatureRow("interaction", "r2", {"history": []}),
            ),
        )
        encoded = pipeline.transform(validation)
        self.assertEqual(encoded[0].values["history"], (0.0, 0.0))
        self.assertEqual(encoded[0].sequence_lengths["history"], 0)
        self.assertEqual(encoded[1].values["genres"], (3, PAD_INDEX, PAD_INDEX))
        self.assertEqual(encoded[1].sequence_lengths["genres"], 1)

    def test_fit_is_order_invariant_and_transform_does_not_change_state(self) -> None:
        first = training()
        second = FeatureDataset(first.schema, reversed(first.rows))
        left = FittedFeaturePipeline.fit(first)
        right = FittedFeaturePipeline.fit(second)
        before = left.state_sha256
        left.transform(second)
        self.assertEqual(left.to_state(), right.to_state())
        self.assertEqual(left.state_sha256, before)

    def test_schema_mismatch_and_value_kind_mismatch_are_rejected(self) -> None:
        pipeline = FittedFeaturePipeline.fit(training())
        other_schema = FeatureSchema(
            (FeatureSpec("different", FeatureKind.FLOAT, FeatureSource.USER),)
        )
        with self.assertRaisesRegex(ValidationError, "exactly match"):
            pipeline.transform(
                FeatureDataset(other_schema, (FeatureRow("user", "u", {"different": 1}),))
            )
        with self.assertRaisesRegex(SerializationError, "must be a token"):
            FeatureDataset.from_state(
                {
                    "format": "orchidrec.feature-dataset",
                    "schema_version": 1,
                    "schema": schema().to_state(),
                    "rows": [
                        {
                            "source": "user",
                            "key": "u",
                            "values": {"country": 1.5, "age": 1},
                        }
                    ],
                }
            )

    def test_scaled_statistics_handle_extreme_finite_values(self) -> None:
        current_schema = FeatureSchema((FeatureSpec("x", FeatureKind.FLOAT, FeatureSource.USER),))
        current = FeatureDataset(
            current_schema,
            (
                FeatureRow("user", "negative", {"x": -1e308}),
                FeatureRow("user", "positive", {"x": 1e308}),
            ),
        )
        pipeline = FittedFeaturePipeline.fit(current)
        self.assertEqual(pipeline.numeric_statistics["x"].mean, 0.0)
        self.assertEqual(pipeline.numeric_statistics["x"].scale, 1e308)
        values = [row.values["x"] for row in pipeline.transform(current)]
        self.assertEqual(values, [-1.0, 1.0])

    def test_empty_training_sequence_has_explicit_default_state(self) -> None:
        current_schema = FeatureSchema(
            (
                FeatureSpec(
                    "tokens",
                    FeatureKind.TOKEN_SEQUENCE,
                    FeatureSource.USER,
                    2,
                    SequenceKeep.HEAD,
                ),
            )
        )
        current = FeatureDataset(
            current_schema,
            (FeatureRow("user", "u", {"tokens": []}),),
        )
        pipeline = FittedFeaturePipeline.fit(current)
        self.assertEqual(pipeline.training_values, 0)
        self.assertEqual(pipeline.token_vocabularies["tokens"], ())
        self.assertEqual(pipeline.transform(current)[0].values["tokens"], (0, 0))

    def test_fit_requires_training_rows_for_each_declared_source(self) -> None:
        current = training()
        without_items = FeatureDataset(
            current.schema,
            tuple(row for row in current if row.source is not FeatureSource.ITEM),
        )
        with self.assertRaisesRegex(ValidationError, "item"):
            FittedFeaturePipeline.fit(without_items)

    def test_vocabulary_limits_apply_across_all_features(self) -> None:
        current_schema = FeatureSchema(
            (
                FeatureSpec("a", FeatureKind.TOKEN, FeatureSource.USER),
                FeatureSpec("b", FeatureKind.TOKEN, FeatureSource.USER),
            )
        )
        current = FeatureDataset(
            current_schema,
            (FeatureRow("user", "u", {"a": "one", "b": "two"}),),
        )
        with self.assertRaisesRegex(ValidationError, "max_vocab_values"):
            FittedFeaturePipeline.fit(current, limits=FeatureLimits(max_vocab_values=1))
        zero_tokens = FeatureDataset(
            current_schema,
            (FeatureRow("user", "u", {"a": 0, "b": 0}),),
        )
        with self.assertRaisesRegex(ValidationError, "max_vocab_token_bytes"):
            FittedFeaturePipeline.fit(
                zero_tokens,
                limits=FeatureLimits(max_vocab_token_bytes=1),
            )

    def test_schema_width_obeys_active_limits(self) -> None:
        feature = FeatureSpec(
            "history",
            FeatureKind.FLOAT_SEQUENCE,
            FeatureSource.USER,
            3,
            SequenceKeep.TAIL,
        )
        with self.assertRaisesRegex(ValidationError, "sequence_length"):
            FeatureSchema((feature,), limits=FeatureLimits(max_sequence_values=2))

    def test_public_limits_have_finite_supported_maxima(self) -> None:
        for field_name in (
            "max_rows",
            "max_total_values",
            "max_vocab_values",
            "max_vocab_token_bytes",
            "max_state_bytes",
        ):
            with (
                self.subTest(field_name=field_name),
                self.assertRaisesRegex(ValidationError, field_name),
            ):
                FeatureLimits(**{field_name: 10**10000})


class FeaturePersistenceTests(unittest.TestCase):
    def test_dataset_pipeline_and_encoded_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = training()
            raw_path = root / "raw.json"
            pipeline_path = root / "pipeline.json"
            encoded_path = root / "encoded.json"
            save_feature_dataset(current, raw_path)
            restored_data = load_feature_dataset(raw_path)
            self.assertEqual(restored_data, current)
            pipeline = FittedFeaturePipeline.fit(restored_data)
            pipeline.save(pipeline_path)
            restored_pipeline = FittedFeaturePipeline.load(pipeline_path)
            self.assertEqual(restored_pipeline.to_state(), pipeline.to_state())
            encoded = restored_pipeline.transform(restored_data)
            save_encoded_features(encoded, encoded_path)
            restored_encoded = load_encoded_features(encoded_path)
            self.assertEqual(restored_encoded.to_state(), encoded.to_state())

    def test_dataset_digest_matches_canonical_state_and_save_streams_rows(self) -> None:
        current = training()
        expected = hashlib.sha256(
            json.dumps(
                current.to_state(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        encoded = FittedFeaturePipeline.fit(current).transform(current)
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.json"
            encoded_path = Path(directory) / "encoded.json"
            with (
                patch.object(FeatureDataset, "to_state", side_effect=AssertionError),
                patch.object(EncodedFeatureDataset, "to_state", side_effect=AssertionError),
            ):
                self.assertEqual(feature_dataset_sha256(current), expected)
                save_feature_dataset(current, raw_path)
                save_encoded_features(encoded, encoded_path)
            self.assertEqual(load_feature_dataset(raw_path), current)
            self.assertEqual(load_encoded_features(encoded_path), encoded)

    def test_streaming_byte_limit_preserves_target_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "raw.json"
            destination.write_bytes(b"sentinel")
            with self.assertRaisesRegex(SerializationError, "max_state_bytes"):
                save_feature_dataset(
                    training(),
                    destination,
                    limits=FeatureLimits(max_state_bytes=64),
                )
            self.assertEqual(destination.read_bytes(), b"sentinel")
            self.assertEqual(list(root.glob(".raw.json.*.tmp")), [])

    def test_pipeline_checksum_and_strict_schema_detect_tampering(self) -> None:
        state = FittedFeaturePipeline.fit(training()).to_state()
        tampered = copy.deepcopy(state)
        tampered["numeric_statistics"]["age"]["mean"] = 999  # type: ignore[index]
        with self.assertRaisesRegex(SerializationError, "checksum"):
            FittedFeaturePipeline.from_state(tampered)
        wrong_version = copy.deepcopy(state)
        wrong_version["schema_version"] = True
        with self.assertRaisesRegex(SerializationError, "unsupported"):
            FittedFeaturePipeline.from_state(wrong_version)
        extra = copy.deepcopy(state)
        extra["unexpected"] = 1
        with self.assertRaisesRegex(SerializationError, "unknown"):
            FittedFeaturePipeline.from_state(extra)

    def test_dataset_state_rejects_boolean_version_and_duplicate_json_fields(self) -> None:
        state = training().to_state()
        state["schema_version"] = True
        with self.assertRaisesRegex(SerializationError, "unsupported"):
            FeatureDataset.from_state(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                '{"format":"orchidrec.feature-dataset","format":"x",'
                '"schema_version":1,"schema":[],"rows":[]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SerializationError, "duplicate"):
                load_feature_dataset(path)

    def test_load_is_byte_bounded_and_rejects_deep_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_bytes(b"{}" * 20)
            with self.assertRaisesRegex(SerializationError, "byte limit"):
                FittedFeaturePipeline.load(path, max_state_bytes=8)
            path.write_text("[" * 2_000 + "]" * 2_000, encoding="utf-8")
            with self.assertRaisesRegex(SerializationError, "invalid feature JSON"):
                load_feature_dataset(path)

    def test_interrupted_atomic_save_preserves_old_target_and_cleans_staging(self) -> None:
        pipeline = FittedFeaturePipeline.fit(training())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "pipeline.json"
            destination.write_bytes(b"sentinel")
            with (
                patch("orchidrec.features.os.replace", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                pipeline.save(destination)
            self.assertEqual(destination.read_bytes(), b"sentinel")
            self.assertEqual(list(root.glob(".pipeline.json.*.tmp")), [])

    def test_encoded_state_validates_pipeline_identity_and_fixed_width(self) -> None:
        encoded = FittedFeaturePipeline.fit(training()).transform(training())
        state = encoded.to_state()
        state["pipeline_sha256"] = "bad"
        with self.assertRaisesRegex(SerializationError, "SHA-256"):
            EncodedFeatureDataset.from_state(state)
        malformed = encoded.to_state()
        malformed["rows"][0]["values"]["history"] = [0.0]  # type: ignore[index]
        with self.assertRaisesRegex(SerializationError, "fixed width"):
            EncodedFeatureDataset.from_state(malformed)

    def test_cli_fits_training_only_pipeline_and_transforms_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.json"
            validation_path = root / "validation.json"
            pipeline_path = root / "pipeline.json"
            encoded_path = root / "encoded.json"
            save_feature_dataset(training(), train_path)
            validation = FeatureDataset(
                schema(),
                (
                    FeatureRow("user", "new", {"country": "DE", "age": 30}),
                    FeatureRow("item", "new-item", {"genres": ["news"]}),
                    FeatureRow("interaction", "new-request", {"history": [3]}),
                ),
            )
            save_feature_dataset(validation, validation_path)
            fit_output = io.StringIO()
            with redirect_stdout(fit_output):
                self.assertEqual(
                    main(
                        [
                            "fit-features",
                            "--input",
                            str(train_path),
                            "--output",
                            str(pipeline_path),
                        ]
                    ),
                    0,
                )
            fit_summary = json.loads(fit_output.getvalue())
            self.assertEqual(fit_summary["training_rows"], 4)
            transform_output = io.StringIO()
            with redirect_stdout(transform_output):
                self.assertEqual(
                    main(
                        [
                            "transform-features",
                            "--pipeline",
                            str(pipeline_path),
                            "--input",
                            str(validation_path),
                            "--output",
                            str(encoded_path),
                        ]
                    ),
                    0,
                )
            transformed = load_encoded_features(encoded_path)
            self.assertEqual(json.loads(transform_output.getvalue())["rows"], 3)
            user = next(row for row in transformed if row.source is FeatureSource.USER)
            self.assertEqual(user.values["country"], UNKNOWN_INDEX)
            self.assertEqual(user.values["age"], 0.0)

    def test_cli_rejects_hardlinked_input_output_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.json"
            output = root / "pipeline.json"
            save_feature_dataset(training(), source)
            os.link(source, output)
            before = source.read_bytes()
            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(
                    main(
                        [
                            "fit-features",
                            "--input",
                            str(source),
                            "--output",
                            str(output),
                        ]
                    ),
                    2,
                )
            self.assertIn("must refer to different files", errors.getvalue())
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
