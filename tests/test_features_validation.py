from __future__ import annotations

import copy
import io
import math
import os
import tempfile
import unittest
from collections.abc import Iterator, Mapping
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import orchidrec.features as features_module
from orchidrec.cli import main
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.features import (
    FEATURE_DATASET_FORMAT,
    FEATURE_PIPELINE_FORMAT,
    EncodedFeatureDataset,
    EncodedFeatureRow,
    FeatureDataset,
    FeatureKind,
    FeatureLimits,
    FeatureRow,
    FeatureSchema,
    FeatureSource,
    FeatureSpec,
    FittedFeaturePipeline,
    NumericStatistics,
    SequenceKeep,
    load_encoded_features,
    load_feature_dataset,
    save_encoded_features,
    save_feature_dataset,
)


def simple_schema() -> FeatureSchema:
    return FeatureSchema(
        (
            FeatureSpec("country", FeatureKind.TOKEN, FeatureSource.USER),
            FeatureSpec("score", FeatureKind.FLOAT, FeatureSource.USER),
        )
    )


def simple_dataset() -> FeatureDataset:
    return FeatureDataset(
        simple_schema(),
        (
            FeatureRow("user", "a", {"country": "US", "score": 1.0}),
            FeatureRow("user", "b", {"country": "CA", "score": 3.0}),
        ),
    )


class StringSubclass(str):
    pass


class StatefulMapping(Mapping[str, object]):
    def __init__(self) -> None:
        self.calls = 0

    def __getitem__(self, key: str) -> object:
        return dict(self.items())[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("country", "score"))

    def __len__(self) -> int:
        return 2

    def items(self):  # type: ignore[no-untyped-def]
        self.calls += 1
        country = "US" if self.calls == 1 else "CHANGED"
        return (("country", country), ("score", 1.0))


class BrokenMapping(StatefulMapping):
    def items(self):  # type: ignore[no-untyped-def]
        raise TypeError("broken mapping")


class FeaturePrimitiveValidationTests(unittest.TestCase):
    def test_primitive_container_and_subclass_errors_are_normalized(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must be a string"):
            FeatureSpec(3, FeatureKind.FLOAT, FeatureSource.USER)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "iterable, not text"):
            FeatureSchema("not-a-schema")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "must be iterable"):
            FeatureSchema(None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "exact FeatureLimits"):
            FeatureSchema((), limits=object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "must be a string or integer token"):
            FeatureRow("user", None, {"x": 1})  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "valid UTF-8"):
            FeatureRow("user", "id", {"x": "\ud800"})
        with self.assertRaisesRegex(ValidationError, "must be a mapping"):
            FeatureRow("user", "id", BrokenMapping())
        with self.assertRaisesRegex(ValidationError, "supported schema"):
            FeatureSpec(
                "x",
                FeatureKind.FLOAT_SEQUENCE,
                FeatureSource.USER,
                65537,
                SequenceKeep.HEAD,
            )

    def test_numeric_conversion_overflow_is_a_validation_error(self) -> None:
        with self.assertRaisesRegex(ValidationError, "finite number"):
            NumericStatistics(1, 10**10000, 1.0)

    def test_limit_relations_and_serialized_limit_state_are_strict(self) -> None:
        invalid_limits = (
            {"max_fields": 4097},
            {"max_sequence_values": 65537},
            {"max_token_chars": 16385},
            {"max_token_bytes": 65537},
            {"max_token_integer_bits": 16385},
        )
        for values in invalid_limits:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                FeatureLimits(**values)

        state = FeatureLimits(max_rows=7).to_state()
        self.assertEqual(FeatureLimits.from_state(state).max_rows, 7)
        state["max_rows"] = True
        with self.assertRaisesRegex(SerializationError, "invalid feature limits"):
            FeatureLimits.from_state(state)
        with self.assertRaisesRegex(SerializationError, "missing or unknown"):
            FeatureLimits.from_state({})

    def test_feature_spec_state_and_enum_validation(self) -> None:
        spec = FeatureSpec(
            "recent",
            FeatureKind.FLOAT_SEQUENCE,
            FeatureSource.INTERACTION,
            2,
            SequenceKeep.TAIL,
        )
        self.assertEqual(FeatureSpec.from_state(spec.to_state()), spec)
        for mutation in (
            {"kind": "not-a-kind"},
            {"source": "not-a-source"},
            {"sequence_length": 0},
            {"keep": "middle"},
        ):
            state = spec.to_state()
            state.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(SerializationError):
                FeatureSpec.from_state(state)
        with self.assertRaisesRegex(SerializationError, "missing or unknown"):
            FeatureSpec.from_state({"name": "x"})

    def test_feature_names_and_tokens_use_exact_builtin_values(self) -> None:
        with self.assertRaisesRegex(ValidationError, "at most 256"):
            FeatureSpec("x" * 257, FeatureKind.FLOAT, FeatureSource.USER)
        with self.assertRaisesRegex(ValidationError, "surrounding whitespace"):
            FeatureSpec(" score", FeatureKind.FLOAT, FeatureSource.USER)
        limits = FeatureLimits(max_token_chars=2, max_token_bytes=2, max_token_integer_bits=2)
        self.assertEqual(
            FeatureRow("user", StringSubclass("id"), {"x": "ok"}, limits=limits).key, "id"
        )
        for value, message in (
            ("long", "max_token_chars"),
            ("éé", "max_token_bytes"),
            (8, "max_token_integer_bits"),
            ("", "must not be empty"),
            (True, "must not be boolean"),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(ValidationError, message):
                FeatureRow("user", "id", {"x": value}, limits=limits)

    def test_numeric_statistics_reject_invalid_state(self) -> None:
        self.assertEqual(
            NumericStatistics.from_state({"count": 1, "mean": 2, "scale": 3}),
            NumericStatistics(1, 2.0, 3.0),
        )
        for state in (
            {"count": -1, "mean": 0.0, "scale": 1.0},
            {"count": 1, "mean": math.inf, "scale": 1.0},
            {"count": 1, "mean": 0.0, "scale": 0.0},
        ):
            with self.subTest(state=state), self.assertRaises(SerializationError):
                NumericStatistics.from_state(state)
        with self.assertRaisesRegex(SerializationError, "missing or unknown"):
            NumericStatistics.from_state({"count": 1})


class FeatureDatasetValidationTests(unittest.TestCase):
    def test_schema_state_and_dataset_public_type_guards(self) -> None:
        with self.assertRaisesRegex(SerializationError, "must be an array"):
            FeatureSchema.from_state({})
        state = [
            FeatureSpec(
                "x", FeatureKind.FLOAT_SEQUENCE, FeatureSource.USER, 2, SequenceKeep.HEAD
            ).to_state()
        ]
        with self.assertRaisesRegex(SerializationError, "max_sequence_values"):
            FeatureSchema.from_state(state, limits=FeatureLimits(max_sequence_values=1))
        with self.assertRaisesRegex(ValidationError, "schema must be"):
            FeatureDataset(object(), ())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "dataset must be"):
            features_module.feature_dataset_sha256(object())  # type: ignore[arg-type]

    def test_stateful_mapping_is_snapshotted_once(self) -> None:
        values = StatefulMapping()
        row = FeatureRow("user", "a", values)
        self.assertEqual(values.calls, 1)
        self.assertEqual(row.values, {"country": "US", "score": 1.0})

    def test_schema_and_dataset_limits_stop_at_limit_plus_one(self) -> None:
        consumed = 0

        def specifications() -> Iterator[FeatureSpec]:
            nonlocal consumed
            for index in range(4):
                consumed += 1
                yield FeatureSpec(f"f{index}", FeatureKind.FLOAT, FeatureSource.USER)

        with self.assertRaisesRegex(ValidationError, "limit of 2"):
            FeatureSchema(specifications(), limits=FeatureLimits(max_fields=2))
        self.assertEqual(consumed, 3)

        row = FeatureRow("user", "a", {"values": [1, 2]})
        current_schema = FeatureSchema(
            (
                FeatureSpec(
                    "values", FeatureKind.FLOAT_SEQUENCE, FeatureSource.USER, 2, SequenceKeep.HEAD
                ),
            )
        )
        with self.assertRaisesRegex(ValidationError, "max_total_values"):
            FeatureDataset(current_schema, (row,), limits=FeatureLimits(max_total_values=1))

    def test_schema_and_row_container_types_are_checked(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must not be empty"):
            FeatureSchema(())
        with self.assertRaisesRegex(ValidationError, "only FeatureSpec"):
            FeatureSchema((object(),))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "must be a mapping"):
            FeatureRow("user", "a", [])  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "only FeatureRow"):
            FeatureDataset(simple_schema(), (object(),))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "unknown"):
            FeatureDataset(
                simple_schema(),
                (FeatureRow("user", "a", {"country": "US", "score": 1, "extra": 2}),),
            )

    def test_raw_value_types_and_finiteness_are_checked(self) -> None:
        for value in (None, object(), math.inf, math.nan, True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                FeatureRow("user", "a", {"x": value})
        with self.assertRaisesRegex(ValidationError, "limit of 1"):
            FeatureRow(
                "user",
                "a",
                {"x": ["ok", "extra"]},
                limits=FeatureLimits(max_sequence_values=1),
            )

    def test_dataset_state_requires_plain_exact_keys_and_valid_rows(self) -> None:
        state = simple_dataset().to_state()
        wrong_keys = {StringSubclass(key): value for key, value in state.items()}
        with self.assertRaisesRegex(SerializationError, "missing or unknown"):
            FeatureDataset.from_state(wrong_keys)
        for field, value in (
            ("format", "wrong"),
            ("schema_version", 2),
            ("schema_version", True),
            ("rows", {}),
        ):
            malformed = copy.deepcopy(state)
            malformed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(SerializationError):
                FeatureDataset.from_state(malformed)
        malformed = copy.deepcopy(state)
        malformed["rows"][0]["source"] = 3  # type: ignore[index]
        with self.assertRaisesRegex(SerializationError, "source"):
            FeatureDataset.from_state(malformed)
        malformed = copy.deepcopy(state)
        malformed["rows"][0]["key"] = True  # type: ignore[index]
        with self.assertRaisesRegex(SerializationError, "key"):
            FeatureDataset.from_state(malformed)
        malformed = copy.deepcopy(state)
        malformed["rows"][0]["values"] = []  # type: ignore[index]
        with self.assertRaisesRegex(SerializationError, "values"):
            FeatureDataset.from_state(malformed)
        malformed = copy.deepcopy(state)
        del malformed["rows"][0]["values"]  # type: ignore[index]
        with self.assertRaisesRegex(SerializationError, "feature row"):
            FeatureDataset.from_state(malformed)


class FittedPipelineValidationTests(unittest.TestCase):
    def _dataset_with(self, spec: FeatureSpec, value: object) -> FeatureDataset:
        return FeatureDataset.from_state(
            {
                "format": FEATURE_DATASET_FORMAT,
                "schema_version": 1,
                "schema": [spec.to_state()],
                "rows": [{"source": "user", "key": "u", "values": {spec.name: value}}],
            }
        )

    def test_dataset_validates_each_declared_physical_kind(self) -> None:
        cases = (
            (FeatureSpec("x", FeatureKind.TOKEN, FeatureSource.USER), 1.5),
            (FeatureSpec("x", FeatureKind.FLOAT, FeatureSource.USER), "bad"),
            (
                FeatureSpec(
                    "x",
                    FeatureKind.TOKEN_SEQUENCE,
                    FeatureSource.USER,
                    2,
                    SequenceKeep.HEAD,
                ),
                1,
            ),
            (
                FeatureSpec(
                    "x",
                    FeatureKind.TOKEN_SEQUENCE,
                    FeatureSource.USER,
                    2,
                    SequenceKeep.HEAD,
                ),
                [1.5],
            ),
            (
                FeatureSpec(
                    "x",
                    FeatureKind.FLOAT_SEQUENCE,
                    FeatureSource.USER,
                    2,
                    SequenceKeep.HEAD,
                ),
                1,
            ),
            (
                FeatureSpec(
                    "x",
                    FeatureKind.FLOAT_SEQUENCE,
                    FeatureSource.USER,
                    2,
                    SequenceKeep.HEAD,
                ),
                ["bad"],
            ),
        )
        for spec, value in cases:
            with self.subTest(kind=spec.kind, value=value), self.assertRaises(SerializationError):
                self._dataset_with(spec, value)

    def test_constant_and_empty_numeric_features_have_unit_scale(self) -> None:
        scalar = self._dataset_with(FeatureSpec("x", FeatureKind.FLOAT, FeatureSource.USER), 0.0)
        self.assertEqual(FittedFeaturePipeline.fit(scalar).numeric_statistics["x"].scale, 1.0)
        sequence = self._dataset_with(
            FeatureSpec(
                "x",
                FeatureKind.FLOAT_SEQUENCE,
                FeatureSource.USER,
                2,
                SequenceKeep.HEAD,
            ),
            [],
        )
        statistics = FittedFeaturePipeline.fit(sequence).numeric_statistics["x"]
        self.assertEqual((statistics.count, statistics.mean, statistics.scale), (0, 0.0, 1.0))

    def test_fit_rejects_empty_data_and_enforces_vocabulary_bytes(self) -> None:
        empty = FeatureDataset(simple_schema(), ())
        with self.assertRaisesRegex(ValidationError, "must not be empty"):
            FittedFeaturePipeline.fit(empty)
        with self.assertRaisesRegex(ValidationError, "max_vocab_token_bytes"):
            FittedFeaturePipeline.fit(
                simple_dataset(),
                limits=FeatureLimits(max_vocab_token_bytes=1),
            )
        with self.assertRaisesRegex(ValidationError, "training must be"):
            FittedFeaturePipeline.fit(object())  # type: ignore[arg-type]

    def test_manual_fitted_state_requires_exact_schema_members(self) -> None:
        current = simple_dataset()
        digest = "0" * 64
        with self.assertRaisesRegex(ValidationError, "differs"):
            FittedFeaturePipeline(current.schema, {}, {}, 2, 4, digest)
        with self.assertRaisesRegex(ValidationError, "stable order"):
            FittedFeaturePipeline(
                current.schema,
                {"country": ("US", "CA")},
                {"score": NumericStatistics(2, 2.0, 1.0)},
                2,
                4,
                digest,
            )
        with self.assertRaisesRegex(ValidationError, "NumericStatistics"):
            FittedFeaturePipeline(
                current.schema,
                {"country": ("CA", "US")},
                {"score": object()},  # type: ignore[dict-item]
                2,
                4,
                digest,
            )
        with self.assertRaisesRegex(ValidationError, "arrays"):
            FittedFeaturePipeline(
                current.schema,
                {"country": object()},  # type: ignore[dict-item]
                {"score": NumericStatistics(2, 2.0, 1.0)},
                2,
                4,
                digest,
            )
        with self.assertRaisesRegex(ValidationError, "non-negative"):
            FittedFeaturePipeline(
                current.schema,
                {"country": ("CA", "US")},
                {"score": NumericStatistics(2, 2.0, 1.0)},
                2,
                -1,
                digest,
            )

    def test_token_lookup_and_numeric_overflow_fail_explicitly(self) -> None:
        pipeline = FittedFeaturePipeline.fit(simple_dataset())
        with self.assertRaisesRegex(ValidationError, "unknown token feature"):
            pipeline.token_at("missing", 2)
        with self.assertRaisesRegex(ValidationError, "outside"):
            pipeline.token_at("country", 99)
        with self.assertRaisesRegex(ValidationError, "dataset must be"):
            pipeline.transform(object())  # type: ignore[arg-type]
        tiny_scale = FittedFeaturePipeline(
            simple_schema(),
            {"country": ("CA", "US")},
            {"score": NumericStatistics(1, 0.0, 5e-324)},
            1,
            2,
            "0" * 64,
        )
        with self.assertRaisesRegex(ValidationError, "normalized finitely"):
            tiny_scale.transform(
                FeatureDataset(
                    simple_schema(),
                    (FeatureRow("user", "a", {"country": "US", "score": 1.0}),),
                )
            )

    def test_pipeline_state_rejects_wrong_format_and_nested_shapes(self) -> None:
        state = FittedFeaturePipeline.fit(simple_dataset()).to_state()
        for field, value in (
            ("format", FEATURE_DATASET_FORMAT),
            ("schema_version", 2),
            ("token_vocabularies", []),
            ("numeric_statistics", []),
            ("training", []),
        ):
            malformed = copy.deepcopy(state)
            malformed[field] = value
            with self.subTest(field=field), self.assertRaises(SerializationError):
                FittedFeaturePipeline.from_state(malformed)
        malformed = copy.deepcopy(state)
        malformed["token_vocabularies"]["country"] = "not-an-array"  # type: ignore[index]
        with self.assertRaisesRegex(SerializationError, "arrays"):
            FittedFeaturePipeline.from_state(malformed)
        malformed = copy.deepcopy(state)
        malformed["numeric_statistics"] = {1: {"count": 1, "mean": 0, "scale": 1}}
        with self.assertRaisesRegex(SerializationError, "names"):
            FittedFeaturePipeline.from_state(malformed)
        malformed = copy.deepcopy(state)
        malformed["training"]["rows"] = -1  # type: ignore[index]
        with self.assertRaisesRegex(SerializationError, "invalid feature pipeline"):
            FittedFeaturePipeline.from_state(malformed)


class EncodedFeatureValidationTests(unittest.TestCase):
    def test_encoded_state_rejects_format_shape_and_value_errors(self) -> None:
        pipeline = FittedFeaturePipeline.fit(simple_dataset())
        encoded = pipeline.transform(simple_dataset())
        state = encoded.to_state()
        for field, value in (
            ("format", FEATURE_PIPELINE_FORMAT),
            ("schema_version", 2),
            ("rows", {}),
            ("pipeline_sha256", "bad"),
        ):
            malformed = copy.deepcopy(state)
            malformed[field] = value
            with self.subTest(field=field), self.assertRaises(SerializationError):
                EncodedFeatureDataset.from_state(malformed)
        malformed = copy.deepcopy(state)
        malformed["rows"][0]["values"]["country"] = -1  # type: ignore[index]
        with self.assertRaisesRegex(SerializationError, "non-negative"):
            EncodedFeatureDataset.from_state(malformed)

    def test_encoded_rows_enforce_numeric_containers_and_lengths(self) -> None:
        limits = FeatureLimits()
        with self.assertRaisesRegex(ValidationError, "must be numeric"):
            EncodedFeatureRow("user", "a", {"x": "bad"}, {}, limits=limits)  # type: ignore[dict-item]
        with self.assertRaisesRegex(ValidationError, "must be numeric"):
            EncodedFeatureRow("user", "a", {"x": [1]}, {}, limits=limits)  # type: ignore[dict-item]
        sequence_schema = FeatureSchema(
            (
                FeatureSpec(
                    "x", FeatureKind.TOKEN_SEQUENCE, FeatureSource.USER, 2, SequenceKeep.HEAD
                ),
            )
        )
        row = EncodedFeatureRow("user", "a", {"x": (2, 0)}, {"x": 3}, limits=limits)
        with self.assertRaisesRegex(ValidationError, "outside"):
            EncodedFeatureDataset(sequence_schema, "0" * 64, (row,), limits=limits)
        with self.assertRaisesRegex(ValidationError, "must be numeric"):
            EncodedFeatureRow(
                "user",
                "a",
                {"x": (2, True)},
                {"x": 2},
                limits=limits,
            )

    def test_encoded_dataset_checks_fields_totals_and_unique_identity(self) -> None:
        limits = FeatureLimits(max_total_values=1)
        scalar_schema = FeatureSchema(
            (FeatureSpec("x", FeatureKind.FLOAT, FeatureSource.USER),),
            limits=limits,
        )
        rows = (
            EncodedFeatureRow("user", "a", {"x": 1.0}, {}, limits=limits),
            EncodedFeatureRow("user", "b", {"x": 2.0}, {}, limits=limits),
        )
        with self.assertRaisesRegex(ValidationError, "max_total_values"):
            EncodedFeatureDataset(scalar_schema, "0" * 64, rows, limits=limits)
        duplicate = EncodedFeatureRow("user", "a", {"x": 1.0}, {}, limits=FeatureLimits())
        with self.assertRaisesRegex(ValidationError, "unique"):
            EncodedFeatureDataset(
                scalar_schema,
                "0" * 64,
                (duplicate, duplicate),
                limits=FeatureLimits(),
            )
        wrong_fields = EncodedFeatureRow("user", "a", {"other": 1.0}, {}, limits=FeatureLimits())
        with self.assertRaisesRegex(ValidationError, "fields differ"):
            EncodedFeatureDataset(
                scalar_schema,
                "0" * 64,
                (wrong_fields,),
                limits=FeatureLimits(),
            )


class FeaturePersistenceBoundaryTests(unittest.TestCase):
    def test_save_type_guards_and_os_errors_are_domain_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "state.json"
            with self.assertRaisesRegex(ValidationError, "FeatureDataset"):
                save_feature_dataset(object(), target)  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValidationError, "EncodedFeatureDataset"):
                save_encoded_features(object(), target)  # type: ignore[arg-type]
            with (
                patch("orchidrec.features.tempfile.mkstemp", side_effect=PermissionError("no")),
                self.assertRaisesRegex(SerializationError, "could not write"),
            ):
                save_feature_dataset(simple_dataset(), target)

    def test_fdopen_and_fsync_interruptions_close_and_clean_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "state.json"
            target.write_bytes(b"old")
            real_close = os.close
            closed: list[int] = []

            def tracking_close(descriptor: int) -> None:
                closed.append(descriptor)
                real_close(descriptor)

            with (
                patch("orchidrec.features.os.fdopen", side_effect=KeyboardInterrupt),
                patch("orchidrec.features.os.close", side_effect=tracking_close),
                self.assertRaises(KeyboardInterrupt),
            ):
                save_feature_dataset(simple_dataset(), target)
            self.assertEqual(len(closed), 1)
            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(list(root.glob(".state.json.*.tmp")), [])

            with (
                patch("orchidrec.features.os.fsync", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                save_feature_dataset(simple_dataset(), target)
            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(list(root.glob(".state.json.*.tmp")), [])

    def test_posix_directory_sync_closes_its_descriptor(self) -> None:
        directory = Path("directory")
        with (
            patch.object(features_module.os, "name", "posix"),
            patch.object(features_module.os, "open", return_value=17) as open_mock,
            patch.object(features_module.os, "fsync") as fsync_mock,
            patch.object(features_module.os, "close") as close_mock,
        ):
            features_module._fsync_directory(directory)
        open_mock.assert_called_once()
        fsync_mock.assert_called_once_with(17)
        close_mock.assert_called_once_with(17)

    def test_missing_invalid_and_oversized_files_fail_as_serialization_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValidationError, "max_state_bytes"):
                FittedFeaturePipeline.load(root / "unused.json", max_state_bytes=10**10000)
            with self.assertRaisesRegex(SerializationError, "could not read"):
                load_feature_dataset(root / "missing.json")
            invalid = root / "invalid.json"
            invalid.write_bytes(b"\xff")
            with self.assertRaisesRegex(SerializationError, "invalid feature JSON"):
                load_feature_dataset(invalid)
            oversized = root / "oversized.json"
            oversized.write_bytes(b"{}" * 20)
            with self.assertRaisesRegex(SerializationError, "byte limit"):
                load_encoded_features(oversized, limits=FeatureLimits(max_state_bytes=8))

    def test_all_save_forms_preserve_old_target_when_state_limit_is_exceeded(self) -> None:
        pipeline = FittedFeaturePipeline.fit(
            simple_dataset(), limits=FeatureLimits(max_state_bytes=64)
        )
        encoded = pipeline.transform(simple_dataset())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, save in (
                ("pipeline", lambda path: pipeline.save(path)),
                (
                    "raw",
                    lambda path: save_feature_dataset(
                        simple_dataset(), path, limits=FeatureLimits(max_state_bytes=64)
                    ),
                ),
                (
                    "encoded",
                    lambda path: save_encoded_features(
                        encoded, path, limits=FeatureLimits(max_state_bytes=64)
                    ),
                ),
            ):
                destination = root / f"{label}.json"
                destination.write_bytes(b"old")
                with (
                    self.subTest(label=label),
                    self.assertRaisesRegex(SerializationError, "max_state_bytes"),
                ):
                    save(destination)
                self.assertEqual(destination.read_bytes(), b"old")
                self.assertEqual(list(root.glob(f".{label}.json.*.tmp")), [])

    def test_dataset_saves_revalidate_all_supplied_limits_before_replacement(self) -> None:
        dataset = simple_dataset()
        pipeline = FittedFeaturePipeline.fit(dataset)
        encoded = pipeline.transform(dataset)
        restrictive = FeatureLimits(max_rows=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, save in (
                (
                    "raw",
                    lambda path: save_feature_dataset(
                        dataset,
                        path,
                        limits=restrictive,
                    ),
                ),
                (
                    "encoded",
                    lambda path: save_encoded_features(
                        encoded,
                        path,
                        limits=restrictive,
                    ),
                ),
            ):
                destination = root / f"{label}.json"
                destination.write_bytes(b"old")
                with (
                    self.subTest(label=label),
                    self.assertRaisesRegex(ValidationError, "limit of 1"),
                ):
                    save(destination)
                self.assertEqual(destination.read_bytes(), b"old")
                self.assertEqual(list(root.glob(f".{label}.json.*.tmp")), [])

    def test_fsync_interrupt_cleans_staging_and_preserves_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "raw.json"
            destination.write_bytes(b"old")
            with (
                patch("orchidrec.features.os.fsync", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                save_feature_dataset(simple_dataset(), destination)
            self.assertEqual(destination.read_bytes(), b"old")
            self.assertEqual(list(root.glob(".raw.json.*.tmp")), [])

    def test_cli_converts_extreme_resource_limit_to_domain_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "features.json"
            save_feature_dataset(simple_dataset(), source)
            errors = io.StringIO()
            with redirect_stderr(errors):
                result = main(
                    [
                        "fit-features",
                        "--input",
                        os.fspath(source),
                        "--output",
                        os.fspath(root / "pipeline.json"),
                        "--max-rows",
                        "9" * 1000,
                    ]
                )
            self.assertEqual(result, 2)
            self.assertIn("max_rows", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
