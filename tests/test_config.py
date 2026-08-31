from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from orchidrec.config import (
    ExperimentConfig,
    config_from_dict,
    load_config,
    save_config,
)
from orchidrec.errors import ConfigurationError


def minimal_payload() -> dict:
    return {"data": {"path": "events.json"}, "model": {"name": "popularity"}}


class ConfigTests(unittest.TestCase):
    def test_minimal_configuration_gets_documented_defaults(self) -> None:
        config = config_from_dict(minimal_payload(), base_dir=".")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.split.method, "leave_one_out")
        self.assertEqual(config.evaluation.k, 10)
        self.assertTrue(config.evaluation.exclude_seen)

    def test_relative_paths_resolve_against_configuration_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = config_from_dict(minimal_payload(), base_dir=directory)
            self.assertEqual(config.data.path, Path(directory).resolve() / "events.json")

    def test_absolute_paths_are_preserved(self) -> None:
        absolute = Path(tempfile.gettempdir()).resolve() / "events.json"
        payload = minimal_payload()
        payload["data"]["path"] = str(absolute)
        self.assertEqual(config_from_dict(payload).data.path, absolute)

    def test_unknown_root_field_is_rejected(self) -> None:
        payload = minimal_payload() | {"typo": True}
        with self.assertRaises(ConfigurationError):
            config_from_dict(payload)

    def test_missing_required_sections_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            config_from_dict({"data": {"path": "x"}})

    def test_unknown_model_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["model"]["name"] = "oracle"
        with self.assertRaises(ConfigurationError):
            config_from_dict(payload)

    def test_unknown_model_parameter_is_rejected(self) -> None:
        payload = minimal_payload()
        payload["model"]["params"] = {"typo": 1}
        with self.assertRaises(ConfigurationError):
            config_from_dict(payload)

    def test_invalid_model_parameters_are_rejected_during_config_parsing(self) -> None:
        cases = [
            ("popularity", {"weighted": "false"}),
            ("item_knn", {"neighbors": 0}),
            ("item_knn", {"shrinkage": float("nan")}),
            ("item_knn", {"shrinkage": 10**400}),
            ("implicit_mf", {"factors": 0}),
            ("implicit_mf", {"learning_rate": float("inf")}),
            ("implicit_mf", {"learning_rate": 10**400}),
            ("implicit_mf", {"seed": True}),
        ]
        for name, params in cases:
            payload = minimal_payload()
            payload["model"] = {"name": name, "params": params}
            with self.subTest(name=name, params=params), self.assertRaises(ConfigurationError):
                config_from_dict(payload)

    def test_invalid_split_method_is_rejected(self) -> None:
        payload = minimal_payload() | {"split": {"method": "future"}}
        with self.assertRaises(ConfigurationError):
            config_from_dict(payload)

    def test_non_string_split_and_model_names_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            config_from_dict(minimal_payload() | {"split": {"method": []}})
        payload = minimal_payload()
        payload["model"]["name"] = []
        with self.assertRaises(ConfigurationError):
            config_from_dict(payload)

    def test_invalid_test_ratio_is_rejected(self) -> None:
        for ratio in (0, 1, True, "0.2", 10**400):
            payload = minimal_payload() | {"split": {"test_ratio": ratio}}
            with self.subTest(ratio=ratio), self.assertRaises(ConfigurationError):
                config_from_dict(payload)

    def test_invalid_evaluation_values_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            config_from_dict(minimal_payload() | {"evaluation": {"k": 0}})
        with self.assertRaises(ConfigurationError):
            config_from_dict(minimal_payload() | {"evaluation": {"exclude_seen": 1}})

    def test_invalid_optional_output_path_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            config_from_dict(minimal_payload() | {"output": {"model_path": ""}})

    def test_load_and_save_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(json.dumps(minimal_payload()), encoding="utf-8")
            config = load_config(source)
            destination = Path(directory) / "normalized.json"
            save_config(config, destination)
            restored = load_config(destination)
        self.assertIsInstance(restored, ExperimentConfig)
        self.assertEqual(restored, config)

    def test_malformed_config_json_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_duplicate_config_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                '{"data":{"path":"a.json"},"data":{"path":"b.json"},'
                '"model":{"name":"popularity"}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "duplicate JSON object key"):
                load_config(path)

    def test_non_standard_json_constant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.json"
            path.write_text(
                '{"data":{"path":"events.json"},"model":{"name":"item_knn",'
                '"params":{"shrinkage":NaN}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "non-standard JSON numeric constant"):
                load_config(path)

    def test_non_object_config_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
