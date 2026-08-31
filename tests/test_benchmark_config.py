from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orchidrec.benchmark_config import (
    benchmark_config_from_dict,
    load_benchmark_config,
    save_benchmark_config,
)
from orchidrec.errors import ConfigurationError


class BenchmarkConfigTests(unittest.TestCase):
    def minimal(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "data": {"path": "ml", "format": "movielens-100k"},
        }

    def test_minimal_config_has_documented_defaults_and_resolved_path(self) -> None:
        config = benchmark_config_from_dict(self.minimal(), base_dir="relative-base")
        self.assertEqual(config.seed, 42)
        self.assertTrue(config.data.path.is_absolute())
        self.assertEqual(config.data.minimum_rating, 4.0)
        self.assertEqual(config.split.method, "leave_one_out")
        self.assertEqual(config.evaluation.bootstrap_samples, 1_000)
        self.assertEqual([model.label for model in config.models], ["popularity", "item-knn", "bpr-mf"])
        self.assertEqual(config.to_dict()["schema_version"], 1)

    def test_config_round_trip_preserves_normalized_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = benchmark_config_from_dict(
                {
                    "schema_version": 1,
                    "seed": 9,
                    "data": {
                        "path": "events.json",
                        "format": "orchidrec-json",
                        "minimum_rating": None,
                    },
                    "split": {"method": "random", "test_ratio": 0.25},
                    "evaluation": {
                        "k": 3,
                        "exclude_seen": False,
                        "bootstrap_samples": 20,
                        "confidence": 0.8,
                    },
                    "models": [
                        {"label": "pop", "name": "popularity"},
                        {
                            "label": "mf",
                            "name": "implicit_mf",
                            "params": {"factors": 2, "epochs": 1},
                        },
                    ],
                },
                base_dir=directory,
            )
            destination = Path(directory) / "benchmark.json"
            save_benchmark_config(config, destination)
            self.assertNotIn(b"\r\n", destination.read_bytes())
            restored = load_benchmark_config(destination)
        self.assertEqual(restored, config)

    def test_loader_rejects_duplicate_keys_and_non_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "duplicate"):
                load_benchmark_config(path)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "object"):
                load_benchmark_config(path)
            with self.assertRaisesRegex(ConfigurationError, "could not read"):
                load_benchmark_config(Path(directory) / "missing.json")

    def test_top_level_schema_and_data_are_strict(self) -> None:
        cases = (
            ({}, "schema_version"),
            ({"schema_version": True}, "schema_version"),
            ({"schema_version": 2}, "schema_version"),
            ({"schema_version": 1, "extra": 1}, "unknown"),
            ({"schema_version": 1}, "data object"),
            ({"schema_version": 1, "data": []}, "data must"),
        )
        for payload, message in cases:
            with self.subTest(payload=payload), self.assertRaisesRegex(ConfigurationError, message):
                benchmark_config_from_dict(payload)  # type: ignore[arg-type]

    def test_data_fields_are_strict(self) -> None:
        invalid_data = (
            ({"path": "x", "format": "movielens-100k", "extra": 1}, "unknown"),
            ({"path": "", "format": "movielens-100k"}, "path"),
            ({"path": "x", "format": "bad"}, "format"),
            ({"path": "x", "format": "orchidrec-json", "minimum_rating": 4}, "only valid"),
            ({"path": "x", "format": "movielens-1m", "minimum_rating": 0}, "between"),
            ({"path": "x", "format": "movielens-1m", "minimum_rating": True}, "finite"),
        )
        for data, message in invalid_data:
            payload = {"schema_version": 1, "data": data}
            with self.subTest(data=data), self.assertRaisesRegex(ConfigurationError, message):
                benchmark_config_from_dict(payload)

    def test_split_and_evaluation_validation(self) -> None:
        updates = (
            ({"split": {"extra": 1}}, "unknown split"),
            ({"split": {"method": "future"}}, "split.method"),
            ({"split": {"test_ratio": True}}, "finite"),
            ({"split": {"test_ratio": 1}}, "between"),
            ({"evaluation": {"extra": 1}}, "unknown evaluation"),
            ({"evaluation": {"k": 0}}, "positive integer"),
            ({"evaluation": {"exclude_seen": 1}}, "boolean"),
            ({"evaluation": {"bootstrap_samples": True}}, "positive integer"),
            ({"evaluation": {"confidence": 1}}, "between"),
        )
        for update, message in updates:
            payload = self.minimal()
            payload.update(update)
            with self.subTest(update=update), self.assertRaisesRegex(ConfigurationError, message):
                benchmark_config_from_dict(payload)

    def test_model_labels_names_and_parameters_are_strict(self) -> None:
        invalid_models = (
            ([], "non-empty"),
            ([{"label": "only", "name": "popularity"}], "at least two"),
            ([{"label": "x", "name": "popularity"}] * 2, "unique"),
            ([{"label": " x", "name": "popularity"}, {"label": "y", "name": "item_knn"}], "label"),
            ([{"label": "x", "name": "bad"}, {"label": "y", "name": "item_knn"}], "name"),
            ([{"label": "x", "name": "popularity", "extra": 1}, {"label": "y", "name": "item_knn"}], "unknown"),
            ([{"label": "x", "name": "popularity", "params": {"bad": 1}}, {"label": "y", "name": "item_knn"}], "unknown"),
            ([{"label": "x", "name": "popularity"}, {"label": "y", "name": "item_knn", "params": {"neighbors": 0}}], "invalid"),
        )
        for models, message in invalid_models:
            payload = self.minimal()
            payload["models"] = models
            with self.subTest(models=models), self.assertRaisesRegex(ConfigurationError, message):
                benchmark_config_from_dict(payload)

    def test_seed_and_mapping_keys_are_validated(self) -> None:
        payload = self.minimal()
        payload["seed"] = True
        with self.assertRaisesRegex(ConfigurationError, "seed"):
            benchmark_config_from_dict(payload)
        with self.assertRaisesRegex(ConfigurationError, "JSON object"):
            benchmark_config_from_dict({1: "bad"})  # type: ignore[dict-item]

    def test_save_rejects_wrong_object_and_wraps_io_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ConfigurationError, "BenchmarkConfig"):
                save_benchmark_config(object(), Path(directory) / "x")  # type: ignore[arg-type]
            config = benchmark_config_from_dict(self.minimal(), base_dir=directory)
            directory_path = Path(directory) / "occupied"
            directory_path.mkdir()
            with self.assertRaisesRegex(ConfigurationError, "could not write"):
                save_benchmark_config(config, directory_path)


if __name__ == "__main__":
    unittest.main()
