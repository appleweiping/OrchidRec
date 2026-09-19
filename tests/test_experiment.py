from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from orchidrec.config import SplitConfig, config_from_dict
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.demo import demo_dataset, run_demo
from orchidrec.errors import ConfigurationError, ValidationError
from orchidrec.experiment import build_model, run_experiment, split_dataset
from orchidrec.models import ImplicitMF, load_model


class ExperimentTests(unittest.TestCase):
    def make_config(self, directory: str, model: str = "popularity"):
        data_path = Path(directory) / "events.json"
        demo_dataset().save_json(data_path)
        if model == "implicit_mf":
            params = {"factors": 3, "epochs": 3}
        elif model == "confidence_als":
            params = {"factors": 3, "epochs": 2}
        else:
            params = {}
        return config_from_dict(
            {
                "seed": 13,
                "data": {"path": "events.json"},
                "split": {"method": "leave_one_out"},
                "model": {"name": model, "params": params},
                "evaluation": {"k": 3},
                "output": {"report_path": "report.json", "model_path": "model.json"},
            },
            base_dir=directory,
        )

    def test_all_builtin_models_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "popularity",
                "item_knn",
                "implicit_mf",
                "confidence_als",
                "ease",
                "user_knn",
                "sequential_markov",
            ):
                with self.subTest(model=name):
                    result = run_experiment(self.make_config(directory, name))
                    self.assertEqual(result.model_type, name)
                    self.assertEqual(result.metrics.users, 6)

    def test_library_entry_points_reject_invalid_model_config_and_split(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "unknown model"):
            build_model("unknown", {}, experiment_seed=1)
        with self.assertRaisesRegex(ConfigurationError, "config must"):
            run_experiment(None)  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(directory)
            malformed = replace(config, split=SplitConfig(method="unknown"))
            with self.assertRaisesRegex(ConfigurationError, "unknown split method"):
                split_dataset(malformed, demo_dataset())

    def test_repeated_run_is_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(directory, "implicit_mf")
            first = run_experiment(config).to_dict()
            second = run_experiment(config).to_dict()
        self.assertEqual(first, second)

    def test_requested_outputs_are_written_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(directory, "item_knn")
            result = run_experiment(config)
            self.assertTrue(config.output.report_path and config.output.report_path.is_file())
            self.assertTrue(config.output.model_path and config.output.model_path.is_file())
            self.assertEqual(load_model(config.output.model_path).model_type, "item_knn")
            saved = json.loads(config.output.report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved, result.to_dict())

    def test_output_aliases_are_rejected_without_overwriting_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "events.json"
            demo_dataset().save_json(data_path)
            original = data_path.read_bytes()
            for output in (
                {"model_path": "events.json"},
                {"report_path": "events.json"},
                {"model_path": "same.json", "report_path": "same.json"},
            ):
                with self.subTest(output=output):
                    config = config_from_dict(
                        {
                            "data": {"path": "events.json"},
                            "model": {"name": "ease", "params": {"regularization": 1}},
                            "output": output,
                        },
                        base_dir=directory,
                    )
                    with self.assertRaisesRegex(ValidationError, "different files"):
                        run_experiment(config)
                    self.assertEqual(data_path.read_bytes(), original)

    def test_hardlink_and_symlink_output_aliases_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "events.json"
            demo_dataset().save_json(data_path)
            original = data_path.read_bytes()
            aliases = (Path(directory) / "hardlink.json", Path(directory) / "symlink.json")
            try:
                os.link(data_path, aliases[0])
                aliases[1].symlink_to(data_path)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"filesystem does not support test links: {error}")
            for alias in aliases:
                with self.subTest(alias=alias):
                    config = config_from_dict(
                        {
                            "data": {"path": "events.json"},
                            "model": {"name": "ease"},
                            "output": {"model_path": str(alias)},
                        },
                        base_dir=directory,
                    )
                    with self.assertRaisesRegex(ValidationError, "different files"):
                        run_experiment(config)
                    self.assertEqual(data_path.read_bytes(), original)

    def test_implicit_model_inherits_experiment_seed(self) -> None:
        model = build_model("implicit_mf", {"factors": 2, "epochs": 1}, experiment_seed=99)
        self.assertIsInstance(model, ImplicitMF)
        self.assertEqual(model.seed, 99)

    def test_report_records_effective_default_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(self.make_config(directory, "popularity"))
        self.assertEqual(result.model_parameters, {"weighted": True})

    def test_explicit_model_seed_takes_precedence(self) -> None:
        model = build_model(
            "implicit_mf", {"factors": 2, "epochs": 1, "seed": 7}, experiment_seed=99
        )
        self.assertEqual(model.seed, 7)

    def test_bad_model_parameters_are_configuration_errors(self) -> None:
        with self.assertRaises(ConfigurationError):
            build_model("implicit_mf", {"factors": 0}, experiment_seed=1)

    def test_leave_one_out_with_only_single_event_users_has_no_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            InteractionDataset([Interaction("u1", "a"), Interaction("u2", "b")]).save_json(
                Path(directory) / "events.json"
            )
            config = config_from_dict(
                {"data": {"path": "events.json"}, "model": {"name": "popularity"}},
                base_dir=directory,
            )
            with self.assertRaises(ConfigurationError):
                run_experiment(config)

    def test_random_split_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo_dataset().save_json(Path(directory) / "events.json")
            config = config_from_dict(
                {
                    "data": {"path": "events.json"},
                    "split": {"method": "random", "test_ratio": 0.25},
                    "model": {"name": "popularity"},
                },
                base_dir=directory,
            )
            self.assertGreater(run_experiment(config).test_size, 0)

    def test_temporal_split_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo_dataset().save_json(Path(directory) / "events.json")
            config = config_from_dict(
                {
                    "data": {"path": "events.json"},
                    "split": {"method": "temporal", "test_ratio": 0.25},
                    "model": {"name": "popularity"},
                },
                base_dir=directory,
            )
            self.assertGreater(run_experiment(config).test_size, 0)

    def test_report_separates_cold_start_test_interactions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            InteractionDataset(
                [
                    Interaction("u1", "a", timestamp=1),
                    Interaction("u2", "b", timestamp=2),
                    Interaction("u3", "a", timestamp=3),
                    Interaction("u4", "cold", timestamp=4),
                ]
            ).save_json(Path(directory) / "events.json")
            config = config_from_dict(
                {
                    "data": {"path": "events.json"},
                    "split": {"method": "temporal", "test_ratio": 0.5},
                    "model": {"name": "popularity"},
                },
                base_dir=directory,
            )
            result = run_experiment(config)

        self.assertEqual(result.test_size, 2)
        self.assertEqual(result.evaluated_test_size, 1)
        self.assertEqual(result.cold_start_test_size, 1)
        self.assertEqual(result.to_dict()["dataset"]["cold_start_test_interactions"], 1)

    def test_experiment_rejects_test_set_with_only_cold_start_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            InteractionDataset(
                [
                    Interaction("u1", "a", timestamp=1),
                    Interaction("u2", "b", timestamp=2),
                    Interaction("u3", "c", timestamp=3),
                    Interaction("u4", "d", timestamp=4),
                ]
            ).save_json(Path(directory) / "events.json")
            config = config_from_dict(
                {
                    "data": {"path": "events.json"},
                    "split": {"method": "temporal", "test_ratio": 0.5},
                    "model": {"name": "popularity"},
                },
                base_dir=directory,
            )
            with self.assertRaises(ConfigurationError):
                run_experiment(config)

    def test_demo_creates_complete_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_demo(directory)
            self.assertTrue(artifacts.data_path.is_file())
            self.assertTrue(artifacts.config_path.is_file())
            self.assertTrue(artifacts.model_path.is_file())
            self.assertTrue(artifacts.report_path.is_file())
            self.assertEqual(artifacts.result.metrics.users, 6)


if __name__ == "__main__":
    unittest.main()
