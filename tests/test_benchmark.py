from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchidrec import benchmark as benchmark_module
from orchidrec.benchmark import METRIC_NAMES, BenchmarkResult, run_benchmark
from orchidrec.benchmark_config import (
    BenchmarkConfig,
    BenchmarkDataConfig,
    BenchmarkEvaluationConfig,
    BenchmarkModelSpec,
    BenchmarkSplitConfig,
    benchmark_config_from_dict,
)
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.datasets import interaction_fingerprint, load_dataset
from orchidrec.demo import demo_dataset
from orchidrec.errors import ConfigurationError, SerializationError, ValidationError
from orchidrec.reporting import benchmark_csv, benchmark_html, save_benchmark_reports

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_config(*, seed: int = 17, label: str = "popularity") -> BenchmarkConfig:
    return benchmark_config_from_dict(
        {
            "schema_version": 1,
            "seed": seed,
            "data": {
                "path": str(FIXTURES / "movielens-100k"),
                "format": "movielens-100k",
                "minimum_rating": 4,
            },
            "split": {"method": "leave_one_out", "test_ratio": 0.2},
            "evaluation": {
                "k": 3,
                "bootstrap_samples": 40,
                "confidence": 0.9,
            },
            "models": [
                {"label": label, "name": "popularity", "params": {"weighted": False}},
                {
                    "label": "item-knn",
                    "name": "item_knn",
                    "params": {"neighbors": 3, "shrinkage": 1.0},
                },
                {
                    "label": "bpr-mf",
                    "name": "implicit_mf",
                    "params": {"factors": 3, "epochs": 2, "negative_samples": 1},
                },
            ],
        }
    )


def tuned_fixture_config(*, direction: str = "maximize") -> BenchmarkConfig:
    payload = fixture_config().to_dict()
    payload["tuning"] = {
        "selection_metric": "ndcg",
        "direction": direction,
        "validation_split": {"method": "leave_one_out", "validation_ratio": 0.2},
        "implicit_mf_seeds": [17, 18],
    }
    payload["models"] = [
        {
            "label": "popularity",
            "name": "popularity",
            "params": {},
            "grid": {"weighted": [False, True]},
        },
        {
            "label": "item-knn",
            "name": "item_knn",
            "params": {"shrinkage": 1.0},
            "grid": {"neighbors": [2, 3]},
        },
        {
            "label": "bpr-mf",
            "name": "implicit_mf",
            "params": {"epochs": 1, "negative_samples": 1},
            "grid": {"factors": [2, 3]},
        },
    ]
    return benchmark_config_from_dict(payload)


class BenchmarkRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_benchmark(fixture_config())

    def test_all_models_share_dataset_split_and_bootstrap(self) -> None:
        result = self.result
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.dataset.source_rows, 18)
        self.assertEqual(result.train_interactions + result.test_interactions, 17)
        self.assertEqual(result.evaluated_users, 6)
        self.assertEqual(len(result.models), 3)
        self.assertEqual(len(result.comparisons), 3)
        self.assertEqual(len(result.config_fingerprint), 64)
        self.assertEqual(len(result.split_fingerprint), 64)
        self.assertEqual(result.to_dict()["schema_version"], 1)
        self.assertNotIn("tuning", result.to_dict())

    def test_every_ranking_metric_has_a_seeded_interval(self) -> None:
        for model in self.result.models:
            points = model.metrics.to_dict()
            self.assertEqual(set(model.confidence_intervals), set(METRIC_NAMES))
            for metric in METRIC_NAMES:
                interval = model.confidence_intervals[metric]
                self.assertAlmostEqual(interval.estimate, float(points[metric]))
                self.assertEqual(interval.samples, 40)
                self.assertEqual(interval.confidence, 0.9)
                self.assertTrue(points[metric] >= 0 or metric == "novelty")
            self.assertGreaterEqual(model.timing.fit_seconds, 0.0)
            self.assertGreaterEqual(model.timing.recommend_seconds, 0.0)

    def test_all_pairs_have_paired_results_for_all_metrics(self) -> None:
        pairs = {
            (comparison.left_label, comparison.right_label)
            for comparison in self.result.comparisons
        }
        self.assertEqual(
            pairs,
            {
                ("popularity", "item-knn"),
                ("popularity", "bpr-mf"),
                ("item-knn", "bpr-mf"),
            },
        )
        for comparison in self.result.comparisons:
            self.assertEqual(set(comparison.metrics), set(METRIC_NAMES))
            self.assertEqual(comparison.to_dict()["difference_direction"], "right_minus_left")
            for statistic in comparison.metrics.values():
                self.assertGreaterEqual(statistic.probability_right_better, 0.0)
                self.assertLessEqual(statistic.probability_right_better, 1.0)
                self.assertGreaterEqual(statistic.two_sided_p_value, 0.0)
                self.assertLessEqual(statistic.two_sided_p_value, 1.0)

    def test_repeated_run_reproduces_everything_except_observed_timing(self) -> None:
        second = run_benchmark(fixture_config())
        first_payload = self.result.to_dict()
        second_payload = second.to_dict()
        for payload in (first_payload, second_payload):
            for model in payload["models"]:  # type: ignore[index,union-attr]
                model.pop("timing")
        self.assertEqual(first_payload, second_payload)

    def test_seed_changes_config_and_model_results(self) -> None:
        other = run_benchmark(fixture_config(seed=18))
        self.assertNotEqual(self.result.config_fingerprint, other.config_fingerprint)
        self.assertEqual(self.result.dataset.interactions_sha256, other.dataset.interactions_sha256)
        self.assertEqual(self.result.split_fingerprint, other.split_fingerprint)
        self.assertEqual(self.result.models[-1].parameters["seed"], 17)
        self.assertEqual(other.models[-1].parameters["seed"], 18)

    def test_config_fingerprint_ignores_local_path_but_not_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "renamed.data"
            copied.write_bytes((FIXTURES / "movielens-100k" / "u.data").read_bytes())
            config_payload = fixture_config().to_dict()
            config_payload["data"]["path"] = str(copied)  # type: ignore[index]
            copied_result = run_benchmark(benchmark_config_from_dict(config_payload))
        self.assertEqual(self.result.config_fingerprint, copied_result.config_fingerprint)
        self.assertEqual(self.result.split_fingerprint, copied_result.split_fingerprint)

    def test_random_and_temporal_splits_run_with_shared_models(self) -> None:
        for method in ("random", "temporal"):
            payload = fixture_config().to_dict()
            payload["split"]["method"] = method  # type: ignore[index]
            payload["models"] = payload["models"][:2]  # type: ignore[index]
            payload["evaluation"]["bootstrap_samples"] = 2  # type: ignore[index]
            with self.subTest(method=method):
                result = run_benchmark(benchmark_config_from_dict(payload))
                self.assertGreater(result.test_interactions, 0)

    def test_cold_start_targets_are_reported_not_silently_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            InteractionDataset(
                [
                    Interaction("u1", "a", timestamp=1),
                    Interaction("u2", "b", timestamp=2),
                    Interaction("u3", "a", timestamp=3),
                    Interaction("u4", "cold", timestamp=4),
                ]
            ).save_json(path)
            config = benchmark_config_from_dict(
                {
                    "schema_version": 1,
                    "data": {"path": str(path), "format": "orchidrec-json"},
                    "split": {"method": "temporal", "test_ratio": 0.5},
                    "evaluation": {"bootstrap_samples": 2},
                    "models": [
                        {"label": "a", "name": "popularity"},
                        {"label": "b", "name": "popularity", "params": {"weighted": False}},
                    ],
                }
            )
            result = run_benchmark(config)
        self.assertEqual(result.cold_start_test_interactions, 1)
        self.assertEqual(result.evaluated_test_interactions, 1)

    def test_all_cold_start_and_wrong_config_fail_cleanly(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "BenchmarkConfig"):
            run_benchmark(object())  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            InteractionDataset(
                [
                    Interaction("u1", "a", timestamp=1),
                    Interaction("u2", "b", timestamp=2),
                    Interaction("u3", "c", timestamp=3),
                    Interaction("u4", "d", timestamp=4),
                ]
            ).save_json(path)
            payload = {
                "schema_version": 1,
                "data": {"path": str(path), "format": "orchidrec-json"},
                "split": {"method": "temporal", "test_ratio": 0.5},
                "evaluation": {"bootstrap_samples": 2},
                "models": [
                    {"label": "a", "name": "popularity"},
                    {"label": "b", "name": "item_knn"},
                ],
            }
            with self.assertRaisesRegex(ConfigurationError, "no test targets"):
                run_benchmark(benchmark_config_from_dict(payload))

    def test_direct_dataclass_construction_is_revalidated_at_runner_boundary(self) -> None:
        malformed = BenchmarkConfig(
            seed=1,
            data=BenchmarkDataConfig(
                path=FIXTURES / "movielens-100k", format="movielens-100k", minimum_rating=4
            ),
            split=BenchmarkSplitConfig(method="not-a-split"),
            evaluation=BenchmarkEvaluationConfig(bootstrap_samples=2),
            models=(
                BenchmarkModelSpec("a", "popularity"),
                BenchmarkModelSpec("b", "item_knn"),
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "split.method"):
            run_benchmark(malformed)

        empty_models = BenchmarkConfig(
            seed=1,
            data=malformed.data,
            split=BenchmarkSplitConfig(),
            evaluation=malformed.evaluation,
            models=(),
        )
        with self.assertRaisesRegex(ConfigurationError, "non-empty"):
            run_benchmark(empty_models)


class BenchmarkTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_benchmark(tuned_fixture_config())

    def test_all_three_models_have_complete_validation_only_search_records(self) -> None:
        tuning = self.result.tuning
        self.assertIsNotNone(tuning)
        assert tuning is not None
        self.assertEqual(tuning.selection_metric, "ndcg")
        self.assertEqual(tuning.direction, "maximize")
        self.assertEqual(tuning.implicit_mf_seeds, (17, 18))
        self.assertEqual(tuning.final_seed, 17)
        self.assertEqual(
            tuning.training_interactions + tuning.validation_interactions,
            tuning.development_interactions,
        )
        self.assertEqual(len(tuning.three_way_split_fingerprint), 64)
        self.assertEqual(
            {model.model_type for model in tuning.models},
            {"popularity", "item_knn", "implicit_mf"},
        )
        by_type = {model.model_type: model for model in tuning.models}
        self.assertEqual(len(by_type["popularity"].trials), 2)
        self.assertEqual(len(by_type["item_knn"].trials), 2)
        self.assertEqual(len(by_type["implicit_mf"].trials), 4)
        self.assertEqual(
            [trial.seed for trial in by_type["implicit_mf"].trials],
            [17, 18, 17, 18],
        )
        for candidate in by_type["implicit_mf"].candidates:
            values = [
                by_type["implicit_mf"].trials[index].selection_value
                for index in candidate.trial_indices
            ]
            self.assertAlmostEqual(candidate.mean_selection_value, sum(values) / len(values))
        self.assertTrue(all(trial.seed is None for trial in by_type["popularity"].trials))
        for model in tuning.models:
            self.assertEqual(sum(candidate.selected for candidate in model.candidates), 1)
            selected = model.candidates[model.selected_candidate_index]
            self.assertTrue(selected.selected)
            self.assertEqual(selected.mean_selection_value, model.selected_validation_score)
            self.assertEqual(
                model.selected_validation_score,
                max(candidate.mean_selection_value for candidate in model.candidates),
            )
        self.assertEqual(by_type["implicit_mf"].final_seed, 17)
        self.assertEqual(by_type["implicit_mf"].final_parameters["seed"], 17)

    def test_outer_test_is_never_passed_to_a_trial_and_is_evaluated_once_per_model(self) -> None:
        config = tuned_fixture_config()
        loaded = load_dataset(
            config.data.path,
            format=config.data.format,
            minimum_rating=config.data.minimum_rating,
        )
        outer = benchmark_module._split(config, loaded.dataset)
        outer_test_hash = interaction_fingerprint(outer.test)
        original = benchmark_module._evaluate_model
        evaluated_hashes: list[str] = []

        def recording_evaluator(*args: object, **kwargs: object) -> object:
            evaluated_split = args[2]
            assert isinstance(evaluated_split, benchmark_module.SplitResult)
            evaluated_hashes.append(interaction_fingerprint(evaluated_split.test))
            return original(*args, **kwargs)  # type: ignore[arg-type]

        with patch("orchidrec.benchmark._evaluate_model", side_effect=recording_evaluator):
            run_benchmark(config)
        outer_positions = [
            index
            for index, fingerprint in enumerate(evaluated_hashes)
            if fingerprint == outer_test_hash
        ]
        self.assertEqual(len(outer_positions), len(config.models))
        self.assertEqual(
            outer_positions,
            list(range(len(evaluated_hashes) - len(config.models), len(evaluated_hashes))),
        )

    def test_changing_only_outer_test_labels_cannot_change_model_selection(self) -> None:
        development = [
            Interaction("u1", "a", timestamp=1),
            Interaction("u1", "b", timestamp=2),
            Interaction("u1", "c", timestamp=3),
            Interaction("u2", "a", timestamp=4),
            Interaction("u2", "c", timestamp=5),
            Interaction("u2", "d", timestamp=6),
            Interaction("u3", "b", timestamp=7),
            Interaction("u3", "d", timestamp=8),
        ]
        tuning_payload: dict[str, object] = {
            "schema_version": 1,
            "split": {"method": "temporal", "test_ratio": 0.1},
            "evaluation": {"k": 2, "bootstrap_samples": 2},
            "tuning": {
                "selection_metric": "ndcg",
                "validation_split": {"method": "leave_one_out"},
            },
            "models": [
                {
                    "label": "pop",
                    "name": "popularity",
                    "grid": {"weighted": [False, True]},
                },
                {
                    "label": "knn",
                    "name": "item_knn",
                    "grid": {"neighbors": [1, 2]},
                },
            ],
        }
        selection_records: list[list[tuple[int, list[float]]]] = []
        with tempfile.TemporaryDirectory() as directory:
            for filename, held_out_item in (("a.json", "a"), ("b.json", "b")):
                path = Path(directory) / filename
                InteractionDataset(
                    [*development, Interaction("outer-test", held_out_item, timestamp=9)]
                ).save_json(path)
                payload = dict(tuning_payload)
                payload["data"] = {"path": str(path), "format": "orchidrec-json"}
                result = run_benchmark(benchmark_config_from_dict(payload))
                assert result.tuning is not None
                selection_records.append(
                    [
                        (
                            model.selected_candidate_index,
                            [candidate.mean_selection_value for candidate in model.candidates],
                        )
                        for model in result.tuning.models
                    ]
                )
        self.assertEqual(selection_records[0], selection_records[1])

    def test_final_fit_uses_the_complete_development_partition(self) -> None:
        config = tuned_fixture_config()
        original = benchmark_module._evaluate_model
        fit_sizes: list[int] = []

        def recording_evaluator(*args: object, **kwargs: object) -> object:
            evaluated_split = args[2]
            assert isinstance(evaluated_split, benchmark_module.SplitResult)
            fit_sizes.append(len(evaluated_split.train))
            return original(*args, **kwargs)  # type: ignore[arg-type]

        with patch("orchidrec.benchmark._evaluate_model", side_effect=recording_evaluator):
            result = run_benchmark(config)
        assert result.tuning is not None
        self.assertEqual(
            fit_sizes[-len(config.models) :],
            [result.tuning.development_interactions] * len(config.models),
        )

    def test_minimize_direction_and_canonical_tie_break_are_deterministic(self) -> None:
        minimized = run_benchmark(tuned_fixture_config(direction="minimize"))
        assert minimized.tuning is not None
        for model in minimized.tuning.models:
            self.assertEqual(
                model.selected_validation_score,
                min(candidate.mean_selection_value for candidate in model.candidates),
            )
        payload = tuned_fixture_config().to_dict()
        payload["models"] = [
            {
                "label": "pop-a",
                "name": "popularity",
                "grid": {"weighted": [False, True]},
            },
            {"label": "pop-b", "name": "popularity"},
        ]
        tied = run_benchmark(benchmark_config_from_dict(payload))
        assert tied.tuning is not None
        self.assertEqual(tied.tuning.models[0].selected_candidate_index, 0)

    def test_tuned_run_is_reproducible_except_all_observed_timings(self) -> None:
        first = self.result.to_dict()
        second = run_benchmark(tuned_fixture_config()).to_dict()
        for payload in (first, second):
            for model in payload["models"]:  # type: ignore[index,union-attr]
                model.pop("timing")
            for model in payload["tuning"]["models"]:  # type: ignore[index,union-attr]
                for trial in model["trials"]:
                    trial.pop("timing")
        self.assertEqual(first, second)

    def test_tuning_report_carries_config_data_and_every_partition_hash(self) -> None:
        payload = self.result.to_dict()
        self.assertEqual(len(payload["fingerprints"]["config_sha256"]), 64)  # type: ignore[index]
        self.assertEqual(len(payload["dataset"]["interactions_sha256"]), 64)  # type: ignore[index]
        fingerprints = payload["tuning"]["fingerprints"]  # type: ignore[index]
        self.assertEqual(
            set(fingerprints),
            {
                "development_sha256",
                "training_sha256",
                "validation_sha256",
                "test_sha256",
                "three_way_split_sha256",
            },
        )
        self.assertTrue(all(len(value) == 64 for value in fingerprints.values()))

    def test_validation_failure_is_not_relabelled_as_a_test_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            InteractionDataset(
                [
                    Interaction("u1", "a", timestamp=1),
                    Interaction("u2", "a", timestamp=2),
                    Interaction("u3", "b", timestamp=3),
                    Interaction("u4", "cold-validation", timestamp=4),
                    Interaction("u5", "a", timestamp=5),
                ]
            ).save_json(path)
            payload: dict[str, object] = {
                "schema_version": 1,
                "data": {"path": str(path), "format": "orchidrec-json"},
                "split": {"method": "temporal", "test_ratio": 0.2},
                "evaluation": {"bootstrap_samples": 2},
                "tuning": {
                    "validation_split": {
                        "method": "temporal",
                        "validation_ratio": 0.25,
                    }
                },
                "models": [
                    {
                        "label": "a",
                        "name": "popularity",
                        "grid": {"weighted": [False, True]},
                    },
                    {"label": "b", "name": "item_knn"},
                ],
            }
            with self.assertRaisesRegex(ConfigurationError, "no test targets"):
                run_benchmark(benchmark_config_from_dict(payload))

    def test_random_inner_validation_split_is_seeded_and_reproducible(self) -> None:
        payload = tuned_fixture_config().to_dict()
        payload["tuning"]["validation_split"] = {  # type: ignore[index]
            "method": "random",
            "validation_ratio": 0.25,
        }
        config = benchmark_config_from_dict(payload)

        first = run_benchmark(config)
        second = run_benchmark(config)
        assert first.tuning is not None and second.tuning is not None
        self.assertGreater(first.tuning.training_interactions, 0)
        self.assertGreater(first.tuning.validation_interactions, 0)
        self.assertEqual(
            first.tuning.training_interactions + first.tuning.validation_interactions,
            first.tuning.development_interactions,
        )
        self.assertEqual(
            first.tuning.three_way_split_fingerprint,
            second.tuning.three_way_split_fingerprint,
        )

    def test_tuned_json_csv_and_html_expose_the_complete_search(self) -> None:
        tuning = self.result.tuning
        assert tuning is not None
        expected_trials = sum(len(model.trials) for model in tuning.models)
        expected_candidates = sum(len(model.candidates) for model in tuning.models)
        csv_text = benchmark_csv(self.result)
        self.assertEqual(csv_text.count("tuning_trial"), expected_trials)
        self.assertEqual(csv_text.count("tuning_candidate"), expected_candidates)
        self.assertIn("parameters_json", csv_text.splitlines()[0])
        html_text = benchmark_html(self.result)
        self.assertIn("Validation-only model selection", html_text)
        self.assertIn(tuning.three_way_split_fingerprint, html_text)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_benchmark_reports(self.result, directory)
            loaded = json.loads(paths.json_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded, self.result.to_dict())


class BenchmarkReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_benchmark(fixture_config(label="<unsafe>"))

    def test_csv_is_tidy_and_contains_models_and_comparisons(self) -> None:
        csv_text = benchmark_csv(self.result)
        self.assertTrue(csv_text.startswith("row_type,left_or_model"))
        self.assertIn("model,<unsafe>", csv_text)
        self.assertIn("comparison_right_minus_left", csv_text)
        self.assertEqual(len(csv_text.splitlines()), 1 + (3 + 3) * len(METRIC_NAMES))

    def test_csv_neutralizes_spreadsheet_formula_labels(self) -> None:
        dangerous = run_benchmark(fixture_config(label="=1+1"))
        csv_text = benchmark_csv(dangerous)
        self.assertIn("model,'=1+1", csv_text)
        self.assertNotIn("model,=1+1", csv_text)

    def test_html_is_standalone_and_escapes_labels(self) -> None:
        html_text = benchmark_html(self.result)
        self.assertTrue(html_text.startswith("<!doctype html>"))
        self.assertIn("&lt;unsafe&gt;", html_text)
        self.assertNotIn("<unsafe>", html_text)
        self.assertNotIn("<script src=", html_text)
        self.assertNotIn("http://", html_text)
        self.assertNotIn("https://", html_text)
        self.assertIn(self.result.split_fingerprint, html_text)

    def test_report_bundle_is_strict_and_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = save_benchmark_reports(self.result, directory)
            payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload, self.result.to_dict())
            self.assertEqual(paths.to_dict()["html"], str(paths.html_path))
            self.assertTrue(paths.csv_path.read_text(encoding="utf-8").endswith("\n"))
            self.assertIn("<!doctype html>", paths.html_path.read_text(encoding="utf-8"))

    def test_reports_reject_dataset_name_and_hardlink_aliases_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "benchmark.json"
            demo_dataset().save_json(source)
            original = source.read_bytes()
            config = benchmark_config_from_dict(
                {
                    "schema_version": 1,
                    "data": {"path": str(source), "format": "orchidrec-json"},
                    "models": [
                        {"label": "ease", "name": "ease", "params": {"regularization": 1}},
                        {"label": "pop", "name": "popularity"},
                    ],
                    "evaluation": {"bootstrap_samples": 10},
                }
            )
            result = run_benchmark(config)
            with self.assertRaisesRegex(ValidationError, "different files"):
                save_benchmark_reports(result, directory)
            self.assertEqual(source.read_bytes(), original)

            source.rename(Path(directory) / "events.json")
            events = Path(directory) / "events.json"
            output = Path(directory) / "benchmark.csv"
            os.link(events, output)
            config = benchmark_config_from_dict(
                {**config.to_dict(), "data": {"path": str(events), "format": "orchidrec-json"}}
            )
            result = run_benchmark(config)
            with self.assertRaisesRegex(ValidationError, "different files"):
                save_benchmark_reports(result, directory)
            self.assertEqual(events.read_bytes(), original)

    def test_reporting_rejects_wrong_types_and_wraps_bad_destination(self) -> None:
        for function in (benchmark_csv, benchmark_html):
            with self.assertRaises(SerializationError):
                function(object())  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SerializationError):
                save_benchmark_reports(object(), directory)  # type: ignore[arg-type]
            blocked = Path(directory) / "blocked"
            blocked.write_text("file", encoding="utf-8")
            with self.assertRaisesRegex(SerializationError, "could not write"):
                save_benchmark_reports(self.result, blocked)


if __name__ == "__main__":
    unittest.main()
