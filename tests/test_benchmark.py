from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
from orchidrec.errors import ConfigurationError, SerializationError
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
        pairs = {(comparison.left_label, comparison.right_label) for comparison in self.result.comparisons}
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
