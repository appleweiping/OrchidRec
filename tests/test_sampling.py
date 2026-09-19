from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from orchidrec import benchmark as benchmark_module
from orchidrec.benchmark import run_benchmark
from orchidrec.benchmark_config import benchmark_config_from_dict
from orchidrec.cli import main
from orchidrec.config import config_from_dict
from orchidrec.datasets import load_dataset
from orchidrec.demo import demo_dataset
from orchidrec.errors import ConfigurationError, ValidationError
from orchidrec.experiment import run_experiment
from orchidrec.sampling import (
    MAX_CANDIDATE_PAIRS,
    MAX_CATALOG_ITEMS,
    MAX_SAMPLING_WORK_UNITS,
    SAMPLER_REGISTRY,
    SamplingConfig,
    parse_sampling,
    sample_candidates,
)


class SamplingTests(unittest.TestCase):
    def test_registry_and_hand_computed_candidate_oracle(self) -> None:
        self.assertEqual(set(SAMPLER_REGISTRY), {"uniform", "popularity"})
        plan = sample_candidates(
            SamplingConfig("uniform", 10),
            seed=7,
            catalog=("p", "seen", "n1", "n2"),
            train_counts={"p": 1, "seen": 2, "n1": 3, "n2": 4},
            relevant={"u": {"p"}},
            seen={"u": {"seen"}},
        )
        self.assertEqual(plan.candidates["u"], ("n1", "n2", "p"))
        self.assertEqual(plan.sampling, SamplingConfig("uniform", 10))
        self.assertEqual(
            (plan.positive_pairs, plan.negative_pairs, plan.candidate_pairs), (1, 2, 3)
        )
        self.assertEqual(len(plan.fingerprint), 64)
        with self.assertRaises(TypeError):
            plan.candidates["u"] = ()  # type: ignore[index]

    def test_uniform_determinism_user_order_seed_and_id_type(self) -> None:
        catalog = tuple(range(20))
        counts = dict.fromkeys(catalog, 1)
        first = sample_candidates(
            SamplingConfig("uniform", 4),
            seed=3,
            catalog=catalog,
            train_counts=counts,
            relevant={"u": {1}, 1: {2}},
            seen={"u": {0}, 1: {0}},
        )
        reordered = sample_candidates(
            SamplingConfig("uniform", 4),
            seed=3,
            catalog=reversed(catalog),
            train_counts=counts,
            relevant={1: {2}, "u": {1}},
            seen={1: {0}, "u": {0}},
        )
        self.assertEqual(first.candidates, reordered.candidates)
        self.assertEqual(first.fingerprint, reordered.fingerprint)
        changed = sample_candidates(
            SamplingConfig("uniform", 4),
            seed=4,
            catalog=catalog,
            train_counts=counts,
            relevant={"u": {1}, 1: {2}},
            seen={"u": {0}, 1: {0}},
        )
        self.assertNotEqual(first.fingerprint, changed.fingerprint)
        self.assertTrue(all(len(row) == len(set(row)) == 5 for row in first.candidates.values()))

    def test_popularity_uses_training_counts_not_holdout(self) -> None:
        catalog = ("p", "heavy", "light")
        counts = {"p": 1, "heavy": 100, "light": 1}
        outcomes = Counter()
        for seed in range(200):
            plan = sample_candidates(
                SamplingConfig("popularity", 1),
                seed=seed,
                catalog=catalog,
                train_counts=counts,
                relevant={"u": {"p"}},
                seen={"u": ()},
            )
            self.assertIn("p", plan.candidates["u"])
            outcomes[next(item for item in plan.candidates["u"] if item != "p")] += 1
        self.assertGreater(outcomes["heavy"], 190)
        self.assertEqual(sum(outcomes.values()), 200)

    def test_no_available_negatives_and_typed_ids(self) -> None:
        plan = sample_candidates(
            SamplingConfig("uniform", 2),
            seed=0,
            catalog=(1, "1"),
            train_counts={1: 1, "1": 1},
            relevant={"u": {"1"}},
            seen={"u": {1}},
        )
        self.assertEqual(plan.candidates["u"], ("1",))
        self.assertEqual(plan.negative_pairs, 0)

    def test_positive_leakage_malformed_counts_and_bounded_inputs(self) -> None:
        args = dict(
            config=SamplingConfig("uniform", 1),
            seed=1,
            catalog=(1, 2, 3),
            train_counts={1: 1, 2: 1, 3: 1},
            relevant={"u": {1}},
            seen={"u": {2}},
        )
        with self.assertRaisesRegex(ValidationError, "overlap"):
            sample_candidates(**{**args, "seen": {"u": {1}}})
        with self.assertRaisesRegex(ValidationError, "training counts"):
            sample_candidates(**{**args, "train_counts": {1: 1, 2: 1}})
        with self.assertRaisesRegex(ValidationError, "training counts"):
            sample_candidates(**{**args, "train_counts": {1: 1, 2: 1, 3: True}})
        with self.assertRaisesRegex(ValidationError, "catalog exceeds"):
            sample_candidates(**{**args, "catalog": iter(range(MAX_CATALOG_ITEMS + 2))})
        # Lower ceilings locally to test fail-closed behavior without
        # allocating millions of candidate records.
        with (
            self.assertRaisesRegex(ValidationError, "candidate pairs exceed"),
            patch("orchidrec.sampling.MAX_CANDIDATE_PAIRS", 1),
        ):
            sample_candidates(**args)
        self.assertGreater(MAX_CANDIDATE_PAIRS, 1)
        with (
            self.assertRaisesRegex(ValidationError, "sampling work exceeds"),
            patch("orchidrec.sampling.MAX_SAMPLING_WORK_UNITS", 1),
        ):
            sample_candidates(**args)
        self.assertGreater(MAX_SAMPLING_WORK_UNITS, 1)

    def test_sampler_rejects_inconsistent_candidate_universe(self) -> None:
        args = dict(
            config=SamplingConfig("popularity", 1),
            seed=5,
            catalog=(1, 2, 3),
            train_counts={1: 1, 2: 2, 3: 3},
            relevant={"u": {1}},
            seen={"u": {2}},
        )
        cases = (
            ({"config": None}, "SamplingConfig"),
            ({"seed": True}, "seed"),
            ({"seed": 2**64}, "seed"),
            ({"catalog": ()}, "catalog must not be empty"),
            ({"catalog": (1, 2, 2)}, "duplicate"),
            ({"catalog": "abc"}, "catalog must be an iterable"),
            ({"catalog": (1, 2, 2**4097)}, "bits"),
            ({"catalog": (1, 2, "\ud800")}, "UTF-8"),
            ({"catalog": (1, 2, "é" * 3000)}, "UTF-8 bytes"),
            ({"train_counts": {1: 1, 2: 2, 9: 3}}, "training counts"),
            ({"train_counts": {1: 1, 2: 0, 3: 3}}, "training counts"),
            ({"train_counts": {1: 1, 2: 2, 3: 1_000_000_001}}, "training counts"),
            ({"train_counts": []}, "mappings"),
            ({"relevant": {}}, "at least one"),
            ({"relevant": {"u": set()}}, "relevant must be nonempty"),
            ({"relevant": {"u": {9}}}, "relevant must be nonempty"),
            ({"seen": {"u": {9}}}, "relevant must be nonempty"),
            ({"seen": {}}, "exactly the evaluated"),
            ({"seen": {"else": set()}}, "exactly the evaluated"),
            ({"seen": {"u": {2}, "else": set()}}, "exactly the evaluated"),
        )
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValidationError, message),
            ):
                sample_candidates(**{**args, **overrides})

    def test_sampled_reports_identify_comparison_universe(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "movielens-100k"
        base = {
            "schema_version": 1,
            "seed": 17,
            "data": {"path": str(fixture), "format": "movielens-100k", "minimum_rating": 4},
            "evaluation": {"k": 2, "bootstrap_samples": 8, "confidence": 0.9},
            "models": [
                {"label": "pop", "name": "popularity", "params": {}},
                {"label": "knn", "name": "item_knn", "params": {"neighbors": 2}},
            ],
        }
        from orchidrec.reporting import benchmark_csv, benchmark_html

        full = run_benchmark(benchmark_config_from_dict(base))
        self.assertNotIn("mode", full.to_dict()["evaluation"])
        self.assertNotIn("evaluation_mode", benchmark_csv(full))
        self.assertIn("full-sort", benchmark_html(full))
        sampled = {
            **base,
            "evaluation": {
                **base["evaluation"],
                "sampling": {"strategy": "uniform", "negatives": 1},
            },
        }
        with patch("orchidrec.benchmark.sample_candidates", wraps=sample_candidates) as make_plan:
            result = run_benchmark(benchmark_config_from_dict(sampled))
        self.assertEqual(make_plan.call_count, 1)
        self.assertIn("candidate_sha256", benchmark_csv(result))
        self.assertIn("sampled", benchmark_html(result))

    def test_config_is_strict_and_default_is_full_sort(self) -> None:
        self.assertIsNone(parse_sampling(None))
        for payload in (
            {},
            {"strategy": "uniform"},
            {"strategy": "bad", "negatives": 1},
            {"strategy": "uniform", "negatives": True},
            {"strategy": "uniform", "negatives": 0},
            {"strategy": "uniform", "negatives": 1, "extra": 1},
        ):
            with self.subTest(payload=payload), self.assertRaises(ConfigurationError):
                parse_sampling(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            demo_dataset().save_json(path)
            base = {"data": {"path": str(path)}, "model": {"name": "popularity"}}
            config = config_from_dict(base)
            self.assertNotIn("sampling", config.to_dict()["evaluation"])
            sampled = config_from_dict(
                {**base, "evaluation": {"sampling": {"strategy": "uniform", "negatives": 2}}}
            )
            self.assertEqual(sampled.evaluation.sampling, SamplingConfig("uniform", 2))
            self.assertEqual(sampled.to_dict()["evaluation"]["sampling"]["negatives"], 2)
            with self.assertRaisesRegex(ConfigurationError, "exclude_seen"):
                config_from_dict(
                    {
                        **base,
                        "evaluation": {
                            "exclude_seen": False,
                            "sampling": {"strategy": "uniform", "negatives": 2},
                        },
                    }
                )

    def test_experiment_sampled_vs_full_contract_and_cli_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "events.json"
            demo_dataset().save_json(data_path)
            base = {
                "seed": 7,
                "data": {"path": str(data_path)},
                "model": {"name": "popularity"},
                "evaluation": {"k": 2},
            }
            full = run_experiment(config_from_dict(base)).to_dict()
            self.assertNotIn("evaluation", full)
            payload = {
                **base,
                "evaluation": {"k": 2, "sampling": {"strategy": "uniform", "negatives": 1}},
            }
            config = config_from_dict(payload)
            result = run_experiment(config)
            report = result.to_dict()
            self.assertEqual(report["evaluation"]["mode"], "sampled")
            self.assertEqual(report["evaluation"]["sampling"]["negatives"], 1)
            self.assertEqual(
                report["evaluation"]["candidate_pairs"],
                report["evaluation"]["positive_pairs"] + report["evaluation"]["negative_pairs"],
            )
            self.assertEqual(report, run_experiment(config).to_dict())
            self.assertIsNotNone(result.candidate_plan)
            for row in result.users:
                self.assertTrue(
                    {item.item_id for item in row.recommendations}
                    <= set(result.candidate_plan.candidates[row.user_id])
                )
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["run", str(config_path)]), 0)
            self.assertEqual(json.loads(output.getvalue()), report)

    def test_benchmark_models_share_sampled_candidates_and_tuning(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "movielens-100k"
        base = {
            "schema_version": 1,
            "seed": 17,
            "data": {"path": str(fixture), "format": "movielens-100k", "minimum_rating": 4},
            "evaluation": {
                "k": 2,
                "bootstrap_samples": 8,
                "confidence": 0.9,
                "sampling": {"strategy": "popularity", "negatives": 2},
            },
            "models": [
                {"label": "pop", "name": "popularity", "params": {}},
                {"label": "knn", "name": "item_knn", "params": {"neighbors": 2}},
            ],
        }
        config = benchmark_config_from_dict(base)
        result = run_benchmark(config)
        report = result.to_dict()
        self.assertEqual(report["evaluation"]["mode"], "sampled")
        self.assertEqual(len(result.models), 2)
        self.assertEqual(len(result.comparisons), 1)
        self.assertIsNotNone(result.candidate_plan)
        # Timing is intentionally non-deterministic, but candidates are not.
        self.assertEqual(
            result.candidate_plan.fingerprint, run_benchmark(config).candidate_plan.fingerprint
        )
        tuned = {
            **base,
            "tuning": {
                "selection_metric": "ndcg",
                "direction": "maximize",
                "validation_split": {"method": "leave_one_out", "validation_ratio": 0.2},
                "implicit_mf_seeds": [17],
            },
            "models": [
                {
                    "label": "pop",
                    "name": "popularity",
                    "params": {},
                    "grid": {"weighted": [False, True]},
                },
                {"label": "knn", "name": "item_knn", "params": {"neighbors": 2}},
            ],
        }
        tuned_config = benchmark_config_from_dict(tuned)
        tuned_result = run_benchmark(tuned_config)
        self.assertIsNotNone(tuned_result.tuning)
        self.assertEqual(tuned_result.to_dict()["evaluation"]["mode"], "sampled")
        from orchidrec.reporting import benchmark_csv, benchmark_html

        csv_report = benchmark_csv(result)
        html_report = benchmark_html(result)
        self.assertIn(
            "evaluation_mode,sampling_strategy,requested_negatives,candidate_partition,candidate_sha256",
            csv_report,
        )
        self.assertIn(result.candidate_plan.fingerprint, csv_report)
        self.assertIn("not directly comparable", html_report)
        self.assertIn(result.candidate_plan.fingerprint, html_report)

        # Reconstruct both partitions independently from the source, then
        # verify every report surface names the pool actually used there.
        loaded = load_dataset(
            tuned_config.data.path,
            format=tuned_config.data.format,
            minimum_rating=tuned_config.data.minimum_rating,
        )
        outer_split = benchmark_module._split(tuned_config, loaded.dataset)
        inner_split = benchmark_module._validation_split(
            tuned_config.tuning, outer_split.train, tuned_config.seed
        )

        def expected_plan(split):
            targets = benchmark_module._targets(split)
            by_user = split.train.by_user()
            return sample_candidates(
                tuned_config.evaluation.sampling,
                seed=tuned_config.seed,
                catalog=split.train.item_ids,
                train_counts=Counter(event.item_id for event in split.train),
                relevant=targets.relevant,
                seen={
                    user: {event.item_id for event in by_user.get(user, ())}
                    for user in targets.users
                },
            )

        outer = expected_plan(outer_split)
        inner = expected_plan(inner_split)
        self.assertNotEqual(outer.fingerprint, inner.fingerprint)
        tuned_json = tuned_result.to_dict()
        self.assertEqual(tuned_json["evaluation"]["candidate_sha256"], outer.fingerprint)
        self.assertEqual(
            tuned_json["tuning"]["fingerprints"]["validation_candidate_sha256"],
            inner.fingerprint,
        )
        rows = list(csv.DictReader(StringIO(benchmark_csv(tuned_result))))
        self.assertTrue(
            {"model", "comparison_right_minus_left", "tuning_candidate", "tuning_trial"}
            <= {row["row_type"] for row in rows}
        )
        for row in rows:
            if row["row_type"].startswith("tuning_"):
                self.assertEqual(row["candidate_partition"], "validation")
                self.assertEqual(row["candidate_sha256"], inner.fingerprint)
            else:
                self.assertEqual(row["candidate_partition"], "test")
                self.assertEqual(row["candidate_sha256"], outer.fingerprint)
        tuned_html = benchmark_html(tuned_result)
        self.assertIn("Inner validation candidate SHA-256", tuned_html)
        self.assertIn("Outer test candidate SHA-256", tuned_html)
        self.assertIn(inner.fingerprint, tuned_html)
        self.assertIn(outer.fingerprint, tuned_html)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "tuned.json"
            config_path.write_text(json.dumps(tuned), encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "benchmark",
                            str(config_path),
                            "--output-dir",
                            str(Path(directory) / "report"),
                        ]
                    ),
                    0,
                )
            cli = json.loads(output.getvalue())
            self.assertEqual(cli["evaluation"]["candidate_sha256"], outer.fingerprint)
            self.assertEqual(cli["tuning"]["validation_candidate_sha256"], inner.fingerprint)
        with self.assertRaisesRegex(ConfigurationError, "exclude_seen"):
            benchmark_config_from_dict(
                {**base, "evaluation": {**base["evaluation"], "exclude_seen": False}}
            )


if __name__ == "__main__":
    unittest.main()
