from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchidrec import propensity
from orchidrec.config import config_from_dict
from orchidrec.demo import demo_dataset
from orchidrec.errors import ConfigurationError, ValidationError
from orchidrec.experiment import run_experiment
from orchidrec.propensity import (
    DEFAULT_EXPONENT,
    DEFAULT_MINIMUM_PROPENSITY,
    ExposureModel,
    popularity_exposure,
    uniform_exposure,
)
from orchidrec.unbiased import (
    UnbiasedMetricReport,
    evaluate_unbiased_ranking,
    ips_ndcg_at_k,
    ips_recall_at_k,
)

COUNTS = {"hot": 100.0, "warm": 10.0, "cold": 1.0, "unseen": 0.0}


def micro_recall(recommendations, relevant, k):
    """Uncorrected recall averaged over interactions rather than users."""
    hits = 0
    total = 0
    for user_id, items in relevant.items():
        ranked = set(recommendations.get(user_id, [])[:k])
        hits += sum(1 for item_id in items if item_id in ranked)
        total += len(items)
    return hits / total


def micro_ndcg(recommendations, relevant, k):
    gain = 0.0
    ideal = 0.0
    for user_id, items in relevant.items():
        ranked = recommendations.get(user_id, [])[:k]
        gain += sum(
            1.0 / math.log2(rank + 2) for rank, item_id in enumerate(ranked) if item_id in items
        )
        ideal += sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(items))))
    return gain / ideal


class PopularityExposureTests(unittest.TestCase):
    def test_the_most_popular_item_anchors_the_scale(self) -> None:
        model = popularity_exposure(COUNTS)
        self.assertAlmostEqual(model.propensities["hot"], 1.0)
        self.assertAlmostEqual(model.weight("hot"), 1.0)

    def test_a_rarer_item_earns_a_larger_weight(self) -> None:
        model = popularity_exposure(COUNTS)
        self.assertGreater(model.weight("cold"), model.weight("warm"))
        self.assertGreater(model.weight("warm"), model.weight("hot"))

    def test_a_never_observed_item_takes_the_floor(self) -> None:
        # 0.02 sits below the rarest observed item, whose propensity is
        # 0.01 ** 0.75 = 0.0316, so only the unobserved item is clipped.
        model = popularity_exposure(COUNTS, minimum=0.02)
        self.assertAlmostEqual(model.propensities["unseen"], 0.02)
        self.assertAlmostEqual(model.weight("unseen"), 50.0)
        self.assertGreater(model.propensities["cold"], 0.02)
        self.assertEqual(model.clipped_items, 1)

    def test_a_higher_floor_clips_more_of_the_tail(self) -> None:
        self.assertEqual(popularity_exposure(COUNTS, minimum=0.05).clipped_items, 2)
        self.assertEqual(popularity_exposure(COUNTS, minimum=0.5).clipped_items, 3)

    def test_a_zero_exponent_is_the_identity(self) -> None:
        model = popularity_exposure(COUNTS, exponent=0.0, minimum=1.0)
        self.assertEqual(set(model.propensities.values()), {1.0})

    def test_a_larger_exponent_spreads_the_weights_further(self) -> None:
        gentle = popularity_exposure(COUNTS, exponent=0.25, minimum=1e-6)
        steep = popularity_exposure(COUNTS, exponent=1.0, minimum=1e-6)
        self.assertGreater(steep.weight("cold"), gentle.weight("cold"))

    def test_the_default_exponent_matches_the_published_eta(self) -> None:
        # Yang et al. take p proportional to n ** ((eta + 1) / 2) with eta = 0.5.
        self.assertAlmostEqual(DEFAULT_EXPONENT, (0.5 + 1.0) / 2.0)

    def test_uniform_exposure_weights_everything_equally(self) -> None:
        model = uniform_exposure(COUNTS)
        self.assertEqual(model.source, "uniform")
        self.assertEqual(model.clipped_items, 0)
        self.assertTrue(all(model.weight(item) == 1.0 for item in COUNTS))

    def test_an_unmodelled_item_is_refused_rather_than_assumed(self) -> None:
        model = popularity_exposure(COUNTS)
        with self.assertRaises(ValidationError) as caught:
            model.weight("absent")
        self.assertIn("absent from the exposure model", str(caught.exception))

    def test_the_summary_omits_the_per_item_table(self) -> None:
        summary = popularity_exposure(COUNTS).to_dict()
        self.assertEqual(summary["source"], "popularity")
        self.assertEqual(summary["items"], 4)
        self.assertEqual(summary["exponent"], DEFAULT_EXPONENT)
        self.assertEqual(summary["minimum"], DEFAULT_MINIMUM_PROPENSITY)
        self.assertNotIn("propensities", summary)

    def test_counts_are_validated(self) -> None:
        for counts, message in (
            ([], "must be a mapping"),
            ({}, "at least one item"),
            ({"a": -1.0}, "non-negative"),
            ({"a": True}, "non-negative"),
            ({"a": float("nan")}, "non-negative"),
            ({"a": 0.0}, "at least one positive count"),
        ):
            with self.subTest(counts=counts):
                with self.assertRaises(ValidationError) as caught:
                    popularity_exposure(counts)  # type: ignore[arg-type]
                self.assertIn(message, str(caught.exception))

    def test_parameters_are_validated(self) -> None:
        for kwargs, message in (
            ({"exponent": -0.1}, "exponent"),
            ({"exponent": 1.1}, "exponent"),
            ({"exponent": True}, "exponent"),
            ({"exponent": float("inf")}, "exponent"),
            ({"minimum": 0.0}, "minimum"),
            ({"minimum": 1.5}, "minimum"),
            ({"minimum": True}, "minimum"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValidationError) as caught:
                    popularity_exposure(COUNTS, **kwargs)  # type: ignore[arg-type]
                self.assertIn(message, str(caught.exception))

    def test_the_model_validates_its_own_fields(self) -> None:
        base = dict(
            propensities={"a": 0.5},
            exponent=0.5,
            minimum=0.1,
            clipped_items=0,
            source="popularity",
        )
        for override, message in (
            ({"propensities": {}}, "at least one item"),
            ({"propensities": []}, "must be a mapping"),
            ({"propensities": {"a": 0.0}}, "in (0, 1]"),
            ({"propensities": {"a": 1.5}}, "in (0, 1]"),
            ({"propensities": {"a": True}}, "in (0, 1]"),
            ({"exponent": float("nan")}, "finite"),
            ({"exponent": 2.0}, "between 0 and 1"),
            ({"minimum": 0.0}, "in (0, 1]"),
            ({"clipped_items": 1.5}, "must be an integer"),
            ({"clipped_items": True}, "must be an integer"),
            ({"clipped_items": 9}, "must not exceed"),
            ({"source": ""}, "non-empty string"),
        ):
            with self.subTest(override=override):
                with self.assertRaises(ValidationError) as caught:
                    ExposureModel(**{**base, **override})  # type: ignore[arg-type]
                self.assertIn(message, str(caught.exception))


class ReductionTests(unittest.TestCase):
    """Under uniform exposure the corrected metrics must be the plain ones."""

    def setUp(self) -> None:
        self.recommendations = {"u1": ["hot", "cold", "warm"], "u2": ["warm", "unseen", "hot"]}
        self.relevant = {"u1": {"cold", "warm"}, "u2": {"hot", "unseen"}}
        self.uniform = uniform_exposure(COUNTS)

    def test_recall_reduces_to_the_micro_average(self) -> None:
        for k in (1, 2, 3):
            with self.subTest(k=k):
                self.assertAlmostEqual(
                    ips_recall_at_k(self.recommendations, self.relevant, self.uniform, k),
                    micro_recall(self.recommendations, self.relevant, k),
                    places=12,
                )

    def test_ndcg_reduces_to_the_micro_average(self) -> None:
        for k in (1, 2, 3):
            with self.subTest(k=k):
                self.assertAlmostEqual(
                    ips_ndcg_at_k(self.recommendations, self.relevant, self.uniform, k),
                    micro_ndcg(self.recommendations, self.relevant, k),
                    places=12,
                )

    def test_a_perfect_ranking_still_scores_one(self) -> None:
        recommendations = {"u": ["cold", "hot"]}
        relevant = {"u": {"cold", "hot"}}
        model = popularity_exposure(COUNTS)
        self.assertAlmostEqual(ips_recall_at_k(recommendations, relevant, model, 2), 1.0)
        self.assertAlmostEqual(ips_ndcg_at_k(recommendations, relevant, model, 2), 1.0)

    def test_an_empty_ranking_scores_zero(self) -> None:
        recommendations = {"u": ["warm", "hot"]}
        relevant = {"u": {"cold"}}
        model = popularity_exposure(COUNTS)
        self.assertAlmostEqual(ips_recall_at_k(recommendations, relevant, model, 2), 0.0)
        self.assertAlmostEqual(ips_ndcg_at_k(recommendations, relevant, model, 2), 0.0)


class WeightingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = popularity_exposure(COUNTS)

    def test_finding_a_rare_item_beats_finding_a_popular_one(self) -> None:
        relevant = {"u1": {"cold"}, "u2": {"hot"}}
        found_rare = {"u1": ["cold"], "u2": ["warm"]}
        found_popular = {"u1": ["warm"], "u2": ["hot"]}
        self.assertGreater(
            ips_recall_at_k(found_rare, relevant, self.model, 1),
            ips_recall_at_k(found_popular, relevant, self.model, 1),
        )

    def test_the_uncorrected_metric_cannot_tell_them_apart(self) -> None:
        relevant = {"u1": {"cold"}, "u2": {"hot"}}
        found_rare = {"u1": ["cold"], "u2": ["warm"]}
        found_popular = {"u1": ["warm"], "u2": ["hot"]}
        self.assertAlmostEqual(
            micro_recall(found_rare, relevant, 1), micro_recall(found_popular, relevant, 1)
        )

    def test_ndcg_still_prefers_an_earlier_hit(self) -> None:
        relevant = {"u": {"cold"}}
        early = ips_ndcg_at_k({"u": ["cold", "hot"]}, relevant, self.model, 2)
        late = ips_ndcg_at_k({"u": ["hot", "cold"]}, relevant, self.model, 2)
        self.assertGreater(early, late)

    def test_a_non_model_exposure_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            ips_recall_at_k({"u": ["hot"]}, {"u": {"hot"}}, object(), 1)  # type: ignore[arg-type]
        self.assertIn("ExposureModel", str(caught.exception))


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recommendations = {"u1": ["hot", "cold"], "u2": ["warm", "hot"]}
        self.relevant = {"u1": {"cold", "hot"}, "u2": {"warm"}}

    def test_a_uniform_model_has_a_full_effective_sample(self) -> None:
        report = evaluate_unbiased_ranking(
            self.recommendations, self.relevant, uniform_exposure(COUNTS), 2
        )
        self.assertAlmostEqual(report.effective_sample_size, 1.0)
        self.assertEqual(report.clipped_interactions, 0)
        self.assertEqual(report.observed_interactions, 3)
        self.assertEqual(report.users, 2)
        self.assertEqual(report.k, 2)

    def test_skewed_weights_shrink_the_effective_sample(self) -> None:
        report = evaluate_unbiased_ranking(
            self.recommendations, self.relevant, popularity_exposure(COUNTS), 2
        )
        self.assertLess(report.effective_sample_size, 1.0)
        self.assertGreater(report.effective_sample_size, 0.0)

    def test_clipped_observations_are_counted(self) -> None:
        relevant = {"u": {"unseen", "hot"}}
        report = evaluate_unbiased_ranking(
            {"u": ["unseen"]}, relevant, popularity_exposure(COUNTS, minimum=0.5), 1
        )
        self.assertEqual(report.clipped_interactions, 1)

    def test_the_report_is_json_compatible(self) -> None:
        report = evaluate_unbiased_ranking(
            self.recommendations, self.relevant, popularity_exposure(COUNTS), 2
        )
        payload = report.to_dict()
        self.assertEqual(payload["exposure"]["source"], "popularity")
        self.assertEqual(payload["users"], 2)
        self.assertIsInstance(report, UnbiasedMetricReport)

    def test_k_and_recommendations_are_validated(self) -> None:
        model = uniform_exposure(COUNTS)
        with self.assertRaises(ValidationError):
            evaluate_unbiased_ranking(self.recommendations, self.relevant, model, 0)
        with self.assertRaises(ValidationError):
            evaluate_unbiased_ranking([], self.relevant, model, 1)  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            evaluate_unbiased_ranking(self.recommendations, {}, model, 1)


class BiasRecoveryTests(unittest.TestCase):
    """The reason the correction exists, checked against a known ground truth.

    Relevance is drawn independently of popularity, so the true micro-averaged
    recall of any ranker is computable. Exposure is then made popularity-biased
    and only exposed relevant items enter the holdout, which is exactly the
    missing-not-at-random situation the estimator is for.
    """

    ITEMS = 300
    USERS = 500
    K = 20
    LIKED_PER_USER = 25

    def _world(self, exposure_exponent: float, seed: int = 11):
        rng = random.Random(seed)
        items = [f"i{index:04d}" for index in range(self.ITEMS)]
        counts = {item: self.ITEMS / (index + 1) for index, item in enumerate(items)}
        largest = max(counts.values())
        probability = {
            item: (count / largest) ** exposure_exponent for item, count in counts.items()
        }
        true_relevant: dict[str, set[str]] = {}
        observed: dict[str, set[str]] = {}
        for index in range(self.USERS):
            user_id = f"u{index:04d}"
            liked = rng.sample(items, self.LIKED_PER_USER)
            true_relevant[user_id] = set(liked)
            seen = {item for item in liked if rng.random() < probability[item]}
            if seen:
                observed[user_id] = seen
        popular_first = sorted(items, key=lambda item: -counts[item])
        recommendations = {user_id: popular_first[: self.K] for user_id in true_relevant}
        observed_counts = dict.fromkeys(items, 0.0)
        for seen in observed.values():
            for item in seen:
                observed_counts[item] += 1.0
        return recommendations, true_relevant, observed, observed_counts

    def test_unbiased_logging_needs_no_correction(self) -> None:
        recommendations, true_relevant, observed, counts = self._world(0.0)
        truth = micro_recall(recommendations, true_relevant, self.K)
        naive = micro_recall(recommendations, observed, self.K)
        corrected = ips_recall_at_k(
            recommendations, observed, popularity_exposure(counts, exponent=0.0), self.K
        )
        self.assertAlmostEqual(truth, naive, places=10)
        self.assertAlmostEqual(truth, corrected, places=10)

    def test_the_correction_recovers_most_of_the_lost_accuracy(self) -> None:
        for exponent in (0.5, 1.0):
            with self.subTest(exposure_exponent=exponent):
                recommendations, true_relevant, observed, counts = self._world(exponent)
                truth = micro_recall(recommendations, true_relevant, self.K)
                naive = micro_recall(recommendations, observed, self.K)
                model = popularity_exposure(counts, exponent=exponent, minimum=0.01)
                corrected = ips_recall_at_k(recommendations, observed, model, self.K)

                naive_error = abs(naive - truth)
                corrected_error = abs(corrected - truth)
                # The naive estimate must be badly wrong for the test to mean
                # anything, and the correction must remove most of that error.
                self.assertGreater(naive_error, 0.1)
                self.assertLess(corrected_error, naive_error * 0.5)

    def test_a_harder_exposure_bias_shrinks_the_effective_sample(self) -> None:
        sizes = []
        for exponent in (0.0, 0.5, 1.0):
            _recommendations, _true, observed, counts = self._world(exponent)
            model = popularity_exposure(counts, exponent=exponent, minimum=0.01)
            report = evaluate_unbiased_ranking(
                {user: [] for user in observed}, observed, model, self.K
            )
            sizes.append(report.effective_sample_size)
        self.assertAlmostEqual(sizes[0], 1.0)
        self.assertGreater(sizes[0], sizes[1])
        self.assertGreater(sizes[1], sizes[2])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class ConfigurationTests(unittest.TestCase):
    def _config(self, evaluation, directory):
        demo_dataset().save_json(Path(directory) / "events.json")
        return config_from_dict(
            {
                "seed": 13,
                "data": {"path": "events.json"},
                "split": {"method": "leave_one_out"},
                "model": {"name": "popularity", "params": {}},
                "evaluation": evaluation,
                "output": {},
            },
            base_dir=directory,
        )

    def test_no_exposure_section_asks_for_no_correction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config({"k": 3}, directory)
            self.assertIsNone(config.evaluation.exposure)
            self.assertIsNone(config.to_dict()["evaluation"]["exposure"])

    def test_an_empty_exposure_section_takes_the_documented_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config({"k": 3, "exposure": {}}, directory)
            self.assertIsNotNone(config.evaluation.exposure)
            assert config.evaluation.exposure is not None
            self.assertAlmostEqual(config.evaluation.exposure.exponent, DEFAULT_EXPONENT)
            self.assertAlmostEqual(config.evaluation.exposure.minimum, DEFAULT_MINIMUM_PROPENSITY)

    def test_exposure_parameters_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(
                {"k": 3, "exposure": {"exponent": 0.4, "minimum": 0.05}}, directory
            )
            payload = config.to_dict()["evaluation"]["exposure"]
            self.assertEqual(payload, {"exponent": 0.4, "minimum": 0.05})

    def test_exposure_parameters_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for exposure, message in (
                ({"exponent": 1.5}, "evaluation.exposure.exponent"),
                ({"exponent": "high"}, "evaluation.exposure.exponent"),
                ({"exponent": True}, "evaluation.exposure.exponent"),
                ({"minimum": 0.0}, "evaluation.exposure.minimum"),
                ({"minimum": 2.0}, "evaluation.exposure.minimum"),
                ({"eta": 0.5}, "unknown evaluation.exposure fields"),
                ([], "evaluation.exposure must be a JSON object"),
            ):
                with self.subTest(exposure=exposure):
                    with self.assertRaises(ConfigurationError) as caught:
                        self._config({"k": 3, "exposure": exposure}, directory)
                    self.assertIn(message, str(caught.exception))


class ExperimentIntegrationTests(unittest.TestCase):
    def _run(self, evaluation, directory):
        demo_dataset().save_json(Path(directory) / "events.json")
        config = config_from_dict(
            {
                "seed": 13,
                "data": {"path": "events.json"},
                "split": {"method": "leave_one_out"},
                "model": {"name": "popularity", "params": {}},
                "evaluation": evaluation,
                "output": {},
            },
            base_dir=directory,
        )
        return run_experiment(config)

    def test_a_run_without_an_exposure_model_reports_exactly_as_before(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self._run({"k": 3}, directory)
            self.assertIsNone(result.unbiased_metrics)
            self.assertNotIn("unbiased_metrics", result.to_dict())

    def test_a_configured_run_adds_the_corrected_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plain = self._run({"k": 3}, directory).to_dict()
            corrected = self._run({"k": 3, "exposure": {}}, directory).to_dict()

        # The section is additive: nothing else about the report moves.
        self.assertEqual(
            {key: value for key, value in corrected.items() if key != "unbiased_metrics"},
            plain,
        )
        section = corrected["unbiased_metrics"]
        self.assertEqual(section["k"], 3)
        self.assertEqual(section["exposure"]["source"], "popularity")
        self.assertGreater(section["observed_interactions"], 0)
        self.assertGreaterEqual(section["effective_sample_size"], 0.0)

    def test_every_evaluated_item_has_a_propensity(self) -> None:
        # Cold-start test items are dropped before evaluation, so every relevant
        # item is in the training catalogue and therefore in the count table the
        # exposure model is built from. If that ever stopped holding, `weight`
        # would raise rather than silently invent a propensity.
        with tempfile.TemporaryDirectory() as directory:
            for model_name in ("popularity", "item_knn", "implicit_mf"):
                with self.subTest(model=model_name):
                    demo_dataset().save_json(Path(directory) / "events.json")
                    params = {"factors": 3, "epochs": 3} if model_name == "implicit_mf" else {}
                    config = config_from_dict(
                        {
                            "seed": 13,
                            "data": {"path": "events.json"},
                            "split": {"method": "leave_one_out"},
                            "model": {"name": model_name, "params": params},
                            "evaluation": {"k": 3, "exposure": {}},
                            "output": {},
                        },
                        base_dir=directory,
                    )
                    result = run_experiment(config)
                    self.assertIsNotNone(result.unbiased_metrics)

    def test_a_configured_run_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo_dataset().save_json(Path(directory) / "events.json")
            config = config_from_dict(
                {
                    "seed": 13,
                    "data": {"path": "events.json"},
                    "split": {"method": "leave_one_out"},
                    "model": {"name": "implicit_mf", "params": {"factors": 3, "epochs": 3}},
                    "evaluation": {"k": 3, "exposure": {"exponent": 0.6}},
                    "output": {},
                },
                base_dir=directory,
            )
            self.assertEqual(run_experiment(config).to_dict(), run_experiment(config).to_dict())


class ResourceGuardTests(unittest.TestCase):
    """The item ceilings exist so a malformed table cannot allocate without bound."""

    def test_a_count_table_beyond_the_ceiling_is_refused(self) -> None:
        with (
            mock.patch.object(propensity, "MAX_PROPENSITY_ITEMS", 2),
            self.assertRaises(ValidationError) as caught,
        ):
            popularity_exposure({"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertIn("must not exceed", str(caught.exception))

    def test_a_propensity_table_beyond_the_ceiling_is_refused(self) -> None:
        with (
            mock.patch.object(propensity, "MAX_PROPENSITY_ITEMS", 2),
            self.assertRaises(ValidationError) as caught,
        ):
            ExposureModel(
                propensities={"a": 0.5, "b": 0.5, "c": 0.5},
                exponent=0.5,
                minimum=0.1,
                clipped_items=0,
                source="popularity",
            )
        self.assertIn("must not exceed", str(caught.exception))
