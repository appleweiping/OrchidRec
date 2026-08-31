from __future__ import annotations

import unittest

from orchidrec.errors import ValidationError
from orchidrec.metrics import (
    MetricReport,
    catalog_coverage,
    evaluate_ranking,
    mrr_at_k,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
    recall_at_k,
)


class RankingMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recommendations = {"u1": ["a", "b", "c"], "u2": ["c", "a", "d"]}
        self.relevant = {"u1": {"a", "c"}, "u2": {"d"}}

    def test_precision_uses_fixed_k_denominator(self) -> None:
        self.assertAlmostEqual(precision_at_k(self.recommendations, self.relevant, 2), 0.25)

    def test_recall_is_macro_averaged(self) -> None:
        self.assertAlmostEqual(recall_at_k(self.recommendations, self.relevant, 3), 1.0)

    def test_ndcg_perfect_ranking_is_one(self) -> None:
        recs = {"u": ["a", "b"]}
        self.assertAlmostEqual(ndcg_at_k(recs, {"u": {"a", "b"}}, 2), 1.0)

    def test_ndcg_penalizes_late_hit(self) -> None:
        early = ndcg_at_k({"u": ["a", "x"]}, {"u": {"a"}}, 2)
        late = ndcg_at_k({"u": ["x", "a"]}, {"u": {"a"}}, 2)
        self.assertGreater(early, late)

    def test_mrr_uses_first_relevant_rank(self) -> None:
        self.assertAlmostEqual(mrr_at_k({"u": ["x", "a", "b"]}, {"u": {"a", "b"}}, 3), 0.5)

    def test_missing_recommendations_score_zero(self) -> None:
        relevant = {"u": {"a"}}
        self.assertEqual(precision_at_k({}, relevant, 2), 0.0)
        self.assertEqual(recall_at_k({}, relevant, 2), 0.0)
        self.assertEqual(ndcg_at_k({}, relevant, 2), 0.0)
        self.assertEqual(mrr_at_k({}, relevant, 2), 0.0)

    def test_short_list_is_not_given_extra_precision(self) -> None:
        self.assertAlmostEqual(precision_at_k({"u": ["a"]}, {"u": {"a"}}, 4), 0.25)

    def test_empty_relevant_users_are_not_evaluated(self) -> None:
        value = recall_at_k({"u": ["a"]}, {"empty": set(), "u": {"a"}}, 1)
        self.assertEqual(value, 1.0)

    def test_invalid_k_is_rejected(self) -> None:
        for k in (0, -1, True):
            with self.subTest(k=k), self.assertRaises(ValidationError):
                precision_at_k(self.recommendations, self.relevant, k)  # type: ignore[arg-type]

    def test_no_nonempty_ground_truth_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            recall_at_k({}, {"u": set()}, 2)

    def test_duplicate_recommendations_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ndcg_at_k({"u": ["a", "a"]}, {"u": {"a"}}, 2)

    def test_duplicate_beyond_cutoff_is_still_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ndcg_at_k({"u": ["a", "b", "a"]}, {"u": {"a"}}, 2)

    def test_string_is_not_treated_as_a_ranking_sequence(self) -> None:
        with self.assertRaises(ValidationError):
            recall_at_k({"u": "abc"}, {"u": {"a"}}, 2)  # type: ignore[dict-item]

    def test_invalid_boolean_item_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            precision_at_k({"u": [True]}, {"u": {"a"}}, 1)  # type: ignore[list-item]

    def test_invalid_recommendation_user_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            precision_at_k({True: ["a"]}, {1: {"a"}}, 1)  # type: ignore[dict-item]


class BeyondAccuracyMetricTests(unittest.TestCase):
    def test_catalog_coverage(self) -> None:
        value = catalog_coverage({"u1": [1, 2], "u2": [2, 3]}, {1, 2, 3, 4}, 2)
        self.assertAlmostEqual(value, 0.75)

    def test_unknown_recommended_item_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            catalog_coverage({"u": [3]}, {1, 2}, 1)

    def test_empty_catalog_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            catalog_coverage({}, set(), 1)

    def test_duplicate_catalog_items_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            catalog_coverage({"u": ["a"]}, ["a", "a"], 1)

    def test_unknown_item_beyond_cutoff_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            catalog_coverage({"u": ["a", "unknown"]}, {"a"}, 1)

    def test_rare_items_have_higher_novelty(self) -> None:
        counts = {"popular": 99, "rare": 1}
        popular = novelty_at_k({"u": ["popular"]}, counts, set(counts), 1)
        rare = novelty_at_k({"u": ["rare"]}, counts, set(counts), 1)
        self.assertGreater(rare, popular)

    def test_novelty_of_empty_output_is_zero(self) -> None:
        self.assertEqual(novelty_at_k({}, {"a": 1}, {"a"}, 2), 0.0)

    def test_negative_count_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            novelty_at_k({"u": ["a"]}, {"a": -1}, {"a"}, 1)

    def test_nonfinite_count_is_rejected(self) -> None:
        for value in (float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                novelty_at_k({"u": ["a"]}, {"a": value}, {"a"}, 1)

    def test_nonfinite_aggregate_count_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            novelty_at_k(
                {"u": ["a"]},
                {"a": 1e308, "b": 1e308},
                {"a", "b"},
                1,
            )

    def test_huge_integer_count_is_rejected_as_nonfinite(self) -> None:
        with self.assertRaises(ValidationError):
            novelty_at_k({"u": ["a"]}, {"a": 10**400}, {"a"}, 1)

    def test_count_for_non_catalog_item_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            novelty_at_k({"u": ["a"]}, {"a": 1, "b": 1}, {"a"}, 1)

    def test_evaluate_ranking_builds_complete_report(self) -> None:
        report = evaluate_ranking(
            {"u": ["a", "b"]},
            {"u": {"a"}},
            {"a", "b", "c"},
            {"a": 3, "b": 2, "c": 1},
            2,
        )
        self.assertIsInstance(report, MetricReport)
        self.assertEqual(report.users, 1)
        self.assertEqual(report.k, 2)
        self.assertEqual(
            set(report.to_dict()),
            {"precision", "recall", "ndcg", "mrr", "coverage", "novelty", "users", "k"},
        )


if __name__ == "__main__":
    unittest.main()
