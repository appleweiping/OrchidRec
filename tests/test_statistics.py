from __future__ import annotations

import math
import unittest

from orchidrec.errors import ValidationError
from orchidrec.statistics import (
    bootstrap_mean,
    interval_from_draws,
    paired_bootstrap_mean,
    paired_comparison_from_draws,
)


class BootstrapStatisticsTests(unittest.TestCase):
    def test_bootstrap_mean_is_seeded_and_contains_metadata(self) -> None:
        first = bootstrap_mean([1, 2, 3, 4], samples=200, confidence=0.9, seed=7)
        second = bootstrap_mean([1, 2, 3, 4], samples=200, confidence=0.9, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first.estimate, 2.5)
        self.assertLessEqual(first.lower, first.estimate)
        self.assertGreaterEqual(first.upper, first.estimate)
        self.assertEqual(first.to_dict()["samples"], 200)

    def test_single_observation_has_degenerate_interval(self) -> None:
        interval = bootstrap_mean([3.5], samples=3, seed=-4)
        self.assertEqual((interval.lower, interval.upper), (3.5, 3.5))

    def test_quantile_interpolates_and_validates_estimate(self) -> None:
        interval = interval_from_draws(2.0, [0, 1, 2, 3, 4], confidence=0.8)
        self.assertAlmostEqual(interval.lower, 0.4)
        self.assertAlmostEqual(interval.upper, 3.6)
        with self.assertRaisesRegex(ValidationError, "estimate"):
            interval_from_draws(float("nan"), [1], confidence=0.9)
        with self.assertRaisesRegex(ValidationError, "estimate"):
            interval_from_draws(True, [1], confidence=0.9)

    def test_paired_bootstrap_detects_consistent_improvement(self) -> None:
        result = paired_bootstrap_mean(
            [1, 2, 3], [2, 3, 4], samples=99, confidence=0.95, seed=8
        )
        self.assertEqual(result.left_mean, 2.0)
        self.assertEqual(result.right_mean, 3.0)
        self.assertEqual(result.difference.estimate, 1.0)
        self.assertAlmostEqual(result.difference.lower, 1.0)
        self.assertEqual(result.probability_right_better, 1.0)
        self.assertAlmostEqual(result.two_sided_p_value, 0.02)
        self.assertEqual(result.to_dict()["right_minus_left"]["estimate"], 1.0)  # type: ignore[index]

    def test_paired_ties_are_half_probability_and_not_significant(self) -> None:
        result = paired_bootstrap_mean([1, 1], [1, 1], samples=10)
        self.assertEqual(result.probability_right_better, 0.5)
        self.assertEqual(result.two_sided_p_value, 1.0)

    def test_comparison_from_aligned_draws(self) -> None:
        result = paired_comparison_from_draws(
            0.2,
            0.3,
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.5],
            confidence=0.8,
        )
        self.assertAlmostEqual(result.difference.estimate, 0.1)
        self.assertEqual(result.probability_right_better, 2 / 3)
        self.assertEqual(result.two_sided_p_value, 1.0)

    def test_large_finite_values_do_not_overflow_the_mean(self) -> None:
        interval = bootstrap_mean([1e308, 1e308], samples=2)
        self.assertTrue(math.isfinite(interval.estimate))
        self.assertEqual(interval.estimate, 1e308)

    def test_invalid_observations_are_rejected(self) -> None:
        invalid = ([], "abc", [True], ["1"], [float("nan")], [float("inf")])
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                bootstrap_mean(values)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "iterable"):
            bootstrap_mean(3)  # type: ignore[arg-type]

    def test_invalid_bootstrap_options_are_rejected(self) -> None:
        for samples in (0, -1, True, 1.5):
            with self.subTest(samples=samples), self.assertRaisesRegex(
                ValidationError, "samples"
            ):
                bootstrap_mean([1], samples=samples)  # type: ignore[arg-type]
        for confidence in (0, 1, True, float("nan"), float("inf"), 10**400):
            with self.subTest(confidence=confidence), self.assertRaisesRegex(
                ValidationError, "confidence"
            ):
                bootstrap_mean([1], confidence=confidence)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "seed"):
            bootstrap_mean([1], seed=True)

    def test_paired_inputs_and_draws_must_align(self) -> None:
        with self.assertRaisesRegex(ValidationError, "same number"):
            paired_bootstrap_mean([1], [1, 2])
        with self.assertRaisesRegex(ValidationError, "same length"):
            paired_comparison_from_draws(1, 2, [1], [1, 2], confidence=0.9)
        with self.assertRaisesRegex(ValidationError, "left_estimate"):
            paired_comparison_from_draws(True, 2, [1], [2], confidence=0.9)
        with self.assertRaisesRegex(ValidationError, "right_estimate"):
            paired_comparison_from_draws(1, True, [1], [2], confidence=0.9)
        with self.assertRaisesRegex(ValidationError, "estimates"):
            paired_comparison_from_draws(float("inf"), 2, [1], [2], confidence=0.9)


if __name__ == "__main__":
    unittest.main()
