from __future__ import annotations

import math
import random
import unittest

from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import NotFittedError, ValidationError
from orchidrec.models import ImplicitMF, ItemKNN, Popularity


def collaborative_dataset() -> InteractionDataset:
    return InteractionDataset(
        [
            Interaction("u1", "a", 1, 1),
            Interaction("u1", "b", 3, 2),
            Interaction("u2", "a", 1, 3),
            Interaction("u2", "b", 1, 4),
            Interaction("u3", "a", 1, 5),
            Interaction("u3", "c", 1, 6),
            Interaction("target", "a", 1, 7),
            Interaction("other", "d", 1, 8),
        ]
    )


class CommonModelBehaviorTests(unittest.TestCase):
    def test_scoring_before_fit_is_rejected(self) -> None:
        with self.assertRaises(NotFittedError):
            Popularity().score_items("u")

    def test_empty_fit_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Popularity().fit(InteractionDataset())

    def test_fit_rejects_wrong_dataset_type(self) -> None:
        with self.assertRaises(ValidationError):
            Popularity().fit([])  # type: ignore[arg-type]

    def test_seen_items_are_excluded_by_default(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        recommended = {entry.item_id for entry in model.recommend("u1", 10)}
        self.assertTrue(recommended.isdisjoint({"a", "b"}))

    def test_seen_items_can_be_included(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        recommended = [entry.item_id for entry in model.recommend("u1", 2, exclude_seen=False)]
        self.assertIn("a", recommended)

    def test_seen_items_accessor_and_unknown_user(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        self.assertEqual(model.seen_items("u1"), frozenset({"a", "b"}))
        self.assertEqual(model.seen_items("unknown"), frozenset())

    def test_non_boolean_exclude_seen_is_rejected(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        with self.assertRaises(ValidationError):
            model.recommend("u1", exclude_seen=1)  # type: ignore[arg-type]

    def test_candidate_subset_is_respected(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        result = model.recommend("new", 5, candidates=["d", "c"])
        self.assertEqual({entry.item_id for entry in result}, {"c", "d"})

    def test_duplicate_candidates_are_rejected(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        with self.assertRaises(ValidationError):
            model.recommend("new", 2, candidates=["c", "c"])

    def test_candidate_string_is_rejected_instead_of_iterated(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        with self.assertRaises(ValidationError):
            model.recommend("new", 2, candidates="abc")

    def test_non_iterable_candidates_are_rejected(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        with self.assertRaises(ValidationError):
            model.recommend("new", 2, candidates=3)  # type: ignore[arg-type]

    def test_unknown_candidate_is_rejected(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        with self.assertRaises(ValidationError):
            model.recommend("new", 2, candidates=["missing"])

    def test_invalid_k_is_rejected(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        for k in (0, -1, True):
            with self.subTest(k=k), self.assertRaises(ValidationError):
                model.recommend("u", k)  # type: ignore[arg-type]

    def test_ranks_are_one_based_and_contiguous(self) -> None:
        result = Popularity().fit(collaborative_dataset()).recommend("new", 3)
        self.assertEqual([entry.rank for entry in result], [1, 2, 3])

    def test_equal_scores_use_stable_id_order(self) -> None:
        dataset = InteractionDataset([Interaction("u1", "z"), Interaction("u2", "a")])
        result = Popularity(weighted=False).fit(dataset).recommend("new", 2)
        self.assertEqual([entry.item_id for entry in result], ["a", "z"])

    def test_nonfinite_model_score_is_rejected(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        model._scores["a"] = float("nan")
        with self.assertRaises(ValidationError):
            model.score_items("new")

    def test_huge_integer_model_score_is_rejected(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        model._scores["a"] = 10**400
        with self.assertRaises(ValidationError):
            model.score_items("new")


class PopularityTests(unittest.TestCase):
    def test_weighted_popularity_sums_values(self) -> None:
        model = Popularity(weighted=True).fit(collaborative_dataset())
        scores = model.score_items("new")
        self.assertEqual(scores["b"], 4.0)
        self.assertEqual(scores["a"], 4.0)

    def test_unweighted_popularity_counts_events(self) -> None:
        model = Popularity(weighted=False).fit(collaborative_dataset())
        self.assertEqual(model.score_items("new")["b"], 2.0)

    def test_unknown_user_receives_normal_ranking(self) -> None:
        model = Popularity().fit(collaborative_dataset())
        self.assertEqual(model.recommend("unknown", 1)[0].item_id, "a")

    def test_weighted_parameter_must_be_boolean(self) -> None:
        with self.assertRaises(ValidationError):
            Popularity(weighted=1)  # type: ignore[arg-type]


class ItemKNNTests(unittest.TestCase):
    def test_cosine_similarity_matches_hand_calculation(self) -> None:
        dataset = InteractionDataset(
            [
                Interaction("u1", "a"),
                Interaction("u1", "b"),
                Interaction("u2", "a"),
                Interaction("u2", "c"),
            ]
        )
        state = ItemKNN(neighbors=3, shrinkage=0).fit(dataset).to_state()
        catalog = state["base"]["catalog"]
        row_b = state["model"]["similarities"][catalog.index("b")]
        similarity_to_a = next(entry["score"] for entry in row_b if entry["item_id"] == "a")
        self.assertAlmostEqual(similarity_to_a, 1.0 / math.sqrt(2.0))

    def test_correlated_item_is_ranked_first(self) -> None:
        model = ItemKNN(neighbors=3, shrinkage=0).fit(collaborative_dataset())
        result = model.recommend("target", 3)
        self.assertEqual(result[0].item_id, "b")

    def test_unknown_user_uses_popularity_fallback(self) -> None:
        model = ItemKNN(neighbors=3).fit(collaborative_dataset())
        self.assertEqual(model.recommend("unknown", 1)[0].item_id, "a")

    def test_zero_similarity_uses_small_fallback(self) -> None:
        model = ItemKNN(neighbors=1).fit(collaborative_dataset())
        scores = model.score_items("target")
        self.assertGreaterEqual(scores["d"], 0.0)

    def test_item_knn_is_deterministic(self) -> None:
        first = ItemKNN(neighbors=3, shrinkage=1).fit(collaborative_dataset()).to_state()
        second = ItemKNN(neighbors=3, shrinkage=1).fit(collaborative_dataset()).to_state()
        self.assertEqual(first, second)

    def test_invalid_neighbors_are_rejected(self) -> None:
        for value in (0, -1, True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ItemKNN(neighbors=value)  # type: ignore[arg-type]

    def test_invalid_shrinkage_is_rejected(self) -> None:
        for value in (-0.1, 10**400):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ItemKNN(shrinkage=value)

    def test_numeric_overflow_during_similarity_training_is_rejected(self) -> None:
        dataset = InteractionDataset(
            [Interaction("u", "a", 1e200), Interaction("u", "b", 1e200)]
        )
        with self.assertRaises(ValidationError):
            ItemKNN(shrinkage=0).fit(dataset)


class ImplicitMFTests(unittest.TestCase):
    def make_model(self, seed: int = 9) -> ImplicitMF:
        return ImplicitMF(factors=5, epochs=8, learning_rate=0.04, regularization=0.01, seed=seed)

    def test_same_seed_produces_identical_state(self) -> None:
        first = self.make_model().fit(collaborative_dataset()).to_state()
        second = self.make_model().fit(collaborative_dataset()).to_state()
        self.assertEqual(first, second)

    def test_training_does_not_change_global_random_state(self) -> None:
        random.seed(773)
        state = random.getstate()
        self.make_model().fit(collaborative_dataset())
        self.assertEqual(random.getstate(), state)

    def test_different_seed_changes_factors(self) -> None:
        first = self.make_model(1).fit(collaborative_dataset()).to_state()["model"]
        second = self.make_model(2).fit(collaborative_dataset()).to_state()["model"]
        self.assertNotEqual(first, second)

    def test_matrix_factorization_returns_finite_ranked_scores(self) -> None:
        result = self.make_model().fit(collaborative_dataset()).recommend("target", 3)
        self.assertTrue(result)
        self.assertEqual([entry.rank for entry in result], list(range(1, len(result) + 1)))

    def test_unknown_user_uses_popularity_fallback(self) -> None:
        result = self.make_model().fit(collaborative_dataset()).recommend("unknown", 1)
        self.assertEqual(result[0].item_id, "a")

    def test_user_who_saw_entire_catalog_has_no_unseen_recommendations(self) -> None:
        dataset = InteractionDataset([Interaction("u", "a"), Interaction("u", "b")])
        model = self.make_model().fit(dataset)
        self.assertEqual(model.recommend("u", 10), [])

    def test_dense_user_does_not_break_negative_sampling(self) -> None:
        dataset = InteractionDataset(
            [Interaction("dense", "a"), Interaction("dense", "b"), Interaction("other", "a")]
        )
        model = self.make_model().fit(dataset)
        self.assertTrue(model.is_fitted)

    def test_invalid_hyperparameters_are_rejected(self) -> None:
        bad_parameters = [
            {"factors": 0},
            {"epochs": 0},
            {"learning_rate": 0},
            {"learning_rate": 10**400},
            {"regularization": -1},
            {"negative_samples": 0},
            {"seed": True},
        ]
        for parameters in bad_parameters:
            with self.subTest(parameters=parameters), self.assertRaises(ValidationError):
                ImplicitMF(**parameters)  # type: ignore[arg-type]

    def test_more_epochs_change_fitted_state(self) -> None:
        short = ImplicitMF(factors=3, epochs=1, seed=3).fit(collaborative_dataset()).to_state()["model"]
        long = ImplicitMF(factors=3, epochs=4, seed=3).fit(collaborative_dataset()).to_state()["model"]
        self.assertNotEqual(short, long)

    def test_bpr_training_ranks_each_observed_item_above_negatives(self) -> None:
        dataset = InteractionDataset(
            [Interaction("u1", "a"), Interaction("u2", "b"), Interaction("u3", "c")]
        )
        model = ImplicitMF(
            factors=8,
            epochs=100,
            learning_rate=0.05,
            regularization=0.01,
            negative_samples=2,
            seed=4,
        ).fit(dataset)
        for user_id, positive in (("u1", "a"), ("u2", "b"), ("u3", "c")):
            with self.subTest(user_id=user_id):
                scores = model.score_items(user_id)
                self.assertGreater(
                    scores[positive],
                    max(score for item_id, score in scores.items() if item_id != positive),
                )


if __name__ == "__main__":
    unittest.main()
