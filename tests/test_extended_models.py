from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from orchidrec.benchmark_config import default_benchmark_models
from orchidrec.config import config_from_dict
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.experiment import build_model
from orchidrec.models import SequentialMarkov, UserKNN, load_model, save_model


def timestamped_dataset() -> InteractionDataset:
    return InteractionDataset(
        [
            Interaction("target", "a", 1, 1),
            Interaction("near", "a", 1, 1),
            Interaction("near", "b", 3, 2),
            Interaction("far", "a", 1, 1),
            Interaction("far", "c", 1, 2),
            Interaction("far", "d", 1, 3),
        ]
    )


class UserKNNTests(unittest.TestCase):
    def test_cosine_neighborhood_is_independently_hand_checkable(self) -> None:
        model = UserKNN(neighbors=4, shrinkage=0).fit(timestamped_dataset())
        weights = dict(model._neighbor_weights["target"])
        self.assertAlmostEqual(weights["near"], 1 / math.sqrt(10))
        self.assertAlmostEqual(weights["far"], 1 / math.sqrt(3))

    def test_neighbor_values_drive_ranking(self) -> None:
        model = UserKNN(neighbors=4, shrinkage=0).fit(timestamped_dataset())
        result = model.recommend("target", 3)
        self.assertEqual(result[0].item_id, "b")
        self.assertGreater(result[0].score, result[1].score)

    def test_neighborhood_limit_is_real_not_an_alias(self) -> None:
        model = UserKNN(neighbors=1, shrinkage=0).fit(timestamped_dataset())
        # The most similar user is "far", so only c/d receive neighbor evidence.
        self.assertEqual([item.item_id for item in model.recommend("target", 2)], ["c", "d"])

    def test_unknown_user_uses_popularity_fallback(self) -> None:
        result = UserKNN().fit(timestamped_dataset()).recommend("unknown", 1)
        self.assertEqual(result[0].item_id, "a")

    def test_state_round_trip_preserves_scores(self) -> None:
        model = UserKNN(neighbors=3, shrinkage=0.5).fit(timestamped_dataset())
        restored = UserKNN.from_state(model.to_state())
        self.assertEqual(restored.to_state(), model.to_state())
        self.assertEqual(restored.score_items("target"), model.score_items("target"))

    def test_invalid_hyperparameters_are_rejected(self) -> None:
        for kwargs in (
            {"neighbors": 0},
            {"neighbors": True},
            {"shrinkage": -1},
            {"shrinkage": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                UserKNN(**kwargs)  # type: ignore[arg-type]

    def test_numeric_overflow_is_rejected(self) -> None:
        huge = InteractionDataset([Interaction("u1", "a", 1e200), Interaction("u2", "a", 1e200)])
        with self.assertRaises(ValidationError):
            UserKNN(shrinkage=0).fit(huge)

    def test_corrupt_state_is_rejected(self) -> None:
        state = UserKNN().fit(timestamped_dataset()).to_state()
        corrupt = copy.deepcopy(state)
        corrupt["model"]["user_values"][0]["values"][0]["value"] = -1
        with self.assertRaises(SerializationError):
            UserKNN.from_state(corrupt)


class SequentialMarkovTests(unittest.TestCase):
    def test_first_order_transition_ranks_observed_successor(self) -> None:
        dataset = InteractionDataset(
            [
                Interaction("u1", "a", 1, 1),
                Interaction("u1", "b", 1, 2),
                Interaction("u2", "a", 1, 1),
                Interaction("u2", "b", 1, 2),
                Interaction("target", "a", 1, 3),
                Interaction("catalog", "c", 1, 1),
            ]
        )
        result = SequentialMarkov(popularity_mix=0).fit(dataset).recommend("target", 2)
        self.assertEqual(result[0].item_id, "b")
        self.assertEqual(result[0].score, 1.0)

    def test_weighted_and_unweighted_transitions_are_distinct(self) -> None:
        dataset = InteractionDataset(
            [
                Interaction("u1", "a", 1, 1),
                Interaction("u1", "b", 9, 2),
                Interaction("u2", "a", 1, 1),
                Interaction("u2", "c", 1, 2),
                Interaction("target", "a", 1, 3),
            ]
        )
        weighted = SequentialMarkov(weighted=True, popularity_mix=0).fit(dataset)
        unweighted = SequentialMarkov(weighted=False, popularity_mix=0).fit(dataset)
        self.assertGreater(weighted.score_items("target")["b"], 0.8)
        self.assertEqual(unweighted.score_items("target")["b"], 0.5)

    def test_missing_timestamp_is_rejected_instead_of_inventing_order(self) -> None:
        with self.assertRaisesRegex(ValidationError, "timestamp"):
            SequentialMarkov().fit(InteractionDataset([Interaction("u", "a")]))

    def test_equal_timestamp_uses_declared_input_order(self) -> None:
        dataset = InteractionDataset(
            [
                Interaction("u", "a", 1, 1),
                Interaction("u", "b", 1, 1),
                Interaction("target", "a", 1, 2),
            ]
        )
        model = SequentialMarkov(popularity_mix=0).fit(dataset)
        self.assertEqual(model.recommend("target", 1)[0].item_id, "b")

    def test_unknown_user_uses_popularity_fallback(self) -> None:
        model = SequentialMarkov().fit(timestamped_dataset())
        self.assertEqual(model.recommend("unknown", 1)[0].item_id, "a")

    def test_state_and_file_round_trip_are_exact(self) -> None:
        model = SequentialMarkov(weighted=False, popularity_mix=0.2).fit(timestamped_dataset())
        restored = SequentialMarkov.from_state(model.to_state())
        self.assertEqual(restored.to_state(), model.to_state())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "markov.json"
            save_model(model, path)
            self.assertEqual(load_model(path).to_state(), model.to_state())

    def test_invalid_hyperparameters_are_rejected(self) -> None:
        for kwargs in (
            {"weighted": 1},
            {"popularity_mix": -0.1},
            {"popularity_mix": 1.1},
            {"popularity_mix": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                SequentialMarkov(**kwargs)  # type: ignore[arg-type]

    def test_corrupt_transition_and_last_item_states_are_rejected(self) -> None:
        state = SequentialMarkov().fit(timestamped_dataset()).to_state()
        corrupt_weight = copy.deepcopy(state)
        first_nonempty = next(row for row in corrupt_weight["model"]["transitions"] if row)
        first_nonempty[0]["weight"] = 0
        with self.assertRaises(SerializationError):
            SequentialMarkov.from_state(corrupt_weight)
        corrupt_last = copy.deepcopy(state)
        corrupt_last["model"]["last_items"][0]["item_id"] = "missing"
        with self.assertRaises(SerializationError):
            SequentialMarkov.from_state(corrupt_last)


class ExtendedModelIntegrationTests(unittest.TestCase):
    def test_experiment_factory_builds_both_models(self) -> None:
        self.assertIsInstance(build_model("user_knn", {}, experiment_seed=4), UserKNN)
        self.assertIsInstance(
            build_model("sequential_markov", {}, experiment_seed=4), SequentialMarkov
        )

    def test_strict_config_accepts_and_serializes_new_models(self) -> None:
        for name, params in (
            ("user_knn", {"neighbors": 2, "shrinkage": 0}),
            ("sequential_markov", {"weighted": False, "popularity_mix": 0.1}),
        ):
            config = config_from_dict(
                {"data": {"path": "events.json"}, "model": {"name": name, "params": params}}
            )
            self.assertEqual(config.model.name, name)
            self.assertEqual(config.model.params, params)
            json.dumps(config.to_dict(), allow_nan=False)

    def test_default_benchmark_covers_general_and_sequential_families(self) -> None:
        names = [spec.name for spec in default_benchmark_models()]
        self.assertEqual(
            names,
            [
                "popularity",
                "item_knn",
                "implicit_mf",
                "confidence_als",
                "user_knn",
                "sequential_markov",
            ],
        )


if __name__ == "__main__":
    unittest.main()
