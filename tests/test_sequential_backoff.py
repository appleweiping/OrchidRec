"""Independent hand-oracle and integration checks for second-order backoff."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from orchidrec.benchmark import run_benchmark
from orchidrec.benchmark_config import benchmark_config_from_dict
from orchidrec.config import config_from_dict
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.experiment import build_model, run_experiment
from orchidrec.models import SequentialBackoff, load_model, save_model

ROOT = Path(__file__).resolve().parents[1]


def hand_dataset(*, c_weight: float = 1.0) -> InteractionDataset:
    return InteractionDataset(
        [
            Interaction("u1", "a", 1, 1),
            Interaction("u1", "b", 1, 2),
            Interaction("u1", "c", c_weight, 3),
            Interaction("u2", "x", 1, 1),
            Interaction("u2", "b", 1, 2),
            Interaction("u2", "d", 1, 3),
            Interaction("target", "a", 1, 1),
            Interaction("target", "b", 1, 2),
        ]
    )


class SequentialBackoffTests(unittest.TestCase):
    def test_hand_count_and_probability_oracle(self) -> None:
        model = SequentialBackoff(weighted=False, backoff_strength=1.0, popularity_mix=0.2).fit(
            hand_dataset()
        )
        scores = model.score_items("target")
        # B -> C/D has one transition each. (A,B) -> C has one, so
        # lambda=1/(1+1)=1/2; C has sequence probability 3/4 and D 1/4.
        # Both C and D have popularity 1/3 (B appears three times).
        self.assertAlmostEqual(scores["c"], 2 / 3)
        self.assertAlmostEqual(scores["d"], 4 / 15)
        self.assertEqual([row.item_id for row in model.recommend("target", 2)], ["c", "d"])
        self.assertEqual(model.score_items("unknown")["c"], 1 / 3)

    def test_weighted_targets_change_both_order_probabilities(self) -> None:
        weighted = SequentialBackoff(weighted=True, backoff_strength=1, popularity_mix=0).fit(
            hand_dataset(c_weight=2)
        )
        unweighted = SequentialBackoff(weighted=False, backoff_strength=1, popularity_mix=0).fit(
            hand_dataset(c_weight=2)
        )
        # Weighted B -> C/D = 2:1; (A,B) -> C support 2 and lambda 2/3.
        self.assertAlmostEqual(weighted.score_items("target")["c"], 8 / 9)
        self.assertAlmostEqual(unweighted.score_items("target")["c"], 3 / 4)

    def test_unseen_second_order_context_backs_off_to_first_order(self) -> None:
        events = [*hand_dataset(), Interaction("other", "c", 1, 4), Interaction("other", "b", 1, 5)]
        model = SequentialBackoff(popularity_mix=0).fit(InteractionDataset(events))
        self.assertAlmostEqual(model.score_items("other")["c"], 0.5)
        self.assertAlmostEqual(model.score_items("other")["d"], 0.5)

    def test_chronology_tie_and_missing_timestamp_boundaries(self) -> None:
        with self.assertRaisesRegex(ValidationError, "timestamp"):
            SequentialBackoff().fit(InteractionDataset([Interaction("u", "a")]))
        tied = InteractionDataset(
            [
                Interaction("u", "a", 1, 1),
                Interaction("u", "b", 1, 1),
                Interaction("u", "c", 1, 1),
                Interaction("target", "a", 1, 2),
                Interaction("target", "b", 1, 2),
            ]
        )
        model = SequentialBackoff(popularity_mix=0, backoff_strength=0).fit(tied)
        self.assertEqual(model.recommend("target", 1)[0].item_id, "c")

    def test_state_and_file_round_trip_and_tamper_rejection(self) -> None:
        model = SequentialBackoff().fit(hand_dataset())
        restored = SequentialBackoff.from_state(model.to_state())
        self.assertEqual(restored.to_state(), model.to_state())
        self.assertEqual(restored.score_items("target"), model.score_items("target"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_model(model, path)
            self.assertEqual(load_model(path).to_state(), model.to_state())
        corrupt = copy.deepcopy(model.to_state())
        corrupt["model"]["second_order"][0]["targets"][0]["weight"] = -1
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(corrupt)
        corrupt = copy.deepcopy(model.to_state())
        corrupt["model"]["last_two"][0]["items"] = ["missing"]
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(corrupt)
        corrupt = copy.deepcopy(model.to_state())
        corrupt["model"]["second_order"][0]["targets"][0]["weight"] = 2
        with self.assertRaisesRegex(SerializationError, "second-order count"):
            SequentialBackoff.from_state(corrupt)
        corrupt = copy.deepcopy(model.to_state())
        # Each context separately fits B -> C, but together they exceed it.
        corrupt["model"]["second_order"][1]["targets"].insert(0, {"item_id": "c", "weight": 1})
        with self.assertRaisesRegex(SerializationError, "second-order count"):
            SequentialBackoff.from_state(corrupt)

    def test_parameter_and_work_bounds(self) -> None:
        for kwargs in (
            {"weighted": 1},
            {"backoff_strength": -1},
            {"backoff_strength": float("nan")},
            {"popularity_mix": 2},
            {"max_interactions": True},
            {"max_interactions": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                SequentialBackoff(**kwargs)
        with self.assertRaisesRegex(ValidationError, "interaction limit"):
            SequentialBackoff(max_interactions=2).fit(hand_dataset())
        with self.assertRaises(ValidationError):
            SequentialBackoff().fit(InteractionDataset([Interaction("u", "a", 1e300, 1)]))
        with self.assertRaisesRegex(ValidationError, "total event weight"):
            SequentialBackoff().fit(
                InteractionDataset([Interaction("u", "a", 6e14, 1), Interaction("u", "b", 6e14, 2)])
            )

    def test_malformed_state_shapes_and_canonical_order(self) -> None:
        original = SequentialBackoff().fit(hand_dataset()).to_state()

        bad = copy.deepcopy(original)
        del bad["parameters"]["weighted"]
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["parameters"]["backoff_strength"] = "not a number"
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["base"]["popularity"][0] = 1e300
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["base"] = {"catalog": [], "popularity": [], "users": []}
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        del bad["model"]["last_two"]
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["model"]["first_order"] = None
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["model"]["first_order"][0] = {"item_id": "b", "weight": 1}
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["model"]["first_order"][0] = [{"item_id": "b"}]
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["model"]["first_order"][0] = [{"item_id": "absent", "weight": 1}]
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["model"]["first_order"][0] = [
            {"item_id": "b", "weight": 6e14},
            {"item_id": "c", "weight": 6e14},
        ]
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["model"]["second_order"][0]["previous_item_id"] = "absent"
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["model"]["second_order"].reverse()
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["model"]["last_two"][0] = {"user_id": "u1"}
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["model"]["last_two"][0]["items"] = []
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)
        bad = copy.deepcopy(original)
        bad["model"]["last_two"].reverse()
        with self.assertRaises(SerializationError):
            SequentialBackoff.from_state(bad)

    def test_experiment_and_benchmark_are_real_consumers(self) -> None:
        self.assertIsInstance(
            build_model("sequential_backoff", {}, experiment_seed=5), SequentialBackoff
        )
        path = ROOT / "examples" / "interactions.json"
        config = config_from_dict(
            {
                "data": {"path": str(path)},
                "split": {"method": "leave_one_out"},
                "model": {"name": "sequential_backoff", "params": {"backoff_strength": 2}},
                "evaluation": {"k": 2},
            }
        )
        experiment = run_experiment(config)
        self.assertEqual(experiment.model_type, "sequential_backoff")
        self.assertGreater(experiment.evaluated_test_size, 0)
        benchmark_config = benchmark_config_from_dict(
            {
                "schema_version": 1,
                "seed": 5,
                "data": {"path": str(path), "format": "orchidrec-json", "minimum_rating": None},
                "split": {"method": "leave_one_out", "test_ratio": 0.2},
                "evaluation": {"k": 2, "bootstrap_samples": 20, "confidence": 0.9},
                "models": [
                    {
                        "label": "backoff",
                        "name": "sequential_backoff",
                        "params": {"backoff_strength": 2},
                    },
                    {"label": "markov", "name": "sequential_markov", "params": {}},
                ],
            }
        )
        report = run_benchmark(benchmark_config)
        self.assertEqual({model.label for model in report.models}, {"backoff", "markov"})
        json.dumps(report.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
