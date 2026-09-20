from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchidrec.benchmark import run_benchmark
from orchidrec.benchmark_config import benchmark_config_from_dict
from orchidrec.config import config_from_dict, load_config
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import ConfigurationError, NotFittedError, SerializationError, ValidationError
from orchidrec.experiment import build_model
from orchidrec.models import SLIMElastic, load_model, save_model


def two_items() -> InteractionDataset:
    return InteractionDataset(
        (Interaction("u1", "a"), Interaction("u1", "b"), Interaction("u2", "a"))
    )


def three_items() -> InteractionDataset:
    return InteractionDataset(
        (
            Interaction("u1", "a"),
            Interaction("u1", "b"),
            Interaction("u1", "c"),
            Interaction("u2", "a"),
            Interaction("u2", "c"),
            Interaction("u3", "b"),
        )
    )


class SLIMElasticTests(unittest.TestCase):
    def test_two_item_solution_matches_direct_one_variable_oracle(self) -> None:
        # For target b, X_a.X_b=1 and ||X_a||²=2, so w_ab=(1-.2)/(2+.3)=8/23.
        # The reverse target has ||X_b||²=1, so w_ba=(1-.2)/(1+.3)=8/13.
        model = SLIMElastic(l1=0.2, l2=0.3).fit(two_items())
        weights = model.to_state()["model"]["weights"]
        self.assertEqual(weights[0][0], 0.0)
        self.assertEqual(weights[1][1], 0.0)
        self.assertAlmostEqual(weights[0][1], 8 / 23, places=12)
        self.assertAlmostEqual(weights[1][0], 8 / 13, places=12)
        self.assertAlmostEqual(model.score_items("u2")["b"], 8 / 23, places=12)
        self.assertEqual(model.recommend("u2", 1)[0].item_id, "b")
        self.assertLessEqual(model.largest_kkt, model.tolerance)

    def test_coupled_target_matches_hand_solved_two_by_two_active_set(self) -> None:
        # Target c: cross products (2, 1), predictor Gram [[2, 1], [1, 2]].
        # KKT active system [[2.3,1],[1,2.3]] w=[1.8,.8].
        # Its independent exact rational solution is (334/429,4/429).
        model = SLIMElastic(l1=0.2, l2=0.3).fit(three_items())
        weights = model.to_state()["model"]["weights"]
        self.assertAlmostEqual(weights[0][2], 334 / 429, places=6)
        self.assertAlmostEqual(weights[1][2], 4 / 429, places=6)
        self.assertEqual(weights[2][2], 0.0)

    def test_zero_cooccurrence_and_one_item_catalog_have_zero_weights(self) -> None:
        disjoint = InteractionDataset((Interaction("u1", "a"), Interaction("u2", "b")))
        model = SLIMElastic(l1=0.0, l2=0.1).fit(disjoint)
        self.assertEqual(model.to_state()["model"]["weights"], [[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(model.score_items("u1")["b"], 0.0)
        self.assertEqual(model.largest_kkt, 0.0)
        single = SLIMElastic().fit(InteractionDataset((Interaction("u", "only"),)))
        self.assertEqual(single.to_state()["model"]["weights"], [[0.0]])
        self.assertEqual(single.score_items("u")["only"], 0.0)

    def test_binary_determinism_and_unknown_user_popularity(self) -> None:
        data = two_items()
        duplicate = Interaction("u2", "a", value=3.0)
        first = SLIMElastic(l1=0.2, l2=0.3).fit(data)
        reversed_rows = SLIMElastic(l1=0.2, l2=0.3).fit(
            InteractionDataset(reversed(data.interactions))
        )
        repeated = SLIMElastic(l1=0.2, l2=0.3).fit(
            InteractionDataset((*data.interactions, duplicate))
        )
        self.assertEqual(first.to_state(), reversed_rows.to_state())
        self.assertEqual(
            first.to_state()["model"]["weights"], repeated.to_state()["model"]["weights"]
        )
        self.assertEqual(first.score_items("new"), {"a": 1.0, "b": 0.5})
        self.assertEqual(repeated.score_items("new"), {"a": 1.0, "b": 0.2})

    def test_nonconvergence_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValidationError, "did not satisfy KKT"):
            SLIMElastic(l1=0.2, l2=0.3, max_sweeps=1, tolerance=1e-8).fit(three_items())

    def test_resource_and_parameter_validation(self) -> None:
        invalid = (
            {"l1": -1.0},
            {"l1": True},
            {"l1": math.inf},
            {"l2": 0.0},
            {"l2": math.nan},
            {"l1": 10**1000},
            {"tolerance": 0.0},
            {"tolerance": 0.2},
            {"max_items": True},
            {"max_items": 257},
            {"max_sweeps": 1001},
            {"max_interactions": 2_000_001},
            {"max_work_units": 1_000_000_001},
        )
        for params in invalid:
            with self.subTest(params=params), self.assertRaises(ValidationError):
                SLIMElastic(**params)
        for params in (
            {"max_items": 1},
            {"max_interactions": 2},
            {"max_work_units": 1604},
        ):
            with self.subTest(params=params), self.assertRaises(ValidationError):
                SLIMElastic(l1=0.2, l2=0.3, **params).fit(two_items())
        self.assertEqual(
            SLIMElastic(l1=0.2, l2=0.3, max_work_units=1605).fit(two_items()).work_units,
            1605,
        )

    def test_unfitted_and_persistence_round_trip(self) -> None:
        model = SLIMElastic(l1=0.2, l2=0.3)
        with self.assertRaises(NotFittedError):
            _ = model.largest_kkt
        with self.assertRaises(NotFittedError):
            model.to_state()
        model.fit(two_items())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slim.json"
            save_model(model, path)
            restored = load_model(path)
        self.assertIsInstance(restored, SLIMElastic)
        self.assertEqual(restored.to_state(), model.to_state())
        self.assertEqual(restored.score_items("u1"), model.score_items("u1"))

    def test_corrupt_state_is_rejected_including_one_ulp_weight_change(self) -> None:
        original = SLIMElastic(l1=0.2, l2=0.3).fit(two_items()).to_state()
        changes = []
        for key, value in (
            ("work_units", 1606),
            ("training_interactions", 1),
            ("sweeps", [0, 1]),
            ("largest_kkt", -1.0),
            ("weights", [[0.0]]),
        ):
            state = copy.deepcopy(original)
            state["model"][key] = value
            changes.append(state)
        for row, col, value in ((0, 0, 0.1), (0, 1, -1.0), (0, 1, math.nan), (0, 1, True)):
            state = copy.deepcopy(original)
            state["model"]["weights"][row][col] = value
            changes.append(state)
        ulp = copy.deepcopy(original)
        current = ulp["model"]["weights"][0][1]
        ulp["model"]["weights"][0][1] = math.nextafter(current, math.inf)
        changes.append(ulp)
        bad_base = copy.deepcopy(original)
        bad_base["base"]["users"][0]["seen"].append("unknown")
        changes.append(bad_base)
        for state in changes:
            with self.subTest(state=state), self.assertRaises(SerializationError):
                SLIMElastic.from_state(state)

        variants: list[tuple[str, dict[str, object]]] = []
        missing_parameter = copy.deepcopy(original)
        del missing_parameter["parameters"]["l1"]
        variants.append(("parameters", missing_parameter))
        invalid_parameter = copy.deepcopy(original)
        invalid_parameter["parameters"]["l2"] = 0.0
        variants.append(("parameters", invalid_parameter))
        extra_model_field = copy.deepcopy(original)
        extra_model_field["model"]["unknown"] = 1
        variants.append(("model", extra_model_field))
        short_weight_row = copy.deepcopy(original)
        short_weight_row["model"]["weights"][0].pop()
        variants.append(("weight columns", short_weight_row))
        boolean_kkt = copy.deepcopy(original)
        boolean_kkt["model"]["largest_kkt"] = True
        variants.append(("KKT", boolean_kkt))
        duplicate_catalog = copy.deepcopy(original)
        duplicate_catalog["base"]["catalog"][1] = "a"
        variants.append(("duplicate", duplicate_catalog))
        for expected, state in variants:
            with (
                self.subTest(expected=expected),
                self.assertRaisesRegex(SerializationError, expected),
            ):
                SLIMElastic.from_state(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(ulp), encoding="utf-8")
            with self.assertRaises(SerializationError):
                load_model(path)

    def test_fit_rejects_unpersistable_identifiers_and_file_size(self) -> None:
        long_id = InteractionDataset((Interaction("u" * 16_385, "a"),))
        with self.assertRaisesRegex(ValidationError, "UTF-8 byte limit"):
            SLIMElastic().fit(long_id)
        for user_id, message in (
            ("\ud800", "Unicode scalar"),
            (10**4096, "JSON digit limit"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValidationError, message):
                SLIMElastic().fit(InteractionDataset((Interaction(user_id, "a"),)))
        with (
            mock.patch("orchidrec.models.slim_elastic.MAX_SLIM_TOTAL_IDENTIFIER_BYTES", 3),
            self.assertRaisesRegex(ValidationError, "aggregate identifier byte limit"),
        ):
            SLIMElastic().fit(two_items())
        with (
            mock.patch("orchidrec.models.slim_elastic.MAX_MODEL_FILE_BYTES", 100),
            self.assertRaisesRegex(ValidationError, "model-file byte limit"),
        ):
            SLIMElastic().fit(two_items())

    def test_config_factory_and_benchmark_grid(self) -> None:
        self.assertIsInstance(
            build_model("slim_elastic", {"l1": 0.2}, experiment_seed=1), SLIMElastic
        )
        config = config_from_dict(
            {
                "data": {"path": "input.json"},
                "model": {"name": "slim_elastic", "params": {"l1": 0.2, "l2": 0.3}},
            }
        )
        self.assertEqual(config.model.name, "slim_elastic")
        with self.assertRaises(ConfigurationError):
            config_from_dict(
                {
                    "data": {"path": "input.json"},
                    "model": {"name": "slim_elastic", "params": {"l2": 0}},
                }
            )
        benchmark = benchmark_config_from_dict(
            {
                "schema_version": 1,
                "data": {"path": "input.json", "format": "orchidrec-json"},
                "models": [
                    {"label": "pop", "name": "popularity"},
                    {"label": "slim", "name": "slim_elastic", "params": {"l2": 0.3}},
                ],
            }
        )
        self.assertEqual(benchmark.models[1].name, "slim_elastic")

    def test_synthetic_example_and_real_benchmark_path(self) -> None:
        examples = Path(__file__).resolve().parents[1] / "examples"
        experiment = load_config(examples / "slim_elastic_config.json")
        self.assertEqual(experiment.model.name, "slim_elastic")
        benchmark = benchmark_config_from_dict(
            {
                "schema_version": 1,
                "seed": 2026,
                "data": {"path": str(examples / "interactions.json"), "format": "orchidrec-json"},
                "evaluation": {"k": 3, "bootstrap_samples": 20, "confidence": 0.9},
                "models": [
                    {"label": "popularity", "name": "popularity"},
                    {
                        "label": "slim",
                        "name": "slim_elastic",
                        "params": {"l1": 0.2, "l2": 0.3},
                    },
                ],
            }
        )
        first = run_benchmark(benchmark)
        second = run_benchmark(benchmark)
        self.assertEqual([row.label for row in first.models], ["popularity", "slim"])
        self.assertEqual(first.split_fingerprint, second.split_fingerprint)
        self.assertEqual(
            [row.metrics for row in first.models], [row.metrics for row in second.models]
        )
        self.assertEqual(first.comparisons, second.comparisons)

    def test_tuning_grid_accepts_two_numeric_penalty_candidates(self) -> None:
        examples = Path(__file__).resolve().parents[1] / "examples"
        benchmark = benchmark_config_from_dict(
            {
                "schema_version": 1,
                "data": {"path": str(examples / "interactions.json"), "format": "orchidrec-json"},
                "tuning": {"selection_metric": "ndcg"},
                "models": [
                    {"label": "pop", "name": "popularity"},
                    {
                        "label": "slim",
                        "name": "slim_elastic",
                        "params": {"l2": 0.3},
                        "grid": {"l1": [0.1, 0.2]},
                    },
                ],
            }
        )
        self.assertEqual(benchmark.models[1].grid, {"l1": (0.1, 0.2)})


if __name__ == "__main__":
    unittest.main()
