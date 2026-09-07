from __future__ import annotations

import copy
import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from orchidrec.benchmark import run_benchmark
from orchidrec.benchmark_config import benchmark_config_from_dict
from orchidrec.cli import main
from orchidrec.config import config_from_dict
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import ConfigurationError, NotFittedError, SerializationError, ValidationError
from orchidrec.experiment import build_model, run_experiment
from orchidrec.models import ConfidenceALS, load_model, model_from_state, save_model
from orchidrec.models.confidence_als import MAX_CONFIDENCE, _solve_spd


def matrix_dataset() -> InteractionDataset:
    return InteractionDataset(
        [
            Interaction("u1", "a", 2, 1),
            Interaction("u1", "b", 1, 2),
            Interaction("u2", "a", 1, 1),
            Interaction("u2", "b", 3, 2),
            Interaction("u3", "c", 1, 1),
            Interaction("u3", "d", 2, 2),
            Interaction("target", "a", 1, 1),
        ]
    )


def benchmark_dataset() -> InteractionDataset:
    return InteractionDataset(
        [
            Interaction("u1", "a", timestamp=1),
            Interaction("u1", "b", timestamp=2),
            Interaction("u1", "c", timestamp=3),
            Interaction("u2", "a", timestamp=1),
            Interaction("u2", "c", timestamp=2),
            Interaction("u2", "d", timestamp=3),
            Interaction("u3", "b", timestamp=1),
            Interaction("u3", "c", timestamp=2),
            Interaction("u3", "d", timestamp=3),
            Interaction("u4", "a", timestamp=1),
            Interaction("u4", "d", timestamp=2),
            Interaction("u4", "b", timestamp=3),
        ]
    )


class CholeskySolverTests(unittest.TestCase):
    def test_hand_solved_system_and_residual(self) -> None:
        solution = _solve_spd([[4.0, 1.0], [1.0, 3.0]], [1.0, 2.0])
        self.assertAlmostEqual(solution[0], 1.0 / 11.0, places=14)
        self.assertAlmostEqual(solution[1], 7.0 / 11.0, places=14)
        self.assertAlmostEqual(4 * solution[0] + solution[1], 1.0, places=14)
        self.assertAlmostEqual(solution[0] + 3 * solution[1], 2.0, places=14)

    def test_bad_systems_fail_closed(self) -> None:
        cases = (
            ([], [], "dimensions"),
            ([[1.0, 0.0]], [1.0, 2.0], "dimensions"),
            ([[1.0, 2.0], [0.0, 1.0]], [1.0, 1.0], "symmetric"),
            ([[1.0, 0.0], [0.0, float("inf")]], [1.0, 1.0], "non-finite"),
            ([[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0], "singular"),
            ([[1.0, 0.0], [0.0, 1e-16]], [1.0, 1.0], "ill-conditioned"),
            ([[1e308, 0.0], [0.0, 1e308]], [1e308, 1e308], "overflow"),
        )
        for matrix, rhs, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValidationError, message):
                _solve_spd(matrix, rhs)


class ConfidenceALSMathematicsTests(unittest.TestCase):
    def test_reported_objective_matches_independent_dense_equation(self) -> None:
        dataset = matrix_dataset()
        model = ConfidenceALS(factors=2, epochs=3, alpha=5.0, regularization=0.2, seed=7).fit(
            dataset
        )
        state = model.to_state()
        users = [entry["user_id"] for entry in state["base"]["users"]]
        items = state["base"]["catalog"]
        user_factors = dict(zip(users, state["model"]["user_factors"], strict=True))
        item_factors = dict(zip(items, state["model"]["item_factors"], strict=True))
        aggregates: dict[tuple[object, object], float] = {}
        for event in dataset:
            key = (event.user_id, event.item_id)
            aggregates[key] = aggregates.get(key, 0.0) + event.value

        terms: list[float] = []
        for user_id in users:
            for item_id in items:
                prediction = math.fsum(
                    left * right
                    for left, right in zip(
                        user_factors[user_id], item_factors[item_id], strict=True
                    )
                )
                value = aggregates.get((user_id, item_id))
                preference = 1.0 if value is not None else 0.0
                confidence = 1.0 + model.alpha * value if value is not None else 1.0
                terms.append(confidence * (preference - prediction) ** 2)
        terms.append(
            model.regularization
            * math.fsum(
                value * value
                for vectors in (user_factors.values(), item_factors.values())
                for vector in vectors
                for value in vector
            )
        )
        self.assertAlmostEqual(math.fsum(terms), model.objective_history[-1], places=10)

    def test_final_item_factors_satisfy_independent_normal_equations(self) -> None:
        dataset = matrix_dataset()
        model = ConfidenceALS(factors=3, epochs=2, alpha=4.0, regularization=0.3, seed=11).fit(
            dataset
        )
        state = model.to_state()
        users = [entry["user_id"] for entry in state["base"]["users"]]
        items = state["base"]["catalog"]
        user_factors = dict(zip(users, state["model"]["user_factors"], strict=True))
        item_factors = dict(zip(items, state["model"]["item_factors"], strict=True))
        values: dict[tuple[object, object], float] = {}
        for event in dataset:
            key = (event.user_id, event.item_id)
            values[key] = values.get(key, 0.0) + event.value

        for item_id in items:
            matrix = [[0.0] * model.factors for _ in range(model.factors)]
            rhs = [0.0] * model.factors
            for row in range(model.factors):
                matrix[row][row] = model.regularization
            for user_id in users:
                confidence = 1.0 + model.alpha * values.get((user_id, item_id), 0.0)
                preference = 1.0 if (user_id, item_id) in values else 0.0
                vector = user_factors[user_id]
                for row in range(model.factors):
                    rhs[row] += confidence * preference * vector[row]
                    for column in range(model.factors):
                        matrix[row][column] += confidence * vector[row] * vector[column]
            solution = item_factors[item_id]
            residual = max(
                abs(math.fsum(a * x for a, x in zip(row, solution, strict=True)) - target)
                for row, target in zip(matrix, rhs, strict=True)
            )
            self.assertLess(residual, 1e-8)

    def test_each_exact_epoch_decreases_the_objective(self) -> None:
        model = ConfidenceALS(factors=3, epochs=5, seed=19).fit(matrix_dataset())
        self.assertEqual(len(model.objective_history), 6)
        self.assertTrue(all(math.isfinite(value) for value in model.objective_history))
        self.assertTrue(
            all(
                right <= left + 1e-7 * max(1.0, abs(left))
                for left, right in zip(
                    model.objective_history[:-1], model.objective_history[1:], strict=True
                )
            )
        )


class ConfidenceALSBehaviorTests(unittest.TestCase):
    def test_collaborative_signal_ranks_the_neighbor_item_first(self) -> None:
        result = (
            ConfidenceALS(factors=3, epochs=8, seed=2).fit(matrix_dataset()).recommend("target", 3)
        )
        self.assertEqual(result[0].item_id, "b")

    def test_unknown_user_uses_weighted_popularity_fallback(self) -> None:
        model = ConfidenceALS(factors=2, epochs=1).fit(matrix_dataset())
        self.assertEqual(model.recommend("cold-user", 1)[0].item_id, "a")

    def test_repeated_events_equal_their_aggregated_strength(self) -> None:
        repeated = InteractionDataset(
            [
                Interaction("u", "a", 1),
                Interaction("u", "a", 2),
                Interaction("other", "b", 1),
            ]
        )
        aggregated = InteractionDataset([Interaction("u", "a", 3), Interaction("other", "b", 1)])
        first = ConfidenceALS(factors=2, epochs=2, seed=3).fit(repeated)
        second = ConfidenceALS(factors=2, epochs=2, seed=3).fit(aggregated)
        self.assertEqual(first.to_state(), second.to_state())

    def test_row_order_does_not_change_the_model(self) -> None:
        original = matrix_dataset()
        reversed_data = InteractionDataset(reversed(original.interactions))
        first = ConfidenceALS(factors=2, epochs=2, seed=3).fit(original)
        second = ConfidenceALS(factors=2, epochs=2, seed=3).fit(reversed_data)
        self.assertEqual(first.to_state(), second.to_state())

    def test_seed_is_local_reproducible_and_effective(self) -> None:
        random.seed(90210)
        global_state = random.getstate()
        first = ConfidenceALS(factors=2, epochs=1, seed=8).fit(matrix_dataset())
        self.assertEqual(random.getstate(), global_state)
        second = ConfidenceALS(factors=2, epochs=1, seed=8).fit(matrix_dataset())
        third = ConfidenceALS(factors=2, epochs=1, seed=9).fit(matrix_dataset())
        self.assertEqual(first.to_state(), second.to_state())
        self.assertNotEqual(first.to_state()["model"], third.to_state()["model"])

    def test_objective_before_fit_is_rejected(self) -> None:
        with self.assertRaises(NotFittedError):
            _ = ConfidenceALS().objective_history

    def test_invalid_hyperparameters_are_rejected(self) -> None:
        cases = (
            {"factors": 0},
            {"factors": 65},
            {"factors": True},
            {"epochs": 0},
            {"epochs": 101},
            {"alpha": -1},
            {"alpha": True},
            {"alpha": float("inf")},
            {"alpha": 1_000_001},
            {"regularization": 0},
            {"regularization": True},
            {"regularization": float("nan")},
            {"regularization": 1_000_001},
            {"seed": True},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                ConfidenceALS(**kwargs)  # type: ignore[arg-type]

    def test_confidence_and_work_limits_fail_before_expensive_training(self) -> None:
        too_confident = InteractionDataset(
            [Interaction("u", "a", MAX_CONFIDENCE), Interaction("v", "b")]
        )
        with self.assertRaisesRegex(ValidationError, "confidence"):
            ConfidenceALS(alpha=2).fit(too_confident)
        too_much_work = InteractionDataset(
            [Interaction(f"u{index:02}", f"i{index:02}") for index in range(33)]
        )
        with self.assertRaisesRegex(ValidationError, "work limit"):
            ConfidenceALS(factors=64, epochs=100).fit(too_much_work)


class ConfidenceALSPersistenceTests(unittest.TestCase):
    def test_registry_and_file_round_trip_are_exact(self) -> None:
        model = ConfidenceALS(factors=2, epochs=2, seed=5).fit(matrix_dataset())
        self.assertEqual(model_from_state(model.to_state()).to_state(), model.to_state())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "als.json"
            save_model(model, path)
            restored = load_model(path)
        self.assertIsInstance(restored, ConfidenceALS)
        self.assertEqual(restored.to_state(), model.to_state())

    def test_tampered_state_is_rejected(self) -> None:
        original = ConfidenceALS(factors=2, epochs=2, seed=5).fit(matrix_dataset()).to_state()

        def reject(name: str, mutation: object) -> None:
            state = copy.deepcopy(original)
            assert callable(mutation)
            mutation(state)
            with self.subTest(name=name), self.assertRaises(SerializationError):
                ConfidenceALS.from_state(state)

        reject("unknown parameter", lambda state: state["parameters"].update({"extra": 1}))
        reject("invalid factor bound", lambda state: state["parameters"].update({"factors": 65}))
        reject("unknown model field", lambda state: state["model"].update({"extra": []}))
        reject("missing user row", lambda state: state["model"]["user_factors"].pop())
        reject("wrong item width", lambda state: state["model"]["item_factors"][0].pop())
        reject(
            "nonfinite factor",
            lambda state: state["model"]["item_factors"][0].__setitem__(0, float("nan")),
        )
        reject(
            "overflowing integer factor",
            lambda state: state["model"]["item_factors"][0].__setitem__(0, 10**400),
        )
        reject(
            "boolean factor",
            lambda state: state["model"]["user_factors"][0].__setitem__(0, True),
        )
        reject("short objectives", lambda state: state["model"]["objective_history"].pop())
        reject(
            "negative objective",
            lambda state: state["model"]["objective_history"].__setitem__(0, -1),
        )
        reject(
            "boolean objective",
            lambda state: state["model"]["objective_history"].__setitem__(0, True),
        )
        reject(
            "increasing objective",
            lambda state: state["model"]["objective_history"].__setitem__(1, 1e9),
        )


class ConfidenceALSIntegrationTests(unittest.TestCase):
    def test_config_factory_experiment_and_cli_use_the_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_dataset().save_json(root / "events.json")
            payload = {
                "seed": 73,
                "data": {"path": "events.json"},
                "split": {"method": "leave_one_out"},
                "model": {
                    "name": "confidence_als",
                    "params": {"factors": 2, "epochs": 2, "alpha": 8.0},
                },
                "evaluation": {"k": 2},
                "output": {"model_path": "model.json", "report_path": "report.json"},
            }
            config = config_from_dict(payload, base_dir=root)
            model = build_model(config.model.name, config.model.params, experiment_seed=config.seed)
            self.assertIsInstance(model, ConfidenceALS)
            self.assertEqual(model.seed, 73)
            result = run_experiment(config)
            self.assertEqual(result.model_type, "confidence_als")
            self.assertEqual(result.model_parameters["seed"], 73)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["run", str(config_path)]), 0)
            self.assertEqual(json.loads(output.getvalue())["model"]["type"], "confidence_als")
            self.assertIsInstance(load_model(root / "model.json"), ConfidenceALS)
            inspection = io.StringIO()
            with redirect_stdout(inspection):
                self.assertEqual(main(["inspect", str(root / "model.json")]), 0)
            self.assertEqual(json.loads(inspection.getvalue())["model_type"], "confidence_als")
            recommendations = io.StringIO()
            with redirect_stdout(recommendations):
                self.assertEqual(
                    main(["recommend", str(root / "model.json"), "cold-user", "--k", "2"]),
                    0,
                )
            self.assertEqual(len(json.loads(recommendations.getvalue())), 2)

    def test_benchmark_grid_repeats_seeded_als_and_refits_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "events.json"
            benchmark_dataset().save_json(data_path)
            config = benchmark_config_from_dict(
                {
                    "schema_version": 1,
                    "seed": 29,
                    "data": {"path": str(data_path), "format": "orchidrec-json"},
                    "split": {"method": "leave_one_out"},
                    "evaluation": {"k": 2, "bootstrap_samples": 10},
                    "tuning": {
                        "selection_metric": "ndcg",
                        "implicit_mf_seeds": [3, 4],
                    },
                    "models": [
                        {
                            "label": "als",
                            "name": "confidence_als",
                            "params": {"epochs": 1, "alpha": 5.0},
                            "grid": {"factors": [1, 2]},
                        },
                        {"label": "pop", "name": "popularity"},
                    ],
                }
            )
            result = run_benchmark(config)
        self.assertIsNotNone(result.tuning)
        assert result.tuning is not None
        als = next(model for model in result.tuning.models if model.model_type == "confidence_als")
        self.assertEqual(len(als.candidates), 2)
        self.assertEqual(len(als.trials), 4)
        self.assertEqual([trial.seed for trial in als.trials], [3, 4, 3, 4])
        self.assertEqual(als.final_seed, 29)
        final = next(model for model in result.models if model.model_type == "confidence_als")
        self.assertEqual(final.parameters["seed"], 29)
        seed_policy = result.to_dict()["tuning"]["seed_policy"]
        self.assertEqual(seed_policy["confidence_als_validation_seeds"], [3, 4])

    def test_tuned_model_seed_has_one_owner(self) -> None:
        payload = {
            "schema_version": 1,
            "data": {"path": "events.json", "format": "orchidrec-json"},
            "tuning": {"implicit_mf_seeds": [1, 2]},
            "models": [
                {"label": "als", "name": "confidence_als", "params": {"seed": 3}},
                {"label": "pop", "name": "popularity", "grid": {"weighted": [True]}},
            ],
        }
        with self.assertRaisesRegex(ConfigurationError, "seed"):
            benchmark_config_from_dict(payload)

    def test_strict_config_rejects_parameters_from_another_model(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "unknown model.params"):
            config_from_dict(
                {
                    "data": {"path": "events.json"},
                    "model": {
                        "name": "confidence_als",
                        "params": {"learning_rate": 0.1},
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
