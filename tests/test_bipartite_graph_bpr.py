"""Independent numerical and end-to-end tests for graph BPR."""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import orchidrec.models.bipartite_graph_bpr as graph_module
from orchidrec.benchmark import run_benchmark
from orchidrec.benchmark_config import benchmark_config_from_dict
from orchidrec.config import config_from_dict
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import ConfigurationError, SerializationError, ValidationError
from orchidrec.experiment import run_experiment
from orchidrec.models import BipartiteGraphBPR, load_model
from orchidrec.models.bipartite_graph_bpr import _batch_loss_gradient, _edges, _propagate

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "bipartite_graph_interactions.json"


def _tiny() -> InteractionDataset:
    return InteractionDataset([Interaction("u1", "a"), Interaction("u2", "b")])


class BipartiteGraphBPRTests(unittest.TestCase):
    def test_nonuniform_degree_normalization_hand_oracle(self) -> None:
        edges = _edges(
            ("u1", "u2"),
            ("a", "b"),
            {"u1": frozenset({"a", "b"}), "u2": frozenset({"b"})},
        )
        self.assertEqual(
            edges,
            ((0, 2, 1 / math.sqrt(2)), (0, 3, 0.5), (1, 3, 1 / math.sqrt(2))),
        )
        states, _ = _propagate([[1.0], [2.0], [3.0], [4.0]], edges, 1)
        for row, expected in zip(
            states[1],
            [3 / math.sqrt(2) + 2, 4 / math.sqrt(2), 1 / math.sqrt(2), 0.5 + 2 / math.sqrt(2)],
            strict=True,
        ):
            self.assertAlmostEqual(row[0], expected)

    def test_two_by_two_graph_and_bpr_hand_oracle(self) -> None:
        # Two disjoint degree-one edges make normalized P swap each pair.
        edges = ((0, 2, 1.0), (1, 3, 1.0))
        ego = [[1.0], [2.0], [3.0], [4.0]]
        states, average = _propagate(ego, edges, 1)
        self.assertEqual(states[1], [[3.0], [4.0], [1.0], [2.0]])
        self.assertEqual(average, [[2.0], [3.0], [2.0], [3.0]])
        # m=2*(2-3)=-2, dm/dE0=(1/2,-1,1/2,-1).
        loss, gradient = _batch_loss_gradient(ego, edges, ((0, 2, 3),), layers=1, regularization=0)
        coefficient = 1 / (1 + math.exp(-2))
        self.assertAlmostEqual(loss, math.log1p(math.exp(2)))
        for actual, expected in zip(
            [row[0] for row in gradient],
            [-coefficient / 2, coefficient, -coefficient / 2, coefficient],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)

    def test_reverse_mode_matches_independent_finite_difference(self) -> None:
        edges = ((0, 2, 1.0), (1, 3, 1.0))
        ego = [[0.2, -0.1], [0.3, 0.4], [-0.5, 0.25], [0.1, 0.7]]
        pairs = ((0, 2, 3), (1, 3, 2))
        _, gradient = _batch_loss_gradient(ego, edges, pairs, layers=2, regularization=0.03)
        epsilon = 1e-6
        for node in range(4):
            for axis in range(2):
                plus, minus = copy.deepcopy(ego), copy.deepcopy(ego)
                plus[node][axis] += epsilon
                minus[node][axis] -= epsilon
                high, _ = _batch_loss_gradient(plus, edges, pairs, layers=2, regularization=0.03)
                low, _ = _batch_loss_gradient(minus, edges, pairs, layers=2, regularization=0.03)
                self.assertAlmostEqual(
                    gradient[node][axis], (high - low) / (2 * epsilon), delta=1e-8
                )

    def test_determinism_negative_exclusion_and_state_roundtrip(self) -> None:
        dataset = InteractionDataset(
            [Interaction("u1", "a"), Interaction("u1", "b"), Interaction("u2", "c")]
        )
        observed: list[tuple[tuple[int, int, int], ...]] = []
        original = graph_module._batch_loss_gradient

        def capture(ego, edges, pairs, *, layers, regularization):
            observed.append(pairs)
            return original(ego, edges, pairs, layers=layers, regularization=regularization)

        with patch.object(graph_module, "_batch_loss_gradient", capture):
            first = BipartiteGraphBPR(factors=2, layers=1, epochs=3, seed=7).fit(dataset)
            second = BipartiteGraphBPR(factors=2, layers=1, epochs=3, seed=7).fit(dataset)
        self.assertEqual(first.to_state(), second.to_state())
        self.assertGreater(first.work_upper, 0)
        self.assertEqual(len(first.loss_history), 3)
        users = tuple(sorted(first._seen))
        for epoch_pairs in observed:
            for user_index, positive, negative in epoch_pairs:
                self.assertNotIn(
                    first.catalog[negative - len(users)], first.seen_items(users[user_index])
                )
                self.assertIn(
                    first.catalog[positive - len(users)], first.seen_items(users[user_index])
                )
        restored = BipartiteGraphBPR.from_state(first.to_state())
        self.assertEqual(restored.to_state(), first.to_state())
        self.assertEqual(restored.score_items("u1"), first.score_items("u1"))
        self.assertEqual(
            first.score_items("unknown"),
            {item: first._popularity_fallback(item) for item in first.catalog},
        )

    def test_public_bounds_and_untrusted_state_fail_closed(self) -> None:
        for params in (
            {"factors": 0},
            {"factors": 17},
            {"layers": 0},
            {"layers": 3},
            {"epochs": 31},
            {"learning_rate": True},
            {"regularization": float("inf")},
            {"seed": 1 << 63},
            {"max_work_units": 0},
        ):
            with self.subTest(params=params), self.assertRaises(ValidationError):
                BipartiteGraphBPR(**params)
        with self.assertRaisesRegex(ValidationError, "unseen"):
            BipartiteGraphBPR().fit(InteractionDataset([Interaction("u", "a")]))
        with self.assertRaisesRegex(ValidationError, "coordinate work"):
            BipartiteGraphBPR(max_work_units=1).fit(_tiny())
        with self.assertRaisesRegex(ValidationError, "item limit"):
            BipartiteGraphBPR().fit(
                InteractionDataset(Interaction("u", f"item-{index}") for index in range(257))
            )
        with self.assertRaisesRegex(ValidationError, "identifier"):
            BipartiteGraphBPR().fit(
                InteractionDataset([Interaction(1 << 513, "a"), Interaction("u", "b")])
            )
        original = BipartiteGraphBPR(factors=2, epochs=2).fit(_tiny()).to_state()
        for mutation in (
            lambda state: state["model"]["edges"][0].__setitem__(1, 3),
            lambda state: state["model"]["ego_embeddings"][0].__setitem__(0, float("inf")),
            lambda state: state["model"].__setitem__("work_upper", 1),
            lambda state: state["model"].__setitem__("loss_history", []),
            lambda state: state["base"].__setitem__("catalog", ["x"] * 257),
        ):
            state = copy.deepcopy(original)
            mutation(state)
            with self.assertRaises(SerializationError):
                BipartiteGraphBPR.from_state(state)

    def test_source_and_checkpoint_adversarial_resource_guards(self) -> None:
        with self.assertRaisesRegex(ValidationError, "2048 bytes"):
            BipartiteGraphBPR().fit(
                InteractionDataset([Interaction("u" * 2049, "a"), Interaction("v", "b")])
            )
        with self.assertRaisesRegex(ValidationError, "UTF-8"):
            BipartiteGraphBPR().fit(
                InteractionDataset([Interaction("\ud800", "a"), Interaction("v", "b")])
            )
        with self.assertRaisesRegex(ValidationError, "total interaction value"):
            BipartiteGraphBPR().fit(
                InteractionDataset(
                    [Interaction("u", "a", 600_000_000), Interaction("v", "b", 600_000_000)]
                )
            )
        with self.assertRaisesRegex(ValidationError, "edge limit"):
            BipartiteGraphBPR().fit(
                InteractionDataset(
                    Interaction(f"u{user}", f"i{item}") for user in range(65) for item in range(64)
                )
            )
        original = BipartiteGraphBPR(factors=2, epochs=2).fit(_tiny()).to_state()
        for mutation in (
            lambda s: s["parameters"].pop("epochs"),
            lambda s: s["parameters"].__setitem__("factors", 17),
            lambda s: s["base"]["popularity"].__setitem__(0, 1_000_000_001),
            lambda s: s["model"].pop("edges"),
            lambda s: s["model"].__setitem__("ego_embeddings", []),
            lambda s: s["model"]["ego_embeddings"][0].pop(),
            lambda s: s["model"]["ego_embeddings"][0].__setitem__(0, True),
            lambda s: s["model"]["loss_history"].__setitem__(0, -1),
            lambda s: s["base"]["users"][0].__setitem__("seen", ["a"] * 257),
        ):
            state = copy.deepcopy(original)
            mutation(state)
            with self.assertRaises(SerializationError):
                BipartiteGraphBPR.from_state(state)

    def test_experiment_benchmark_and_saved_model_use_train_split_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "graph-model.json"
            config = config_from_dict(
                {
                    "data": {"path": str(EXAMPLE)},
                    "split": {"method": "leave_one_out"},
                    "model": {"name": "bipartite_graph_bpr", "params": {"factors": 2, "epochs": 2}},
                    "evaluation": {"k": 2},
                    "output": {"model_path": str(model_path)},
                }
            )
            result = run_experiment(config)
            self.assertEqual(result.model_type, "bipartite_graph_bpr")
            saved = load_model(model_path)
            self.assertIsInstance(saved, BipartiteGraphBPR)
            self.assertEqual(saved.to_state()["model"]["edges"], [[0, 3], [1, 4], [2, 5]])
            self.assertEqual(saved.seen_items("u1"), frozenset({"a"}))
            self.assertEqual(saved.seen_items("u2"), frozenset({"b"}))
            self.assertEqual(saved.seen_items("u3"), frozenset({"c"}))
            benchmark = benchmark_config_from_dict(
                {
                    "schema_version": 1,
                    "data": {"path": str(EXAMPLE), "format": "orchidrec-json"},
                    "split": {"method": "leave_one_out"},
                    "evaluation": {"k": 2, "bootstrap_samples": 20},
                    "models": [
                        {"label": "pop", "name": "popularity"},
                        {"label": "graph", "name": "bipartite_graph_bpr", "params": {"epochs": 2}},
                    ],
                }
            )
            report = run_benchmark(benchmark)
            self.assertEqual({model.label for model in report.models}, {"pop", "graph"})
            with self.assertRaises(ConfigurationError):
                config_from_dict(
                    {
                        **config.to_dict(),
                        "model": {"name": "bipartite_graph_bpr", "params": {"layers": 3}},
                    }
                )


if __name__ == "__main__":
    unittest.main()
