"""Independent graph-walk oracle and integration tests."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orchidrec.benchmark import run_benchmark
from orchidrec.benchmark_config import benchmark_config_from_dict
from orchidrec.config import config_from_dict
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import ConfigurationError, SerializationError, ValidationError
from orchidrec.experiment import run_experiment
from orchidrec.models import KGWalkRec, load_model, save_model
from orchidrec.recbole_knowledge import (
    import_recbole_knowledge,
    save_recbole_knowledge,
)
from orchidrec.reporting import benchmark_html

ROOT = Path(__file__).resolve().parents[1]


def knowledge():
    return import_recbole_knowledge(
        kg_path=ROOT / "examples/kg_walk_synthetic.kg",
        link_path=ROOT / "examples/kg_walk_synthetic.link",
    )


class KGWalkRecTests(unittest.TestCase):
    def test_two_hop_relation_weight_oracle(self) -> None:
        train = InteractionDataset(
            [
                Interaction("u", "A"),
                Interaction("v", "B"),
                Interaction("w", "C"),
                Interaction("x", "D"),
            ]
        )
        model = KGWalkRec(hops=2, relation_weights={"r": 2, "q": 1}, popularity_mix=0)
        model.fit(train, knowledge())
        scores = model.score_items("u")
        self.assertAlmostEqual(scores["B"], 1 / 3)
        self.assertAlmostEqual(scores["C"], 1 / 6)
        self.assertEqual(scores["D"], 0)
        self.assertEqual(model.recommend("u", 2)[0].item_id, "B")

    def test_training_value_weighting_has_independent_hand_oracle(self) -> None:
        train = InteractionDataset(
            [
                Interaction("u", "A", 3),
                Interaction("u", "C", 1),
                Interaction("v", "B"),
            ]
        )
        options = {"hops": 2, "relation_weights": {"r": 2, "q": 1}, "popularity_mix": 0}
        weighted = KGWalkRec(**options, weighted=True).fit(train, knowledge())
        unit = KGWalkRec(**options, weighted=False).fit(train, knowledge())
        self.assertAlmostEqual(weighted.score_items("u")["B"], 1 / 4)
        self.assertAlmostEqual(unit.score_items("u")["B"], 1 / 6)
        self.assertEqual(
            KGWalkRec.from_state(weighted.to_state()).score_items("u"), weighted.score_items("u")
        )

    def test_dangling_and_unseen_user_fallback_and_round_trip(self) -> None:
        train = InteractionDataset(
            [Interaction("u", "A"), Interaction("x", "D"), Interaction("v", "B")]
        )
        model = KGWalkRec(hops=2, popularity_mix=0).fit(train, knowledge())
        self.assertEqual(model.score_items("x")["D"], 1)
        self.assertEqual(model.score_items("new")["D"], 1)
        self.assertEqual(KGWalkRec.from_state(model.to_state()).to_state(), model.to_state())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_model(model, path)
            self.assertEqual(load_model(path).score_items("u"), model.score_items("u"))
            rejected_path = Path(directory) / "too-large.json"
            with (
                patch("orchidrec.models.io.MAX_MODEL_FILE_BYTES", 100),
                self.assertRaisesRegex(SerializationError, "safety limit"),
            ):
                save_model(model, rejected_path)
            self.assertFalse(rejected_path.exists())

    def test_rejects_bounds_ids_unknown_relation_and_tampered_state(self) -> None:
        train = InteractionDataset(
            [Interaction("u", "A"), Interaction("v", "B"), Interaction("w", "C")]
        )
        for params in (
            {"hops": 0},
            {"hops": 4},
            {"weighted": 1},
            {"popularity_mix": float("inf")},
            {"relation_weights": {"r": -1}},
            {"max_work_units": 0},
        ):
            with self.subTest(params=params), self.assertRaises(ValidationError):
                KGWalkRec(**params)
        with self.assertRaisesRegex(ValidationError, "unknown relation"):
            KGWalkRec(relation_weights={"not-there": 1}).fit(train, knowledge())
        with self.assertRaisesRegex(ValidationError, "work limit"):
            KGWalkRec(max_work_units=1).fit(train, knowledge())
        with self.assertRaisesRegex(ValidationError, "item IDs must be strings"):
            KGWalkRec().fit(InteractionDataset([Interaction("u", 42)]), knowledge())
        with self.assertRaisesRegex(ValidationError, "linked training positive"):
            KGWalkRec().fit(InteractionDataset([Interaction("u", "unlinked")]), knowledge())

        original = KGWalkRec().fit(train, knowledge()).to_state()
        bad = copy.deepcopy(original)
        bad["model"]["triples"][0][0] = "different"
        with self.assertRaisesRegex(SerializationError, "fingerprint"):
            KGWalkRec.from_state(bad)
        bad = copy.deepcopy(original)
        bad["model"]["user_seeds"][0]["items"][0]["item_id"] = "unlinked"
        with self.assertRaises(SerializationError):
            KGWalkRec.from_state(bad)

    def test_public_type_and_numeric_boundaries(self) -> None:
        for params in (
            {"hops": True},
            {"weighted": "yes"},
            {"popularity_mix": True},
            {"relation_weights": []},
            {"relation_weights": {"": 1}},
            {"relation_weights": {"r": 0}},
            {"max_work_units": True},
        ):
            with self.subTest(params=params), self.assertRaises(ValidationError):
                KGWalkRec(**params)
        with self.assertRaises(ValidationError):
            KGWalkRec().fit(InteractionDataset())
        with self.assertRaises(ValidationError):
            KGWalkRec().fit(InteractionDataset([Interaction("u", "A")]))
        with self.assertRaises(ValidationError):
            KGWalkRec().fit(InteractionDataset([Interaction(1 << 513, "A")]), knowledge())
        with self.assertRaises(ValidationError):
            KGWalkRec().fit(InteractionDataset([Interaction("u" * 2049, "A")]), knowledge())
        with self.assertRaisesRegex(ValidationError, "valid UTF-8"):
            KGWalkRec().fit(InteractionDataset([Interaction("\ud800", "A")]), knowledge())
        with self.assertRaisesRegex(ValidationError, "valid UTF-8"):
            KGWalkRec(relation_weights={"\ud800": 1})
        long_integer = 1 << 511
        self.assertAlmostEqual(
            KGWalkRec()
            .fit(InteractionDataset([Interaction(long_integer, "A")]), knowledge())
            .score_items(long_integer)["A"],
            0.525,
        )
        with self.assertRaises(ValidationError):
            KGWalkRec().fit(
                InteractionDataset([Interaction("u", "A", 6e11), Interaction("u", "B", 6e11)]),
                knowledge(),
            )
        with self.assertRaisesRegex(ValidationError, "user limit"):
            KGWalkRec().fit(
                InteractionDataset(Interaction(f"u{index}", "A") for index in range(2001)),
                knowledge(),
            )
        with self.assertRaisesRegex(ValidationError, "catalog limit"):
            KGWalkRec().fit(
                InteractionDataset(Interaction("u", f"item-{index}") for index in range(2001)),
                knowledge(),
            )

    def test_malformed_state_shapes_are_rejected(self) -> None:
        train = InteractionDataset(
            [
                Interaction("u", "A"),
                Interaction("u", "B"),
                Interaction("v", "C"),
                Interaction("w", "D"),
            ]
        )
        original = KGWalkRec().fit(train, knowledge()).to_state()

        def rejected(state: dict) -> None:
            with self.assertRaises(SerializationError):
                KGWalkRec.from_state(state)

        bad = copy.deepcopy(original)
        del bad["parameters"]["hops"]
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["parameters"]["hops"] = 99
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["base"] = {"catalog": [], "popularity": [], "users": []}
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["base"]["catalog"][0] = 1
        for entry in bad["base"]["users"]:
            entry["seen"] = [1 if item == "A" else item for item in entry["seen"]]
        rejected(bad)
        bad = copy.deepcopy(original)
        del bad["model"]["links"]
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["triples"] = "not an array"
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["triples"][0] = ["eA"]
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["triples"][0][0] = "bad token"
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["triples"].reverse()
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["links"][1][1] = bad["model"]["links"][0][1]
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["user_seeds"] = {}
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["user_seeds"][0] = {"user_id": "u"}
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["user_seeds"][0]["items"] = []
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["user_seeds"][0]["items"][0] = {"item_id": "A"}
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["user_seeds"][0]["items"][0]["weight"] = True
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["user_seeds"][0]["items"][0]["weight"] = 0
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["user_seeds"][0]["items"].reverse()
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["user_seeds"].reverse()
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["user_seeds"][0]["user_id"] = "not-in-train"
        rejected(bad)
        bad = copy.deepcopy(original)
        bad["model"]["source_sha256"][0] = "not-a-digest"
        with self.assertRaises(SerializationError):
            KGWalkRec.from_state(bad)

    def test_sparse_hub_work_accounting_without_runtime_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kg = root / "hub.kg"
            links = root / "hub.link"
            kg.write_text(
                "head_id:token\trelation_id:token\ttail_id:token\n"
                + "".join(f"hub\tr\tleaf-{index:04d}\n" for index in range(512)),
                encoding="utf-8",
            )
            links.write_text(
                "item_id:token\tentity_id:token\nA\thub\nB\tleaf-0000\n",
                encoding="utf-8",
            )
            graph = import_recbole_knowledge(kg_path=kg, link_path=links)
            train = InteractionDataset([Interaction("u", "A"), Interaction("v", "B")])
            model = KGWalkRec(hops=1, popularity_mix=0, max_work_units=513).fit(train, graph)
            self.assertEqual(model.work_units, 513)
            self.assertAlmostEqual(model.score_items("u")["B"], 1 / 512)
            with self.assertRaisesRegex(ValidationError, "work limit"):
                KGWalkRec(hops=1, max_work_units=512).fit(train, graph)

    def test_dangling_self_loop_is_charged_to_work_budget(self) -> None:
        train = InteractionDataset([Interaction("u", "A"), Interaction("x", "D")])
        model = KGWalkRec(hops=1, max_work_units=3).fit(train, knowledge())
        self.assertEqual(model.work_units, 3)
        with self.assertRaisesRegex(ValidationError, "work limit"):
            KGWalkRec(hops=1, max_work_units=2).fit(train, knowledge())

    def test_oversized_graph_and_base_arrays_fail_before_deep_validation(self) -> None:
        graph = knowledge()
        train = InteractionDataset([Interaction("u", "A")])
        oversized = replace(graph, triples=graph.triples * 5001)
        with (
            patch("orchidrec.models.kg_walk_rec._validate_semantics") as validate,
            self.assertRaisesRegex(ValidationError, "graph or link limit"),
        ):
            KGWalkRec().fit(train, oversized)
        validate.assert_not_called()

        original = KGWalkRec().fit(train, graph).to_state()
        for key, value in (
            ("catalog", ["A"] * 2001),
            ("users", original["base"]["users"] * 2001),
        ):
            with self.subTest(key=key):
                bad = copy.deepcopy(original)
                bad["base"][key] = value
                with (
                    patch.object(KGWalkRec, "_restore_base_state") as restore,
                    self.assertRaisesRegex(SerializationError, "base arrays exceed limits"),
                ):
                    KGWalkRec.from_state(bad)
                restore.assert_not_called()
        bad = copy.deepcopy(original)
        bad["base"]["users"][0]["seen"] = ["A"] * 2001
        with (
            patch.object(KGWalkRec, "_restore_base_state") as restore,
            self.assertRaisesRegex(SerializationError, "seen array exceeds limits"),
        ):
            KGWalkRec.from_state(bad)
        restore.assert_not_called()

    def test_loaded_aggregate_values_preserve_fit_bound(self) -> None:
        train = InteractionDataset([Interaction("u", "A"), Interaction("v", "B")])
        original = KGWalkRec().fit(train, knowledge()).to_state()
        bad = copy.deepcopy(original)
        bad["base"]["popularity"] = [6e11, 6e11]
        with self.assertRaisesRegex(SerializationError, "popularity total exceeds bound"):
            KGWalkRec.from_state(bad)
        bad = copy.deepcopy(original)
        for entry in bad["model"]["user_seeds"]:
            entry["items"][0]["weight"] = 6e11
        with self.assertRaisesRegex(SerializationError, "aggregate seed total exceeds bound"):
            KGWalkRec.from_state(bad)

    def test_importer_experiment_and_shared_benchmark_provenance(self) -> None:
        graph = import_recbole_knowledge(
            kg_path=ROOT / "examples/kg_walk_synthetic.kg",
            link_path=ROOT / "examples/kg_walk_synthetic.link",
            inter_path=ROOT / "examples/kg_walk_synthetic.inter",
            minimum_rating=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "knowledge.json"
            save_recbole_knowledge(graph, artifact)
            config = config_from_dict(
                {
                    "data": {
                        "path": str(ROOT / "examples/kg_walk_interactions.json"),
                        "knowledge_path": str(artifact),
                    },
                    "split": {"method": "leave_one_out"},
                    "model": {"name": "kg_walk_rec", "params": {"hops": 2}},
                    "evaluation": {"k": 2},
                }
            )
            result = run_experiment(config)
            self.assertEqual(result.model_type, "kg_walk_rec")
            self.assertEqual(result.knowledge["fingerprint_sha256"], graph.fingerprint)
            self.assertEqual(result.evaluated_test_size, 4)
            benchmark_data = {
                "schema_version": 1,
                "data": {
                    "path": str(ROOT / "examples/kg_walk_synthetic.inter"),
                    "format": "recbole-inter",
                    "minimum_rating": 1,
                    "knowledge_path": str(artifact),
                },
                "split": {"method": "leave_one_out"},
                "evaluation": {"k": 2, "bootstrap_samples": 20},
                "models": [
                    {"label": "pop", "name": "popularity"},
                    {"label": "kg", "name": "kg_walk_rec", "params": {"hops": 2}},
                ],
            }
            report = run_benchmark(benchmark_config_from_dict(benchmark_data))
            self.assertEqual({model.label for model in report.models}, {"pop", "kg"})
            self.assertEqual(report.knowledge["fingerprint_sha256"], graph.fingerprint)
            self.assertIn(graph.fingerprint, benchmark_html(report))
            second_artifact = root / "same-content.json"
            save_recbole_knowledge(graph, second_artifact)
            benchmark_data["data"]["knowledge_path"] = str(second_artifact)
            same = run_benchmark(benchmark_config_from_dict(benchmark_data))
            self.assertEqual(report.config_fingerprint, same.config_fingerprint)
            config_with_alias = config_from_dict(
                {**config.to_dict(), "output": {"model_path": str(artifact)}}
            )
            with self.assertRaisesRegex(ValidationError, "different files"):
                run_experiment(config_with_alias)
            benchmark_data["data"]["path"] = str(ROOT / "examples/recbole-synthetic.inter")
            with self.assertRaises(ConfigurationError):
                run_benchmark(benchmark_config_from_dict(benchmark_data))


if __name__ == "__main__":
    unittest.main()
