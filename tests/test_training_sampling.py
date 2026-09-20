"""Independent CDF oracles and train-only integration for BPR sampling."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchidrec.config import config_from_dict
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import ConfigurationError, SerializationError, ValidationError
from orchidrec.experiment import run_experiment
from orchidrec.features import (
    FeatureDataset,
    FeatureKind,
    FeatureRow,
    FeatureSchema,
    FeatureSource,
    FeatureSpec,
)
from orchidrec.models import BipartiteGraphBPR, ImplicitMF, SideFeatureFM, load_model
from orchidrec.training_sampling import TrainingNegativeSampler


class FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def _training() -> InteractionDataset:
    return InteractionDataset(
        [
            Interaction("u1", "a"),
            *(Interaction("u2", "b") for _ in range(4)),
            Interaction("u3", "c"),
        ]
    )


def _seen(dataset: InteractionDataset) -> dict[str, frozenset[str]]:
    return {
        user: frozenset(event.item_id for event in events)
        for user, events in dataset.by_user().items()
    }


def _features() -> FeatureDataset:
    schema = FeatureSchema(
        (
            FeatureSpec("group", FeatureKind.TOKEN, FeatureSource.USER),
            FeatureSpec("topic", FeatureKind.TOKEN, FeatureSource.ITEM),
        )
    )
    return FeatureDataset(
        schema,
        (
            *(FeatureRow("user", user, {"group": user}) for user in ("u1", "u2", "u3")),
            *(FeatureRow("item", item, {"topic": item}) for item in ("a", "b", "c")),
        ),
    )


class TrainingSamplingTests(unittest.TestCase):
    def test_default_states_match_released_v018_golden_bytes(self) -> None:
        dataset = InteractionDataset([Interaction("u1", "a"), Interaction("u2", "b")])
        schema = FeatureSchema(
            (
                FeatureSpec("group", FeatureKind.TOKEN, FeatureSource.USER),
                FeatureSpec("topic", FeatureKind.TOKEN, FeatureSource.ITEM),
            )
        )
        features = FeatureDataset(
            schema,
            (
                FeatureRow("user", "u1", {"group": "u1"}),
                FeatureRow("user", "u2", {"group": "u2"}),
                FeatureRow("item", "a", {"topic": "a"}),
                FeatureRow("item", "b", {"topic": "b"}),
            ),
        )
        # Digests independently frozen from the signed v0.18.0 release wheel.
        # Graph BPR contains libm-sensitive intermediate values, so its
        # Windows and Linux release-wheel byte strings differ slightly.
        golden = {
            "implicit_mf": "52aba108d45a776d636f7cadc09fd26a162b9464ef9695385569b7d5527c372f",
            "bipartite_graph_bpr": (
                "d761b39c908605ba1aab71526f9b97aad39a04690c6449b55c157fce87b43fc4"
                if sys.platform == "linux"
                else "05005f3aff657bc118bf546dc4f7913cbc41fec552f152b0a10fb196d5084aac"
            ),
            "side_feature_fm": "18986b79d08456cac2c73fff42b3dea90c908ebe513ced63de8cb8f0f4ccd3df",
        }
        models = (
            ImplicitMF(factors=2, epochs=2, seed=7).fit(dataset),
            BipartiteGraphBPR(factors=2, epochs=2, seed=7).fit(dataset),
            SideFeatureFM(factors=2, epochs=2, seed=7).fit(dataset, features),
        )
        for model in models:
            raw = json.dumps(
                model.to_state(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), golden[model.model_type])

    def test_hand_cdf_boundaries_and_small_pool_enumeration(self) -> None:
        dataset = _training()
        for alpha, boundary in ((1.0, 0.8), (0.5, 2 / 3)):
            sampler = TrainingNegativeSampler(
                dataset,
                seen=_seen(dataset),
                catalog=dataset.item_ids,
                strategy="popularity",
                alpha=alpha,
                epochs=1,
                draws_per_positive=1,
            )
            self.assertEqual(sampler.pool("u1"), ("b", "c"))
            self.assertEqual(sampler.sample("u1", FixedRandom(0.0)), "b")  # type: ignore[arg-type]
            self.assertEqual(sampler.sample("u1", FixedRandom(boundary - 1e-7)), "b")  # type: ignore[arg-type]
            self.assertEqual(sampler.sample("u1", FixedRandom(boundary)), "c")  # type: ignore[arg-type]
            self.assertEqual(sampler.sample("u1", FixedRandom(0.999999)), "c")  # type: ignore[arg-type]
            self.assertAlmostEqual(4**alpha / (4**alpha + 1**alpha), boundary)
            with self.assertRaisesRegex(ValidationError, "outside the training sampler"):
                sampler.sample("unseen", FixedRandom(0.5))  # type: ignore[arg-type]

    def test_three_models_are_deterministic_and_exclude_seen(self) -> None:
        dataset = _training()
        observed: list[tuple[str, str]] = []
        original = TrainingNegativeSampler.sample

        def capture(sampler, user, rng):
            chosen = original(sampler, user, rng)
            observed.append((user, chosen))
            return chosen

        with patch.object(TrainingNegativeSampler, "sample", capture):
            graph = BipartiteGraphBPR(
                factors=2, epochs=2, seed=7, negative_strategy="popularity"
            ).fit(dataset)
            implicit = ImplicitMF(factors=2, epochs=2, seed=7, negative_strategy="popularity").fit(
                dataset
            )
            fm = SideFeatureFM(factors=2, epochs=2, seed=7, negative_strategy="popularity").fit(
                dataset, _features()
            )
        self.assertTrue(observed)
        for user, item in observed:
            self.assertNotIn(item, _seen(dataset)[user])
        for model in (graph, implicit, fm):
            state = model.to_state()
            self.assertEqual(state["parameters"]["negative_strategy"], "popularity")
            self.assertEqual(type(model).from_state(state).to_state(), state)
        self.assertEqual(
            graph.to_state(),
            BipartiteGraphBPR(factors=2, epochs=2, seed=7, negative_strategy="popularity")
            .fit(dataset)
            .to_state(),
        )

    def test_old_uniform_parameter_shapes_and_roundtrip_remain_unchanged(self) -> None:
        dataset = _training()
        models = (
            BipartiteGraphBPR(factors=2, epochs=2, seed=7).fit(dataset),
            ImplicitMF(factors=2, epochs=2, seed=7).fit(dataset),
            SideFeatureFM(factors=2, epochs=2, seed=7).fit(dataset, _features()),
        )
        for model in models:
            state = model.to_state()
            self.assertNotIn("negative_strategy", state["parameters"])
            self.assertNotIn("popularity_alpha", state["parameters"])
            self.assertEqual(type(model).from_state(state).to_state(), state)
            self.assertEqual(
                type(model).from_state(state).score_items("u1"), model.score_items("u1")
            )

    def test_heldout_label_change_does_not_change_fit(self) -> None:
        # Both inputs have identical first (training) interactions per user.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            states = []
            for index, test_value in enumerate((1, 99)):
                source = root / f"case-{index}.json"
                source.write_text(
                    json.dumps(
                        [
                            {"user_id": "u1", "item_id": "a", "timestamp": 1},
                            {"user_id": "u1", "item_id": "b", "value": test_value, "timestamp": 2},
                            {"user_id": "u2", "item_id": "b", "timestamp": 1},
                            {"user_id": "u2", "item_id": "c", "timestamp": 2},
                            {"user_id": "u3", "item_id": "c", "timestamp": 1},
                            {"user_id": "u3", "item_id": "a", "timestamp": 2},
                        ]
                    ),
                    encoding="utf-8",
                )
                output = root / f"model-{index}.json"
                config = config_from_dict(
                    {
                        "seed": 7,
                        "data": {"path": str(source)},
                        "split": {"method": "leave_one_out"},
                        "model": {
                            "name": "bipartite_graph_bpr",
                            "params": {
                                "factors": 2,
                                "epochs": 2,
                                "negative_strategy": "popularity",
                            },
                        },
                        "evaluation": {"k": 2},
                        "output": {"model_path": str(output)},
                    }
                )
                run_experiment(config)
                states.append(load_model(output).to_state())
            self.assertEqual(states[0], states[1])

    def test_validation_resource_and_state_tamper_guards(self) -> None:
        for kwargs in (
            {"negative_strategy": "unknown"},
            {"negative_strategy": "popularity", "popularity_alpha": 0},
            {"negative_strategy": "popularity", "popularity_alpha": math.inf},
            {"negative_strategy": "popularity", "popularity_alpha": 10**1000},
            {"negative_strategy": "popularity", "popularity_alpha": "1"},
            {"negative_strategy": "uniform", "popularity_alpha": 2},
        ):
            for model in (ImplicitMF, SideFeatureFM, BipartiteGraphBPR):
                with (
                    self.subTest(model=model.__name__, kwargs=kwargs),
                    self.assertRaises(ValidationError),
                ):
                    model(**kwargs)
        dataset = _training()
        with self.assertRaisesRegex(ValidationError, "legacy uniform"):
            TrainingNegativeSampler(
                dataset,
                seen=_seen(dataset),
                catalog=dataset.item_ids,
                strategy="uniform",
                alpha=1.0,
                epochs=1,
                draws_per_positive=1,
            )
        with self.assertRaisesRegex(ValidationError, "nonempty training"):
            TrainingNegativeSampler(
                InteractionDataset(),
                seen={},
                catalog=(),
                strategy="popularity",
                alpha=1.0,
                epochs=1,
                draws_per_positive=1,
            )
        with self.assertRaisesRegex(ValidationError, "positive bounded draw"):
            TrainingNegativeSampler(
                dataset,
                seen=_seen(dataset),
                catalog=dataset.item_ids,
                strategy="popularity",
                alpha=1.0,
                epochs=0,
                draws_per_positive=1,
            )
        all_seen = InteractionDataset([Interaction("u1", "a"), Interaction("u1", "b")])
        full_sampler = TrainingNegativeSampler(
            all_seen,
            seen=_seen(all_seen),
            catalog=all_seen.item_ids,
            strategy="popularity",
            alpha=1.0,
            epochs=1,
            draws_per_positive=1,
        )
        with self.assertRaisesRegex(ValidationError, "no unseen"):
            full_sampler.sample("u1", FixedRandom(0.5))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "pools must match"):
            TrainingNegativeSampler(
                dataset,
                seen={**_seen(dataset), "u1": frozenset({"a", "b"})},
                catalog=dataset.item_ids,
                strategy="popularity",
                alpha=1.0,
                epochs=1,
                draws_per_positive=1,
            )
        with self.assertRaisesRegex(ValidationError, "frozen training pools"):
            TrainingNegativeSampler(
                dataset,
                seen={"u1": {"a"}},  # type: ignore[arg-type]
                catalog=dataset.item_ids,
                strategy="popularity",
                alpha=1.0,
                epochs=1,
                draws_per_positive=1,
            )
        with self.assertRaisesRegex(ValidationError, "resource limit"):
            TrainingNegativeSampler(
                dataset,
                seen=_seen(dataset),
                catalog=dataset.item_ids,
                strategy="popularity",
                alpha=1.0,
                epochs=1_000_000,
                draws_per_positive=1,
            )
        with self.assertRaisesRegex(ValidationError, "coordinate work"):
            ImplicitMF(factors=65, epochs=1, negative_strategy="popularity").fit(dataset)
        # This guard is specific to the opt-in path; the existing uniform
        # constructor and its checkpoint parameter schema remain untouched.
        self.assertEqual(ImplicitMF(factors=65, epochs=1).factors, 65)
        for fitted in (
            BipartiteGraphBPR(factors=2, epochs=1, negative_strategy="popularity").fit(dataset),
            ImplicitMF(factors=2, epochs=1, negative_strategy="popularity").fit(dataset),
            SideFeatureFM(factors=2, epochs=1, negative_strategy="popularity").fit(
                dataset, _features()
            ),
        ):
            state = copy.deepcopy(fitted.to_state())
            state["parameters"]["negative_strategy"] = "uniform"
            with self.assertRaises(SerializationError):
                type(fitted).from_state(state)
        with self.assertRaises(ConfigurationError):
            config_from_dict(
                {
                    "data": {"path": "examples/interactions.json"},
                    "model": {
                        "name": "bipartite_graph_bpr",
                        "params": {"negative_strategy": "unknown"},
                    },
                    "output": {"model_path": "unused.json"},
                }
            )


if __name__ == "__main__":
    unittest.main()
