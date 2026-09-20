from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

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
    SequenceKeep,
    save_feature_dataset,
)
from orchidrec.models import SideFeatureFM, load_model, save_model
from orchidrec.models.side_feature_fm import _bounded_identifier


def _schema() -> FeatureSchema:
    return FeatureSchema(
        (
            FeatureSpec("region", FeatureKind.TOKEN, FeatureSource.USER),
            FeatureSpec(
                "topic", FeatureKind.TOKEN_SEQUENCE, FeatureSource.ITEM, 2, SequenceKeep.HEAD
            ),
            FeatureSpec("quality", FeatureKind.FLOAT, FeatureSource.ITEM),
        )
    )


def _features() -> FeatureDataset:
    return FeatureDataset(
        _schema(),
        (
            FeatureRow("user", "u1", {"region": "north"}),
            FeatureRow("user", "u2", {"region": "south"}),
            FeatureRow("item", "i1", {"topic": ["math", "art"], "quality": 1}),
            FeatureRow("item", "i2", {"topic": ["art", "art"], "quality": 3}),
        ),
    )


def _dataset() -> InteractionDataset:
    return InteractionDataset((Interaction("u1", "i1"), Interaction("u2", "i2")))


def _refresh_digest(state: dict[str, object]) -> None:
    model = state["model"]
    encoded = model["encoded"]
    payload = json.dumps(
        encoded, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    model["encoded_sha256"] = hashlib.sha256(payload).hexdigest()


class SideFeatureFMTests(unittest.TestCase):
    def test_sparse_score_and_one_bpr_gradient_have_hand_oracles(self) -> None:
        model = SideFeatureFM(factors=1, epochs=1, learning_rate=0.1, regularization=0.2)
        model._linear = [0.0, 0.0, 0.0]
        model._latent = [[1.0], [2.0], [3.0]]
        model._user_vectors = {"u": ((0, 1.0),)}
        model._item_vectors = {"p": ((1, 1.0),), "n": ((2, 1.0),)}
        self.assertEqual(model._score_sparse(((0, 1.0), (1, 1.0))), 2.0)
        self.assertEqual(model._score_sparse(((0, 1.0), (2, 1.0))), 3.0)
        model._update_pair("u", "p", "n")
        gradient = 1 / (1 + math.exp(-1))
        self.assertAlmostEqual(model._linear[0], 0.0)
        self.assertAlmostEqual(model._linear[1], 0.1 * gradient)
        self.assertAlmostEqual(model._linear[2], -0.1 * gradient)
        self.assertAlmostEqual(model._latent[0][0], 1 + 0.1 * (-gradient - 0.2))
        self.assertAlmostEqual(model._latent[1][0], 2 + 0.1 * (gradient - 0.4))
        self.assertAlmostEqual(model._latent[2][0], 3 + 0.1 * (-gradient - 0.6))

    def test_actual_side_features_affect_score_and_round_trip(self) -> None:
        model = SideFeatureFM(factors=2, epochs=4, seed=3).fit(_dataset(), _features())
        self.assertEqual(
            model.to_state(),
            SideFeatureFM(factors=2, epochs=4, seed=3).fit(_dataset(), _features()).to_state(),
        )
        self.assertEqual(model._pipeline.training_rows, 4)  # type: ignore[union-attr]
        self.assertEqual(model._encoded.pipeline_sha256, model._pipeline.state_sha256)  # type: ignore[union-attr]
        first = model._item_vectors["i1"]
        second = model._item_vectors["i2"]
        self.assertNotEqual(first, second)
        self.assertIn(0.5, dict(first).values())
        self.assertIn(1.0, dict(second).values())
        baseline = model.score_items("u1")
        self.assertEqual(SideFeatureFM.from_state(model.to_state()).score_items("u1"), baseline)
        state = model.to_state()
        state["model"]["linear"] = [0.0] * len(model._linear)
        state["model"]["latent"] = [[0.0] * model.factors for _ in model._latent]
        for index, value in second:
            if index not in dict(first):
                state["model"]["linear"][index] = 2.0 * value
        changed = SideFeatureFM.from_state(state)
        self.assertNotEqual(changed.score_items("u1"), baseline)
        self.assertEqual(changed.score_items("absent")["i1"], model.score_items("absent")["i1"])

    def test_validation_and_tamper_detection(self) -> None:
        with self.assertRaisesRegex(ValidationError, "features must"):
            SideFeatureFM().fit(_dataset())
        with self.assertRaisesRegex(ValidationError, "exactly match"):
            SideFeatureFM().fit(_dataset(), FeatureDataset(_schema(), _features().rows[:-1]))
        interaction_schema = FeatureSchema(
            (FeatureSpec("context", FeatureKind.TOKEN, FeatureSource.INTERACTION),)
        )
        interaction_features = FeatureDataset(
            interaction_schema, (FeatureRow("interaction", "event-1", {"context": "x"}),)
        )
        with self.assertRaisesRegex(ValidationError, "event IDs"):
            SideFeatureFM().fit(_dataset(), interaction_features)
        for kwargs in (
            {"factors": True},
            {"epochs": 101},
            {"learning_rate": 0},
            {"regularization": -1},
            {"seed": True},
            {"seed": 1 << 63},
            {"seed": -(1 << 63) - 1},
            {"max_work_units": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                SideFeatureFM(**kwargs)
        state = SideFeatureFM(factors=1, epochs=1).fit(_dataset(), _features()).to_state()
        for mutation in (
            lambda item: item["model"].update(encoded_sha256="0" * 64),
            lambda item: item["model"].update(latent=[]),
            lambda item: item["model"].update(updates=0),
            lambda item: item["model"].update(
                linear=[float("nan")] * len(state["model"]["linear"])
            ),
        ):
            altered = copy.deepcopy(state)
            mutation(altered)
            with self.assertRaises(SerializationError):
                SideFeatureFM.from_state(altered)

    def test_fit_bounds_and_no_negative_failure(self) -> None:
        with self.assertRaisesRegex(ValidationError, "max_work_units"):
            SideFeatureFM(factors=2, epochs=100, max_work_units=1).fit(_dataset(), _features())
        full = InteractionDataset((Interaction("u1", "i1"), Interaction("u1", "i2")))
        subset = FeatureDataset(_schema(), tuple(row for row in _features() if row.key != "u2"))
        with self.assertRaisesRegex(ValidationError, "unobserved"):
            SideFeatureFM().fit(full, subset)

    def test_constructor_and_preflight_limits(self) -> None:
        with self.assertRaisesRegex(ValidationError, "finite bounded"):
            SideFeatureFM(learning_rate="fast")
        with self.assertRaisesRegex(ValidationError, "non-empty"):
            SideFeatureFM().fit(InteractionDataset(), _features())
        with self.assertRaisesRegex(ValidationError, "integer bits"):
            _bounded_identifier(1 << 513)
        self.assertEqual(_bounded_identifier(42), 42)
        with self.assertRaisesRegex(ValidationError, "UTF-8"):
            _bounded_identifier("\ud800")
        event = Interaction("u1", "i1")
        with self.assertRaisesRegex(ValidationError, "interaction limit"):
            SideFeatureFM().fit(InteractionDataset([event] * 100_001), _features())
        over_users = InteractionDataset(Interaction(user, "i1") for user in range(2049))
        with self.assertRaisesRegex(ValidationError, "user or item limit"):
            SideFeatureFM().fit(over_users, _features())
        row_schema = FeatureSchema((FeatureSpec("topic", FeatureKind.TOKEN, FeatureSource.ITEM),))
        over_rows = FeatureDataset(
            row_schema,
            (FeatureRow("item", index, {"topic": "a"}) for index in range(4097)),
        )
        with self.assertRaisesRegex(ValidationError, "row limit"):
            SideFeatureFM().fit(_dataset(), over_rows)
        wide_schema = FeatureSchema(
            (
                FeatureSpec(
                    "wide", FeatureKind.TOKEN_SEQUENCE, FeatureSource.USER, 128, SequenceKeep.HEAD
                ),
            )
        )
        wide = FeatureDataset(
            wide_schema,
            (FeatureRow("user", user, {"wide": []}) for user in ("u1", "u2")),
        )
        with self.assertRaisesRegex(ValidationError, "active-coordinate"):
            SideFeatureFM().fit(_dataset(), wide)

    def test_duplicate_events_and_full_catalog_user_follow_unique_pair_contract(self) -> None:
        unique = InteractionDataset(
            (Interaction("u1", "i1"), Interaction("u2", "i1"), Interaction("u2", "i2"))
        )
        duplicate = InteractionDataset((*unique, Interaction("u1", "i1", value=5)))
        base = SideFeatureFM(epochs=3, factors=1, seed=9).fit(unique, _features())
        repeated = SideFeatureFM(epochs=3, factors=1, seed=9).fit(duplicate, _features())
        self.assertEqual(base._updates, 3)
        self.assertEqual(repeated._updates, 3)
        self.assertEqual(base._linear, repeated._linear)
        self.assertEqual(base._latent, repeated._latent)
        self.assertNotEqual(base.to_state()["base"], repeated.to_state()["base"])

    def test_item_only_and_numeric_sequences(self) -> None:
        schema = FeatureSchema(
            (
                FeatureSpec("mood", FeatureKind.TOKEN, FeatureSource.ITEM),
                FeatureSpec(
                    "trajectory",
                    FeatureKind.FLOAT_SEQUENCE,
                    FeatureSource.ITEM,
                    2,
                    SequenceKeep.HEAD,
                ),
            )
        )
        features = FeatureDataset(
            schema,
            (
                FeatureRow("item", "i1", {"mood": "blue", "trajectory": [1]}),
                FeatureRow("item", "i2", {"mood": "red", "trajectory": [3, 5]}),
            ),
        )
        model = SideFeatureFM(factors=1, epochs=1).fit(_dataset(), features)
        self.assertEqual(len(model._user_vectors["u1"]), 1)
        self.assertEqual(len(model._item_vectors["i1"]), 3)
        self.assertNotEqual(model._item_vectors["i1"], model._item_vectors["i2"])
        self.assertEqual(
            SideFeatureFM.from_state(model.to_state()).score_items("u1"), model.score_items("u1")
        )

    def test_serialized_state_adversarial_boundary_matrix(self) -> None:
        original = SideFeatureFM(factors=2, epochs=2).fit(_dataset(), _features()).to_state()

        def invalid(mutator, expected: str = "FM") -> None:
            state = copy.deepcopy(original)
            mutator(state)
            with self.assertRaisesRegex(SerializationError, expected):
                SideFeatureFM.from_state(state)

        invalid(lambda s: s["parameters"].update(extra=1), "parameters")
        invalid(lambda s: s["parameters"].update(seed=1 << 63), "signed 64-bit")
        invalid(lambda s: s["model"].pop("updates"), "fields")
        invalid(lambda s: s["base"].update(catalog=list(range(2049))), "catalog limit")
        invalid(lambda s: s["base"].update(users=[s["base"]["users"][0]] * 2049), "user limit")
        invalid(lambda s: s["base"]["catalog"].__setitem__(0, "x" * 2049), "UTF-8 bytes")
        invalid(lambda s: s["base"]["users"].__setitem__(0, {}), "user state")
        invalid(lambda s: s["model"].update(encoded=[]), "encoded feature rows")
        invalid(lambda s: s["model"]["encoded"].update(rows=[{}] * 4097), "row limit")
        invalid(lambda s: s["model"]["pipeline"].update(state_sha256="0" * 64), "checksum")
        invalid(lambda s: s["model"]["encoded"].update(pipeline_sha256="0" * 64), "pipeline")
        invalid(lambda s: s["model"].update(linear=[]), "linear weight length")
        invalid(lambda s: s["model"].update(latent=[]), "latent weight row count")
        invalid(lambda s: s["model"]["latent"][0].append(1), "latent weight width")
        invalid(lambda s: s["model"]["latent"][0].__setitem__(0, 1e30), "latent weight")
        invalid(lambda s: s["model"]["linear"].__setitem__(0, 1e30), "linear weight")
        invalid(lambda s: s["model"].update(updates=1), "update count")
        invalid(lambda s: s["model"].update(encoded_sha256="0" * 64), "digest")
        invalid(lambda s: s["parameters"].update(max_work_units=1), "work bound")

        def bad_token(state: dict[str, object]) -> None:
            rows = state["model"]["encoded"]["rows"]
            row = next(row for row in rows if row["source"] == "user")
            row["values"]["region"] = 999
            _refresh_digest(state)

        invalid(bad_token, "vocabulary")

        def scalar_padding(state: dict[str, object]) -> None:
            rows = state["model"]["encoded"]["rows"]
            row = next(row for row in rows if row["source"] == "user")
            row["values"]["region"] = 0
            _refresh_digest(state)

        invalid(scalar_padding, "padding index")

        def token_sequence_oov(state: dict[str, object]) -> None:
            rows = state["model"]["encoded"]["rows"]
            row = next(row for row in rows if row["source"] == "item")
            row["values"]["topic"][0] = 999
            _refresh_digest(state)

        invalid(token_sequence_oov, "vocabulary")

        def bad_padding(state: dict[str, object]) -> None:
            rows = state["model"]["encoded"]["rows"]
            row = next(row for row in rows if row["source"] == "item")
            row["sequence_lengths"]["topic"] = 1
            _refresh_digest(state)

        invalid(bad_padding, "padding")

        def bad_join(state: dict[str, object]) -> None:
            rows = state["model"]["encoded"]["rows"]
            row = next(row for row in rows if row["source"] == "user")
            row["key"] = "different"
            _refresh_digest(state)

        invalid(bad_join, "join")

    def test_numeric_sequence_padding_and_fitted_state_integrity(self) -> None:
        schema = FeatureSchema(
            (
                FeatureSpec(
                    "curve", FeatureKind.FLOAT_SEQUENCE, FeatureSource.ITEM, 2, SequenceKeep.HEAD
                ),
            )
        )
        features = FeatureDataset(
            schema,
            (
                FeatureRow("item", "i1", {"curve": [1]}),
                FeatureRow("item", "i2", {"curve": [2, 3]}),
            ),
        )
        model = SideFeatureFM(factors=1, epochs=1).fit(_dataset(), features)
        state = model.to_state()
        row = next(row for row in state["model"]["encoded"]["rows"] if row["key"] == "i1")
        row["values"]["curve"][1] = 1.0
        _refresh_digest(state)
        with self.assertRaisesRegex(SerializationError, "numeric sequence padding"):
            SideFeatureFM.from_state(state)
        model._pipeline = None
        with self.assertRaisesRegex(SerializationError, "missing feature state"):
            model.to_state()

    def test_divergent_update_is_rejected(self) -> None:
        model = SideFeatureFM(factors=1, epochs=1, learning_rate=1, regularization=1)
        model._linear = [0.0, 0.0, 0.0]
        model._latent = [[1_000_000.0], [-1_000_000.0], [1_000_000.0]]
        model._user_vectors = {"u": ((0, 1.0),)}
        model._item_vectors = {"p": ((1, 1_000.0),), "n": ((2, 1_000.0),)}
        with self.assertRaisesRegex(ValidationError, "diverged"):
            model._update_pair("u", "p", "n")

    def test_failed_fit_cannot_be_saved_or_recommended(self) -> None:
        model = SideFeatureFM(factors=1, epochs=1)
        original = model._update_pair

        def fail(*_args: object) -> None:
            raise ValidationError("injected update failure")

        model._update_pair = fail  # type: ignore[method-assign]
        try:
            with self.assertRaisesRegex(ValidationError, "injected"):
                model.fit(_dataset(), _features())
        finally:
            model._update_pair = original  # type: ignore[method-assign]
        self.assertFalse(model.is_fitted)
        with self.assertRaisesRegex(Exception, "not been fitted"):
            model.recommend("u1")
        with self.assertRaisesRegex(Exception, "not been fitted"):
            model.to_state()
        oversized = InteractionDataset((Interaction("u1", "i1"), Interaction("u2", "x" * 2049)))
        with self.assertRaisesRegex(ValidationError, "UTF-8 bytes"):
            SideFeatureFM().fit(oversized, _features())

    def test_experiment_train_only_pipeline_cli_path_and_saved_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = InteractionDataset(
                (
                    Interaction("u1", "i1", timestamp=1),
                    Interaction("u1", "i2", timestamp=2),
                    Interaction("u2", "i2", timestamp=1),
                    Interaction("u2", "i1", timestamp=2),
                    Interaction("u3", "i1", timestamp=1),
                    Interaction("u3", "i2", timestamp=2),
                )
            )
            events.save_json(root / "events.json")
            save_feature_dataset(
                FeatureDataset(
                    _schema(),
                    (*_features().rows, FeatureRow("user", "u3", {"region": "west"})),
                ),
                root / "features.json",
            )
            config = config_from_dict(
                {
                    "data": {"path": "events.json", "features_path": "features.json"},
                    "split": {"method": "temporal", "test_ratio": 0.33},
                    "model": {"name": "side_feature_fm", "params": {"factors": 2, "epochs": 2}},
                    "output": {"model_path": "model.json", "report_path": "report.json"},
                },
                base_dir=root,
            )
            first = run_experiment(config)
            self.assertEqual(first.to_dict(), run_experiment(config).to_dict())
            self.assertEqual(first.model_type, "side_feature_fm")
            self.assertEqual(first.metrics.users, 2)
            saved = load_model(root / "model.json")
            self.assertIsInstance(saved, SideFeatureFM)
            self.assertEqual(
                json.loads((root / "report.json").read_text())["model"]["type"], "side_feature_fm"
            )
            save_model(saved, root / "roundtrip.json")
            self.assertEqual(
                (root / "model.json").read_bytes(), (root / "roundtrip.json").read_bytes()
            )
            with self.assertRaisesRegex(ConfigurationError, "required only"):
                config_from_dict(
                    {
                        "data": {"path": "events.json", "features_path": "features.json"},
                        "model": {"name": "popularity"},
                    },
                    base_dir=root,
                )

    def test_holdout_only_item_is_excluded_from_fitted_feature_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = InteractionDataset(
                (
                    Interaction("u1", "i1", timestamp=1),
                    Interaction("u2", "i2", timestamp=2),
                    Interaction("u1", "i2", timestamp=3),
                    Interaction("u2", "i1", timestamp=4),
                    Interaction("u1", "i3", timestamp=5),
                )
            )
            events.save_json(root / "events.json")
            schema = FeatureSchema((FeatureSpec("topic", FeatureKind.TOKEN, FeatureSource.ITEM),))
            save_feature_dataset(
                FeatureDataset(
                    schema,
                    (
                        FeatureRow("item", "i1", {"topic": "known-a"}),
                        FeatureRow("item", "i2", {"topic": "known-b"}),
                        FeatureRow("item", "i3", {"topic": "holdout-only"}),
                    ),
                ),
                root / "features.json",
            )
            config = config_from_dict(
                {
                    "data": {"path": "events.json", "features_path": "features.json"},
                    "split": {"method": "temporal", "test_ratio": 0.4},
                    "model": {"name": "side_feature_fm", "params": {"epochs": 2}},
                    "output": {"model_path": "model.json"},
                },
                base_dir=root,
            )
            result = run_experiment(config)
            self.assertEqual(result.cold_start_test_size, 1)
            self.assertEqual(result.evaluated_test_size, 1)
            model = load_model(root / "model.json")
            self.assertEqual(model.catalog, ("i1", "i2"))
            vocabulary = model.to_state()["model"]["pipeline"]["token_vocabularies"]["topic"]
            self.assertEqual(vocabulary, ["known-a", "known-b"])


if __name__ == "__main__":
    unittest.main()
