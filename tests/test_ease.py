from __future__ import annotations

import copy
import json
import math
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

from orchidrec.benchmark import run_benchmark
from orchidrec.benchmark_config import benchmark_config_from_dict
from orchidrec.config import config_from_dict
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.demo import demo_dataset
from orchidrec.errors import ConfigurationError, NotFittedError, SerializationError, ValidationError
from orchidrec.experiment import build_model
from orchidrec.models import load_model, save_model
from orchidrec.models.ease import EASE, MAX_EASE_USERS, _invert_spd


def training() -> InteractionDataset:
    return InteractionDataset(
        (
            Interaction("u1", "a"),
            Interaction("u1", "b"),
            Interaction("u2", "a"),
        )
    )


class EaseLinearAlgebraTests(unittest.TestCase):
    def test_checked_inverse_matches_hand_computed_two_by_two_oracle(self) -> None:
        inverse, residual = _invert_spd(((3.0, 1.0), (1.0, 2.0)))
        expected = ((0.4, -0.2), (-0.2, 0.6))
        for actual_row, expected_row in zip(inverse, expected, strict=True):
            for actual, wanted in zip(actual_row, expected_row, strict=True):
                self.assertAlmostEqual(actual, wanted, places=12)
        self.assertLess(residual, 1e-12)

    def test_checked_inverse_rejects_invalid_systems(self) -> None:
        invalid = (
            "not a matrix",
            (),
            ((1.0, 0.0),),
            ((0.0,),),
            ((1.0, 2.0), (0.0, 1.0)),
            ((1.0, 1.0), (1.0, 1.0)),
            ((math.inf,),),
            ((True,),),
        )
        for matrix in invalid:
            with self.subTest(matrix=matrix), self.assertRaises(ValidationError):
                _invert_spd(matrix)


class EaseModelTests(unittest.TestCase):
    def test_coefficients_and_ranking_match_closed_form_oracle(self) -> None:
        model = EASE(regularization=1.0).fit(training())
        state = model.to_state()
        coefficients = state["model"]["coefficients"]
        self.assertAlmostEqual(coefficients[0][1], 1.0 / 3.0, places=12)
        self.assertAlmostEqual(coefficients[1][0], 0.5, places=12)
        self.assertEqual(coefficients[0][0], 0.0)
        self.assertEqual(coefficients[1][1], 0.0)
        recommendation = model.recommend("u2", k=1)
        self.assertEqual(recommendation[0].item_id, "b")
        self.assertAlmostEqual(recommendation[0].score, 1.0 / 3.0, places=12)

    def test_unknown_user_uses_popularity_and_known_scores_are_deterministic(self) -> None:
        model = EASE(regularization=1.0).fit(training())
        self.assertEqual([row.item_id for row in model.recommend("new", 2)], ["a", "b"])
        first = model.score_items("u1")
        second = model.score_items("u1", reversed(model.catalog))
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["a"], 0.5, places=12)
        self.assertAlmostEqual(first["b"], 1.0 / 3.0, places=12)

    def test_binary_fit_is_order_independent_and_records_diagnostics(self) -> None:
        data = training()
        first = EASE(regularization=1.0).fit(data)
        second = EASE(regularization=1.0).fit(InteractionDataset(reversed(data.interactions)))
        self.assertEqual(first.to_state(), second.to_state())
        self.assertEqual(first.training_interactions, 3)
        self.assertEqual(first.work_units, 29)
        self.assertLess(first.inverse_residual, 1e-12)

    def test_extreme_finite_weights_preserve_fit_load_order_independence(self) -> None:
        rows = [
            Interaction("u1", "a", 1e16),
            Interaction("u2", "a", 1.0),
            Interaction("u3", "a", 1.0),
            Interaction("u1", "b", 2.0),
        ]
        forward = EASE(regularization=1.0).fit(InteractionDataset(rows))
        backward = EASE(regularization=1.0).fit(InteractionDataset(reversed(rows)))
        self.assertEqual(forward.to_state(), backward.to_state())
        self.assertEqual(EASE.from_state(forward.to_state()).to_state(), backward.to_state())
        self.assertEqual(forward.score_items("new"), backward.score_items("new"))

    def test_repeated_events_are_binary_for_regression(self) -> None:
        base = training()
        repeated = InteractionDataset((*base.interactions, Interaction("u2", "a", value=5.0)))
        first = EASE(regularization=1.0).fit(base).to_state()["model"]["coefficients"]
        second = EASE(regularization=1.0).fit(repeated).to_state()["model"]["coefficients"]
        self.assertEqual(first, second)

    def test_resource_limits_fail_before_matrix_construction(self) -> None:
        data = training()
        with self.assertRaisesRegex(ValidationError, "at most 1 catalog"):
            EASE(regularization=1.0, max_items=1).fit(data)
        with self.assertRaisesRegex(ValidationError, "at most 2 interactions"):
            EASE(regularization=1.0, max_interactions=2).fit(data)
        with self.assertRaisesRegex(ValidationError, "above max_work_units=28"):
            EASE(regularization=1.0, max_work_units=28).fit(data)
        accepted = EASE(regularization=1.0, max_work_units=29).fit(data)
        self.assertEqual(accepted.work_units, 29)

    def test_fit_rejects_states_its_loader_cannot_accept(self) -> None:
        too_many_users = InteractionDataset(
            Interaction(user_id, "a") for user_id in range(MAX_EASE_USERS + 1)
        )
        with self.assertRaisesRegex(ValidationError, "user limit"):
            EASE(regularization=1.0).fit(too_many_users)

        too_long_id = InteractionDataset((Interaction("u" * 16_385, "a"),))
        with self.assertRaisesRegex(ValidationError, "UTF-8 limit"):
            EASE(regularization=1.0).fit(too_long_id)

        too_many_digits = InteractionDataset((Interaction(10**4096, "a"),))
        with self.assertRaisesRegex(ValidationError, "JSON digit limit"):
            EASE(regularization=1.0).fit(too_many_digits)

    def test_fit_checks_aggregate_identifier_and_serialized_file_budgets(self) -> None:
        with (
            mock.patch("orchidrec.models.ease.MAX_EASE_TOTAL_IDENTIFIER_BYTES", 2),
            self.assertRaisesRegex(ValidationError, "aggregate UTF-8 byte limit"),
        ):
            EASE(regularization=1.0).fit(training())
        with (
            mock.patch("orchidrec.models.ease.MAX_MODEL_FILE_BYTES", 100),
            self.assertRaisesRegex(ValidationError, "model-file limit"),
        ):
            EASE(regularization=1.0).fit(training())
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ease.json"
            save_model(EASE(regularization=1.0).fit(training()), destination)
            exact_bytes = destination.stat().st_size
        with mock.patch("orchidrec.models.ease.MAX_MODEL_FILE_BYTES", exact_bytes):
            EASE(regularization=1.0).fit(training())
        with (
            mock.patch("orchidrec.models.ease.MAX_MODEL_FILE_BYTES", exact_bytes - 1),
            self.assertRaisesRegex(ValidationError, "model-file limit"),
        ):
            EASE(regularization=1.0).fit(training())

    def test_parameters_are_bounded_and_boolean_safe(self) -> None:
        invalid = (
            {"regularization": 0.0},
            {"regularization": math.inf},
            {"regularization": True},
            {"max_items": 0},
            {"max_items": True},
            {"max_interactions": 2_000_001},
            {"max_work_units": 1_000_000_001},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValidationError):
                EASE(**parameters)

    def test_unfitted_diagnostics_and_state_are_guarded(self) -> None:
        model = EASE()
        for access in (
            lambda: model.training_interactions,
            lambda: model.work_units,
            lambda: model.inverse_residual,
            model.to_state,
        ):
            with self.subTest(access=access), self.assertRaises(NotFittedError):
                access()

    def test_json_round_trip_preserves_scores(self) -> None:
        model = EASE(regularization=1.0).fit(training())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ease.json"
            save_model(model, path)
            restored = load_model(path)
        self.assertIsInstance(restored, EASE)
        self.assertEqual(restored.to_state(), model.to_state())
        self.assertEqual(restored.recommend("u2", 1), model.recommend("u2", 1))

    def test_mixed_identifier_types_have_stable_binary_state(self) -> None:
        data = InteractionDataset(
            (
                Interaction("user", "1"),
                Interaction("user", 1),
                Interaction(2, "1"),
            )
        )
        model = EASE(regularization=1.0).fit(data)
        self.assertEqual(model.catalog, (1, "1"))
        restored = EASE.from_state(model.to_state())
        self.assertEqual(restored.score_items("user"), model.score_items("user"))

    def test_state_rejects_inconsistent_dimensions_and_diagnostics(self) -> None:
        state = EASE(regularization=1.0).fit(training()).to_state()
        mutations: tuple[tuple[str, object], ...] = (
            ("parameters", {"regularization": 1.0}),
            ("coefficients", [[0.0]]),
            ("training_interactions", 1),
            ("work_units", 30),
            ("inverse_residual", -1.0),
        )
        for field, value in mutations:
            changed = copy.deepcopy(state)
            if field == "parameters":
                changed[field] = value
            else:
                changed["model"][field] = value
            with self.subTest(field=field), self.assertRaises(SerializationError):
                EASE.from_state(changed)

        diagonal = copy.deepcopy(state)
        diagonal["model"]["coefficients"][0][0] = 0.1
        with self.assertRaisesRegex(SerializationError, "diagonal"):
            EASE.from_state(diagonal)
        non_finite = copy.deepcopy(state)
        non_finite["model"]["coefficients"][0][1] = math.nan
        with self.assertRaisesRegex(SerializationError, "finite"):
            EASE.from_state(non_finite)

    def test_state_rejects_malformed_model_shape_and_numeric_types(self) -> None:
        original = EASE(regularization=1.0).fit(training()).to_state()
        variants = []
        unknown_field = copy.deepcopy(original)
        unknown_field["model"]["unexpected"] = 1
        variants.append(("model state", unknown_field))
        short_row = copy.deepcopy(original)
        short_row["model"]["coefficients"][0].pop()
        variants.append(("column count", short_row))
        boolean_coefficient = copy.deepcopy(original)
        boolean_coefficient["model"]["coefficients"][0][1] = True
        variants.append(("finite numbers", boolean_coefficient))
        boolean_residual = copy.deepcopy(original)
        boolean_residual["model"]["inverse_residual"] = True
        variants.append(("inverse_residual", boolean_residual))
        for expected, state in variants:
            with (
                self.subTest(expected=expected),
                self.assertRaisesRegex(SerializationError, expected),
            ):
                EASE.from_state(state)

    def test_valid_json_rejects_single_ulp_coefficient_and_residual_tampering(self) -> None:
        original = EASE(regularization=1.0).fit(training()).to_state()
        coefficient = copy.deepcopy(original)
        current = coefficient["model"]["coefficients"][0][1]
        coefficient["model"]["coefficients"][0][1] = math.nextafter(current, math.inf)
        residual = copy.deepcopy(original)
        current_residual = residual["model"]["inverse_residual"]
        residual["model"]["inverse_residual"] = math.nextafter(current_residual, math.inf)
        with tempfile.TemporaryDirectory() as directory:
            for name, changed, expected in (
                ("coefficient", coefficient, "coefficients.*canonical"),
                ("residual", residual, "inverse_residual.*canonical"),
            ):
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps(changed, allow_nan=False), encoding="utf-8")
                with self.subTest(name=name), self.assertRaisesRegex(SerializationError, expected):
                    load_model(path)

    def test_state_rebuilds_the_canonical_linear_system_once(self) -> None:
        original = EASE(regularization=1.0).fit(training()).to_state()
        with mock.patch("orchidrec.models.ease._invert_spd", wraps=_invert_spd) as invert:
            restored = EASE.from_state(original)
        self.assertEqual(invert.call_count, 1)
        self.assertEqual(restored.to_state(), original)

    def test_state_preflight_rejects_user_fanout_before_base_restoration(self) -> None:
        state = EASE(regularization=1.0).fit(training()).to_state()
        self.assertLess(MAX_EASE_USERS, 50_000)
        state["base"]["users"] = [state["base"]["users"][0]] * 50_000
        tracemalloc.start()
        try:
            with (
                mock.patch.object(
                    EASE,
                    "_restore_base_state",
                    side_effect=AssertionError("base restorer must not run"),
                ) as restore,
                self.assertRaisesRegex(SerializationError, "user limit"),
            ):
                EASE.from_state(state)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        restore.assert_not_called()
        self.assertLess(peak, 4_000_000)

    def test_state_preflight_rejects_structure_utf8_and_work_before_restore(self) -> None:
        original = EASE(regularization=1.0, max_work_units=29).fit(training()).to_state()
        variants = []
        malformed = copy.deepcopy(original)
        malformed["base"]["users"][0]["seen"] = ("a", "b")
        variants.append(("arrays", malformed))
        invalid_unicode = copy.deepcopy(original)
        invalid_unicode["base"]["catalog"][0] = "\ud800"
        variants.append(("Unicode", invalid_unicode))
        excessive_work = copy.deepcopy(original)
        excessive_work["base"]["users"].append({"user_id": "u3", "seen": ["a", "b"]})
        variants.append(("work", excessive_work))
        for expected, state in variants:
            with (
                self.subTest(expected=expected),
                mock.patch.object(
                    EASE,
                    "_restore_base_state",
                    side_effect=AssertionError("base restorer must not run"),
                ) as restore,
                self.assertRaisesRegex(SerializationError, expected),
            ):
                EASE.from_state(state)
            restore.assert_not_called()

    def test_state_preflight_rejects_malformed_arrays_and_tight_limits(self) -> None:
        original = EASE(regularization=1.0).fit(training()).to_state()
        variants: tuple[tuple[str, str, object], ...] = (
            ("base", "popularity", None),
            ("base", "catalog", []),
            ("base", "popularity", []),
            ("parameters", "max_work_units", 1),
            ("parameters", "max_interactions", 1),
        )
        for section, field, value in variants:
            changed = copy.deepcopy(original)
            changed[section][field] = value
            with (
                self.subTest(section=section, field=field, value=value),
                self.assertRaises(SerializationError),
            ):
                EASE.from_state(changed)
        missing = copy.deepcopy(original)
        del missing["base"]["popularity"]
        with self.assertRaisesRegex(SerializationError, "invalid fields"):
            EASE.from_state(missing)
        empty_seen = copy.deepcopy(original)
        empty_seen["base"]["users"][0]["seen"] = []
        with self.assertRaisesRegex(SerializationError, "invalid size"):
            EASE.from_state(empty_seen)

    def test_state_preflight_rejects_catalog_bytes_and_unknown_seen_item(self) -> None:
        original = EASE(regularization=1.0).fit(training()).to_state()
        with (
            mock.patch("orchidrec.models.ease.MAX_EASE_TOTAL_IDENTIFIER_BYTES", 1),
            self.assertRaisesRegex(SerializationError, "aggregate UTF-8 byte limit"),
        ):
            EASE.from_state(original)
        unknown_item = copy.deepcopy(original)
        unknown_item["base"]["users"][0]["seen"][0] = "unknown"
        with self.assertRaisesRegex(SerializationError, "invalid item"):
            EASE.from_state(unknown_item)

    def test_integer_subclasses_cannot_bypass_constructor_or_state_bounds(self) -> None:
        class LyingInt(int):
            def __le__(self, other: object) -> bool:
                return True

            def __ge__(self, other: object) -> bool:
                return True

            def __eq__(self, other: object) -> bool:
                return True

        for name in ("max_items", "max_interactions", "max_work_units"):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                EASE(**{name: LyingInt(-1)})

        original = EASE(regularization=1.0).fit(training()).to_state()
        for name in ("training_interactions", "work_units"):
            changed = copy.deepcopy(original)
            changed["model"][name] = LyingInt(-1)
            with self.subTest(name=name), self.assertRaises(SerializationError):
                EASE.from_state(changed)


class EaseIntegrationTests(unittest.TestCase):
    def test_strict_configuration_and_factory_accept_only_ease_parameters(self) -> None:
        config = config_from_dict(
            {
                "data": {"path": "events.json"},
                "model": {
                    "name": "ease",
                    "params": {"regularization": 5.0, "max_items": 32},
                },
            }
        )
        model = build_model(config.model.name, config.model.params, experiment_seed=7)
        self.assertIsInstance(model, EASE)
        self.assertEqual(model.regularization, 5.0)
        for params in ({"regularization": 0}, {"factors": 2}):
            with self.subTest(params=params), self.assertRaises(ConfigurationError):
                config_from_dict(
                    {
                        "data": {"path": "events.json"},
                        "model": {"name": "ease", "params": params},
                    }
                )

    def test_benchmark_grid_selects_and_refits_ease_on_shared_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            demo_dataset().save_json(path)
            config = benchmark_config_from_dict(
                {
                    "schema_version": 1,
                    "seed": 17,
                    "data": {"path": str(path), "format": "orchidrec-json"},
                    "evaluation": {"k": 3, "bootstrap_samples": 10},
                    "tuning": {
                        "selection_metric": "ndcg",
                        "validation_split": {"method": "leave_one_out"},
                        "implicit_mf_seeds": [17],
                    },
                    "models": [
                        {"label": "pop", "name": "popularity"},
                        {
                            "label": "ease",
                            "name": "ease",
                            "params": {"max_items": 32},
                            "grid": {"regularization": [1.0, 10.0]},
                        },
                    ],
                }
            )
            result = run_benchmark(config)
        self.assertEqual([entry.model_type for entry in result.models], ["popularity", "ease"])
        self.assertIsNotNone(result.tuning)
        assert result.tuning is not None
        ease = next(entry for entry in result.tuning.models if entry.model_type == "ease")
        self.assertEqual(len(ease.trials), 2)
        self.assertEqual(ease.final_parameters["max_items"], 32)


if __name__ == "__main__":
    unittest.main()
