from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import NotFittedError, SerializationError
from orchidrec.models import (
    ImplicitMF,
    ItemKNN,
    Popularity,
    load_model,
    model_from_state,
    save_model,
)


def dataset() -> InteractionDataset:
    return InteractionDataset(
        [
            Interaction("u1", "a"),
            Interaction("u1", "b"),
            Interaction("u2", "a"),
            Interaction("u2", "c"),
            Interaction("u3", "b"),
        ]
    )


class ModelSerializationTests(unittest.TestCase):
    def models(self):
        return [
            Popularity().fit(dataset()),
            ItemKNN(neighbors=2, shrinkage=1).fit(dataset()),
            ImplicitMF(factors=3, epochs=3, seed=5).fit(dataset()),
        ]

    def test_state_round_trip_preserves_recommendations(self) -> None:
        for model in self.models():
            with self.subTest(model=type(model).__name__):
                restored = model_from_state(model.to_state())
                self.assertEqual(restored.recommend("u1", 3), model.recommend("u1", 3))

    def test_file_round_trip_preserves_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for model in self.models():
                path = Path(directory) / f"{model.model_type}.json"
                save_model(model, path)
                self.assertEqual(load_model(path).to_state(), model.to_state())

    def test_serialized_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = ItemKNN(neighbors=2).fit(dataset())
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            save_model(model, first)
            save_model(model, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_unknown_model_type_is_rejected(self) -> None:
        state = Popularity().fit(dataset()).to_state()
        state["model_type"] = "mystery"
        with self.assertRaises(SerializationError):
            model_from_state(state)

    def test_wrong_schema_version_is_rejected(self) -> None:
        state = Popularity().fit(dataset()).to_state()
        state["schema_version"] = 999
        with self.assertRaises(SerializationError):
            model_from_state(state)

    def test_boolean_schema_version_is_rejected(self) -> None:
        state = Popularity().fit(dataset()).to_state()
        state["schema_version"] = True
        with self.assertRaises(SerializationError):
            model_from_state(state)

    def test_nonfinite_factor_value_is_rejected(self) -> None:
        state = ImplicitMF(factors=3, epochs=1).fit(dataset()).to_state()
        state["model"]["item_factors"][0][0] = float("nan")
        with self.assertRaises(SerializationError):
            model_from_state(state)

    def test_huge_integer_factor_value_is_rejected(self) -> None:
        state = ImplicitMF(factors=3, epochs=1).fit(dataset()).to_state()
        state["model"]["item_factors"][0][0] = 10**400
        with self.assertRaises(SerializationError):
            model_from_state(state)

    def test_zero_base_popularity_is_rejected(self) -> None:
        state = Popularity().fit(dataset()).to_state()
        state["base"]["popularity"][0] = 0.0
        with self.assertRaises(SerializationError):
            model_from_state(state)

    def test_empty_seen_item_state_is_rejected(self) -> None:
        state = Popularity().fit(dataset()).to_state()
        state["base"]["users"][0]["seen"] = []
        with self.assertRaises(SerializationError):
            model_from_state(state)

    def test_out_of_range_item_knn_similarity_is_rejected(self) -> None:
        state = ItemKNN(neighbors=2, shrinkage=0).fit(dataset()).to_state()
        nonempty_row = next(row for row in state["model"]["similarities"] if row)
        nonempty_row[0]["score"] = 2.0
        with self.assertRaises(SerializationError):
            model_from_state(state)

    def test_malformed_base_state_variants_are_rejected(self) -> None:
        original = Popularity().fit(dataset()).to_state()

        def reject(name, mutate) -> None:
            state = copy.deepcopy(original)
            mutate(state)
            with self.subTest(name=name), self.assertRaises(SerializationError):
                model_from_state(state)

        reject("unknown base field", lambda state: state["base"].update({"extra": 1}))
        reject("malformed arrays", lambda state: state["base"].update({"users": "bad"}))
        reject("unsorted catalog", lambda state: state["base"]["catalog"].reverse())
        reject("popularity length", lambda state: state["base"]["popularity"].pop())
        reject("boolean popularity", lambda state: state["base"]["popularity"].__setitem__(0, True))
        reject("malformed user", lambda state: state["base"]["users"].__setitem__(0, {}))
        reject("invalid user ID", lambda state: state["base"]["users"][0].update({"user_id": True}))
        reject(
            "duplicate user",
            lambda state: state["base"]["users"].append(copy.deepcopy(state["base"]["users"][0])),
        )
        reject("unknown seen item", lambda state: state["base"]["users"][0].update({"seen": ["z"]}))
        reject("unstable user order", lambda state: state["base"]["users"].reverse())

    def test_malformed_popularity_state_variants_are_rejected(self) -> None:
        original = Popularity().fit(dataset()).to_state()
        variants = []
        state = copy.deepcopy(original)
        state["parameters"]["weighted"] = 1
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["scores"] = "bad"
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["scores"].pop()
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["scores"][0] = True
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["scores"][0] = -1.0
        variants.append(state)
        for state in variants:
            with self.subTest(state=state), self.assertRaises(SerializationError):
                model_from_state(state)

    def test_malformed_item_knn_state_variants_are_rejected(self) -> None:
        original = ItemKNN(neighbors=2, shrinkage=0).fit(dataset()).to_state()
        variants = []
        state = copy.deepcopy(original)
        state["parameters"]["neighbors"] = 0
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["extra"] = []
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["similarities"] = "bad"
        variants.append(state)
        state = copy.deepcopy(original)
        nonempty_row = next(row for row in state["model"]["similarities"] if row)
        nonempty_row[0] = {}
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["user_values"][0]["user_id"] = True
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["user_values"][0]["values"][0]["value"] = True
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["user_values"].pop()
        variants.append(state)
        for state in variants:
            with self.subTest(state=state), self.assertRaises(SerializationError):
                model_from_state(state)

    def test_malformed_implicit_mf_state_variants_are_rejected(self) -> None:
        original = ImplicitMF(factors=3, epochs=1).fit(dataset()).to_state()
        variants = []
        state = copy.deepcopy(original)
        state["parameters"]["factors"] = 0
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["extra"] = []
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["user_factors"].pop()
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["item_factors"][0].pop()
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["item_bias"].pop()
        variants.append(state)
        state = copy.deepcopy(original)
        state["model"]["item_bias"][0] = True
        variants.append(state)
        for state in variants:
            with self.subTest(state=state), self.assertRaises(SerializationError):
                model_from_state(state)

    def test_unknown_top_level_field_is_rejected(self) -> None:
        state = Popularity().fit(dataset()).to_state()
        state["surprise"] = True
        with self.assertRaises(SerializationError):
            model_from_state(state)

    def test_tampered_factor_shape_is_rejected(self) -> None:
        state = ImplicitMF(factors=3, epochs=1).fit(dataset()).to_state()
        state["model"]["item_factors"][0] = [1.0]
        with self.assertRaises(SerializationError):
            model_from_state(state)

    def test_malformed_json_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("[", encoding="utf-8")
            with self.assertRaises(SerializationError):
                load_model(path)

    def test_duplicate_model_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                '{"model_type":"popularity","model_type":"item_knn"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SerializationError, "duplicate JSON object key"):
                load_model(path)

    def test_non_object_model_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaises(SerializationError):
                load_model(path)

    def test_missing_model_file_is_rejected(self) -> None:
        with self.assertRaises(SerializationError):
            load_model("missing-orchidrec-model.json")

    def test_unfitted_model_cannot_be_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(NotFittedError):
            save_model(Popularity(), Path(directory) / "model.json")

    def test_non_model_cannot_be_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SerializationError):
            save_model(object(), Path(directory) / "model.json")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
