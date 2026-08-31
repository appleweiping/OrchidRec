"""Deterministic Bayesian personalized ranking with latent factors."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any, Self

from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, InteractionDataset, stable_id_key
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.models.base import BaseRecommender, make_envelope, parse_envelope


class ImplicitMF(BaseRecommender):
    """Implicit matrix factorization trained with pairwise BPR updates."""

    model_type = "implicit_mf"

    def __init__(
        self,
        *,
        factors: int = 16,
        epochs: int = 20,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        negative_samples: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if isinstance(factors, bool) or not isinstance(factors, int) or factors <= 0:
            raise ValidationError("factors must be a positive integer")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValidationError("epochs must be a positive integer")
        if isinstance(negative_samples, bool) or not isinstance(negative_samples, int) or negative_samples <= 0:
            raise ValidationError("negative_samples must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValidationError("seed must be an integer")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise ValidationError("learning_rate must be a finite positive number")
        if isinstance(regularization, bool) or not isinstance(regularization, (int, float)):
            raise ValidationError("regularization must be a finite non-negative number")
        numeric_rate = safe_float(learning_rate)
        numeric_regularization = safe_float(regularization)
        if not math.isfinite(numeric_rate) or numeric_rate <= 0:
            raise ValidationError("learning_rate must be a finite positive number")
        if not math.isfinite(numeric_regularization) or numeric_regularization < 0:
            raise ValidationError("regularization must be a finite non-negative number")
        self.factors = factors
        self.epochs = epochs
        self.learning_rate = numeric_rate
        self.regularization = numeric_regularization
        self.negative_samples = negative_samples
        self.seed = seed
        self._user_factors: dict[EntityId, list[float]] = {}
        self._item_factors: dict[EntityId, list[float]] = {}
        self._item_bias: dict[EntityId, float] = {}

    def _fit_model(self, dataset: InteractionDataset) -> None:
        del dataset
        rng = random.Random(self.seed)
        users = sorted(self._seen, key=stable_id_key)
        scale = 0.1 / math.sqrt(self.factors)
        self._user_factors = {
            user_id: [rng.uniform(-scale, scale) for _ in range(self.factors)] for user_id in users
        }
        self._item_factors = {
            item_id: [rng.uniform(-scale, scale) for _ in range(self.factors)] for item_id in self._catalog
        }
        self._item_bias = {item_id: 0.0 for item_id in self._catalog}
        positives = [
            (user_id, item_id)
            for user_id in users
            for item_id in sorted(self._seen[user_id], key=stable_id_key)
        ]
        negatives = {
            user_id: tuple(item_id for item_id in self._catalog if item_id not in self._seen[user_id])
            for user_id in users
        }
        for _epoch in range(self.epochs):
            epoch_pairs = positives.copy()
            rng.shuffle(epoch_pairs)
            for user_id, positive_id in epoch_pairs:
                available = negatives[user_id]
                if not available:
                    continue
                for _sample in range(self.negative_samples):
                    negative_id = available[rng.randrange(len(available))]
                    self._update(user_id, positive_id, negative_id)
        values = (
            value
            for vector in (*self._user_factors.values(), *self._item_factors.values())
            for value in vector
        )
        if any(not math.isfinite(value) for value in values) or any(
            not math.isfinite(value) for value in self._item_bias.values()
        ):
            raise ValidationError("ImplicitMF training diverged; use a smaller learning rate")

    def _update(self, user_id: EntityId, positive_id: EntityId, negative_id: EntityId) -> None:
        user = self._user_factors[user_id]
        positive = self._item_factors[positive_id]
        negative = self._item_factors[negative_id]
        difference = self._item_bias[positive_id] - self._item_bias[negative_id]
        difference += sum(u * (p - n) for u, p, n in zip(user, positive, negative, strict=True))
        if difference >= 0:
            exp_negative = math.exp(-difference)
            gradient = exp_negative / (1.0 + exp_negative)
        else:
            gradient = 1.0 / (1.0 + math.exp(difference))
        old_user = user.copy()
        old_positive = positive.copy()
        old_negative = negative.copy()
        rate = self.learning_rate
        regularization = self.regularization
        for index in range(self.factors):
            user[index] += rate * (
                gradient * (old_positive[index] - old_negative[index]) - regularization * old_user[index]
            )
            positive[index] += rate * (gradient * old_user[index] - regularization * old_positive[index])
            negative[index] += rate * (-gradient * old_user[index] - regularization * old_negative[index])
        self._item_bias[positive_id] += rate * (gradient - regularization * self._item_bias[positive_id])
        self._item_bias[negative_id] += rate * (-gradient - regularization * self._item_bias[negative_id])

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        user = self._user_factors.get(user_id)
        if user is None:
            return self._popularity_fallback(item_id)
        score = self._item_bias[item_id]
        score += sum(
            left * right for left, right in zip(user, self._item_factors[item_id], strict=True)
        )
        return score + self._popularity_fallback(item_id) * 1e-12

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        users = sorted(self._seen, key=stable_id_key)
        return make_envelope(
            self.model_type,
            {
                "factors": self.factors,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "negative_samples": self.negative_samples,
                "seed": self.seed,
            },
            self._base_state(),
            {
                "user_factors": [self._user_factors[user_id] for user_id in users],
                "item_factors": [self._item_factors[item_id] for item_id in self._catalog],
                "item_bias": [self._item_bias[item_id] for item_id in self._catalog],
            },
        )

    @staticmethod
    def _restore_matrix(raw: object, rows: int, columns: int, name: str) -> list[list[float]]:
        if not isinstance(raw, list) or len(raw) != rows:
            raise SerializationError(f"{name} row count is malformed")
        matrix: list[list[float]] = []
        for row in raw:
            if not isinstance(row, list) or len(row) != columns:
                raise SerializationError(f"{name} column count is malformed")
            converted: list[float] = []
            for value in row:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(safe_float(value))
                ):
                    raise SerializationError(f"{name} contains a non-finite numeric value")
                converted.append(safe_float(value))
            matrix.append(converted)
        return matrix

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        expected_parameters = {
            "factors",
            "epochs",
            "learning_rate",
            "regularization",
            "negative_samples",
            "seed",
        }
        if set(parameters) != expected_parameters:
            raise SerializationError("ImplicitMF parameters are malformed")
        try:
            instance = cls(**parameters)
        except (TypeError, ValidationError) as exc:
            raise SerializationError(f"invalid ImplicitMF parameters: {exc}") from exc
        instance._restore_base_state(base)
        if set(model) != {"user_factors", "item_factors", "item_bias"}:
            raise SerializationError("ImplicitMF model state is malformed")
        users = sorted(instance._seen, key=stable_id_key)
        user_matrix = cls._restore_matrix(model["user_factors"], len(users), instance.factors, "user_factors")
        item_matrix = cls._restore_matrix(
            model["item_factors"], len(instance._catalog), instance.factors, "item_factors"
        )
        raw_bias = model["item_bias"]
        if not isinstance(raw_bias, list) or len(raw_bias) != len(instance._catalog):
            raise SerializationError("item_bias length is malformed")
        bias: list[float] = []
        for value in raw_bias:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(safe_float(value))
            ):
                raise SerializationError("item_bias contains a non-finite numeric value")
            bias.append(safe_float(value))
        instance._user_factors = dict(zip(users, user_matrix, strict=True))
        instance._item_factors = dict(zip(instance._catalog, item_matrix, strict=True))
        instance._item_bias = dict(zip(instance._catalog, bias, strict=True))
        return instance
