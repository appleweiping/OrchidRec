"""Confidence-weighted implicit-feedback matrix factorization by ALS."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from typing import Any, Self

from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, InteractionDataset, stable_id_key
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.models.base import BaseRecommender, make_envelope, parse_envelope

MAX_FACTORS = 64
MAX_EPOCHS = 100
MAX_ALPHA = 1_000_000.0
MAX_REGULARIZATION = 1_000_000.0
MAX_CONFIDENCE = 1_000_000_000_000.0
MAX_INTERACTIONS = 2_000_000
MAX_ENTITIES = 200_000
MAX_FACTOR_ENTRIES = 5_000_000
MAX_ALS_WORK_UNITS = 1_000_000_000
_MIN_REGULARIZATION = 1e-8
_PIVOT_RELATIVE_FLOOR = 1e-14
_RESIDUAL_RELATIVE_TOLERANCE = 1e-8
_OBJECTIVE_RELATIVE_TOLERANCE = 1e-7


def _bounded_positive_int(value: object, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValidationError(f"{name} must be an integer between 1 and {maximum}")
    return value


def _bounded_float(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite number between {minimum} and {maximum}")
    numeric = safe_float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValidationError(f"{name} must be a finite number between {minimum} and {maximum}")
    return numeric


def _checked_sum(values: Iterable[float], context: str) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValidationError(f"{context} overflowed") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{context} is not finite")
    return result


def _dot(left: Sequence[float], right: Sequence[float], context: str) -> float:
    return _checked_sum(
        (left_value * right_value for left_value, right_value in zip(left, right, strict=True)),
        context,
    )


def _gram(vectors: Sequence[Sequence[float]], factors: int) -> list[list[float]]:
    """Return ``V.T @ V`` using a fixed summation order."""

    result = [[0.0] * factors for _ in range(factors)]
    for row in range(factors):
        for column in range(row + 1):
            value = _checked_sum(
                (vector[row] * vector[column] for vector in vectors),
                "ALS Gram matrix",
            )
            result[row][column] = value
            result[column][row] = value
    return result


def _solve_spd(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve a small symmetric-positive-definite system by checked Cholesky."""

    size = len(rhs)
    if size == 0 or len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValidationError("ALS normal equation has invalid dimensions")
    if any(not math.isfinite(value) for row in matrix for value in row) or any(
        not math.isfinite(value) for value in rhs
    ):
        raise ValidationError("ALS normal equation contains a non-finite value")
    for row in range(size):
        for column in range(row):
            scale = max(1.0, abs(matrix[row][column]), abs(matrix[column][row]))
            if abs(matrix[row][column] - matrix[column][row]) > 1e-12 * scale:
                raise ValidationError("ALS normal equation is not symmetric")

    largest_diagonal = max(abs(matrix[index][index]) for index in range(size))
    if largest_diagonal == 0.0:
        raise ValidationError("ALS normal equation is singular")
    pivot_floor = max(1e-15, largest_diagonal * _PIVOT_RELATIVE_FLOOR)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            products = (lower[row][index] * lower[column][index] for index in range(column))
            remainder = matrix[row][column] - _checked_sum(products, "ALS factorization")
            if not math.isfinite(remainder):
                raise ValidationError("ALS factorization produced a non-finite pivot")
            if row == column:
                if remainder <= pivot_floor:
                    raise ValidationError("ALS normal equation is singular or ill-conditioned")
                lower[row][column] = math.sqrt(remainder)
            else:
                value = remainder / lower[column][column]
                if not math.isfinite(value):
                    raise ValidationError("ALS factorization produced a non-finite value")
                lower[row][column] = value

    forward = [0.0] * size
    for row in range(size):
        numerator = rhs[row] - _checked_sum(
            (lower[row][column] * forward[column] for column in range(row)),
            "ALS forward substitution",
        )
        forward[row] = numerator / lower[row][row]
        if not math.isfinite(forward[row]):
            raise ValidationError("ALS forward substitution produced a non-finite value")

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        numerator = forward[row] - _checked_sum(
            (lower[column][row] * solution[column] for column in range(row + 1, size)),
            "ALS backward substitution",
        )
        solution[row] = numerator / lower[row][row]
        if not math.isfinite(solution[row]):
            raise ValidationError("ALS solve produced a non-finite factor")

    max_residual = 0.0
    max_row_norm = 0.0
    for row in range(size):
        estimate = _dot(matrix[row], solution, "ALS residual")
        max_residual = max(max_residual, abs(estimate - rhs[row]))
        max_row_norm = max(
            max_row_norm, _checked_sum((abs(value) for value in matrix[row]), "ALS matrix norm")
        )
    scale = (
        1.0
        + max(abs(value) for value in rhs)
        + max_row_norm * max(abs(value) for value in solution)
    )
    if not math.isfinite(scale):
        raise ValidationError("ALS residual scale overflowed")
    if max_residual > _RESIDUAL_RELATIVE_TOLERANCE * scale:
        raise ValidationError(
            "ALS normal equation solve failed its residual check "
            f"({max_residual:g} > {_RESIDUAL_RELATIVE_TOLERANCE * scale:g})"
        )
    return solution


class ConfidenceALS(BaseRecommender):
    """Factor implicit feedback with Hu-Koren-Volinsky confidence weighting.

    Every observed pair has preference ``p_ui = 1`` and confidence
    ``c_ui = 1 + alpha * sum(value)``. Unobserved pairs retain preference zero
    and confidence one. Each alternating step solves the exact regularized
    normal equation for one user or item.
    """

    model_type = "confidence_als"

    def __init__(
        self,
        *,
        factors: int = 16,
        epochs: int = 5,
        alpha: float = 40.0,
        regularization: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.factors = _bounded_positive_int(factors, "factors", MAX_FACTORS)
        self.epochs = _bounded_positive_int(epochs, "epochs", MAX_EPOCHS)
        self.alpha = _bounded_float(alpha, "alpha", minimum=0.0, maximum=MAX_ALPHA)
        self.regularization = _bounded_float(
            regularization,
            "regularization",
            minimum=_MIN_REGULARIZATION,
            maximum=MAX_REGULARIZATION,
        )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValidationError("seed must be an integer")
        self.seed = seed
        self._user_factors: dict[EntityId, list[float]] = {}
        self._item_factors: dict[EntityId, list[float]] = {}
        self._objective_history: tuple[float, ...] = ()

    @property
    def objective_history(self) -> tuple[float, ...]:
        """Return the initial and post-epoch confidence objective values."""

        self._require_fitted()
        return self._objective_history

    def _validate_problem_size(self, user_count: int, item_count: int, events: int | None) -> None:
        if events is not None and events > MAX_INTERACTIONS:
            raise ValidationError(f"confidence_als accepts at most {MAX_INTERACTIONS} interactions")
        if user_count + item_count > MAX_ENTITIES:
            raise ValidationError(f"confidence_als accepts at most {MAX_ENTITIES} total entities")
        factor_entries = (user_count + item_count) * self.factors
        if factor_entries > MAX_FACTOR_ENTRIES:
            raise ValidationError(
                f"confidence_als factor state exceeds {MAX_FACTOR_ENTRIES} numeric entries"
            )
        work_units = self.epochs * (user_count + item_count) * self.factors**3
        if work_units > MAX_ALS_WORK_UNITS:
            raise ValidationError(
                "confidence_als configuration exceeds the dependency-free ALS work limit"
            )

    def _aggregate_confidence(
        self, dataset: InteractionDataset
    ) -> tuple[
        dict[EntityId, tuple[tuple[EntityId, float], ...]],
        dict[EntityId, tuple[tuple[EntityId, float], ...]],
    ]:
        values: dict[EntityId, dict[EntityId, list[float]]] = defaultdict(lambda: defaultdict(list))
        for event in dataset:
            values[event.user_id][event.item_id].append(event.value)
        by_user: dict[EntityId, tuple[tuple[EntityId, float], ...]] = {}
        reverse: dict[EntityId, list[tuple[EntityId, float]]] = defaultdict(list)
        for user_id in sorted(values, key=stable_id_key):
            rows: list[tuple[EntityId, float]] = []
            for item_id in sorted(values[user_id], key=stable_id_key):
                aggregate = _checked_sum(
                    sorted(values[user_id][item_id]),
                    f"aggregate interaction value for {(user_id, item_id)!r}",
                )
                confidence = 1.0 + self.alpha * aggregate
                if not math.isfinite(confidence) or confidence > MAX_CONFIDENCE:
                    raise ValidationError(
                        f"confidence for {(user_id, item_id)!r} exceeds {MAX_CONFIDENCE:g}"
                    )
                rows.append((item_id, confidence))
                reverse[item_id].append((user_id, confidence))
            by_user[user_id] = tuple(rows)
        by_item = {
            item_id: tuple(sorted(reverse[item_id], key=lambda row: stable_id_key(row[0])))
            for item_id in self._catalog
        }
        return by_user, by_item

    def _normal_equation(
        self,
        gram: Sequence[Sequence[float]],
        observations: Sequence[tuple[EntityId, float]],
        other_factors: Mapping[EntityId, Sequence[float]],
    ) -> tuple[list[list[float]], list[float]]:
        matrix = [list(row) for row in gram]
        rhs = [0.0] * self.factors
        for index in range(self.factors):
            matrix[index][index] += self.regularization
        for entity_id, confidence in observations:
            vector = other_factors[entity_id]
            adjustment = confidence - 1.0
            for row in range(self.factors):
                rhs[row] += confidence * vector[row]
                if not math.isfinite(rhs[row]):
                    raise ValidationError("ALS normal-equation right-hand side overflowed")
                for column in range(row + 1):
                    value = matrix[row][column] + adjustment * vector[row] * vector[column]
                    if not math.isfinite(value):
                        raise ValidationError("ALS normal-equation matrix overflowed")
                    matrix[row][column] = value
                    matrix[column][row] = value
        return matrix, rhs

    def _objective(
        self,
        by_user: Mapping[EntityId, Sequence[tuple[EntityId, float]]],
    ) -> float:
        item_vectors = [self._item_factors[item_id] for item_id in self._catalog]
        item_gram = _gram(item_vectors, self.factors)
        terms: list[float] = []
        for user_id in sorted(self._user_factors, key=stable_id_key):
            user = self._user_factors[user_id]
            gram_user = [_dot(row, user, "ALS objective") for row in item_gram]
            terms.append(_dot(user, gram_user, "ALS objective"))
            for item_id, confidence in by_user[user_id]:
                prediction = _dot(user, self._item_factors[item_id], "ALS prediction")
                error = 1.0 - prediction
                terms.append(confidence * error * error - prediction * prediction)
        terms.append(
            self.regularization
            * _checked_sum(
                (
                    value * value
                    for vector in chain(self._user_factors.values(), self._item_factors.values())
                    for value in vector
                ),
                "ALS regularization",
            )
        )
        objective = _checked_sum(terms, "ALS objective")
        tolerance = _OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, abs(objective))
        if objective < -tolerance:
            raise ValidationError("confidence_als produced a negative objective")
        return max(0.0, objective)

    def _fit_model(self, dataset: InteractionDataset) -> None:
        users = tuple(sorted(self._seen, key=stable_id_key))
        self._validate_problem_size(len(users), len(self._catalog), len(dataset))
        by_user, by_item = self._aggregate_confidence(dataset)
        generator = random.Random(self.seed)
        scale = 0.01 / math.sqrt(self.factors)
        self._user_factors = {
            user_id: [generator.uniform(-scale, scale) for _ in range(self.factors)]
            for user_id in users
        }
        self._item_factors = {
            item_id: [generator.uniform(-scale, scale) for _ in range(self.factors)]
            for item_id in self._catalog
        }
        objectives = [self._objective(by_user)]
        for _epoch in range(self.epochs):
            item_gram = _gram(
                [self._item_factors[item_id] for item_id in self._catalog], self.factors
            )
            for user_id in users:
                matrix, rhs = self._normal_equation(item_gram, by_user[user_id], self._item_factors)
                self._user_factors[user_id] = _solve_spd(matrix, rhs)

            user_gram = _gram([self._user_factors[user_id] for user_id in users], self.factors)
            for item_id in self._catalog:
                matrix, rhs = self._normal_equation(user_gram, by_item[item_id], self._user_factors)
                self._item_factors[item_id] = _solve_spd(matrix, rhs)

            objective = self._objective(by_user)
            previous = objectives[-1]
            tolerance = _OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, abs(previous))
            if objective > previous + tolerance:
                raise ValidationError("confidence_als objective increased during exact ALS updates")
            objectives.append(objective)
        self._objective_history = tuple(objectives)

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        user = self._user_factors.get(user_id)
        if user is None:
            return self._popularity_fallback(item_id)
        score = _dot(user, self._item_factors[item_id], "confidence_als score")
        return score + self._popularity_fallback(item_id) * 1e-12

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        users = sorted(self._seen, key=stable_id_key)
        return make_envelope(
            self.model_type,
            {
                "factors": self.factors,
                "epochs": self.epochs,
                "alpha": self.alpha,
                "regularization": self.regularization,
                "seed": self.seed,
            },
            self._base_state(),
            {
                "user_factors": [self._user_factors[user_id] for user_id in users],
                "item_factors": [self._item_factors[item_id] for item_id in self._catalog],
                "objective_history": list(self._objective_history),
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
            for raw_value in row:
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    raise SerializationError(f"{name} values must be finite numbers")
                value = safe_float(raw_value)
                if not math.isfinite(value):
                    raise SerializationError(f"{name} values must be finite numbers")
                converted.append(value)
            matrix.append(converted)
        return matrix

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        if set(parameters) != {"factors", "epochs", "alpha", "regularization", "seed"}:
            raise SerializationError("ConfidenceALS parameters are malformed")
        try:
            instance = cls(**parameters)
        except (TypeError, ValidationError) as exc:
            raise SerializationError(f"invalid ConfidenceALS parameters: {exc}") from exc
        instance._restore_base_state(base)
        users = sorted(instance._seen, key=stable_id_key)
        try:
            instance._validate_problem_size(len(users), len(instance._catalog), None)
        except ValidationError as exc:
            raise SerializationError(f"invalid ConfidenceALS problem size: {exc}") from exc
        if set(model) != {"user_factors", "item_factors", "objective_history"}:
            raise SerializationError("ConfidenceALS model state is malformed")
        user_matrix = cls._restore_matrix(
            model["user_factors"], len(users), instance.factors, "user_factors"
        )
        item_matrix = cls._restore_matrix(
            model["item_factors"], len(instance._catalog), instance.factors, "item_factors"
        )
        raw_objectives = model["objective_history"]
        if not isinstance(raw_objectives, list) or len(raw_objectives) != instance.epochs + 1:
            raise SerializationError("objective_history length is malformed")
        objectives: list[float] = []
        for raw_value in raw_objectives:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise SerializationError(
                    "objective_history values must be finite non-negative numbers"
                )
            value = safe_float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise SerializationError(
                    "objective_history values must be finite non-negative numbers"
                )
            if objectives:
                tolerance = _OBJECTIVE_RELATIVE_TOLERANCE * max(1.0, abs(objectives[-1]))
                if value > objectives[-1] + tolerance:
                    raise SerializationError("objective_history must be non-increasing")
            objectives.append(value)
        instance._user_factors = dict(zip(users, user_matrix, strict=True))
        instance._item_factors = dict(zip(instance._catalog, item_matrix, strict=True))
        instance._objective_history = tuple(objectives)
        return instance
