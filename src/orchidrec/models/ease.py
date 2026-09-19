"""Deterministic closed-form EASE recommender for binary implicit feedback."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Self, cast

from orchidrec._json import MAX_JSON_INTEGER_DIGITS
from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, InteractionDataset, stable_id_key, validate_entity_id
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.models.base import BaseRecommender, make_envelope, parse_envelope
from orchidrec.models.io import MAX_MODEL_FILE_BYTES

MAX_EASE_ITEMS = 512
MAX_EASE_INTERACTIONS = 2_000_000
MAX_EASE_WORK_UNITS = 1_000_000_000
MAX_EASE_USERS = 32_768
MAX_EASE_IDENTIFIER_UTF8_BYTES = 16_384
MAX_EASE_TOTAL_IDENTIFIER_BYTES = 32 * 1024 * 1024
MAX_EASE_REGULARIZATION = 1_000_000_000_000.0
_MIN_EASE_REGULARIZATION = 1e-8
_PIVOT_RELATIVE_FLOOR = 1e-14
_INVERSE_RELATIVE_TOLERANCE = 1e-8


def _bounded_positive_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValidationError(f"{name} must be an integer between 1 and {maximum}")
    return value


def _bounded_regularization(value: object) -> float:
    if type(value) not in {int, float}:
        raise ValidationError(
            "regularization must be a finite number between "
            f"{_MIN_EASE_REGULARIZATION} and {MAX_EASE_REGULARIZATION}"
        )
    numeric = safe_float(cast(int | float, value))
    if (
        not math.isfinite(numeric)
        or numeric < _MIN_EASE_REGULARIZATION
        or numeric > MAX_EASE_REGULARIZATION
    ):
        raise ValidationError(
            "regularization must be a finite number between "
            f"{_MIN_EASE_REGULARIZATION} and {MAX_EASE_REGULARIZATION}"
        )
    return numeric


def _finite_sum(values: Iterable[float], context: str) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValidationError(f"{context} overflowed") from error
    if not math.isfinite(result):
        raise ValidationError(f"{context} is not finite")
    return result


def _normalized_square_matrix(
    matrix: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not isinstance(matrix, (list, tuple)):
        raise ValidationError("EASE system matrix must be a non-empty square matrix")
    size = len(matrix)
    if size == 0:
        raise ValidationError("EASE system matrix must be a non-empty square matrix")
    normalized: list[list[float]] = []
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != size:
            raise ValidationError("EASE system matrix must be a non-empty square matrix")
        converted: list[float] = []
        for value in row:
            if type(value) not in {int, float}:
                raise ValidationError("EASE system matrix must contain finite numbers")
            number = safe_float(value)
            if not math.isfinite(number):
                raise ValidationError("EASE system matrix must contain finite numbers")
            converted.append(number)
        normalized.append(converted)
    return normalized


def _invert_spd(matrix: Sequence[Sequence[float]]) -> tuple[list[list[float]], float]:
    """Invert a small SPD matrix through one checked Cholesky factorization."""

    system = _normalized_square_matrix(matrix)
    size = len(system)
    for row in range(size):
        for column in range(row):
            scale = max(1.0, abs(system[row][column]), abs(system[column][row]))
            if abs(system[row][column] - system[column][row]) > 1e-12 * scale:
                raise ValidationError("EASE system matrix must be symmetric")

    largest_diagonal = max(abs(system[index][index]) for index in range(size))
    if largest_diagonal == 0.0:
        raise ValidationError("EASE system matrix is singular")
    pivot_floor = max(1e-15, largest_diagonal * _PIVOT_RELATIVE_FLOOR)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            products = (lower[row][index] * lower[column][index] for index in range(column))
            remainder = system[row][column] - _finite_sum(products, "EASE factorization")
            if not math.isfinite(remainder):
                raise ValidationError("EASE factorization produced a non-finite pivot")
            if row == column:
                if remainder <= pivot_floor:
                    raise ValidationError("EASE system matrix is singular or ill-conditioned")
                lower[row][column] = math.sqrt(remainder)
            else:
                factor = remainder / lower[column][column]
                if not math.isfinite(factor):
                    raise ValidationError("EASE factorization produced a non-finite value")
                lower[row][column] = factor

    inverse = [[0.0] * size for _ in range(size)]
    for target in range(size):
        forward = [0.0] * size
        for row in range(size):
            numerator = (1.0 if row == target else 0.0) - _finite_sum(
                (lower[row][column] * forward[column] for column in range(row)),
                "EASE forward substitution",
            )
            forward[row] = numerator / lower[row][row]
            if not math.isfinite(forward[row]):
                raise ValidationError("EASE forward substitution is not finite")
        solution = [0.0] * size
        for row in range(size - 1, -1, -1):
            numerator = forward[row] - _finite_sum(
                (lower[column][row] * solution[column] for column in range(row + 1, size)),
                "EASE backward substitution",
            )
            solution[row] = numerator / lower[row][row]
            if not math.isfinite(solution[row]):
                raise ValidationError("EASE backward substitution is not finite")
        for row, value in enumerate(solution):
            inverse[row][target] = value

    matrix_norm = max(
        _finite_sum((abs(value) for value in row), "EASE matrix norm") for row in system
    )
    inverse_norm = max(
        _finite_sum((abs(value) for value in row), "EASE inverse norm") for row in inverse
    )
    residual = 0.0
    symmetry_error = 0.0
    for row in range(size):
        for column in range(size):
            product = _finite_sum(
                (system[row][index] * inverse[index][column] for index in range(size)),
                "EASE inverse residual",
            )
            residual = max(residual, abs(product - (1.0 if row == column else 0.0)))
            symmetry_error = max(symmetry_error, abs(inverse[row][column] - inverse[column][row]))
    tolerance = _INVERSE_RELATIVE_TOLERANCE * (1.0 + matrix_norm * inverse_norm)
    if not math.isfinite(tolerance) or residual > tolerance or symmetry_error > tolerance:
        raise ValidationError(
            "EASE inverse failed its residual check "
            f"(residual={residual:g}, symmetry={symmetry_error:g}, tolerance={tolerance:g})"
        )
    return inverse, residual


def _required_work(seen: Mapping[EntityId, frozenset[EntityId]], items: int) -> int:
    gram_work = sum(len(profile) * len(profile) for profile in seen.values())
    return gram_work + 3 * items * items * items


def _identifier_storage_bytes(value: object, name: str) -> int:
    try:
        identifier = validate_entity_id(value, name)
    except ValidationError as error:
        raise SerializationError(str(error)) from error
    if type(identifier) is int:
        try:
            digits = len(str(abs(identifier)))
        except ValueError as error:
            raise SerializationError("integer identifier exceeds the JSON digit limit") from error
        if digits > MAX_JSON_INTEGER_DIGITS:
            raise SerializationError("integer identifier exceeds the JSON digit limit")
        return max(1, (abs(identifier).bit_length() + 7) // 8)
    try:
        size = len(cast(str, identifier).encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise SerializationError(f"{name} is not valid Unicode scalar text") from error
    if size > MAX_EASE_IDENTIFIER_UTF8_BYTES:
        raise SerializationError(
            f"{name} exceeds the {MAX_EASE_IDENTIFIER_UTF8_BYTES}-byte UTF-8 limit"
        )
    return size


def _preflight_base_state(
    state: Mapping[str, Any],
    *,
    max_items: int,
    max_interactions: int,
    max_work_units: int,
) -> tuple[int, int]:
    """Bound raw EASE base state before the generic restorer allocates copies."""

    if set(state) != {"catalog", "popularity", "users"}:
        raise SerializationError("base model state has invalid fields")
    catalog = state["catalog"]
    popularity = state["popularity"]
    users = state["users"]
    if type(catalog) is not list or type(popularity) is not list or type(users) is not list:
        raise SerializationError("base model arrays are malformed")
    items = len(catalog)
    if items == 0 or items > max_items:
        raise SerializationError("EASE catalog size is outside max_items")
    if len(popularity) != items:
        raise SerializationError("popularity length does not match catalog")
    if len(users) == 0 or len(users) > MAX_EASE_USERS:
        raise SerializationError(f"EASE base state exceeds the {MAX_EASE_USERS}-user limit")

    identifier_bytes = 0
    catalog_ids: set[EntityId] = set()
    for item_id in catalog:
        identifier_bytes += _identifier_storage_bytes(item_id, "catalog item ID")
        catalog_ids.add(validate_entity_id(item_id, "catalog item ID"))
    if identifier_bytes > MAX_EASE_TOTAL_IDENTIFIER_BYTES:
        raise SerializationError("EASE base identifiers exceed the aggregate UTF-8 byte limit")

    unique_interactions = 0
    work_units = 3 * items * items * items
    if work_units > max_work_units:
        raise SerializationError("EASE base state exceeds max_work_units")
    for entry in users:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"user_id", "seen"}
            or type(entry["seen"]) is not list
        ):
            raise SerializationError("user seen-item arrays are malformed")
        identifier_bytes += _identifier_storage_bytes(entry["user_id"], "user_id")
        seen = entry["seen"]
        if not seen or len(seen) > items:
            raise SerializationError("user seen-item state has an invalid size")
        unique_interactions += len(seen)
        if unique_interactions > max_interactions:
            raise SerializationError("EASE base state exceeds max_interactions")
        work_units += len(seen) * len(seen)
        if work_units > max_work_units:
            raise SerializationError("EASE base state exceeds max_work_units")
        seen_ids: set[EntityId] = set()
        for item_id in seen:
            identifier_bytes += _identifier_storage_bytes(item_id, "seen item ID")
            validated_item = validate_entity_id(item_id, "seen item ID")
            if validated_item not in catalog_ids or validated_item in seen_ids:
                raise SerializationError("user seen-item state references an invalid item")
            seen_ids.add(validated_item)
        if identifier_bytes > MAX_EASE_TOTAL_IDENTIFIER_BYTES:
            raise SerializationError("EASE base identifiers exceed the aggregate UTF-8 byte limit")
    return unique_interactions, work_units


def _check_serialized_size(state: Mapping[str, Any]) -> None:
    """Ensure a successful fit can be written and read under the model-file limit."""

    encoder = json.JSONEncoder(indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    total = 1  # save_model appends one newline.
    try:
        for fragment in encoder.iterencode(state):
            total += len(fragment.encode("utf-8", errors="strict"))
            if total > MAX_MODEL_FILE_BYTES:
                raise ValidationError(
                    f"EASE state exceeds the {MAX_MODEL_FILE_BYTES}-byte model-file limit"
                )
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValidationError(f"EASE state cannot be serialized as strict JSON: {error}") from error


def _derive_ease_state(
    seen: Mapping[EntityId, frozenset[EntityId]],
    catalog: Sequence[EntityId],
    regularization: float,
) -> tuple[list[list[float]], dict[EntityId, int], int, float]:
    items = len(catalog)
    item_indices = {item_id: index for index, item_id in enumerate(catalog)}
    gram = [[0.0] * items for _ in range(items)]
    for user_id in sorted(seen, key=stable_id_key):
        profile = tuple(sorted(item_indices[item_id] for item_id in seen[user_id]))
        for left in profile:
            row = gram[left]
            for right in profile:
                row[right] += 1.0
    for index in range(items):
        gram[index][index] += regularization
    inverse, residual = _invert_spd(gram)
    coefficients = [[0.0] * items for _ in range(items)]
    for source in range(items):
        for target in range(items):
            if source == target:
                continue
            denominator = inverse[target][target]
            if denominator <= 0.0 or not math.isfinite(denominator):
                raise ValidationError("EASE inverse has a non-positive diagonal")
            symmetric_entry = 0.5 * (inverse[source][target] + inverse[target][source])
            coefficient = -symmetric_entry / denominator
            if not math.isfinite(coefficient):
                raise ValidationError("EASE coefficient matrix is not finite")
            coefficients[source][target] = coefficient
    return coefficients, item_indices, _required_work(seen, items), residual


class EASE(BaseRecommender):
    """Closed-form item-to-item regression over binary user histories.

    The model solves ``G = X.T @ X + regularization * I`` once, where ``X`` is
    the deduplicated binary user-item matrix, and derives a zero-diagonal item
    coefficient matrix from ``G^-1``. Repeated events therefore affect only the
    common popularity fallback, not the EASE regression itself.
    """

    model_type = "ease"

    def __init__(
        self,
        *,
        regularization: float = 100.0,
        max_items: int = 256,
        max_interactions: int = MAX_EASE_INTERACTIONS,
        max_work_units: int = 100_000_000,
    ) -> None:
        super().__init__()
        self.regularization = _bounded_regularization(regularization)
        self.max_items = _bounded_positive_int(max_items, "max_items", MAX_EASE_ITEMS)
        self.max_interactions = _bounded_positive_int(
            max_interactions,
            "max_interactions",
            MAX_EASE_INTERACTIONS,
        )
        self.max_work_units = _bounded_positive_int(
            max_work_units,
            "max_work_units",
            MAX_EASE_WORK_UNITS,
        )
        self._coefficients: list[list[float]] = []
        self._item_indices: dict[EntityId, int] = {}
        self._training_interactions = 0
        self._work_units = 0
        self._inverse_residual = 0.0

    @property
    def training_interactions(self) -> int:
        self._require_fitted()
        return self._training_interactions

    @property
    def work_units(self) -> int:
        self._require_fitted()
        return self._work_units

    @property
    def inverse_residual(self) -> float:
        self._require_fitted()
        return self._inverse_residual

    def _fit_model(self, dataset: InteractionDataset) -> None:
        if len(dataset) > self.max_interactions:
            raise ValidationError(f"EASE supports at most {self.max_interactions} interactions")
        items = len(self._catalog)
        if items > self.max_items:
            raise ValidationError(f"EASE supports at most {self.max_items} catalog items")
        work_units = _required_work(self._seen, items)
        if work_units > self.max_work_units:
            raise ValidationError(
                f"EASE requires {work_units} work units, above max_work_units={self.max_work_units}"
            )
        base = self._snapshot_base_state()
        try:
            unique_interactions, preflight_work = _preflight_base_state(
                base,
                max_items=self.max_items,
                max_interactions=self.max_interactions,
                max_work_units=self.max_work_units,
            )
        except SerializationError as error:
            raise ValidationError(
                f"EASE training state exceeds persistence limits: {error}"
            ) from error
        if unique_interactions > len(dataset) or preflight_work != work_units:
            raise ValidationError("EASE training state is inconsistent")
        coefficients, item_indices, derived_work, residual = _derive_ease_state(
            self._seen,
            self._catalog,
            self.regularization,
        )
        if derived_work != work_units:
            raise ValidationError("EASE work accounting is inconsistent")
        _check_serialized_size(
            make_envelope(
                self.model_type,
                {
                    "regularization": self.regularization,
                    "max_items": self.max_items,
                    "max_interactions": self.max_interactions,
                    "max_work_units": self.max_work_units,
                },
                base,
                {
                    "coefficients": coefficients,
                    "training_interactions": len(dataset),
                    "work_units": work_units,
                    "inverse_residual": residual,
                },
            )
        )
        self._coefficients = coefficients
        self._item_indices = item_indices
        self._training_interactions = len(dataset)
        self._work_units = work_units
        self._inverse_residual = residual

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        seen = self._seen.get(user_id)
        if seen is None:
            return self._popularity_fallback(item_id)
        target = self._item_indices[item_id]
        score = _finite_sum(
            (
                self._coefficients[source][target]
                for source in sorted(self._item_indices[item_id] for item_id in seen)
            ),
            "EASE score",
        )
        return score

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        return make_envelope(
            self.model_type,
            {
                "regularization": self.regularization,
                "max_items": self.max_items,
                "max_interactions": self.max_interactions,
                "max_work_units": self.max_work_units,
            },
            self._base_state(),
            {
                "coefficients": [row.copy() for row in self._coefficients],
                "training_interactions": self._training_interactions,
                "work_units": self._work_units,
                "inverse_residual": self._inverse_residual,
            },
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        if set(parameters) != {
            "regularization",
            "max_items",
            "max_interactions",
            "max_work_units",
        }:
            raise SerializationError("EASE parameters are malformed")
        try:
            instance = cls(**parameters)
        except (TypeError, ValidationError) as error:
            raise SerializationError(f"invalid EASE parameters: {error}") from error
        if set(model) != {
            "coefficients",
            "training_interactions",
            "work_units",
            "inverse_residual",
        }:
            raise SerializationError("EASE model state is malformed")
        training_interactions = model["training_interactions"]
        if type(training_interactions) is not int:
            raise SerializationError("EASE training_interactions is malformed")
        work_units = model["work_units"]
        if type(work_units) is not int:
            raise SerializationError("EASE work_units is malformed")
        unique_interactions, preflight_work = _preflight_base_state(
            base,
            max_items=instance.max_items,
            max_interactions=instance.max_interactions,
            max_work_units=instance.max_work_units,
        )
        if not unique_interactions <= training_interactions <= instance.max_interactions:
            raise SerializationError("EASE training_interactions is malformed")
        if work_units != preflight_work:
            raise SerializationError("EASE work_units is malformed")
        instance._restore_base_state(base)
        items = len(instance._catalog)
        raw_coefficients = model["coefficients"]
        if type(raw_coefficients) is not list or len(raw_coefficients) != items:
            raise SerializationError("EASE coefficient matrix row count is malformed")
        coefficients: list[list[float]] = []
        for row_index, raw_row in enumerate(raw_coefficients):
            if type(raw_row) is not list or len(raw_row) != items:
                raise SerializationError("EASE coefficient matrix column count is malformed")
            row: list[float] = []
            for column_index, raw_value in enumerate(raw_row):
                if type(raw_value) not in {int, float}:
                    raise SerializationError("EASE coefficient matrix must contain finite numbers")
                value = safe_float(raw_value)
                if not math.isfinite(value):
                    raise SerializationError("EASE coefficient matrix must contain finite numbers")
                if row_index == column_index and value != 0.0:
                    raise SerializationError("EASE coefficient matrix diagonal must be zero")
                row.append(value)
            coefficients.append(row)
        inverse_residual = model["inverse_residual"]
        if type(inverse_residual) not in {int, float}:
            raise SerializationError("EASE inverse_residual is malformed")
        residual = safe_float(inverse_residual)
        if not math.isfinite(residual) or residual < 0.0:
            raise SerializationError("EASE inverse_residual is malformed")
        try:
            expected_coefficients, item_indices, expected_work, expected_residual = (
                _derive_ease_state(instance._seen, instance._catalog, instance.regularization)
            )
        except ValidationError as error:
            raise SerializationError(f"invalid canonical EASE state: {error}") from error
        if expected_work != preflight_work:
            raise SerializationError("EASE canonical work accounting is inconsistent")
        if coefficients != expected_coefficients:
            raise SerializationError("EASE coefficients do not match canonical base state")
        if residual != expected_residual:
            raise SerializationError("EASE inverse_residual does not match canonical base state")
        instance._coefficients = expected_coefficients
        instance._item_indices = item_indices
        instance._training_interactions = training_interactions
        instance._work_units = expected_work
        instance._inverse_residual = expected_residual
        return instance
