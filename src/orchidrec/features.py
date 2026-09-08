"""Typed, leakage-safe feature schemas and fitted preprocessing pipelines."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, StrEnum
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, TypeVar, cast, overload

from orchidrec._json import strict_json_loads
from orchidrec.data import EntityId, stable_id_key
from orchidrec.errors import SerializationError, ValidationError

FEATURE_DATASET_FORMAT = "orchidrec.feature-dataset"
FEATURE_PIPELINE_FORMAT = "orchidrec.feature-pipeline"
ENCODED_FEATURES_FORMAT = "orchidrec.encoded-features"
FEATURE_SCHEMA_VERSION = 1
PAD_INDEX = 0
UNKNOWN_INDEX = 1

_MAX_FEATURE_ROWS = 10_000_000
_MAX_TOTAL_VALUES = 100_000_000
_MAX_VOCAB_VALUES = 10_000_000
_MAX_VOCAB_TOKEN_BYTES = 1_073_741_824
_MAX_STATE_BYTES = 1_073_741_824
_MAX_JSON_DEPTH = 128

Token: TypeAlias = str | int
RawScalar: TypeAlias = str | int | float
RawFeatureValue: TypeAlias = RawScalar | tuple[RawScalar, ...]
EncodedFeatureValue: TypeAlias = int | float | tuple[int, ...] | tuple[float, ...]


class FeatureKind(StrEnum):
    """Physical representation of one raw feature."""

    # These are public schema labels, not credentials.
    TOKEN = "token"  # nosec B105
    FLOAT = "float"
    TOKEN_SEQUENCE = "token-sequence"  # nosec B105
    FLOAT_SEQUENCE = "float-sequence"

    @property
    def is_sequence(self) -> bool:
        return self in {FeatureKind.TOKEN_SEQUENCE, FeatureKind.FLOAT_SEQUENCE}

    @property
    def is_token(self) -> bool:
        return self in {FeatureKind.TOKEN, FeatureKind.TOKEN_SEQUENCE}


class FeatureSource(StrEnum):
    """Entity namespace that owns a feature."""

    INTERACTION = "interaction"
    USER = "user"
    ITEM = "item"


class SequenceKeep(StrEnum):
    """Which side survives when a sequence is longer than its fitted width."""

    HEAD = "head"
    TAIL = "tail"


def _plain_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    result = str.__str__(value)
    if not result or result != result.strip():
        raise ValidationError(f"{name} must be non-empty without surrounding whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in result):
        raise ValidationError(f"{name} must not contain control characters")
    return result


def _plain_int(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        qualifier = "a positive " if positive else "an "
        raise ValidationError(f"{name} must be {qualifier}integer")
    result = int(int.__index__(value))
    if positive and result <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return result


def _plain_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite number")
    try:
        if isinstance(value, int):
            result = float(int.__index__(value))
        else:
            result = float(float.__float__(value))
    except (OverflowError, ValueError) as error:
        raise ValidationError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be a finite number")
    return result


EnumValue = TypeVar("EnumValue", bound=Enum)


def _enum_value(value: object, enum_type: type[EnumValue], name: str) -> EnumValue:
    if isinstance(value, str):
        value = str.__str__(value)
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(member.value for member in enum_type)
        raise ValidationError(f"{name} must be one of: {choices}") from error


def _bounded_tuple(values: Iterable[Any], limit: int, name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValidationError(f"{name} must be an iterable, not text or bytes")
    try:
        if isinstance(values, list):
            iterator: Iterator[Any] = list.__iter__(values)
        elif isinstance(values, tuple):
            iterator = tuple.__iter__(values)
        else:
            iterator = iter(values)
        materialized = tuple(islice(iterator, limit + 1))
    except TypeError as error:
        raise ValidationError(f"{name} must be iterable") from error
    if len(materialized) > limit:
        raise ValidationError(f"{name} exceeds the limit of {limit}")
    return materialized


def _state_mapping(
    state: object,
    expected: set[str],
    message: str,
) -> dict[str, object]:
    """Accept only a plain object with plain-string, exactly matching keys."""

    if type(state) is not dict:
        raise SerializationError(message)
    keys = tuple(dict.__iter__(state))
    if any(type(key) is not str for key in keys) or set(keys) != expected:
        raise SerializationError(message)
    return cast(dict[str, object], state)


@dataclass(frozen=True, slots=True)
class FeatureLimits:
    """Persisted ceilings for raw, fitted, and encoded feature state."""

    max_fields: int = 256
    max_rows: int = 1_000_000
    max_total_values: int = 5_000_000
    max_sequence_values: int = 4_096
    max_token_chars: int = 512
    max_token_bytes: int = 2_048
    max_token_integer_bits: int = 512
    max_vocab_values: int = 1_000_000
    max_vocab_token_bytes: int = 16 * 1024 * 1024
    max_state_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _plain_int(getattr(self, field_name), field_name, positive=True),
            )
        if self.max_fields > 4_096:
            raise ValidationError("max_fields must be at most 4096")
        if self.max_sequence_values > 65_536:
            raise ValidationError("max_sequence_values must be at most 65536")
        if self.max_token_chars > 16_384 or self.max_token_bytes > 65_536:
            raise ValidationError("token text limits exceed the supported schema")
        if self.max_token_integer_bits > 16_384:
            raise ValidationError("max_token_integer_bits must be at most 16384")
        if self.max_rows > _MAX_FEATURE_ROWS:
            raise ValidationError(f"max_rows must be at most {_MAX_FEATURE_ROWS}")
        if self.max_total_values > _MAX_TOTAL_VALUES:
            raise ValidationError(f"max_total_values must be at most {_MAX_TOTAL_VALUES}")
        if self.max_vocab_values > _MAX_VOCAB_VALUES:
            raise ValidationError(f"max_vocab_values must be at most {_MAX_VOCAB_VALUES}")
        if self.max_vocab_token_bytes > _MAX_VOCAB_TOKEN_BYTES:
            raise ValidationError(f"max_vocab_token_bytes must be at most {_MAX_VOCAB_TOKEN_BYTES}")
        if self.max_state_bytes > _MAX_STATE_BYTES:
            raise ValidationError(f"max_state_bytes must be at most {_MAX_STATE_BYTES}")

    def to_state(self) -> dict[str, int]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}

    @classmethod
    def from_state(cls, state: object) -> FeatureLimits:
        expected = set(cls.__dataclass_fields__)
        normalized = _state_mapping(
            state,
            expected,
            "feature limits have missing or unknown fields",
        )
        try:
            return cls(**cast(dict[str, int], normalized))
        except ValidationError as error:
            raise SerializationError(f"invalid feature limits: {error}") from error


DEFAULT_FEATURE_LIMITS = FeatureLimits()


def _copy_limits(limits: FeatureLimits) -> FeatureLimits:
    if type(limits) is not FeatureLimits:
        raise ValidationError("limits must be an exact FeatureLimits value")
    return FeatureLimits(**FeatureLimits.to_state(limits))


def _plain_token(value: object, name: str, limits: FeatureLimits) -> Token:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValidationError(f"{name} must be a string or integer token")
    if isinstance(value, str):
        token_text = str.__str__(value)
        if not token_text:
            raise ValidationError(f"{name} must not be empty")
        if len(token_text) > limits.max_token_chars:
            raise ValidationError(f"{name} exceeds max_token_chars")
        try:
            encoded = token_text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValidationError(f"{name} must be valid UTF-8 text") from error
        if len(encoded) > limits.max_token_bytes:
            raise ValidationError(f"{name} exceeds max_token_bytes")
        return token_text
    token_integer = int(int.__index__(value))
    if abs(token_integer).bit_length() > limits.max_token_integer_bits:
        raise ValidationError(f"{name} exceeds max_token_integer_bits")
    return token_integer


def _plain_entity_id(value: object, name: str, limits: FeatureLimits) -> EntityId:
    return _plain_token(value, name, limits)


def _mapping_snapshot(value: object, limit: int, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be a mapping")
    try:
        iterator = dict.items(value) if isinstance(value, dict) else iter(value.items())
        items = tuple(islice(iterator, limit + 1))
    except (AttributeError, TypeError) as error:
        raise ValidationError(f"{name} must be a mapping") from error
    if len(items) > limit:
        raise ValidationError(f"{name} exceeds the limit of {limit} fields")
    result: dict[str, object] = {}
    for key, item in items:
        plain_key = _plain_text(key, f"{name} field name")
        if plain_key in result:
            raise ValidationError(f"{name} contains duplicate field {plain_key!r}")
        result[plain_key] = item
    return result


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One named feature and its entity/type contract."""

    name: str
    kind: FeatureKind
    source: FeatureSource
    sequence_length: int | None = None
    keep: SequenceKeep | None = None

    def __post_init__(self) -> None:
        name = _plain_text(self.name, "feature name")
        if len(name) > 256:
            raise ValidationError("feature name must contain at most 256 characters")
        kind = _enum_value(self.kind, FeatureKind, "feature kind")
        source = _enum_value(self.source, FeatureSource, "feature source")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source", source)
        if kind.is_sequence:
            length = _plain_int(self.sequence_length, "sequence_length", positive=True)
            if length > DEFAULT_FEATURE_LIMITS.max_sequence_values:
                raise ValidationError("sequence_length exceeds the supported schema")
            keep = _enum_value(self.keep, SequenceKeep, "sequence keep policy")
            object.__setattr__(self, "sequence_length", length)
            object.__setattr__(self, "keep", keep)
        elif self.sequence_length is not None or self.keep is not None:
            raise ValidationError("scalar features cannot declare sequence_length or keep")

    def to_state(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "sequence_length": self.sequence_length,
            "keep": None if self.keep is None else self.keep.value,
            "source": self.source.value,
        }

    @classmethod
    def from_state(cls, state: object) -> FeatureSpec:
        expected = {"kind", "name", "sequence_length", "keep", "source"}
        normalized = _state_mapping(
            state,
            expected,
            "feature specification has missing or unknown fields",
        )
        try:
            return cls(
                name=cast(str, normalized["name"]),
                kind=cast(FeatureKind, normalized["kind"]),
                source=cast(FeatureSource, normalized["source"]),
                sequence_length=cast(int | None, normalized["sequence_length"]),
                keep=cast(SequenceKeep | None, normalized["keep"]),
            )
        except ValidationError as error:
            raise SerializationError(f"invalid feature specification: {error}") from error


@dataclass(frozen=True, slots=True, init=False)
class FeatureSchema(Sequence[FeatureSpec]):
    """Ordered, immutable feature specifications with globally unique names."""

    features: tuple[FeatureSpec, ...]

    def __init__(
        self,
        features: Iterable[FeatureSpec],
        *,
        limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
    ) -> None:
        active_limits = _copy_limits(limits)
        raw = _bounded_tuple(features, active_limits.max_fields, "feature schema")
        if not raw:
            raise ValidationError("feature schema must not be empty")
        copied: list[FeatureSpec] = []
        for feature in raw:
            if type(feature) is not FeatureSpec:
                raise ValidationError("feature schema accepts only FeatureSpec values")
            copied_feature = FeatureSpec(
                feature.name,
                feature.kind,
                feature.source,
                feature.sequence_length,
                feature.keep,
            )
            if (
                copied_feature.sequence_length is not None
                and copied_feature.sequence_length > active_limits.max_sequence_values
            ):
                raise ValidationError("schema sequence_length exceeds max_sequence_values")
            copied.append(copied_feature)
        names = [feature.name for feature in copied]
        if len(names) != len(set(names)):
            raise ValidationError("feature names must be globally unique")
        object.__setattr__(self, "features", tuple(copied))

    @overload
    def __getitem__(self, index: int) -> FeatureSpec: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FeatureSpec, ...]: ...

    def __getitem__(self, index: int | slice) -> FeatureSpec | tuple[FeatureSpec, ...]:
        return self.features[index]

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self.features)

    def for_source(self, source: FeatureSource | str) -> tuple[FeatureSpec, ...]:
        normalized = _enum_value(source, FeatureSource, "feature source")
        return tuple(feature for feature in self if feature.source is normalized)

    def to_state(self) -> list[dict[str, object]]:
        return [feature.to_state() for feature in self]

    @classmethod
    def from_state(
        cls,
        state: object,
        *,
        limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
    ) -> FeatureSchema:
        if type(state) is not list:
            raise SerializationError("feature schema must be an array")
        try:
            return cls(
                (FeatureSpec.from_state(item) for item in list.__iter__(state)),
                limits=limits,
            )
        except ValidationError as error:
            raise SerializationError(f"invalid feature schema: {error}") from error


def _raw_scalar(value: object, name: str, limits: FeatureLimits) -> RawScalar:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must not be boolean")
    if isinstance(value, str):
        return _plain_token(value, name, limits)
    if isinstance(value, int):
        return _plain_token(value, name, limits)
    if isinstance(value, float):
        return _plain_float(value, name)
    raise ValidationError(f"{name} has an unsupported feature value")


def _raw_value(value: object, name: str, limits: FeatureLimits) -> RawFeatureValue:
    if isinstance(value, (list, tuple)):
        items = _bounded_tuple(value, limits.max_sequence_values, name)
        return tuple(_raw_scalar(item, f"{name} element", limits) for item in items)
    return _raw_scalar(value, name, limits)


def _validate_raw_kind(feature: FeatureSpec, value: RawFeatureValue) -> None:
    if feature.kind is FeatureKind.TOKEN:
        if isinstance(value, (tuple, float)):
            raise ValidationError(f"feature {feature.name!r} must be a token")
        return
    if feature.kind is FeatureKind.FLOAT:
        if isinstance(value, (tuple, str)):
            raise ValidationError(f"feature {feature.name!r} must be numeric")
        return
    if not isinstance(value, tuple):
        raise ValidationError(f"feature {feature.name!r} must be a sequence")
    if feature.kind is FeatureKind.TOKEN_SEQUENCE:
        if any(isinstance(item, float) for item in value):
            raise ValidationError(f"feature {feature.name!r} must be a token sequence")
        return
    if any(isinstance(item, str) for item in value):
        raise ValidationError(f"feature {feature.name!r} must be a numeric sequence")


@dataclass(frozen=True, slots=True, init=False)
class FeatureRow:
    """One immutable row in an interaction, user, or item namespace."""

    source: FeatureSource
    key: EntityId
    values: Mapping[str, RawFeatureValue]

    def __init__(
        self,
        source: FeatureSource | str,
        key: EntityId,
        values: Mapping[str, object],
        *,
        limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
    ) -> None:
        active_limits = _copy_limits(limits)
        normalized_source = _enum_value(source, FeatureSource, "row source")
        normalized_key = _plain_entity_id(key, "row key", active_limits)
        raw_values = _mapping_snapshot(values, active_limits.max_fields, "row values")
        normalized_values = {
            name: _raw_value(value, f"feature {name!r}", active_limits)
            for name, value in raw_values.items()
        }
        object.__setattr__(self, "source", normalized_source)
        object.__setattr__(self, "key", normalized_key)
        object.__setattr__(self, "values", MappingProxyType(normalized_values))

    def to_state(self) -> dict[str, object]:
        return {
            "key": self.key,
            "source": self.source.value,
            "values": {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in self.values.items()
            },
        }


def _source_key(row: FeatureRow) -> tuple[str, tuple[int, int | str]]:
    return (row.source.value, stable_id_key(row.key))


@dataclass(frozen=True, slots=True, init=False)
class FeatureDataset(Sequence[FeatureRow]):
    """Canonical raw rows whose fields are validated against one schema."""

    schema: FeatureSchema
    rows: tuple[FeatureRow, ...]

    def __init__(
        self,
        schema: FeatureSchema,
        rows: Iterable[FeatureRow],
        *,
        limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
    ) -> None:
        if type(schema) is not FeatureSchema:
            raise ValidationError("schema must be a FeatureSchema")
        active_limits = _copy_limits(limits)
        copied_schema = FeatureSchema(schema.features, limits=active_limits)
        raw_rows = _bounded_tuple(rows, active_limits.max_rows, "feature rows")
        copied_rows: list[FeatureRow] = []
        expected = {
            source: {feature.name for feature in copied_schema.for_source(source)}
            for source in FeatureSource
        }
        total_values = 0
        for row in raw_rows:
            if type(row) is not FeatureRow:
                raise ValidationError("feature rows accept only FeatureRow values")
            copied = FeatureRow(row.source, row.key, row.values, limits=active_limits)
            actual = set(copied.values)
            if actual != expected[copied.source]:
                missing = sorted(expected[copied.source] - actual)
                unknown = sorted(actual - expected[copied.source])
                details: list[str] = []
                if missing:
                    details.append(f"missing: {', '.join(missing)}")
                if unknown:
                    details.append(f"unknown: {', '.join(unknown)}")
                raise ValidationError(
                    f"{copied.source.value} row fields differ from schema ({'; '.join(details)})"
                )
            for feature in copied_schema.for_source(copied.source):
                _validate_raw_kind(feature, copied.values[feature.name])
            total_values += sum(
                len(value) if isinstance(value, tuple) else 1 for value in copied.values.values()
            )
            if total_values > active_limits.max_total_values:
                raise ValidationError("feature rows exceed max_total_values")
            copied_rows.append(copied)
        canonical = tuple(sorted(copied_rows, key=_source_key))
        row_keys = [(row.source, row.key) for row in canonical]
        if len(row_keys) != len(set(row_keys)):
            raise ValidationError("feature row keys must be unique within each source")
        object.__setattr__(self, "schema", copied_schema)
        object.__setattr__(self, "rows", canonical)

    @overload
    def __getitem__(self, index: int) -> FeatureRow: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FeatureRow, ...]: ...

    def __getitem__(self, index: int | slice) -> FeatureRow | tuple[FeatureRow, ...]:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[FeatureRow]:
        return iter(self.rows)

    @property
    def total_values(self) -> int:
        return sum(
            len(value) if isinstance(value, tuple) else 1
            for row in self
            for value in row.values.values()
        )

    def to_state(self) -> dict[str, object]:
        return {
            "format": FEATURE_DATASET_FORMAT,
            "schema_version": FEATURE_SCHEMA_VERSION,
            "schema": self.schema.to_state(),
            "rows": [row.to_state() for row in self],
        }

    @classmethod
    def from_state(
        cls,
        state: object,
        *,
        limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
    ) -> FeatureDataset:
        expected = {"format", "schema_version", "schema", "rows"}
        normalized = _state_mapping(
            state, expected, "feature dataset has missing or unknown fields"
        )
        schema_version = normalized["schema_version"]
        format_name = normalized["format"]
        if (
            type(format_name) is not str
            or str.__str__(format_name) != FEATURE_DATASET_FORMAT
            or type(schema_version) is not int
            or int.__index__(schema_version) != FEATURE_SCHEMA_VERSION
        ):
            raise SerializationError("unsupported feature dataset format or schema version")
        schema = FeatureSchema.from_state(normalized["schema"], limits=limits)
        rows = normalized["rows"]
        if type(rows) is not list:
            raise SerializationError("feature dataset rows must be an array")
        try:
            parsed_rows = (_feature_row_from_state(row, limits) for row in list.__iter__(rows))
            return cls(schema, parsed_rows, limits=limits)
        except ValidationError as error:
            raise SerializationError(f"invalid feature dataset: {error}") from error


def _feature_row_from_state(state: object, limits: FeatureLimits) -> FeatureRow:
    expected = {"key", "source", "values"}
    normalized = _state_mapping(state, expected, "feature row has missing or unknown fields")
    source = normalized["source"]
    key = normalized["key"]
    values = normalized["values"]
    if not isinstance(source, str):
        raise SerializationError("feature row source must be a string")
    if isinstance(key, bool) or not isinstance(key, (str, int)):
        raise SerializationError("feature row key must be a string or integer")
    if type(values) is not dict:
        raise SerializationError("feature row values must be an object")
    return FeatureRow(source, key, cast(Mapping[str, object], values), limits=limits)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, UnicodeEncodeError, ValueError) as error:
        raise SerializationError(f"feature state is not canonical JSON: {error}") from error


def feature_dataset_sha256(dataset: FeatureDataset) -> str:
    """Hash a canonical, order-independent raw feature dataset."""

    if type(dataset) is not FeatureDataset:
        raise ValidationError("dataset must be a FeatureDataset")
    digest = hashlib.sha256()
    try:
        for chunk in _feature_dataset_json_chunks(dataset):
            digest.update(chunk.encode("utf-8"))
    except (OverflowError, TypeError, UnicodeEncodeError, ValueError) as error:
        raise SerializationError(f"feature state is not canonical JSON: {error}") from error
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NumericStatistics:
    """Population normalization fitted from training values only."""

    count: int
    mean: float
    scale: float

    def __post_init__(self) -> None:
        count = _plain_int(self.count, "numeric statistic count")
        if count < 0:
            raise ValidationError("numeric statistic count must be non-negative")
        mean = _plain_float(self.mean, "numeric statistic mean")
        scale = _plain_float(self.scale, "numeric statistic scale")
        if scale <= 0:
            raise ValidationError("numeric statistic scale must be positive")
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def to_state(self) -> dict[str, int | float]:
        return {"count": self.count, "mean": self.mean, "scale": self.scale}

    @classmethod
    def from_state(cls, state: object) -> NumericStatistics:
        normalized = _state_mapping(
            state,
            {"count", "mean", "scale"},
            "numeric statistics have missing or unknown fields",
        )
        try:
            return cls(
                count=cast(int, normalized["count"]),
                mean=cast(float, normalized["mean"]),
                scale=cast(float, normalized["scale"]),
            )
        except ValidationError as error:
            raise SerializationError(f"invalid numeric statistics: {error}") from error


def _token_key(value: Token) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, value)


def _token_storage_bytes(value: Token) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return max(1, (abs(value).bit_length() + 7) // 8)


def _numeric_values(dataset: FeatureDataset, feature: FeatureSpec) -> tuple[float, ...]:
    result: list[float] = []
    for row in dataset:
        if row.source is not feature.source:
            continue
        value = row.values[feature.name]
        if feature.kind is FeatureKind.FLOAT:
            if isinstance(value, (tuple, str)):
                raise ValidationError(f"feature {feature.name!r} must contain finite numbers")
            result.append(_plain_float(value, f"feature {feature.name!r}"))
        elif feature.kind is FeatureKind.FLOAT_SEQUENCE:
            if not isinstance(value, tuple):
                raise ValidationError(f"feature {feature.name!r} must contain numeric sequences")
            for item in value:
                if isinstance(item, str):
                    raise ValidationError(
                        f"feature {feature.name!r} must contain numeric sequences"
                    )
                result.append(_plain_float(item, f"feature {feature.name!r} element"))
    return tuple(result)


def _tokens(dataset: FeatureDataset, feature: FeatureSpec) -> tuple[Token, ...]:
    result: set[Token] = set()
    for row in dataset:
        if row.source is not feature.source:
            continue
        value = row.values[feature.name]
        if feature.kind is FeatureKind.TOKEN:
            if isinstance(value, (tuple, float)):
                raise ValidationError(f"feature {feature.name!r} must contain tokens")
            result.add(value)
        elif feature.kind is FeatureKind.TOKEN_SEQUENCE:
            if not isinstance(value, tuple):
                raise ValidationError(f"feature {feature.name!r} must contain token sequences")
            for item in value:
                if isinstance(item, float):
                    raise ValidationError(f"feature {feature.name!r} must contain token sequences")
                result.add(item)
    return tuple(sorted(result, key=_token_key))


def _fit_numeric(values: Sequence[float]) -> NumericStatistics:
    if not values:
        return NumericStatistics(0, 0.0, 1.0)
    maximum = max(abs(value) for value in values)
    if maximum == 0.0:
        return NumericStatistics(len(values), 0.0, 1.0)
    scaled = tuple(value / maximum for value in values)
    try:
        scaled_mean = math.fsum(scaled) / len(scaled)
        mean = scaled_mean * maximum
        scaled_variance = math.fsum((value - scaled_mean) ** 2 for value in scaled) / len(scaled)
        scale = maximum * math.sqrt(max(0.0, scaled_variance))
    except (OverflowError, ValueError) as error:
        raise ValidationError("numeric feature statistics are not finite") from error
    if not math.isfinite(mean) or not math.isfinite(scale):
        raise ValidationError("numeric feature statistics are not finite")
    return NumericStatistics(len(values), mean, 1.0 if scale == 0.0 else scale)


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a SHA-256 string")
    normalized = str.__str__(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValidationError(f"{name} must be a lowercase SHA-256 string")
    return normalized


@dataclass(frozen=True, slots=True, init=False)
class EncodedFeatureRow:
    """One fixed-width, model-ready row aligned to a fitted schema."""

    source: FeatureSource
    key: EntityId
    values: Mapping[str, EncodedFeatureValue]
    sequence_lengths: Mapping[str, int]

    def __init__(
        self,
        source: FeatureSource,
        key: EntityId,
        values: Mapping[str, EncodedFeatureValue],
        sequence_lengths: Mapping[str, int],
        *,
        limits: FeatureLimits,
    ) -> None:
        object.__setattr__(self, "source", _enum_value(source, FeatureSource, "row source"))
        object.__setattr__(self, "key", _plain_entity_id(key, "row key", limits))
        copied_values: dict[str, EncodedFeatureValue] = {}
        for name, value in _mapping_snapshot(values, limits.max_fields, "encoded values").items():
            if isinstance(value, tuple):
                items = _bounded_tuple(value, limits.max_sequence_values, f"encoded {name!r}")
                normalized_items: list[int | float] = []
                for item in items:
                    if isinstance(item, bool) or not isinstance(item, (int, float)):
                        raise ValidationError("encoded feature values must be numeric")
                    if isinstance(item, int):
                        normalized_items.append(int(int.__index__(item)))
                    else:
                        normalized_items.append(
                            _plain_float(item, f"encoded feature {name!r} element")
                        )
                copied_values[name] = tuple(normalized_items)
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError("encoded feature values must be numeric")
            elif isinstance(value, int):
                copied_values[name] = int(int.__index__(value))
            else:
                copied_values[name] = _plain_float(value, f"encoded feature {name!r}")
        copied_lengths = {
            name: _plain_int(value, f"sequence length {name!r}")
            for name, value in _mapping_snapshot(
                sequence_lengths, limits.max_fields, "sequence lengths"
            ).items()
        }
        object.__setattr__(self, "values", MappingProxyType(copied_values))
        object.__setattr__(self, "sequence_lengths", MappingProxyType(copied_lengths))

    def to_state(self) -> dict[str, object]:
        return {
            "key": self.key,
            "source": self.source.value,
            "values": {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in self.values.items()
            },
            "sequence_lengths": dict(self.sequence_lengths),
        }


@dataclass(frozen=True, slots=True, init=False)
class EncodedFeatureDataset(Sequence[EncodedFeatureRow]):
    """Encoded rows plus the exact pipeline identity that produced them."""

    schema: FeatureSchema
    pipeline_sha256: str
    rows: tuple[EncodedFeatureRow, ...]

    def __init__(
        self,
        schema: FeatureSchema,
        pipeline_sha256: str,
        rows: Iterable[EncodedFeatureRow],
        *,
        limits: FeatureLimits,
    ) -> None:
        if type(schema) is not FeatureSchema:
            raise ValidationError("schema must be an exact FeatureSchema value")
        active_limits = _copy_limits(limits)
        copied_schema = FeatureSchema(schema.features, limits=active_limits)
        digest = _sha256(pipeline_sha256, "pipeline_sha256")
        raw_rows = _bounded_tuple(rows, active_limits.max_rows, "encoded feature rows")
        copied_rows: list[EncodedFeatureRow] = []
        expected = {
            source: {feature.name for feature in copied_schema.for_source(source)}
            for source in FeatureSource
        }
        expected_sequences = {
            source: {
                feature.name
                for feature in copied_schema.for_source(source)
                if feature.kind.is_sequence
            }
            for source in FeatureSource
        }
        total_values = 0
        for row in raw_rows:
            if type(row) is not EncodedFeatureRow:
                raise ValidationError("encoded rows accept only EncodedFeatureRow values")
            copied = EncodedFeatureRow(
                row.source,
                row.key,
                row.values,
                row.sequence_lengths,
                limits=active_limits,
            )
            if set(copied.values) != expected[copied.source]:
                raise ValidationError("encoded row fields differ from the feature schema")
            if set(copied.sequence_lengths) != expected_sequences[copied.source]:
                raise ValidationError("encoded sequence lengths differ from the feature schema")
            for feature in copied_schema.for_source(copied.source):
                value = copied.values[feature.name]
                if feature.kind.is_sequence:
                    if not isinstance(value, tuple) or len(value) != feature.sequence_length:
                        raise ValidationError(
                            "encoded sequences must have their declared fixed width"
                        )
                    length = copied.sequence_lengths[feature.name]
                    if length < 0 or length > feature.sequence_length:
                        raise ValidationError(
                            "encoded sequence length is outside its declared width"
                        )
                    total_values += len(value)
                elif isinstance(value, tuple):
                    raise ValidationError("encoded scalar features must be scalar")
                else:
                    total_values += 1
                if feature.kind.is_token:
                    token_values = value if isinstance(value, tuple) else (value,)
                    if any(
                        isinstance(item, bool) or not isinstance(item, int) or item < 0
                        for item in token_values
                    ):
                        raise ValidationError("encoded token indices must be non-negative integers")
                else:
                    numeric_values = value if isinstance(value, tuple) else (value,)
                    for item in numeric_values:
                        _plain_float(item, "encoded numeric value")
            if total_values > active_limits.max_total_values:
                raise ValidationError("encoded rows exceed max_total_values")
            copied_rows.append(copied)
        canonical = tuple(
            sorted(copied_rows, key=lambda row: (row.source.value, stable_id_key(row.key)))
        )
        identities = [(row.source, row.key) for row in canonical]
        if len(identities) != len(set(identities)):
            raise ValidationError("encoded row keys must be unique within each source")
        object.__setattr__(self, "schema", copied_schema)
        object.__setattr__(self, "pipeline_sha256", digest)
        object.__setattr__(self, "rows", canonical)

    @overload
    def __getitem__(self, index: int) -> EncodedFeatureRow: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[EncodedFeatureRow, ...]: ...

    def __getitem__(self, index: int | slice) -> EncodedFeatureRow | tuple[EncodedFeatureRow, ...]:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[EncodedFeatureRow]:
        return iter(self.rows)

    def to_state(self) -> dict[str, object]:
        return {
            "format": ENCODED_FEATURES_FORMAT,
            "schema_version": FEATURE_SCHEMA_VERSION,
            "pipeline_sha256": self.pipeline_sha256,
            "schema": self.schema.to_state(),
            "rows": [row.to_state() for row in self],
        }

    @classmethod
    def from_state(
        cls,
        state: object,
        *,
        limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
    ) -> EncodedFeatureDataset:
        expected = {"format", "schema_version", "pipeline_sha256", "schema", "rows"}
        normalized = _state_mapping(
            state, expected, "encoded features have missing or unknown fields"
        )
        schema_version = normalized["schema_version"]
        format_name = normalized["format"]
        if (
            type(format_name) is not str
            or str.__str__(format_name) != ENCODED_FEATURES_FORMAT
            or type(schema_version) is not int
            or int.__index__(schema_version) != FEATURE_SCHEMA_VERSION
        ):
            raise SerializationError("unsupported encoded feature format or schema version")
        active_limits = _copy_limits(limits)
        schema = FeatureSchema.from_state(normalized["schema"], limits=active_limits)
        raw_rows = normalized["rows"]
        if type(raw_rows) is not list:
            raise SerializationError("encoded feature rows must be an array")
        try:
            rows = (_encoded_row_from_state(row, active_limits) for row in list.__iter__(raw_rows))
            return cls(
                schema,
                cast(str, normalized["pipeline_sha256"]),
                rows,
                limits=active_limits,
            )
        except ValidationError as error:
            raise SerializationError(f"invalid encoded feature state: {error}") from error


def _encoded_row_from_state(state: object, limits: FeatureLimits) -> EncodedFeatureRow:
    expected = {"key", "source", "values", "sequence_lengths"}
    normalized = _state_mapping(
        state, expected, "encoded feature row has missing or unknown fields"
    )
    source = normalized["source"]
    key = normalized["key"]
    values = normalized["values"]
    lengths = normalized["sequence_lengths"]
    if not isinstance(source, str):
        raise SerializationError("encoded feature row source must be a string")
    if isinstance(key, bool) or not isinstance(key, (str, int)):
        raise SerializationError("encoded feature row key must be a string or integer")
    if type(values) is not dict or type(lengths) is not dict:
        raise SerializationError("encoded feature row values and lengths must be objects")
    normalized_values = {
        cast(str, name): (
            tuple(list.__iter__(value)) if type(value) is list else cast(int | float, value)
        )
        for name, value in dict.items(values)
    }
    return EncodedFeatureRow(
        cast(FeatureSource, source),
        key,
        normalized_values,
        cast(Mapping[str, int], dict(lengths)),
        limits=limits,
    )


@dataclass(frozen=True, slots=True, init=False)
class FittedFeaturePipeline:
    """A deterministic vocabulary and numeric transform fitted on one train split."""

    schema: FeatureSchema
    token_vocabularies: Mapping[str, tuple[Token, ...]]
    token_indices: Mapping[str, Mapping[Token, int]]
    numeric_statistics: Mapping[str, NumericStatistics]
    training_rows: int
    training_values: int
    training_sha256: str
    limits: FeatureLimits

    def __init__(
        self,
        schema: FeatureSchema,
        token_vocabularies: Mapping[str, tuple[Token, ...]],
        numeric_statistics: Mapping[str, NumericStatistics],
        training_rows: int,
        training_values: int,
        training_sha256: str,
        limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
    ) -> None:
        if type(schema) is not FeatureSchema:
            raise ValidationError("schema must be an exact FeatureSchema value")
        copied_limits = _copy_limits(limits)
        copied_schema = FeatureSchema(schema.features, limits=copied_limits)
        expected_tokens = {feature.name for feature in copied_schema if feature.kind.is_token}
        expected_numeric = {feature.name for feature in copied_schema if not feature.kind.is_token}
        raw_vocabularies = _mapping_snapshot(
            token_vocabularies, copied_limits.max_fields, "token vocabularies"
        )
        raw_statistics = _mapping_snapshot(
            numeric_statistics, copied_limits.max_fields, "numeric statistics"
        )
        if set(raw_vocabularies) != expected_tokens or set(raw_statistics) != expected_numeric:
            raise ValidationError("fitted feature state differs from the feature schema")
        vocabularies: dict[str, tuple[Token, ...]] = {}
        total_vocab_values = 0
        total_vocab_bytes = 0
        for name, raw_values in raw_vocabularies.items():
            if not isinstance(raw_values, (list, tuple)):
                raise ValidationError("token vocabularies must contain arrays")
            vocabulary_values = _bounded_tuple(
                raw_values, copied_limits.max_vocab_values, f"vocabulary {name!r}"
            )
            normalized = tuple(
                _plain_token(value, f"vocabulary {name!r} token", copied_limits)
                for value in vocabulary_values
            )
            if len(normalized) != len(set(normalized)) or normalized != tuple(
                sorted(normalized, key=_token_key)
            ):
                raise ValidationError("token vocabularies must be unique and in stable order")
            total_vocab_values += len(normalized)
            total_vocab_bytes += sum(_token_storage_bytes(value) for value in normalized)
            if total_vocab_values > copied_limits.max_vocab_values:
                raise ValidationError("all vocabularies together exceed max_vocab_values")
            if total_vocab_bytes > copied_limits.max_vocab_token_bytes:
                raise ValidationError("all vocabularies together exceed max_vocab_token_bytes")
            vocabularies[name] = normalized
        statistics: dict[str, NumericStatistics] = {}
        for name, value in raw_statistics.items():
            if type(value) is not NumericStatistics:
                raise ValidationError("numeric statistics must contain NumericStatistics values")
            statistics[name] = NumericStatistics(value.count, value.mean, value.scale)
        rows = _plain_int(training_rows, "training_rows", positive=True)
        training_value_count = _plain_int(training_values, "training_values")
        if training_value_count < 0:
            raise ValidationError("training_values must be non-negative")
        if rows > copied_limits.max_rows or training_value_count > copied_limits.max_total_values:
            raise ValidationError("training provenance exceeds the persisted feature limits")
        object.__setattr__(self, "schema", copied_schema)
        object.__setattr__(self, "token_vocabularies", MappingProxyType(vocabularies))
        object.__setattr__(
            self,
            "token_indices",
            MappingProxyType(
                {
                    name: MappingProxyType(
                        {token: index + 2 for index, token in enumerate(vocabulary)}
                    )
                    for name, vocabulary in vocabularies.items()
                }
            ),
        )
        object.__setattr__(self, "numeric_statistics", MappingProxyType(statistics))
        object.__setattr__(self, "training_rows", rows)
        object.__setattr__(self, "training_values", training_value_count)
        object.__setattr__(self, "training_sha256", _sha256(training_sha256, "training_sha256"))
        object.__setattr__(self, "limits", copied_limits)

    @classmethod
    def fit(
        cls,
        training: FeatureDataset,
        *,
        limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
    ) -> FittedFeaturePipeline:
        if type(training) is not FeatureDataset:
            raise ValidationError("training must be a FeatureDataset")
        active_limits = _copy_limits(limits)
        snapshot = FeatureDataset(training.schema, training.rows, limits=active_limits)
        if not snapshot:
            raise ValidationError("training feature dataset must not be empty")
        present_sources = {row.source for row in snapshot}
        required_sources = {feature.source for feature in snapshot.schema}
        missing_sources = sorted(source.value for source in required_sources - present_sources)
        if missing_sources:
            raise ValidationError(
                f"training feature dataset has no rows for: {', '.join(missing_sources)}"
            )
        vocabularies: dict[str, tuple[Token, ...]] = {}
        statistics: dict[str, NumericStatistics] = {}
        total_vocab_values = 0
        total_vocab_bytes = 0
        for feature in snapshot.schema:
            if feature.kind.is_token:
                vocabulary = _tokens(snapshot, feature)
                total_vocab_values += len(vocabulary)
                total_vocab_bytes += sum(_token_storage_bytes(value) for value in vocabulary)
                if total_vocab_values > active_limits.max_vocab_values:
                    raise ValidationError("fitted vocabularies exceed max_vocab_values")
                if total_vocab_bytes > active_limits.max_vocab_token_bytes:
                    raise ValidationError("fitted vocabularies exceed max_vocab_token_bytes")
                vocabularies[feature.name] = vocabulary
            else:
                statistics[feature.name] = _fit_numeric(_numeric_values(snapshot, feature))
        return cls(
            snapshot.schema,
            vocabularies,
            statistics,
            len(snapshot),
            snapshot.total_values,
            feature_dataset_sha256(snapshot),
            active_limits,
        )

    def _encode_token(self, feature: FeatureSpec, value: object) -> int:
        token = _plain_token(value, f"feature {feature.name!r}", self.limits)
        return self.token_indices[feature.name].get(token, UNKNOWN_INDEX)

    def _encode_numeric(self, feature: FeatureSpec, value: object) -> float:
        number = _plain_float(value, f"feature {feature.name!r}")
        statistics = self.numeric_statistics[feature.name]
        normalized = (number / statistics.scale) - (statistics.mean / statistics.scale)
        if not math.isfinite(normalized):
            raise ValidationError(f"feature {feature.name!r} cannot be normalized finitely")
        return normalized

    def transform(self, dataset: FeatureDataset) -> EncodedFeatureDataset:
        """Encode rows without learning anything from the transformed split."""

        if type(dataset) is not FeatureDataset:
            raise ValidationError("dataset must be a FeatureDataset")
        snapshot = FeatureDataset(dataset.schema, dataset.rows, limits=self.limits)
        if snapshot.schema != self.schema:
            raise ValidationError("transform schema must exactly match the fitted schema")
        rows: list[EncodedFeatureRow] = []
        for row in snapshot:
            encoded: dict[str, EncodedFeatureValue] = {}
            lengths: dict[str, int] = {}
            for feature in self.schema.for_source(row.source):
                value = row.values[feature.name]
                if feature.kind is FeatureKind.TOKEN:
                    if isinstance(value, (tuple, float)):
                        raise ValidationError(f"feature {feature.name!r} must be a token")
                    encoded[feature.name] = self._encode_token(feature, value)
                elif feature.kind is FeatureKind.FLOAT:
                    if isinstance(value, (tuple, str)):
                        raise ValidationError(f"feature {feature.name!r} must be numeric")
                    encoded[feature.name] = self._encode_numeric(feature, value)
                else:
                    if not isinstance(value, tuple):
                        raise ValidationError(f"feature {feature.name!r} must be a sequence")
                    width = feature.sequence_length
                    if width is None:
                        raise ValidationError("sequence feature has no declared width")
                    retained = (
                        value[:width] if feature.keep is SequenceKeep.HEAD else value[-width:]
                    )
                    lengths[feature.name] = len(retained)
                    if feature.kind is FeatureKind.TOKEN_SEQUENCE:
                        if any(isinstance(item, float) for item in retained):
                            raise ValidationError(
                                f"feature {feature.name!r} must be a token sequence"
                            )
                        encoded_values = tuple(
                            self._encode_token(feature, item) for item in retained
                        )
                        encoded[feature.name] = encoded_values + (PAD_INDEX,) * (
                            width - len(encoded_values)
                        )
                    else:
                        if any(isinstance(item, str) for item in retained):
                            raise ValidationError(
                                f"feature {feature.name!r} must be a numeric sequence"
                            )
                        numeric = tuple(self._encode_numeric(feature, item) for item in retained)
                        encoded[feature.name] = numeric + (0.0,) * (width - len(numeric))
            rows.append(
                EncodedFeatureRow(
                    row.source,
                    row.key,
                    encoded,
                    lengths,
                    limits=self.limits,
                )
            )
        return EncodedFeatureDataset(
            self.schema,
            self.state_sha256,
            rows,
            limits=self.limits,
        )

    def token_at(self, feature_name: str, index: int) -> Token | None:
        """Decode a learned token index; PAD and UNKNOWN intentionally return ``None``."""

        name = _plain_text(feature_name, "feature_name")
        normalized_index = _plain_int(index, "index")
        if name not in self.token_vocabularies:
            raise ValidationError(f"unknown token feature: {name!r}")
        if normalized_index in {PAD_INDEX, UNKNOWN_INDEX}:
            return None
        position = normalized_index - 2
        if position < 0 or position >= len(self.token_vocabularies[name]):
            raise ValidationError("token index is outside the fitted vocabulary")
        return self.token_vocabularies[name][position]

    def _state_without_digest(self) -> dict[str, object]:
        return {
            "format": FEATURE_PIPELINE_FORMAT,
            "schema_version": FEATURE_SCHEMA_VERSION,
            "schema": self.schema.to_state(),
            "limits": self.limits.to_state(),
            "token_vocabularies": {
                name: list(values) for name, values in self.token_vocabularies.items()
            },
            "numeric_statistics": {
                name: statistics.to_state() for name, statistics in self.numeric_statistics.items()
            },
            "training": {
                "rows": self.training_rows,
                "values": self.training_values,
                "sha256": self.training_sha256,
            },
        }

    @property
    def state_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self._state_without_digest())).hexdigest()

    def to_state(self) -> dict[str, object]:
        state = self._state_without_digest()
        state["state_sha256"] = self.state_sha256
        return state

    @classmethod
    def from_state(cls, state: object) -> FittedFeaturePipeline:
        expected = {
            "format",
            "schema_version",
            "schema",
            "limits",
            "token_vocabularies",
            "numeric_statistics",
            "training",
            "state_sha256",
        }
        normalized = _state_mapping(
            state, expected, "feature pipeline has missing or unknown fields"
        )
        schema_version = normalized["schema_version"]
        format_name = normalized["format"]
        if (
            type(format_name) is not str
            or str.__str__(format_name) != FEATURE_PIPELINE_FORMAT
            or type(schema_version) is not int
            or int.__index__(schema_version) != FEATURE_SCHEMA_VERSION
        ):
            raise SerializationError("unsupported feature pipeline format or schema version")
        limits = FeatureLimits.from_state(normalized["limits"])
        schema = FeatureSchema.from_state(normalized["schema"], limits=limits)
        training = normalized["training"]
        training_state = _state_mapping(
            training,
            {"rows", "values", "sha256"},
            "feature pipeline training provenance is malformed",
        )
        raw_vocabularies = normalized["token_vocabularies"]
        raw_statistics = normalized["numeric_statistics"]
        if type(raw_vocabularies) is not dict or type(raw_statistics) is not dict:
            raise SerializationError("feature pipeline fitted state must contain mappings")
        try:
            pipeline = cls(
                schema=schema,
                token_vocabularies=_vocabularies_from_state(raw_vocabularies),
                numeric_statistics=_statistics_from_state(raw_statistics),
                training_rows=cast(int, training_state["rows"]),
                training_values=cast(int, training_state["values"]),
                training_sha256=cast(str, training_state["sha256"]),
                limits=limits,
            )
            expected_digest = _sha256(normalized["state_sha256"], "state_sha256")
        except (TypeError, ValidationError) as error:
            raise SerializationError(f"invalid feature pipeline state: {error}") from error
        if pipeline.state_sha256 != expected_digest:
            raise SerializationError("feature pipeline state checksum does not match")
        return pipeline

    def save(self, path: str | Path) -> None:
        _atomic_write_json(
            Path(path),
            _json_chunks(self.to_state()),
            self.limits.max_state_bytes,
            "feature pipeline",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        max_state_bytes: int = DEFAULT_FEATURE_LIMITS.max_state_bytes,
    ) -> FittedFeaturePipeline:
        limit = _plain_int(max_state_bytes, "max_state_bytes", positive=True)
        if limit > _MAX_STATE_BYTES:
            raise ValidationError(f"max_state_bytes must be at most {_MAX_STATE_BYTES}")
        payload = _load_json_file(Path(path), limit)
        return cls.from_state(payload)


def _vocabularies_from_state(state: dict[object, object]) -> dict[str, tuple[Token, ...]]:
    result: dict[str, tuple[Token, ...]] = {}
    for name, values in dict.items(state):
        if not isinstance(name, str) or type(values) is not list:
            raise SerializationError("token vocabularies must map names to arrays")
        result[str.__str__(name)] = cast(tuple[Token, ...], tuple(list.__iter__(values)))
    return result


def _statistics_from_state(
    state: dict[object, object],
) -> dict[str, NumericStatistics]:
    result: dict[str, NumericStatistics] = {}
    for name, value in dict.items(state):
        if not isinstance(name, str):
            raise SerializationError("numeric statistic names must be strings")
        result[str.__str__(name)] = NumericStatistics.from_state(value)
    return result


def save_feature_dataset(
    dataset: FeatureDataset,
    path: str | Path,
    *,
    limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
) -> None:
    """Persist a strict raw feature dataset atomically."""

    if type(dataset) is not FeatureDataset:
        raise ValidationError("dataset must be a FeatureDataset")
    active_limits = _copy_limits(limits)
    snapshot = FeatureDataset(dataset.schema, dataset.rows, limits=active_limits)
    _atomic_write_json(
        Path(path),
        _feature_dataset_json_chunks(snapshot),
        active_limits.max_state_bytes,
        "feature dataset",
    )


def load_feature_dataset(
    path: str | Path,
    *,
    limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
) -> FeatureDataset:
    """Read a bounded strict raw feature dataset."""

    copied_limits = _copy_limits(limits)
    return FeatureDataset.from_state(
        _load_json_file(Path(path), copied_limits.max_state_bytes),
        limits=copied_limits,
    )


def save_encoded_features(
    dataset: EncodedFeatureDataset,
    path: str | Path,
    *,
    limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
) -> None:
    """Persist model-ready rows with their producing pipeline identity."""

    if type(dataset) is not EncodedFeatureDataset:
        raise ValidationError("dataset must be an EncodedFeatureDataset")
    active_limits = _copy_limits(limits)
    snapshot = EncodedFeatureDataset(
        dataset.schema,
        dataset.pipeline_sha256,
        dataset.rows,
        limits=active_limits,
    )
    _atomic_write_json(
        Path(path),
        _encoded_feature_json_chunks(snapshot),
        active_limits.max_state_bytes,
        "encoded features",
    )


def load_encoded_features(
    path: str | Path,
    *,
    limits: FeatureLimits = DEFAULT_FEATURE_LIMITS,
) -> EncodedFeatureDataset:
    """Read bounded encoded rows and retain their producing pipeline identity."""

    copied_limits = _copy_limits(limits)
    return EncodedFeatureDataset.from_state(
        _load_json_file(Path(path), copied_limits.max_state_bytes),
        limits=copied_limits,
    )


def _load_json_file(path: Path, limit: int) -> object:
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except OSError as error:
        raise SerializationError(f"could not read feature state from {path}: {error}") from error
    if len(payload) > limit:
        raise SerializationError(f"feature state exceeds the byte limit of {limit}")
    try:
        state = strict_json_loads(payload.decode("utf-8"))
        _validate_json_depth(state)
        return state
    except (RecursionError, UnicodeDecodeError, ValueError) as error:
        raise SerializationError(f"invalid feature JSON in {path}: {error}") from error


def _validate_json_depth(state: object) -> None:
    """Reject excessively nested state without relying on interpreter recursion limits."""

    pending: list[tuple[Iterator[object], int]] = [(iter((state,)), 0)]
    while pending:
        values, depth = pending[-1]
        try:
            value = next(values)
        except StopIteration:
            pending.pop()
            continue
        if type(value) is dict:
            if depth >= _MAX_JSON_DEPTH and value:
                raise ValueError(f"feature JSON exceeds the maximum depth of {_MAX_JSON_DEPTH}")
            pending.append((iter(dict.values(value)), depth + 1))
        elif type(value) is list:
            if depth >= _MAX_JSON_DEPTH and value:
                raise ValueError(f"feature JSON exceeds the maximum depth of {_MAX_JSON_DEPTH}")
            pending.append((list.__iter__(value), depth + 1))


_JSON_ENCODER = json.JSONEncoder(
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
)


def _json_chunks(value: object) -> Iterator[str]:
    yield from _JSON_ENCODER.iterencode(value)


def _feature_dataset_json_chunks(dataset: FeatureDataset) -> Iterator[str]:
    """Yield the exact canonical dataset JSON without copying the row collection."""

    yield '{"format":'
    yield from _json_chunks(FEATURE_DATASET_FORMAT)
    yield ',"rows":['
    for index, row in enumerate(dataset):
        if index:
            yield ","
        yield from _json_chunks(row.to_state())
    yield '],"schema":'
    yield from _json_chunks(dataset.schema.to_state())
    yield ',"schema_version":'
    yield from _json_chunks(FEATURE_SCHEMA_VERSION)
    yield "}"


def _encoded_feature_json_chunks(dataset: EncodedFeatureDataset) -> Iterator[str]:
    """Yield deterministic encoded feature JSON one row at a time."""

    yield '{"format":'
    yield from _json_chunks(ENCODED_FEATURES_FORMAT)
    yield ',"pipeline_sha256":'
    yield from _json_chunks(dataset.pipeline_sha256)
    yield ',"rows":['
    for index, row in enumerate(dataset):
        if index:
            yield ","
        yield from _json_chunks(row.to_state())
    yield '],"schema":'
    yield from _json_chunks(dataset.schema.to_state())
    yield ',"schema_version":'
    yield from _json_chunks(FEATURE_SCHEMA_VERSION)
    yield "}"


def _fsync_directory(directory: Path) -> None:
    """Persist a directory entry on platforms that expose directory descriptors."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(
    path: Path,
    chunks: Iterable[str],
    max_bytes: int,
    label: str,
) -> None:
    descriptor: int | None = None
    temporary: Path | None = None
    installed = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        stream = os.fdopen(descriptor, "wb")
        descriptor = None
        total_bytes = 0
        with stream:
            for text in chunks:
                encoded = text.encode("utf-8")
                total_bytes += len(encoded)
                if total_bytes + 1 > max_bytes:
                    raise SerializationError(f"{label} exceeds max_state_bytes")
                stream.write(encoded)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if temporary is None:
            raise SerializationError("could not create a feature-state staging file")
        os.replace(temporary, path)
        installed = True
        _fsync_directory(path.parent)
    except BaseException as error:
        if isinstance(error, OSError):
            raise SerializationError(f"could not write feature state to {path}: {error}") from error
        raise
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if not installed and temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
