"""Strict, local RecBole-style .user/.item side-feature interchange.

This intentionally supports only atomic user/item tables, not RecBole's dataset
registry, interaction joins, knowledge graphs, or model preprocessing semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass, fields
from pathlib import Path
from typing import cast

from orchidrec._json import strict_json_loads
from orchidrec.errors import DatasetError, SerializationError, ValidationError
from orchidrec.features import (
    FEATURE_DATASET_FORMAT,
    FEATURE_SCHEMA_VERSION,
    FeatureDataset,
    FeatureKind,
    FeatureLimits,
    FeatureRow,
    FeatureSchema,
    FeatureSource,
    FeatureSpec,
    SequenceKeep,
    feature_dataset_sha256,
)

SIDE_FORMAT = "orchidrec.recbole-side-features"
SIDE_VERSION = 1
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_FLOAT = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_TYPE = {
    "token": FeatureKind.TOKEN,
    "float": FeatureKind.FLOAT,
    "token_seq": FeatureKind.TOKEN_SEQUENCE,
    "float_seq": FeatureKind.FLOAT_SEQUENCE,
}


@dataclass(frozen=True, slots=True)
class RecBoleSideLimits:
    """Explicit finite limits, applied before retaining decoded rows."""

    max_file_bytes: int = 16 * 1024 * 1024
    max_line_bytes: int = 64 * 1024
    max_rows: int = 100_000
    max_fields: int = 128
    max_sequence_values: int = 4_096
    max_total_values: int = 1_000_000
    max_output_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        ceilings = {
            "max_file_bytes": 64 * 1024 * 1024,
            "max_line_bytes": 1024 * 1024,
            "max_rows": 1_000_000,
            "max_fields": 256,
            "max_sequence_values": 4_096,
            "max_total_values": 5_000_000,
            "max_output_bytes": 256 * 1024 * 1024,
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or not 0 < value <= ceilings[field.name]:
                raise ValidationError(
                    f"{field.name} must be an integer in 1..{ceilings[field.name]}"
                )

    def to_state(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_state(cls, value: object) -> RecBoleSideLimits:
        if type(value) is not dict or set(value) != {field.name for field in fields(cls)}:
            raise SerializationError("side-feature limits have missing or unknown fields")
        try:
            return cls(**cast(dict[str, int], value))
        except ValidationError as error:
            raise SerializationError(f"invalid side-feature limits: {error}") from error

    def feature_limits(self) -> FeatureLimits:
        return FeatureLimits(
            max_fields=self.max_fields,
            max_rows=self.max_rows,
            max_total_values=self.max_total_values,
            max_sequence_values=self.max_sequence_values,
            max_state_bytes=self.max_output_bytes,
        )


DEFAULT_RECBOLE_SIDE_LIMITS = RecBoleSideLimits()


@dataclass(frozen=True, slots=True)
class SideSource:
    kind: str
    sha256: str
    bytes: int
    rows: int

    def to_state(self) -> dict[str, str | int]:
        return {
            "kind": self.kind,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "rows": self.rows,
        }

    @classmethod
    def from_state(cls, value: object) -> SideSource:
        if type(value) is not dict or set(value) != {"kind", "sha256", "bytes", "rows"}:
            raise SerializationError("side-feature source has missing or unknown fields")
        source = cast(dict[str, object], value)
        kind, digest, size, rows = (
            source["kind"],
            source["sha256"],
            source["bytes"],
            source["rows"],
        )
        if (
            type(kind) is not str
            or kind not in {"user", "item"}
            or type(digest) is not str
            or not _SHA.fullmatch(digest)
            or type(size) is not int
            or size <= 0
            or type(rows) is not int
            or rows <= 0
        ):
            raise SerializationError("invalid side-feature source provenance")
        return cls(kind, digest, size, rows)


@dataclass(frozen=True, slots=True)
class LoadedRecBoleSideFeatures:
    dataset: FeatureDataset
    sources: tuple[SideSource, ...]
    limits: RecBoleSideLimits
    schema_reference_sha256: str | None = None

    def to_state(self) -> dict[str, object]:
        body: dict[str, object] = {
            "format": SIDE_FORMAT,
            "schema_version": SIDE_VERSION,
            "dataset": self.dataset.to_state(),
            "dataset_sha256": feature_dataset_sha256(self.dataset),
            "sources": [source.to_state() for source in self.sources],
            "limits": self.limits.to_state(),
            "schema_reference_sha256": self.schema_reference_sha256,
        }
        body["state_sha256"] = _digest(body)
        return body


_ENCODER = json.JSONEncoder(
    sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
)


def _canonical(value: object, *, max_bytes: int) -> bytes:
    result = bytearray()
    try:
        for part in _ENCODER.iterencode(value):
            encoded = part.encode("utf-8")
            if len(result) + len(encoded) > max_bytes:
                raise SerializationError("side-feature artifact exceeds max_output_bytes")
            result.extend(encoded)
    except (TypeError, ValueError, UnicodeEncodeError, OverflowError) as error:
        raise SerializationError(f"side-feature state is not canonical JSON: {error}") from error
    return bytes(result)


def _digest(value: object) -> str:
    digest = hashlib.sha256()
    try:
        for part in _ENCODER.iterencode(value):
            digest.update(part.encode("utf-8"))
    except (TypeError, ValueError, UnicodeEncodeError, OverflowError) as error:
        raise SerializationError(f"side-feature state is not canonical JSON: {error}") from error
    return digest.hexdigest()


def _preflight_dataset(dataset: FeatureDataset, max_bytes: int) -> None:
    """Reject oversized expanded row keys before making a whole dataset state."""

    size = 0
    try:
        parts = (
            '{"format":',
            *_ENCODER.iterencode(FEATURE_DATASET_FORMAT),
            ',"rows":[',
        )
        for part in parts:
            size += len(part.encode("utf-8"))
            if size > max_bytes:
                raise SerializationError("side-feature artifact exceeds max_output_bytes")
        for index, row in enumerate(dataset):
            if index:
                size += 1
                if size > max_bytes:
                    raise SerializationError("side-feature artifact exceeds max_output_bytes")
            for part in _ENCODER.iterencode(row.to_state()):
                size += len(part.encode("utf-8"))
                if size > max_bytes:
                    raise SerializationError("side-feature artifact exceeds max_output_bytes")
        size += len('],"schema":')
        for part in _ENCODER.iterencode(dataset.schema.to_state()):
            size += len(part.encode("utf-8"))
            if size > max_bytes:
                raise SerializationError("side-feature artifact exceeds max_output_bytes")
        size += len(',"schema_version":')
        for part in _ENCODER.iterencode(FEATURE_SCHEMA_VERSION):
            size += len(part.encode("utf-8"))
        size += 1
        if size > max_bytes:
            raise SerializationError("side-feature artifact exceeds max_output_bytes")
    except (TypeError, ValueError, UnicodeEncodeError, OverflowError) as error:
        raise SerializationError(f"side-feature state is not canonical JSON: {error}") from error


def _token(value: str, label: str) -> str:
    if not value or len(value) > 512 or len(value.encode("utf-8")) > 2_048:
        raise DatasetError(f"{label} must be a non-empty token of at most 512 characters")
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise DatasetError(f"{label} must not contain whitespace or controls")
    return value


def _number(value: str, label: str) -> float:
    if not _FLOAT.fullmatch(value):
        raise DatasetError(f"{label} must be a finite decimal number")
    number = float(value)
    if not math.isfinite(number):
        raise DatasetError(f"{label} must be a finite decimal number")
    return number


def _parse_cell(value: str, kind: FeatureKind, label: str, limits: RecBoleSideLimits) -> object:
    if kind is FeatureKind.TOKEN:
        return _token(value, label)
    if kind is FeatureKind.FLOAT:
        return _number(value, label)
    if not value:
        return ()
    parts = value.split(" ")
    if len(parts) > limits.max_sequence_values:
        raise DatasetError(f"{label} exceeds max_sequence_values")
    if kind is FeatureKind.TOKEN_SEQUENCE:
        return tuple(_token(part, label) for part in parts)
    return tuple(_number(part, label) for part in parts)


def _physical_lines(payload: bytes, label: str, limits: RecBoleSideLimits) -> list[str]:
    if not payload or b"\r" in payload.replace(b"\r\n", b""):
        raise DatasetError(f"{label} is empty or contains a bare carriage return")
    if payload.count(b"\n") > limits.max_rows + 1:
        raise DatasetError(f"{label} exceeds max_rows")
    lines = payload.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    if len(lines) < 2 or len(lines) - 1 > limits.max_rows:
        raise DatasetError(f"{label} must contain 1..{limits.max_rows} data rows")
    decoded: list[str] = []
    for number, line in enumerate(lines, 1):
        if len(line) > limits.max_line_bytes:
            raise DatasetError(f"{label} line {number} exceeds max_line_bytes")
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line:
            raise DatasetError(f"{label} line {number} is blank")
        try:
            decoded.append(line.decode("utf-8", errors="strict"))
        except UnicodeDecodeError as error:
            raise DatasetError(f"{label} line {number} is not UTF-8") from error
    return decoded


def _read_table(
    path: Path,
    source: FeatureSource,
    limits: RecBoleSideLimits,
) -> tuple[list[FeatureRow], list[FeatureSpec], SideSource, int]:
    if path.suffix != f".{source.value}" or not path.is_file():
        raise DatasetError(f"{source.value} input must be a local .{source.value} file")
    try:
        with path.open("rb") as stream:
            payload = stream.read(limits.max_file_bytes + 1)
    except OSError as error:
        raise DatasetError(f"could not read {source.value} input: {error}") from error
    if len(payload) > limits.max_file_bytes:
        raise DatasetError(f"{source.value} input exceeds max_file_bytes")
    lines = _physical_lines(payload, source.value, limits)
    header = lines[0].split("\t")
    if len(header) < 2 or len(header) - 1 > limits.max_fields:
        raise DatasetError(
            f"{source.value} header must have an ID and 1..{limits.max_fields} fields"
        )
    columns: list[tuple[str, FeatureKind]] = []
    seen: set[str] = set()
    id_name = f"{source.value}_id"
    for column in header:
        pieces = column.split(":")
        if len(pieces) != 2 or not _NAME.fullmatch(pieces[0]) or pieces[0] in seen:
            raise DatasetError(f"{source.value} header has malformed or duplicate field")
        name, type_name = pieces
        if name == id_name:
            if type_name != "token":
                raise DatasetError(f"{id_name} must have token type")
        elif type_name not in _TYPE:
            raise DatasetError(f"{source.value} header has unsupported field type {type_name!r}")
        columns.append((name, _TYPE[type_name]))
        seen.add(name)
    if id_name not in seen:
        raise DatasetError(f"{source.value} header is missing {id_name}:token")
    id_index = next(index for index, (name, _) in enumerate(columns) if name == id_name)
    widths: dict[str, int] = {name: 0 for name, kind in columns if kind.is_sequence}
    rows: list[FeatureRow] = []
    keys: set[str] = set()
    total_values = 0
    feature_limits = limits.feature_limits()
    for line_number, line in enumerate(lines[1:], 2):
        cells = line.split("\t")
        if len(cells) != len(columns):
            raise DatasetError(f"{source.value} line {line_number} has wrong field count")
        key = _token(cells[id_index], f"{source.value} line {line_number} ID")
        if key in keys:
            raise DatasetError(f"{source.value} line {line_number} repeats ID {key!r}")
        keys.add(key)
        values: dict[str, object] = {}
        for (name, kind), cell in zip(columns, cells, strict=True):
            if name == id_name:
                continue
            full_name = f"{source.value}.{name}"
            value = _parse_cell(cell, kind, f"{source.value} line {line_number} {name}", limits)
            if isinstance(value, tuple):
                widths[name] = max(widths[name], len(value))
                total_values += len(value)
            else:
                total_values += 1
            if total_values > limits.max_total_values:
                raise DatasetError("side features exceed max_total_values")
            values[full_name] = value
        rows.append(FeatureRow(source, key, values, limits=feature_limits))
    specs = [
        FeatureSpec(
            f"{source.value}.{name}",
            kind,
            source,
            max(1, widths[name]) if kind.is_sequence else None,
            SequenceKeep.HEAD if kind.is_sequence else None,
        )
        for name, kind in columns
        if name != id_name
    ]
    provenance = SideSource(
        source.value, hashlib.sha256(payload).hexdigest(), len(payload), len(rows)
    )
    return rows, specs, provenance, total_values


def import_recbole_side_features(
    *,
    user_path: str | Path | None = None,
    item_path: str | Path | None = None,
    limits: RecBoleSideLimits = DEFAULT_RECBOLE_SIDE_LIMITS,
    schema_from: LoadedRecBoleSideFeatures | None = None,
) -> LoadedRecBoleSideFeatures:
    """Import a bounded local snapshot; IDs stay strings and no join is performed."""

    if type(limits) is not RecBoleSideLimits:
        raise ValidationError("limits must be exact RecBoleSideLimits")
    if user_path is None and item_path is None:
        raise ValidationError("at least one .user or .item input is required")
    if schema_from is not None and type(schema_from) is not LoadedRecBoleSideFeatures:
        raise ValidationError("schema_from must be a loaded side-feature artifact")
    rows: list[FeatureRow] = []
    specs: list[FeatureSpec] = []
    sources: list[SideSource] = []
    values = 0
    for path, source in ((user_path, FeatureSource.USER), (item_path, FeatureSource.ITEM)):
        if path is None:
            continue
        table_rows, table_specs, provenance, table_values = _read_table(Path(path), source, limits)
        rows.extend(table_rows)
        specs.extend(table_specs)
        sources.append(provenance)
        values += table_values
        if len(rows) > limits.max_rows or len(specs) > limits.max_fields:
            raise DatasetError("side features exceed max_rows or max_fields")
        if values > limits.max_total_values:
            raise DatasetError("side features exceed max_total_values")
    feature_limits = limits.feature_limits()
    observed_schema = FeatureSchema(
        sorted(specs, key=lambda spec: spec.name), limits=feature_limits
    )
    reference_digest: str | None = None
    if schema_from is not None:
        reference = {spec.name: spec for spec in schema_from.dataset.schema}
        observed = {spec.name: spec for spec in observed_schema}
        if set(reference) != set(observed) or any(
            reference[name].kind != spec.kind or reference[name].source != spec.source
            for name, spec in observed.items()
        ):
            raise DatasetError("side-feature fields and kinds differ from training schema")
        for name, spec in observed.items():
            if spec.kind.is_sequence and cast(int, spec.sequence_length) > cast(
                int, reference[name].sequence_length
            ):
                raise DatasetError(f"side-feature {name!r} exceeds trained sequence width")
        schema = FeatureSchema(schema_from.dataset.schema.features, limits=feature_limits)
        reference_digest = cast(str, schema_from.to_state()["state_sha256"])
    else:
        schema = observed_schema
    dataset = FeatureDataset(schema, rows, limits=feature_limits)
    return LoadedRecBoleSideFeatures(dataset, tuple(sources), limits, reference_digest)


def save_recbole_side_features(loaded: LoadedRecBoleSideFeatures, path: str | Path) -> None:
    """Publish a bounded, checksummed artifact atomically without overwriting."""

    if type(loaded) is not LoadedRecBoleSideFeatures:
        raise ValidationError("loaded must be LoadedRecBoleSideFeatures")
    destination = Path(path)
    _preflight_dataset(loaded.dataset, loaded.limits.max_output_bytes - 1)
    payload = _canonical(loaded.to_state(), max_bytes=loaded.limits.max_output_bytes - 1) + b"\n"
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            written = stream.write(payload)
            if written != len(payload):
                raise SerializationError("short write of side-feature artifact")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValidationError("side-feature output already exists") from error
    except OSError as error:
        raise SerializationError(f"could not save side-feature artifact: {error}") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def load_recbole_side_features(
    path: str | Path, *, max_output_bytes: int = 64 * 1024 * 1024
) -> LoadedRecBoleSideFeatures:
    """Load and verify an interchange artifact, including normalized fingerprint."""

    if type(max_output_bytes) is not int or not 0 < max_output_bytes <= 256 * 1024 * 1024:
        raise ValidationError("max_output_bytes must be an integer in 1..268435456")
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(max_output_bytes + 1)
    except OSError as error:
        raise SerializationError(f"could not read side-feature artifact: {error}") from error
    if len(raw) > max_output_bytes:
        raise SerializationError("side-feature artifact exceeds max_output_bytes")
    try:
        state = strict_json_loads(raw)
    except ValueError as error:
        raise SerializationError(f"invalid side-feature JSON: {error}") from error
    expected = {
        "format",
        "schema_version",
        "dataset",
        "dataset_sha256",
        "sources",
        "limits",
        "schema_reference_sha256",
        "state_sha256",
    }
    if type(state) is not dict or set(state) != expected:
        raise SerializationError("side-feature artifact has missing or unknown fields")
    body = cast(dict[str, object], state)
    if (
        body["format"] != SIDE_FORMAT
        or type(body["schema_version"]) is not int
        or body["schema_version"] != SIDE_VERSION
    ):
        raise SerializationError("unsupported side-feature format or schema version")
    digest = body.pop("state_sha256")
    if type(digest) is not str or not _SHA.fullmatch(digest) or _digest(body) != digest:
        raise SerializationError("side-feature artifact checksum mismatch")
    limits = RecBoleSideLimits.from_state(body["limits"])
    if len(raw) > limits.max_output_bytes:
        raise SerializationError("side-feature artifact exceeds its declared max_output_bytes")
    dataset = FeatureDataset.from_state(body["dataset"], limits=limits.feature_limits())
    if body["dataset_sha256"] != feature_dataset_sha256(dataset):
        raise SerializationError("side-feature dataset fingerprint mismatch")
    reference_digest = body["schema_reference_sha256"]
    if reference_digest is not None and (
        type(reference_digest) is not str or not _SHA.fullmatch(reference_digest)
    ):
        raise SerializationError("invalid side-feature training schema reference")
    sources_value = body["sources"]
    if type(sources_value) is not list or not 1 <= len(sources_value) <= 2:
        raise SerializationError("side-feature sources must contain one or two entries")
    sources = tuple(SideSource.from_state(item) for item in sources_value)
    if tuple(source.kind for source in sources) not in {("user",), ("item",), ("user", "item")}:
        raise SerializationError("side-feature source ordering or duplication is invalid")
    if sum(source.rows for source in sources) != len(dataset):
        raise SerializationError("side-feature source row counts do not match dataset")
    for source in sources:
        if source.bytes > limits.max_file_bytes or source.rows > limits.max_rows:
            raise SerializationError("side-feature source exceeds declared limits")
        if len(tuple(row for row in dataset if row.source.value == source.kind)) != source.rows:
            raise SerializationError("side-feature source rows do not match dataset")
    if {feature.source.value for feature in dataset.schema} != {source.kind for source in sources}:
        raise SerializationError("side-feature source schema does not match provenance")
    return LoadedRecBoleSideFeatures(dataset, sources, limits, reference_digest)
