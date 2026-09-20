"""Strict bounded interchange for a local RecBole-style social edge file.

This is a directed edge snapshot, not a social recommender or graph split.
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
from orchidrec.datasets import load_recbole_inter
from orchidrec.errors import DatasetError, SerializationError, ValidationError

NETWORK_FORMAT = "orchidrec.recbole-network"
NETWORK_VERSION = 1
_COLUMNS = ("source_id:token", "target_id:token")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TEXT_CHARS = 16_384
_MAX_UTF8_BYTES = 65_536
_ENCODER = json.JSONEncoder(
    sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
)


@dataclass(frozen=True, slots=True)
class NetworkLimits:
    """Limits enforced on source bytes, expanded records, and JSON output."""

    max_file_bytes: int = 16 * 1024 * 1024
    max_line_bytes: int = 64 * 1024
    max_rows: int = 100_000
    max_token_chars: int = 512
    max_token_bytes: int = 2_048
    max_output_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        ceilings = {
            "max_file_bytes": 64 * 1024 * 1024,
            "max_line_bytes": 1024 * 1024,
            "max_rows": 1_000_000,
            "max_token_chars": _MAX_TEXT_CHARS,
            "max_token_bytes": _MAX_UTF8_BYTES,
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
    def from_state(cls, value: object) -> NetworkLimits:
        if type(value) is not dict or set(value) != {field.name for field in fields(cls)}:
            raise SerializationError("network limits have missing or unknown fields")
        try:
            return cls(**cast(dict[str, int], value))
        except ValidationError as error:
            raise SerializationError(f"invalid network limits: {error}") from error


DEFAULT_NETWORK_LIMITS = NetworkLimits()


def _token(value: object, label: str, limits: NetworkLimits) -> str:
    if type(value) is not str or not value:
        raise DatasetError(f"{label} must be a non-empty token string")
    if len(value) > limits.max_token_chars or len(value.encode("utf-8")) > limits.max_token_bytes:
        raise DatasetError(f"{label} exceeds token length limits")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise DatasetError(f"{label} must not contain whitespace or controls")
    return value


def _digest(value: object) -> str:
    digest = hashlib.sha256()
    try:
        for part in _ENCODER.iterencode(value):
            digest.update(part.encode("utf-8"))
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError) as error:
        raise SerializationError(f"network state is not canonical JSON: {error}") from error
    return digest.hexdigest()


def _canonical(value: object, max_bytes: int) -> bytes:
    result = bytearray()
    try:
        for part in _ENCODER.iterencode(value):
            encoded = part.encode("utf-8")
            if len(result) + len(encoded) > max_bytes:
                raise SerializationError("network artifact exceeds max_output_bytes")
            result.extend(encoded)
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError) as error:
        raise SerializationError(f"network state is not canonical JSON: {error}") from error
    return bytes(result)


@dataclass(frozen=True, order=True, slots=True)
class SocialEdge:
    source_id: str
    target_id: str

    def to_state(self) -> list[str]:
        return [self.source_id, self.target_id]


@dataclass(frozen=True, slots=True)
class NetworkSource:
    sha256: str
    bytes: int
    rows: int

    def to_state(self) -> dict[str, str | int]:
        return {"sha256": self.sha256, "bytes": self.bytes, "rows": self.rows}

    @classmethod
    def from_state(cls, value: object) -> NetworkSource:
        if type(value) is not dict or set(value) != {"sha256", "bytes", "rows"}:
            raise SerializationError("network source has missing or unknown fields")
        state = cast(dict[str, object], value)
        digest, size, rows = state["sha256"], state["bytes"], state["rows"]
        if (
            type(digest) is not str
            or not _SHA.fullmatch(digest)
            or type(size) is not int
            or size <= 0
            or type(rows) is not int
            or rows <= 0
        ):
            raise SerializationError("invalid network source provenance")
        return cls(digest, size, rows)


@dataclass(frozen=True, slots=True)
class NetworkCatalogReference:
    """Descriptive user overlap with one separately validated `.inter` file."""

    source_sha256: str
    normalized_sha256: str
    catalog_users: int
    network_users_in_catalog: int
    minimum_rating: float | None

    def to_state(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_sha256,
            "normalized_sha256": self.normalized_sha256,
            "catalog_users": self.catalog_users,
            "network_users_in_catalog": self.network_users_in_catalog,
            "minimum_rating": self.minimum_rating,
        }

    @classmethod
    def from_state(cls, value: object, network_users: int) -> NetworkCatalogReference:
        expected = {
            "source_sha256",
            "normalized_sha256",
            "catalog_users",
            "network_users_in_catalog",
            "minimum_rating",
        }
        if type(value) is not dict or set(value) != expected:
            raise SerializationError("network reference has missing or unknown fields")
        state = cast(dict[str, object], value)
        for name in ("source_sha256", "normalized_sha256"):
            digest = state[name]
            if type(digest) is not str or not _SHA.fullmatch(digest):
                raise SerializationError(f"invalid network reference {name}")
        count, matched = state["catalog_users"], state["network_users_in_catalog"]
        if (
            type(count) is not int
            or count <= 0
            or type(matched) is not int
            or not 0 <= matched <= min(count, network_users)
        ):
            raise SerializationError("invalid network reference overlap counts")
        threshold = state["minimum_rating"]
        if threshold is not None and (type(threshold) is not float or not math.isfinite(threshold)):
            raise SerializationError("invalid network reference rating threshold")
        return cls(
            cast(str, state["source_sha256"]),
            cast(str, state["normalized_sha256"]),
            count,
            matched,
            threshold,
        )


@dataclass(frozen=True, slots=True)
class LoadedSocialNetwork:
    edges: tuple[SocialEdge, ...]
    source: NetworkSource
    limits: NetworkLimits
    reference: NetworkCatalogReference | None = None

    @property
    def users(self) -> tuple[str, ...]:
        return tuple(sorted({id_ for edge in self.edges for id_ in edge.to_state()}))

    @property
    def fingerprint(self) -> str:
        return _digest([edge.to_state() for edge in self.edges])

    def to_state(self) -> dict[str, object]:
        body: dict[str, object] = {
            "format": NETWORK_FORMAT,
            "schema_version": NETWORK_VERSION,
            "edges": [edge.to_state() for edge in self.edges],
            "source": self.source.to_state(),
            "limits": self.limits.to_state(),
            "reference": self.reference.to_state() if self.reference is not None else None,
            "users": len(self.users),
            "fingerprint_sha256": self.fingerprint,
        }
        body["state_sha256"] = _digest(body)
        return body


def _read_net(
    path: str | Path, limits: NetworkLimits
) -> tuple[tuple[SocialEdge, ...], NetworkSource]:
    source = Path(path)
    if source.suffix != ".net" or not source.is_file():
        raise DatasetError("network input must be a local .net file")
    try:
        with source.open("rb") as stream:
            payload = stream.read(limits.max_file_bytes + 1)
    except OSError as error:
        raise DatasetError(f"could not read .net input: {error}") from error
    if not payload or len(payload) > limits.max_file_bytes:
        raise DatasetError(".net input is empty or exceeds max_file_bytes")
    if payload.endswith(b"\r"):
        raise DatasetError(".net input ends with bare CR; use LF or CRLF")
    physical_rows = payload.count(b"\n") + (not payload.endswith(b"\n"))
    if physical_rows < 2 or physical_rows - 1 > limits.max_rows:
        raise DatasetError(".net input violates max_rows")
    lines = payload.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    decoded: list[str] = []
    for number, line in enumerate(lines, 1):
        if len(line) > limits.max_line_bytes:
            raise DatasetError(f".net line {number} exceeds max_line_bytes")
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line or b"\r" in line:
            raise DatasetError(f".net line {number} is blank or contains bare CR")
        try:
            decoded.append(line.decode("utf-8", errors="strict"))
        except UnicodeDecodeError as error:
            raise DatasetError(f".net line {number} is not UTF-8") from error
    columns = decoded[0].split("\t")
    if len(columns) != 2 or set(columns) != set(_COLUMNS):
        raise DatasetError(".net header requires exactly source_id:token and target_id:token")
    edges: list[SocialEdge] = []
    for number, text_line in enumerate(decoded[1:], 2):
        cells = text_line.split("\t")
        if len(cells) != 2:
            raise DatasetError(f".net line {number} has wrong tab field count")
        row = dict(zip(columns, cells, strict=True))
        edges.append(
            SocialEdge(
                _token(row[_COLUMNS[0]], f".net line {number} source", limits),
                _token(row[_COLUMNS[1]], f".net line {number} target", limits),
            )
        )
    if len(edges) != len(set(edges)):
        raise DatasetError(".net input contains duplicate directed edges")
    return (
        tuple(sorted(edges)),
        NetworkSource(hashlib.sha256(payload).hexdigest(), len(payload), len(edges)),
    )


def import_recbole_network(
    *,
    net_path: str | Path,
    inter_path: str | Path | None = None,
    minimum_rating: float | None = None,
    limits: NetworkLimits = DEFAULT_NETWORK_LIMITS,
) -> LoadedSocialNetwork:
    """Import local directed edges, optionally measuring `.inter` user overlap."""

    if type(limits) is not NetworkLimits:
        raise ValidationError("limits must be exact NetworkLimits")
    if inter_path is None and minimum_rating is not None:
        raise ValidationError("minimum_rating requires inter_path")
    edges, source = _read_net(net_path, limits)
    reference = None
    if inter_path is not None:
        interaction = load_recbole_inter(inter_path, minimum_rating=minimum_rating)
        catalog = set(interaction.dataset.user_ids)
        users = {id_ for edge in edges for id_ in edge.to_state()}
        reference = NetworkCatalogReference(
            interaction.summary.source_sha256,
            interaction.summary.interactions_sha256,
            len(catalog),
            len(users & catalog),
            interaction.summary.minimum_rating,
        )
    return LoadedSocialNetwork(edges, source, limits, reference)


def _validate_semantics(loaded: LoadedSocialNetwork) -> None:
    if type(loaded.limits) is not NetworkLimits or type(loaded.source) is not NetworkSource:
        raise SerializationError("network limits or source has wrong type")
    limits = loaded.limits
    if type(loaded.edges) is not tuple or not 0 < len(loaded.edges) <= limits.max_rows:
        raise SerializationError("network records exceed declared row limits")
    try:
        for edge in loaded.edges:
            if type(edge) is not SocialEdge:
                raise SerializationError("network edge has wrong type")
            _token(edge.source_id, "source ID", limits)
            _token(edge.target_id, "target ID", limits)
    except (DatasetError, UnicodeEncodeError) as error:
        raise SerializationError(f"invalid network token: {error}") from error
    if loaded.edges != tuple(sorted(set(loaded.edges))):
        raise SerializationError("network edges are not unique and canonically sorted")
    NetworkSource.from_state(loaded.source.to_state())
    if loaded.source.rows != len(loaded.edges) or loaded.source.bytes > limits.max_file_bytes:
        raise SerializationError("network source exceeds declared row or byte limits")
    if loaded.reference is not None:
        if type(loaded.reference) is not NetworkCatalogReference:
            raise SerializationError("network reference has wrong type")
        NetworkCatalogReference.from_state(loaded.reference.to_state(), len(loaded.users))


def save_recbole_network(loaded: LoadedSocialNetwork, path: str | Path) -> None:
    """Publish a checksummed artifact atomically, refusing existing paths."""

    if type(loaded) is not LoadedSocialNetwork:
        raise ValidationError("loaded must be exact LoadedSocialNetwork")
    _validate_semantics(loaded)
    size = 0
    for edge in loaded.edges:
        for part in _ENCODER.iterencode(edge.to_state()):
            size += len(part.encode("utf-8"))
            if size >= loaded.limits.max_output_bytes:
                raise SerializationError("network artifact exceeds max_output_bytes")
    payload = _canonical(loaded.to_state(), loaded.limits.max_output_bytes - 1) + b"\n"
    destination = Path(path)
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
            if stream.write(payload) != len(payload):
                raise SerializationError("short write of network artifact")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValidationError("network output already exists") from error
    except OSError as error:
        raise SerializationError(f"could not save network artifact: {error}") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def load_recbole_network(
    path: str | Path, *, max_output_bytes: int = 64 * 1024 * 1024
) -> LoadedSocialNetwork:
    """Verify a bounded artifact, including semantic invariants after checksum."""

    if type(max_output_bytes) is not int or not 0 < max_output_bytes <= 256 * 1024 * 1024:
        raise ValidationError("max_output_bytes must be an integer in 1..268435456")
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(max_output_bytes + 1)
    except OSError as error:
        raise SerializationError(f"could not read network artifact: {error}") from error
    if len(raw) > max_output_bytes:
        raise SerializationError("network artifact exceeds max_output_bytes")
    try:
        state = strict_json_loads(raw)
    except ValueError as error:
        raise SerializationError(f"invalid network JSON: {error}") from error
    expected = {
        "format",
        "schema_version",
        "edges",
        "source",
        "limits",
        "reference",
        "users",
        "fingerprint_sha256",
        "state_sha256",
    }
    if type(state) is not dict or set(state) != expected:
        raise SerializationError("network artifact has missing or unknown fields")
    body = cast(dict[str, object], state)
    if (
        body["format"] != NETWORK_FORMAT
        or type(body["schema_version"]) is not int
        or body["schema_version"] != NETWORK_VERSION
    ):
        raise SerializationError("unsupported network format or schema version")
    digest = body.pop("state_sha256")
    if type(digest) is not str or not _SHA.fullmatch(digest) or _digest(body) != digest:
        raise SerializationError("network artifact checksum mismatch")
    limits = NetworkLimits.from_state(body["limits"])
    if len(raw) > limits.max_output_bytes:
        raise SerializationError("network artifact exceeds declared max_output_bytes")
    edge_values = body["edges"]
    if type(edge_values) is not list or not 0 < len(edge_values) <= limits.max_rows:
        raise SerializationError("network records exceed declared row limits")
    edges: list[SocialEdge] = []
    try:
        for index, row in enumerate(edge_values):
            if type(row) is not list or len(row) != 2:
                raise SerializationError(f"network edge {index} has wrong width")
            edges.append(SocialEdge(*(_token(cell, "edge token", limits) for cell in row)))
    except (DatasetError, UnicodeEncodeError) as error:
        raise SerializationError(f"invalid network token: {error}") from error
    source = NetworkSource.from_state(body["source"])
    reference = (
        NetworkCatalogReference.from_state(
            body["reference"], len({id_ for edge in edges for id_ in edge.to_state()})
        )
        if body["reference"] is not None
        else None
    )
    loaded = LoadedSocialNetwork(tuple(edges), source, limits, reference)
    _validate_semantics(loaded)
    if body["fingerprint_sha256"] != loaded.fingerprint:
        raise SerializationError("network normalized fingerprint mismatch")
    if type(body["users"]) is not int or body["users"] != len(loaded.users):
        raise SerializationError("network unique-user count mismatch")
    return loaded
