"""Bounded local RecBole-style knowledge triplet and item-link interchange.

This records a source snapshot and provenance. It does not remap IDs, train a
knowledge-aware model, filter a graph, or infer a recommendation split.
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
from orchidrec.features import FeatureSource
from orchidrec.recbole_side import load_recbole_side_features

KNOWLEDGE_FORMAT = "orchidrec.recbole-knowledge-links"
KNOWLEDGE_VERSION = 1
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_KG_COLUMNS = ("head_id:token", "relation_id:token", "tail_id:token")
_LINK_COLUMNS = ("item_id:token", "entity_id:token")
_MAX_TEXT_CHARS = 16_384
_MAX_UTF8_BYTES = 65_536
_ENCODER = json.JSONEncoder(
    sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
)


@dataclass(frozen=True, slots=True)
class KnowledgeLimits:
    """Finite source, row, token, and artifact ceilings."""

    max_file_bytes: int = 16 * 1024 * 1024
    max_line_bytes: int = 64 * 1024
    max_rows_per_file: int = 100_000
    max_total_rows: int = 200_000
    max_token_chars: int = 512
    max_token_bytes: int = 2_048
    max_output_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        ceilings = {
            "max_file_bytes": 64 * 1024 * 1024,
            "max_line_bytes": 1024 * 1024,
            "max_rows_per_file": 1_000_000,
            "max_total_rows": 2_000_000,
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
    def from_state(cls, value: object) -> KnowledgeLimits:
        if type(value) is not dict or set(value) != {field.name for field in fields(cls)}:
            raise SerializationError("knowledge limits have missing or unknown fields")
        try:
            return cls(**cast(dict[str, int], value))
        except ValidationError as error:
            raise SerializationError(f"invalid knowledge limits: {error}") from error


DEFAULT_KNOWLEDGE_LIMITS = KnowledgeLimits()


def _token(value: object, name: str, limits: KnowledgeLimits) -> str:
    if type(value) is not str or not value:
        raise DatasetError(f"{name} must be a non-empty token string")
    if len(value) > limits.max_token_chars or len(value.encode("utf-8")) > limits.max_token_bytes:
        raise DatasetError(f"{name} exceeds token length limits")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise DatasetError(f"{name} must not contain whitespace or controls")
    return value


@dataclass(frozen=True, order=True, slots=True)
class KnowledgeTriple:
    head_id: str
    relation_id: str
    tail_id: str

    def to_state(self) -> list[str]:
        return [self.head_id, self.relation_id, self.tail_id]


@dataclass(frozen=True, order=True, slots=True)
class ItemEntityLink:
    item_id: str
    entity_id: str

    def to_state(self) -> list[str]:
        return [self.item_id, self.entity_id]


@dataclass(frozen=True, slots=True)
class AtomicSource:
    kind: str
    sha256: str
    bytes: int
    rows: int

    def to_state(self) -> dict[str, str | int]:
        return {"kind": self.kind, "sha256": self.sha256, "bytes": self.bytes, "rows": self.rows}

    @classmethod
    def from_state(cls, value: object) -> AtomicSource:
        if type(value) is not dict or set(value) != {"kind", "sha256", "bytes", "rows"}:
            raise SerializationError("atomic source has missing or unknown fields")
        source = cast(dict[str, object], value)
        kind, digest, size, rows = (
            source["kind"],
            source["sha256"],
            source["bytes"],
            source["rows"],
        )
        if (
            type(kind) is not str
            or kind not in {"kg", "link"}
            or type(digest) is not str
            or not _SHA.fullmatch(digest)
            or type(size) is not int
            or size <= 0
            or type(rows) is not int
            or rows <= 0
        ):
            raise SerializationError("invalid atomic source provenance")
        return cls(kind, digest, size, rows)


@dataclass(frozen=True, slots=True)
class CatalogReference:
    """Optional read-only overlap with a separately validated local catalog."""

    kind: str
    source_sha256: str
    normalized_sha256: str
    catalog_items: int
    linked_items_in_catalog: int
    minimum_rating: float | None = None

    def to_state(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "source_sha256": self.source_sha256,
            "normalized_sha256": self.normalized_sha256,
            "catalog_items": self.catalog_items,
            "linked_items_in_catalog": self.linked_items_in_catalog,
            "minimum_rating": self.minimum_rating,
        }

    @classmethod
    def from_state(cls, value: object, link_count: int) -> CatalogReference:
        expected = {
            "kind",
            "source_sha256",
            "normalized_sha256",
            "catalog_items",
            "linked_items_in_catalog",
            "minimum_rating",
        }
        if type(value) is not dict or set(value) != expected:
            raise SerializationError("catalog reference has missing or unknown fields")
        state = cast(dict[str, object], value)
        kind = state["kind"]
        if type(kind) is not str or kind not in {"recbole-inter", "recbole-side"}:
            raise SerializationError("invalid catalog reference kind")
        for name in ("source_sha256", "normalized_sha256"):
            digest = state[name]
            if type(digest) is not str or not _SHA.fullmatch(digest):
                raise SerializationError(f"invalid catalog reference {name}")
        count, matched = state["catalog_items"], state["linked_items_in_catalog"]
        if (
            type(count) is not int
            or count < 0
            or type(matched) is not int
            or not 0 <= matched <= min(count, link_count)
        ):
            raise SerializationError("invalid catalog reference overlap counts")
        threshold = state["minimum_rating"]
        if kind == "recbole-side" and threshold is not None:
            raise SerializationError(
                "side-feature catalog reference cannot have a rating threshold"
            )
        if threshold is not None and (type(threshold) is not float or not math.isfinite(threshold)):
            raise SerializationError("invalid catalog reference rating threshold")
        return cls(
            kind,
            cast(str, state["source_sha256"]),
            cast(str, state["normalized_sha256"]),
            count,
            matched,
            threshold,
        )


def _digest(value: object) -> str:
    digest = hashlib.sha256()
    try:
        for part in _ENCODER.iterencode(value):
            digest.update(part.encode("utf-8"))
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError) as error:
        raise SerializationError(f"knowledge state is not canonical JSON: {error}") from error
    return digest.hexdigest()


def _canonical(value: object, max_bytes: int) -> bytes:
    result = bytearray()
    try:
        for part in _ENCODER.iterencode(value):
            encoded = part.encode("utf-8")
            if len(result) + len(encoded) > max_bytes:
                raise SerializationError("knowledge artifact exceeds max_output_bytes")
            result.extend(encoded)
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError) as error:
        raise SerializationError(f"knowledge state is not canonical JSON: {error}") from error
    return bytes(result)


@dataclass(frozen=True, slots=True)
class LoadedKnowledgeLinks:
    triples: tuple[KnowledgeTriple, ...]
    links: tuple[ItemEntityLink, ...]
    sources: tuple[AtomicSource, AtomicSource]
    limits: KnowledgeLimits
    references: tuple[CatalogReference, ...] = ()

    @property
    def linked_entities_in_kg(self) -> int:
        entities = {
            entity for triple in self.triples for entity in (triple.head_id, triple.tail_id)
        }
        return sum(link.entity_id in entities for link in self.links)

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "triples": [triple.to_state() for triple in self.triples],
                "links": [link.to_state() for link in self.links],
            }
        )

    def to_state(self) -> dict[str, object]:
        body: dict[str, object] = {
            "format": KNOWLEDGE_FORMAT,
            "schema_version": KNOWLEDGE_VERSION,
            "triples": [triple.to_state() for triple in self.triples],
            "links": [link.to_state() for link in self.links],
            "sources": [source.to_state() for source in self.sources],
            "limits": self.limits.to_state(),
            "references": [reference.to_state() for reference in self.references],
            "linked_entities_in_kg": self.linked_entities_in_kg,
            "fingerprint_sha256": self.fingerprint,
        }
        body["state_sha256"] = _digest(body)
        return body


def _read_table(
    path: str | Path, suffix: str, expected: tuple[str, ...], limits: KnowledgeLimits
) -> tuple[list[tuple[str, ...]], AtomicSource]:
    source = Path(path)
    if source.suffix != f".{suffix}" or not source.is_file():
        raise DatasetError(f"knowledge {suffix} input must be a local .{suffix} file")
    try:
        with source.open("rb") as stream:
            payload = stream.read(limits.max_file_bytes + 1)
    except OSError as error:
        raise DatasetError(f"could not read .{suffix} input: {error}") from error
    if not payload or len(payload) > limits.max_file_bytes:
        raise DatasetError(f".{suffix} input is empty or exceeds max_file_bytes")
    if payload.endswith(b"\r"):
        raise DatasetError(f".{suffix} input ends with bare CR; use LF or CRLF")
    physical_rows = payload.count(b"\n") + (not payload.endswith(b"\n"))
    if physical_rows < 2 or physical_rows - 1 > limits.max_rows_per_file:
        raise DatasetError(f".{suffix} input violates max_rows_per_file")
    lines = payload.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    decoded: list[str] = []
    for number, line in enumerate(lines, 1):
        if len(line) > limits.max_line_bytes:
            raise DatasetError(f".{suffix} line {number} exceeds max_line_bytes")
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line or b"\r" in line:
            raise DatasetError(f".{suffix} line {number} is blank or contains bare CR")
        try:
            decoded.append(line.decode("utf-8", errors="strict"))
        except UnicodeDecodeError as error:
            raise DatasetError(f".{suffix} line {number} is not UTF-8") from error
    columns = decoded[0].split("\t")
    if len(columns) != len(expected) or set(columns) != set(expected):
        raise DatasetError(f".{suffix} header requires exactly {', '.join(expected)}")
    rows: list[tuple[str, ...]] = []
    for number, text_line in enumerate(decoded[1:], 2):
        cells = text_line.split("\t")
        if len(cells) != len(expected):
            raise DatasetError(f".{suffix} line {number} has wrong tab field count")
        row = dict(zip(columns, cells, strict=True))
        rows.append(
            tuple(
                _token(row[column], f".{suffix} line {number} {column}", limits)
                for column in expected
            )
        )
    return rows, AtomicSource(suffix, hashlib.sha256(payload).hexdigest(), len(payload), len(rows))


def import_recbole_knowledge(
    *,
    kg_path: str | Path,
    link_path: str | Path,
    inter_path: str | Path | None = None,
    minimum_rating: float | None = None,
    side_features_path: str | Path | None = None,
    limits: KnowledgeLimits = DEFAULT_KNOWLEDGE_LIMITS,
) -> LoadedKnowledgeLinks:
    """Import a local KG/link pair and optional read-only catalog references."""

    if type(limits) is not KnowledgeLimits:
        raise ValidationError("limits must be exact KnowledgeLimits")
    if inter_path is None and minimum_rating is not None:
        raise ValidationError("minimum_rating requires inter_path")
    kg_rows, kg_source = _read_table(kg_path, "kg", _KG_COLUMNS, limits)
    link_rows, link_source = _read_table(link_path, "link", _LINK_COLUMNS, limits)
    if len(kg_rows) + len(link_rows) > limits.max_total_rows:
        raise DatasetError("knowledge inputs exceed max_total_rows")
    if len(kg_rows) != len(set(kg_rows)):
        raise DatasetError(".kg input contains duplicate triplets")
    items = [row[0] for row in link_rows]
    entities = [row[1] for row in link_rows]
    if len(items) != len(set(items)) or len(entities) != len(set(entities)):
        raise DatasetError(".link must be a one-to-one item/entity mapping")
    triples = tuple(sorted(KnowledgeTriple(*row) for row in kg_rows))
    links = tuple(sorted(ItemEntityLink(*row) for row in link_rows))
    linked_items = set(items)
    references: list[CatalogReference] = []
    if inter_path is not None:
        interaction = load_recbole_inter(inter_path, minimum_rating=minimum_rating)
        catalog = set(interaction.dataset.item_ids)
        references.append(
            CatalogReference(
                "recbole-inter",
                interaction.summary.source_sha256,
                interaction.summary.interactions_sha256,
                len(catalog),
                len(linked_items & catalog),
                interaction.summary.minimum_rating,
            )
        )
    if side_features_path is not None:
        side = load_recbole_side_features(side_features_path)
        side_state = side.to_state()
        catalog = {
            row.key
            for row in side.dataset
            if row.source is FeatureSource.ITEM and type(row.key) is str
        }
        references.append(
            CatalogReference(
                "recbole-side",
                cast(str, side_state["state_sha256"]),
                cast(str, side_state["dataset_sha256"]),
                len(catalog),
                len(linked_items & catalog),
            )
        )
    return LoadedKnowledgeLinks(triples, links, (kg_source, link_source), limits, tuple(references))


def _preflight(loaded: LoadedKnowledgeLinks, max_bytes: int) -> None:
    """Bound expanded records before materializing a complete state tree."""

    size = 0
    for rows in (loaded.triples, loaded.links):
        for row in rows:
            for part in _ENCODER.iterencode(row.to_state()):
                size += len(part.encode("utf-8"))
                if size > max_bytes:
                    raise SerializationError("knowledge artifact exceeds max_output_bytes")


def _validate_semantics(loaded: LoadedKnowledgeLinks) -> None:
    """Enforce the same record and provenance contract before save and after load."""

    if type(loaded.limits) is not KnowledgeLimits:
        raise SerializationError("knowledge limits must be exact KnowledgeLimits")
    limits = loaded.limits
    if type(loaded.triples) is not tuple or type(loaded.links) is not tuple:
        raise SerializationError("knowledge triples and links must be tuples")
    if (
        not 0 < len(loaded.triples) <= limits.max_rows_per_file
        or not 0 < len(loaded.links) <= limits.max_rows_per_file
        or len(loaded.triples) + len(loaded.links) > limits.max_total_rows
    ):
        raise SerializationError("knowledge records exceed declared row limits")
    try:
        for triple in loaded.triples:
            if type(triple) is not KnowledgeTriple:
                raise SerializationError("knowledge triple has wrong type")
            for token in triple.to_state():
                _token(token, "triple token", limits)
        for link in loaded.links:
            if type(link) is not ItemEntityLink:
                raise SerializationError("knowledge link has wrong type")
            for token in link.to_state():
                _token(token, "link token", limits)
    except (DatasetError, UnicodeEncodeError) as error:
        raise SerializationError(f"invalid knowledge token: {error}") from error
    if loaded.triples != tuple(sorted(set(loaded.triples))) or loaded.links != tuple(
        sorted(set(loaded.links))
    ):
        raise SerializationError("knowledge records are not unique and canonically sorted")
    if len({link.item_id for link in loaded.links}) != len(loaded.links) or len(
        {link.entity_id for link in loaded.links}
    ) != len(loaded.links):
        raise SerializationError("knowledge links are not one-to-one")
    if type(loaded.sources) is not tuple or len(loaded.sources) != 2:
        raise SerializationError("knowledge sources must have .kg and .link entries")
    for source in loaded.sources:
        if type(source) is not AtomicSource:
            raise SerializationError("invalid atomic source provenance")
        AtomicSource.from_state(source.to_state())
    if loaded.sources[0].kind != "kg" or loaded.sources[1].kind != "link":
        raise SerializationError("knowledge sources must be ordered kg, link")
    if loaded.sources[0].rows != len(loaded.triples) or loaded.sources[1].rows != len(loaded.links):
        raise SerializationError("knowledge source row counts do not match records")
    if any(source.bytes > limits.max_file_bytes for source in loaded.sources):
        raise SerializationError("knowledge source bytes exceed declared limits")
    if type(loaded.references) is not tuple or len(loaded.references) > 2:
        raise SerializationError("knowledge references must be an array of at most two")
    for reference in loaded.references:
        if type(reference) is not CatalogReference:
            raise SerializationError("invalid catalog reference kind")
        CatalogReference.from_state(reference.to_state(), len(loaded.links))
    if tuple(reference.kind for reference in loaded.references) not in {
        (),
        ("recbole-inter",),
        ("recbole-side",),
        ("recbole-inter", "recbole-side"),
    }:
        raise SerializationError("knowledge catalog references are duplicated or unordered")


def save_recbole_knowledge(loaded: LoadedKnowledgeLinks, path: str | Path) -> None:
    """Atomically publish a checksummed artifact, refusing existing destinations."""

    if type(loaded) is not LoadedKnowledgeLinks:
        raise ValidationError("loaded must be exact LoadedKnowledgeLinks")
    _validate_semantics(loaded)
    _preflight(loaded, loaded.limits.max_output_bytes - 1)
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
                raise SerializationError("short write of knowledge artifact")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValidationError("knowledge output already exists") from error
    except OSError as error:
        raise SerializationError(f"could not save knowledge artifact: {error}") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def load_recbole_knowledge(
    path: str | Path, *, max_output_bytes: int = 64 * 1024 * 1024
) -> LoadedKnowledgeLinks:
    """Strictly verify an artifact without loading its original source files."""

    if type(max_output_bytes) is not int or not 0 < max_output_bytes <= 256 * 1024 * 1024:
        raise ValidationError("max_output_bytes must be an integer in 1..268435456")
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(max_output_bytes + 1)
    except OSError as error:
        raise SerializationError(f"could not read knowledge artifact: {error}") from error
    if len(raw) > max_output_bytes:
        raise SerializationError("knowledge artifact exceeds max_output_bytes")
    try:
        state = strict_json_loads(raw)
    except ValueError as error:
        raise SerializationError(f"invalid knowledge JSON: {error}") from error
    expected = {
        "format",
        "schema_version",
        "triples",
        "links",
        "sources",
        "limits",
        "references",
        "linked_entities_in_kg",
        "fingerprint_sha256",
        "state_sha256",
    }
    if type(state) is not dict or set(state) != expected:
        raise SerializationError("knowledge artifact has missing or unknown fields")
    body = cast(dict[str, object], state)
    if (
        body["format"] != KNOWLEDGE_FORMAT
        or type(body["schema_version"]) is not int
        or body["schema_version"] != KNOWLEDGE_VERSION
    ):
        raise SerializationError("unsupported knowledge format or schema version")
    digest = body.pop("state_sha256")
    if type(digest) is not str or not _SHA.fullmatch(digest) or _digest(body) != digest:
        raise SerializationError("knowledge artifact checksum mismatch")
    limits = KnowledgeLimits.from_state(body["limits"])
    if len(raw) > limits.max_output_bytes:
        raise SerializationError("knowledge artifact exceeds declared max_output_bytes")
    triples_value, links_value = body["triples"], body["links"]
    if type(triples_value) is not list or type(links_value) is not list:
        raise SerializationError("knowledge triples and links must be arrays")
    if (
        not 0 < len(triples_value) <= limits.max_rows_per_file
        or not 0 < len(links_value) <= limits.max_rows_per_file
        or len(triples_value) + len(links_value) > limits.max_total_rows
    ):
        raise SerializationError("knowledge records exceed declared row limits")
    triples: list[KnowledgeTriple] = []
    links: list[ItemEntityLink] = []
    try:
        for index, row in enumerate(triples_value):
            if type(row) is not list or len(row) != 3:
                raise SerializationError(f"knowledge triple {index} has wrong width")
            triples.append(KnowledgeTriple(*(_token(cell, "triple token", limits) for cell in row)))
        for index, row in enumerate(links_value):
            if type(row) is not list or len(row) != 2:
                raise SerializationError(f"knowledge link {index} has wrong width")
            links.append(ItemEntityLink(*(_token(cell, "link token", limits) for cell in row)))
    except (DatasetError, UnicodeEncodeError) as error:
        raise SerializationError(f"invalid knowledge token: {error}") from error
    sources_value = body["sources"]
    if type(sources_value) is not list or len(sources_value) != 2:
        raise SerializationError("knowledge sources must have .kg and .link entries")
    sources = tuple(AtomicSource.from_state(value) for value in sources_value)
    references_value = body["references"]
    if type(references_value) is not list or len(references_value) > 2:
        raise SerializationError("knowledge references must be an array of at most two")
    references = tuple(CatalogReference.from_state(value, len(links)) for value in references_value)
    loaded = LoadedKnowledgeLinks(
        tuple(triples),
        tuple(links),
        cast(tuple[AtomicSource, AtomicSource], sources),
        limits,
        references,
    )
    _validate_semantics(loaded)
    if body["fingerprint_sha256"] != loaded.fingerprint:
        raise SerializationError("knowledge normalized fingerprint mismatch")
    if (
        type(body["linked_entities_in_kg"]) is not int
        or body["linked_entities_in_kg"] != loaded.linked_entities_in_kg
    ):
        raise SerializationError("knowledge linked-entity count mismatch")
    return loaded
