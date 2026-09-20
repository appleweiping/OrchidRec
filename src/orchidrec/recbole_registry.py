"""Bounded local registry for named RecBole-style atomic file families.

The registry composes OrchidRec's strict interchange adapters. It records
namespace-specific overlaps without claiming RecBole's full preprocessing,
filtering, token remapping, or model/dataloader compatibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, fields
from pathlib import Path
from typing import cast

from orchidrec.data import EntityId
from orchidrec.datasets import DatasetSummary, load_recbole_inter
from orchidrec.errors import DatasetError, SerializationError, ValidationError
from orchidrec.features import FeatureSource
from orchidrec.recbole_knowledge import import_recbole_knowledge
from orchidrec.recbole_network import import_recbole_network
from orchidrec.recbole_side import import_recbole_side_features

REGISTRY_PROTOCOL = "orchidrec-atomic-registry-v1"
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_SUFFIXES = ("inter", "user", "item", "kg", "link", "net")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ENCODER = json.JSONEncoder(
    sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
)


def _digest(value: object) -> str:
    digest = hashlib.sha256()
    try:
        for part in _ENCODER.iterencode(value):
            digest.update(part.encode("utf-8"))
    except (TypeError, ValueError, UnicodeEncodeError, OverflowError) as error:
        raise SerializationError(f"registry state is not canonical JSON: {error}") from error
    return digest.hexdigest()


def _canonical(value: object, limit: int) -> bytes:
    result = bytearray()
    try:
        for part in _ENCODER.iterencode(value):
            encoded = part.encode("utf-8")
            if len(result) + len(encoded) + 1 > limit:
                raise SerializationError("registry artifact exceeds max_output_bytes")
            result.extend(encoded)
    except (TypeError, ValueError, UnicodeEncodeError, OverflowError) as error:
        raise SerializationError(f"registry state is not canonical JSON: {error}") from error
    result.extend(b"\n")
    return bytes(result)


@dataclass(frozen=True, slots=True)
class RegistryLimits:
    max_datasets: int = 8
    max_file_bytes: int = 16 * 1024 * 1024
    max_total_bytes: int = 64 * 1024 * 1024
    max_output_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        ceilings = {
            "max_datasets": 32,
            "max_file_bytes": 16 * 1024 * 1024,
            "max_total_bytes": 256 * 1024 * 1024,
            "max_output_bytes": 16 * 1024 * 1024,
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or not 0 < value <= ceilings[field.name]:
                raise ValidationError(
                    f"{field.name} must be an integer in 1..{ceilings[field.name]}"
                )
        if self.max_file_bytes > self.max_total_bytes:
            raise ValidationError("max_file_bytes cannot exceed max_total_bytes")

    def to_state(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


DEFAULT_REGISTRY_LIMITS = RegistryLimits()


@dataclass(frozen=True, slots=True)
class NamedAtomicDataset:
    name: str
    directory: Path
    minimum_rating: float | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not _NAME.fullmatch(self.name):
            raise ValidationError("dataset name must match [A-Za-z][A-Za-z0-9_-]{0,63}")
        if not isinstance(self.directory, Path) or self.directory.name != self.name:
            raise ValidationError("dataset directory basename must equal its declared name")
        threshold = self.minimum_rating
        if threshold is not None:
            if type(threshold) not in (int, float):
                raise ValidationError("minimum_rating must be a finite number")
            try:
                number = float(threshold)
            except OverflowError as error:
                raise ValidationError("minimum_rating must be a finite number") from error
            if not math.isfinite(number):
                raise ValidationError("minimum_rating must be a finite number")
            object.__setattr__(self, "minimum_rating", number)


@dataclass(frozen=True, slots=True)
class AtomicFileProvenance:
    suffix: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if (
            self.suffix not in _SUFFIXES
            or type(self.sha256) is not str
            or not _SHA.fullmatch(self.sha256)
        ):
            raise ValidationError("invalid atomic file provenance")
        if type(self.bytes) is not int or self.bytes < 1:
            raise ValidationError("atomic file byte length must be positive")

    def to_state(self) -> dict[str, object]:
        return {"suffix": self.suffix, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class NamespaceOverlap:
    name: str
    observed: int
    matched: int

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValidationError("overlap name must be non-empty")
        if (
            type(self.observed) is not int
            or type(self.matched) is not int
            or not 0 <= self.matched <= self.observed
        ):
            raise ValidationError("namespace overlap counts are inconsistent")

    def to_state(self) -> dict[str, object]:
        return {"name": self.name, "observed": self.observed, "matched": self.matched}


@dataclass(frozen=True, slots=True)
class RegisteredAtomicDataset:
    name: str
    minimum_rating: float | None
    files: tuple[AtomicFileProvenance, ...]
    interactions: DatasetSummary
    side_sha256: str | None
    knowledge_sha256: str | None
    network_sha256: str | None
    overlaps: tuple[NamespaceOverlap, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not _NAME.fullmatch(self.name):
            raise ValidationError("registered dataset name is invalid")
        if type(self.files) is not tuple or any(
            type(file) is not AtomicFileProvenance for file in self.files
        ):
            raise ValidationError("atomic files must have exact provenance types")
        suffixes = tuple(file.suffix for file in self.files)
        if (
            not suffixes
            or suffixes[0] != "inter"
            or suffixes != tuple(suffix for suffix in _SUFFIXES if suffix in suffixes)
            or ("kg" in suffixes) != ("link" in suffixes)
        ):
            raise ValidationError("atomic families must be unique, ordered, and include .inter")
        if (
            not isinstance(self.interactions, DatasetSummary)
            or self.interactions.format != "recbole-inter"
        ):
            raise ValidationError("registered interactions require a RecBole .inter summary")
        if self.interactions.source_sha256 != self.files[0].sha256:
            raise ValidationError("interaction provenance does not match the file snapshot")
        if self.interactions.source_bytes != self.files[0].bytes:
            raise ValidationError("interaction source length does not match the file snapshot")
        if self.interactions.minimum_rating != self.minimum_rating:
            raise ValidationError("interaction threshold does not match the dataset declaration")
        if type(self.overlaps) is not tuple or any(
            type(item) is not NamespaceOverlap for item in self.overlaps
        ):
            raise ValidationError("namespace overlaps must have exact types")
        for present, digest in (
            ("user" in suffixes or "item" in suffixes, self.side_sha256),
            ("kg" in suffixes, self.knowledge_sha256),
            ("net" in suffixes, self.network_sha256),
        ):
            if present != (type(digest) is str and _SHA.fullmatch(digest) is not None):
                raise ValidationError("atomic family fingerprint is inconsistent")
        overlap_names = tuple(item.name for item in self.overlaps)
        if overlap_names != tuple(sorted(set(overlap_names))):
            raise ValidationError("namespace overlaps must be uniquely sorted")

    def to_state(self) -> dict[str, object]:
        return {
            "name": self.name,
            "minimum_rating": self.minimum_rating,
            "files": [file.to_state() for file in self.files],
            "interactions": self.interactions.to_dict(),
            "side_sha256": self.side_sha256,
            "knowledge_sha256": self.knowledge_sha256,
            "network_sha256": self.network_sha256,
            "overlaps": [item.to_state() for item in self.overlaps],
        }


@dataclass(frozen=True, slots=True)
class AtomicRegistryManifest:
    datasets: tuple[RegisteredAtomicDataset, ...]
    limits: RegistryLimits
    protocol: str = REGISTRY_PROTOCOL

    def __post_init__(self) -> None:
        if type(self.limits) is not RegistryLimits or type(self.datasets) is not tuple:
            raise ValidationError("registry limits and datasets must have exact types")
        if type(self.protocol) is not str or not re.fullmatch(
            r"orchidrec-atomic-registry-v[1-9][0-9]*", self.protocol
        ):
            raise ValidationError("registry protocol must declare a positive revision")
        if any(type(item) is not RegisteredAtomicDataset for item in self.datasets):
            raise ValidationError("registry datasets must have exact types")
        names = tuple(item.name for item in self.datasets)
        if (
            not 1 <= len(names) <= self.limits.max_datasets
            or names != tuple(sorted(names))
            or len({name.casefold() for name in names}) != len(names)
        ):
            raise ValidationError("registry datasets must be bounded and uniquely sorted")

    @property
    def registry_id(self) -> str:
        return _digest(
            {
                "protocol": self.protocol,
                "limits": self.limits.to_state(),
                "datasets": [
                    {
                        "name": dataset.name,
                        "minimum_rating": dataset.minimum_rating,
                        "files": [file.to_state() for file in dataset.files],
                    }
                    for dataset in self.datasets
                ],
            }
        )

    def to_state(self) -> dict[str, object]:
        body: dict[str, object] = {
            "format": self.protocol,
            "schema_version": 1,
            "registry_id": self.registry_id,
            "limits": self.limits.to_state(),
            "datasets": [dataset.to_state() for dataset in self.datasets],
        }
        body["state_sha256"] = _digest(body)
        return body


def _capture(spec: NamedAtomicDataset, limits: RegistryLimits, remaining: int) -> dict[str, bytes]:
    directory = spec.directory
    if directory.is_symlink() or not directory.is_dir():
        raise DatasetError(f"{spec.name}: directory must be a real local directory")
    result: dict[str, bytes] = {}
    for suffix in _SUFFIXES:
        source = directory / f"{spec.name}.{suffix}"
        if not source.exists() and not source.is_symlink():
            continue
        if source.is_symlink() or not source.is_file():
            raise DatasetError(f"{spec.name}.{suffix} must be a regular, non-symlink file")
        try:
            with source.open("rb") as stream:
                payload = stream.read(min(limits.max_file_bytes, remaining) + 1)
        except OSError as error:
            raise DatasetError(f"could not read {spec.name}.{suffix}: {error}") from error
        if not payload or len(payload) > limits.max_file_bytes:
            raise DatasetError(f"{spec.name}.{suffix} is empty or exceeds max_file_bytes")
        if len(payload) > remaining:
            raise DatasetError("registry inputs exceed max_total_bytes")
        remaining -= len(payload)
        result[suffix] = payload
    if "inter" not in result:
        raise DatasetError(f"{spec.name}.inter is required")
    if ("kg" in result) != ("link" in result):
        raise DatasetError(f"{spec.name}: .kg and .link must be supplied together")
    return result


def _ids(values: Sequence[EntityId]) -> set[str]:
    if any(type(value) is not str for value in values):
        raise SerializationError("atomic interaction IDs must be string tokens")
    return cast(set[str], set(values))


def _overlap(name: str, observed: set[str], catalog: set[str]) -> NamespaceOverlap:
    return NamespaceOverlap(name, len(observed), len(observed & catalog))


def _import_one(
    spec: NamedAtomicDataset, snapshots: dict[str, bytes], stage: Path
) -> RegisteredAtomicDataset:
    staged = stage / spec.name
    staged.mkdir()
    for suffix, payload in snapshots.items():
        (staged / f"{spec.name}.{suffix}").write_bytes(payload)

    def source_path(suffix: str) -> Path:
        return staged / f"{spec.name}.{suffix}"

    inter = load_recbole_inter(source_path("inter"), minimum_rating=spec.minimum_rating)
    inter_users, inter_items = _ids(inter.dataset.user_ids), _ids(inter.dataset.item_ids)
    overlaps = [_overlap("lexical_user_item_ids", inter_users, inter_items)]
    side_digest: str | None = None
    knowledge_digest: str | None = None
    network_digest: str | None = None
    source_hashes = {
        suffix: hashlib.sha256(payload).hexdigest() for suffix, payload in snapshots.items()
    }
    if inter.summary.source_sha256 != source_hashes["inter"]:
        raise SerializationError("interaction adapter provenance differs from captured source")
    if "user" in snapshots or "item" in snapshots:
        side = import_recbole_side_features(
            user_path=source_path("user") if "user" in snapshots else None,
            item_path=source_path("item") if "item" in snapshots else None,
        )
        side_state = side.to_state()
        side_digest = cast(str, side_state["dataset_sha256"])
        for side_source in side.sources:
            if side_source.sha256 != source_hashes[side_source.kind]:
                raise SerializationError("side adapter provenance differs from captured source")
        for namespace, label, catalog in (
            (FeatureSource.USER, "side_users_in_inter", inter_users),
            (FeatureSource.ITEM, "side_items_in_inter", inter_items),
        ):
            if namespace.value in snapshots:
                keys = {cast(str, row.key) for row in side.dataset if row.source is namespace}
                overlaps.append(_overlap(label, keys, catalog))
    if "kg" in snapshots:
        knowledge = import_recbole_knowledge(
            kg_path=source_path("kg"), link_path=source_path("link")
        )
        knowledge_digest = knowledge.fingerprint
        for knowledge_source in knowledge.sources:
            if knowledge_source.sha256 != source_hashes[knowledge_source.kind]:
                raise SerializationError(
                    "knowledge adapter provenance differs from captured source"
                )
        links = {link.item_id for link in knowledge.links}
        linked_entities = {link.entity_id for link in knowledge.links}
        kg_entities = {
            entity for triple in knowledge.triples for entity in (triple.head_id, triple.tail_id)
        }
        overlaps.append(_overlap("linked_items_in_inter", links, inter_items))
        overlaps.append(_overlap("linked_entities_in_kg", linked_entities, kg_entities))
    if "net" in snapshots:
        network = import_recbole_network(net_path=source_path("net"))
        network_digest = network.fingerprint
        if network.source.sha256 != source_hashes["net"]:
            raise SerializationError("network adapter provenance differs from captured source")
        overlaps.append(_overlap("network_users_in_inter", set(network.users), inter_users))
    files = tuple(
        AtomicFileProvenance(suffix, source_hashes[suffix], len(snapshots[suffix]))
        for suffix in _SUFFIXES
        if suffix in snapshots
    )
    return RegisteredAtomicDataset(
        spec.name,
        spec.minimum_rating,
        files,
        inter.summary,
        side_digest,
        knowledge_digest,
        network_digest,
        tuple(sorted(overlaps, key=lambda item: item.name)),
    )


def register_atomic_datasets(
    datasets: Sequence[NamedAtomicDataset], *, limits: RegistryLimits = DEFAULT_REGISTRY_LIMITS
) -> AtomicRegistryManifest:
    """Capture named local files once, then validate staged immutable snapshots."""

    if type(limits) is not RegistryLimits or not isinstance(datasets, Sequence):
        raise ValidationError("datasets and limits have invalid types")
    specs = tuple(datasets)
    if not 1 <= len(specs) <= limits.max_datasets or any(
        type(spec) is not NamedAtomicDataset for spec in specs
    ):
        raise ValidationError("registry requires bounded NamedAtomicDataset entries")
    names = [spec.name for spec in specs]
    if len({name.casefold() for name in names}) != len(names):
        raise ValidationError("registry dataset names must be unique, ignoring case")
    try:
        resolved = [spec.directory.resolve() for spec in specs]
    except OSError as error:
        raise DatasetError(f"could not resolve dataset directories: {error}") from error
    if len(set(resolved)) != len(resolved):
        raise ValidationError("registry dataset directories must be distinct")
    snapshots: list[tuple[NamedAtomicDataset, dict[str, bytes]]] = []
    total = 0
    for spec in sorted(specs, key=lambda item: item.name):
        captured = _capture(spec, limits, limits.max_total_bytes - total)
        total += sum(len(data) for data in captured.values())
        snapshots.append((spec, captured))
    try:
        with tempfile.TemporaryDirectory(prefix="orchidrec-registry-") as stage_name:
            stage = Path(stage_name)
            registered = tuple(_import_one(spec, captured, stage) for spec, captured in snapshots)
    except OSError as error:
        raise SerializationError(f"could not stage atomic file snapshots: {error}") from error
    return AtomicRegistryManifest(registered, limits)


def save_atomic_registry(manifest: AtomicRegistryManifest, directory: str | Path) -> Path:
    """Publish one content-addressed JSON record atomically without overwrite."""

    if type(manifest) is not AtomicRegistryManifest:
        raise ValidationError("manifest must be AtomicRegistryManifest")
    payload = _canonical(manifest.to_state(), manifest.limits.max_output_bytes)
    registry = Path(directory)
    if registry.exists() and not registry.is_dir():
        raise ValidationError("registry output must be a directory")
    try:
        registry.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SerializationError(f"could not create registry directory: {error}") from error
    destination = registry / f"{manifest.registry_id}.json"
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".registry-", suffix=".tmp", dir=registry)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            if stream.write(payload) != len(payload):
                raise SerializationError("short write of registry artifact")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValidationError("registry record already exists") from error
        if os.name == "posix":
            parent = os.open(registry, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
    except OSError as error:
        raise SerializationError(f"could not save registry artifact: {error}") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
    return destination


def verify_atomic_registry(
    path: str | Path,
    datasets: Sequence[NamedAtomicDataset],
    *,
    limits: RegistryLimits = DEFAULT_REGISTRY_LIMITS,
) -> bool:
    """Re-run adapters against current inputs and compare exact artifact bytes."""

    manifest = register_atomic_datasets(datasets, limits=limits)
    record = Path(path)
    if record.name != f"{manifest.registry_id}.json":
        return False
    try:
        with record.open("rb") as stream:
            payload = stream.read(limits.max_output_bytes + 1)
    except OSError as error:
        raise SerializationError(f"could not read registry artifact: {error}") from error
    return len(payload) <= limits.max_output_bytes and payload == _canonical(
        manifest.to_state(), limits.max_output_bytes
    )
