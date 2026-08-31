"""Validated interaction data and deterministic identifier mapping."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from orchidrec._json import strict_json_loads
from orchidrec._numeric import safe_float
from orchidrec.errors import SerializationError, ValidationError

EntityId = str | int


def validate_entity_id(value: object, field_name: str = "entity_id") -> EntityId:
    """Validate and return an identifier supported by OrchidRec."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValidationError(f"{field_name} must be a string or integer")
    if isinstance(value, str) and not value:
        raise ValidationError(f"{field_name} must not be an empty string")
    return value


def stable_id_key(value: EntityId) -> tuple[int, int | str]:
    """Return a total ordering that never conflates integers and strings."""

    validated = validate_entity_id(value)
    if isinstance(validated, int):
        return (0, validated)
    return (1, validated)


@dataclass(frozen=True, slots=True)
class Interaction:
    """One positive implicit-feedback event."""

    user_id: EntityId
    item_id: EntityId
    value: float = 1.0
    timestamp: float | None = None

    def __post_init__(self) -> None:
        validate_entity_id(self.user_id, "user_id")
        validate_entity_id(self.item_id, "item_id")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValidationError("value must be a finite positive number")
        numeric_value = safe_float(self.value)
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            raise ValidationError("value must be a finite positive number")
        object.__setattr__(self, "value", numeric_value)
        if self.timestamp is not None:
            if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)):
                raise ValidationError("timestamp must be a finite number or null")
            numeric_timestamp = safe_float(self.timestamp)
            if not math.isfinite(numeric_timestamp):
                raise ValidationError("timestamp must be a finite number or null")
            object.__setattr__(self, "timestamp", numeric_timestamp)

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> Interaction:
        """Build an interaction from a strict JSON-style mapping."""

        if not isinstance(record, Mapping):
            raise ValidationError("each interaction record must be an object")
        if any(not isinstance(key, str) for key in record):
            raise ValidationError("interaction field names must be strings")
        allowed = {"user_id", "item_id", "value", "timestamp"}
        unknown = set(record) - allowed
        if unknown:
            raise ValidationError(f"unknown interaction fields: {', '.join(sorted(unknown))}")
        missing = {"user_id", "item_id"} - set(record)
        if missing:
            raise ValidationError(f"missing interaction fields: {', '.join(sorted(missing))}")
        return cls(
            user_id=record["user_id"],  # type: ignore[arg-type]
            item_id=record["item_id"],  # type: ignore[arg-type]
            value=record.get("value", 1.0),  # type: ignore[arg-type]
            timestamp=record.get("timestamp"),  # type: ignore[arg-type]
        )

    def to_record(self) -> dict[str, EntityId | float | None]:
        """Return a JSON-compatible record."""

        return {
            "user_id": self.user_id,
            "item_id": self.item_id,
            "value": self.value,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class InteractionDataset(Sequence[Interaction]):
    """An immutable sequence of validated interactions."""

    interactions: tuple[Interaction, ...]

    def __init__(self, interactions: Iterable[Interaction] = ()) -> None:
        if isinstance(interactions, (str, bytes)):
            raise ValidationError("interactions must be an iterable of Interaction instances")
        materialized = tuple(interactions)
        if any(not isinstance(interaction, Interaction) for interaction in materialized):
            raise ValidationError("InteractionDataset accepts only Interaction instances")
        object.__setattr__(self, "interactions", materialized)

    @overload
    def __getitem__(self, index: int) -> Interaction: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Interaction, ...]: ...

    def __getitem__(self, index: int | slice) -> Interaction | tuple[Interaction, ...]:
        return self.interactions[index]

    def __iter__(self) -> Iterator[Interaction]:
        return iter(self.interactions)

    def __len__(self) -> int:
        return len(self.interactions)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> InteractionDataset:
        """Build a dataset from JSON-style records."""

        try:
            return cls(Interaction.from_record(record) for record in records)
        except TypeError as exc:
            raise ValidationError("interaction records must be iterable") from exc

    @classmethod
    def load_json(cls, path: str | Path) -> InteractionDataset:
        """Load a JSON array of interaction objects."""

        source = Path(path)
        try:
            payload = strict_json_loads(source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise SerializationError(f"could not read interactions from {source}: {exc}") from exc
        except ValueError as exc:
            raise SerializationError(f"invalid interaction JSON in {source}: {exc}") from exc
        if not isinstance(payload, list):
            raise SerializationError("interaction JSON must contain an array")
        try:
            return cls.from_records(payload)
        except ValidationError as exc:
            raise SerializationError(f"invalid interaction data in {source}: {exc}") from exc

    def save_json(self, path: str | Path) -> None:
        """Write the dataset as deterministic, human-readable JSON."""

        destination = Path(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(self.to_records(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise SerializationError(f"could not write interactions to {destination}: {exc}") from exc

    def to_records(self) -> list[dict[str, EntityId | float | None]]:
        """Return all events as independent JSON-compatible mappings."""

        return [interaction.to_record() for interaction in self]

    @property
    def user_ids(self) -> tuple[EntityId, ...]:
        """Return user IDs in deterministic order."""

        return StableIdMap.from_values(interaction.user_id for interaction in self).ids

    @property
    def item_ids(self) -> tuple[EntityId, ...]:
        """Return item IDs in deterministic order."""

        return StableIdMap.from_values(interaction.item_id for interaction in self).ids

    def by_user(self) -> dict[EntityId, tuple[Interaction, ...]]:
        """Group interactions by user while preserving event order."""

        groups: dict[EntityId, list[Interaction]] = defaultdict(list)
        for interaction in self:
            groups[interaction.user_id].append(interaction)
        return {user_id: tuple(groups[user_id]) for user_id in sorted(groups, key=stable_id_key)}

    def seen_items(self, user_id: EntityId) -> frozenset[EntityId]:
        """Return items observed for ``user_id``; unknown users yield an empty set."""

        validate_entity_id(user_id, "user_id")
        return frozenset(interaction.item_id for interaction in self if interaction.user_id == user_id)

    def item_counts(self, *, weighted: bool = False) -> dict[EntityId, float]:
        """Count item events, optionally summing interaction values."""

        counts: dict[EntityId, float] = defaultdict(float)
        for interaction in self:
            updated = counts[interaction.item_id] + (interaction.value if weighted else 1.0)
            if not math.isfinite(updated):
                raise ValidationError(
                    f"aggregate count for item {interaction.item_id!r} is not finite"
                )
            counts[interaction.item_id] = updated
        return {item_id: counts[item_id] for item_id in sorted(counts, key=stable_id_key)}


class StableIdMap:
    """A reversible ID-to-index map with input-order-independent indices."""

    __slots__ = ("_ids", "_indices")

    def __init__(self, ids: tuple[EntityId, ...]) -> None:
        if isinstance(ids, (str, bytes)):
            raise ValidationError("IDs must be an iterable of IDs, not a string")
        try:
            materialized = tuple(ids)
            canonical = tuple(
                sorted({validate_entity_id(entity_id) for entity_id in materialized}, key=stable_id_key)
            )
        except TypeError as exc:
            raise ValidationError("IDs must be an iterable of strings or integers") from exc
        if materialized != canonical:
            raise ValidationError("IDs must be unique and in stable order")
        self._ids = canonical
        self._indices = {entity_id: index for index, entity_id in enumerate(canonical)}

    @classmethod
    def from_values(cls, values: Iterable[EntityId]) -> StableIdMap:
        """Build a map sorted by ID type and then value."""

        if isinstance(values, (str, bytes)):
            raise ValidationError("ID values must be an iterable of IDs, not a string")
        unique: set[EntityId] = set()
        try:
            for value in values:
                unique.add(validate_entity_id(value))
        except TypeError as exc:
            raise ValidationError("ID values must be iterable") from exc
        return cls(tuple(sorted(unique, key=stable_id_key)))

    @property
    def ids(self) -> tuple[EntityId, ...]:
        return self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, value: object) -> bool:
        try:
            validated = validate_entity_id(value)
        except ValidationError:
            return False
        return validated in self._indices

    def index_of(self, entity_id: EntityId) -> int:
        """Return the stable integer index for an ID."""

        validated = validate_entity_id(entity_id)
        try:
            return self._indices[validated]
        except KeyError as exc:
            raise ValidationError(f"unknown entity ID: {validated!r}") from exc

    def id_at(self, index: int) -> EntityId:
        """Return the ID at an index."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise ValidationError("index must be an integer")
        if index < 0:
            raise ValidationError(f"ID index out of range: {index}")
        try:
            return self._ids[index]
        except IndexError as exc:
            raise ValidationError(f"ID index out of range: {index}") from exc

    def transform(self, values: Iterable[EntityId]) -> tuple[int, ...]:
        return tuple(self.index_of(value) for value in values)

    def inverse_transform(self, indices: Iterable[int]) -> tuple[EntityId, ...]:
        return tuple(self.id_at(index) for index in indices)

    def to_state(self) -> dict[str, list[EntityId]]:
        """Return a JSON-compatible state object."""

        return {"ids": list(self._ids)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> StableIdMap:
        """Restore and validate a serialized stable map."""

        if not isinstance(state, Mapping) or set(state) != {"ids"} or not isinstance(state["ids"], list):
            raise SerializationError("ID map state must contain only an 'ids' array")
        try:
            restored = cls.from_values(state["ids"])
        except ValidationError as exc:
            raise SerializationError(f"invalid ID map state: {exc}") from exc
        if list(restored.ids) != state["ids"]:
            raise SerializationError("ID map state is duplicated or not in stable order")
        return restored

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StableIdMap) and self.ids == other.ids

    def __repr__(self) -> str:
        return f"StableIdMap(ids={self._ids!r})"
