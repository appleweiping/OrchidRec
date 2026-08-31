"""Local dataset adapters and content-addressed dataset summaries.

The adapters in this module never download data.  They turn an explicitly
provided local file into OrchidRec's immutable interaction representation and
record both the source bytes and normalized interactions with SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from orchidrec._json import strict_json_loads
from orchidrec._numeric import safe_float
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import DatasetError, ValidationError

DatasetFormat = Literal["orchidrec-json", "movielens-100k", "movielens-1m"]
DATASET_SUMMARY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Reproducibility metadata for one normalized local dataset."""

    format: DatasetFormat
    source_name: str
    source_sha256: str
    interactions_sha256: str
    source_bytes: int
    source_rows: int
    retained_interactions: int
    dropped_interactions: int
    users: int
    items: int
    minimum_rating: float | None
    rating_min: float
    rating_max: float
    timestamp_min: float | None
    timestamp_max: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a versioned JSON-compatible summary."""

        return {
            "schema_version": DATASET_SUMMARY_SCHEMA_VERSION,
            "format": self.format,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "interactions_sha256": self.interactions_sha256,
            "source_bytes": self.source_bytes,
            "source_rows": self.source_rows,
            "retained_interactions": self.retained_interactions,
            "dropped_interactions": self.dropped_interactions,
            "users": self.users,
            "items": self.items,
            "minimum_rating": self.minimum_rating,
            "rating_min": self.rating_min,
            "rating_max": self.rating_max,
            "timestamp_min": self.timestamp_min,
            "timestamp_max": self.timestamp_max,
        }


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """A normalized interaction dataset together with its provenance."""

    dataset: InteractionDataset
    summary: DatasetSummary


def interaction_fingerprint(dataset: InteractionDataset) -> str:
    """Hash the ordered, normalized interaction records."""

    if not isinstance(dataset, InteractionDataset):
        raise DatasetError("dataset must be an InteractionDataset")
    encoded = json.dumps(
        dataset.to_records(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_file(path: str | Path, dataset_format: DatasetFormat) -> Path:
    source = Path(path)
    if source.is_dir():
        filename = {
            "movielens-100k": "u.data",
            "movielens-1m": "ratings.dat",
            "orchidrec-json": "interactions.json",
        }[dataset_format]
        source = source / filename
    if not source.is_file():
        raise DatasetError(f"dataset file does not exist: {source}")
    return source


def _read_source(source: Path) -> bytes:
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise DatasetError(f"could not read dataset file {source}: {exc}") from exc
    if not payload:
        raise DatasetError(f"dataset file is empty: {source}")
    return payload


def _positive_decimal(raw: str, name: str, line_number: int) -> int:
    if not raw.isascii() or not raw.isdecimal():
        raise DatasetError(f"line {line_number}: {name} must be an unsigned decimal integer")
    try:
        value = int(raw)
    except ValueError as exc:
        raise DatasetError(f"line {line_number}: {name} is too large") from exc
    if value <= 0:
        raise DatasetError(f"line {line_number}: {name} must be greater than zero")
    return value


def _minimum_rating(value: int | float | None) -> float:
    if value is None:
        return 4.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError("minimum_rating must be a finite number between 1 and 5")
    numeric = safe_float(value)
    if not math.isfinite(numeric) or not 1.0 <= numeric <= 5.0:
        raise DatasetError("minimum_rating must be a finite number between 1 and 5")
    return numeric


def _movielens_rows(
    payload: bytes,
    dataset_format: Literal["movielens-100k", "movielens-1m"],
) -> list[tuple[int, int, int, int]]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise DatasetError("MovieLens ratings data must contain ASCII text") from exc
    delimiter = "\t" if dataset_format == "movielens-100k" else "::"
    rows: list[tuple[int, int, int, int]] = []
    pairs: set[tuple[int, int]] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise DatasetError(f"line {line_number}: blank rows are not allowed")
        fields = line.split(delimiter)
        if len(fields) != 4:
            raise DatasetError(
                f"line {line_number}: expected 4 {delimiter!r}-separated fields, got {len(fields)}"
            )
        user_id = _positive_decimal(fields[0], "user ID", line_number)
        item_id = _positive_decimal(fields[1], "item ID", line_number)
        rating = _positive_decimal(fields[2], "rating", line_number)
        timestamp = _positive_decimal(fields[3], "timestamp", line_number)
        if rating > 5:
            raise DatasetError(f"line {line_number}: rating must be between 1 and 5")
        if not math.isfinite(safe_float(timestamp)):
            raise DatasetError(f"line {line_number}: timestamp is too large")
        pair = (user_id, item_id)
        if pair in pairs:
            raise DatasetError(f"line {line_number}: duplicate user-item rating {pair!r}")
        pairs.add(pair)
        rows.append((user_id, item_id, rating, timestamp))
    if not rows:
        raise DatasetError("MovieLens ratings data contains no rows")
    return rows


def load_movielens(
    path: str | Path,
    *,
    format: Literal["movielens-100k", "movielens-1m"],
    minimum_rating: int | float | None = 4.0,
) -> LoadedDataset:
    """Load a local MovieLens ratings file as binary implicit feedback.

    Ratings at or above ``minimum_rating`` become unit-valued positive events.
    Lower ratings are counted in the summary but are not treated as negative
    examples.  Both supported formats are parsed strictly and duplicate
    user-item rows are rejected.
    """

    if format not in {"movielens-100k", "movielens-1m"}:
        raise DatasetError("format must be movielens-100k or movielens-1m")
    threshold = _minimum_rating(minimum_rating)
    source = _source_file(path, format)
    payload = _read_source(source)
    rows = _movielens_rows(payload, format)
    retained = [
        Interaction(user_id, item_id, 1.0, float(timestamp))
        for user_id, item_id, rating, timestamp in rows
        if rating >= threshold
    ]
    if not retained:
        raise DatasetError("minimum_rating removed every interaction")
    dataset = InteractionDataset(retained)
    ratings = [rating for _, _, rating, _ in rows]
    timestamps = [timestamp for _, _, _, timestamp in rows]
    return LoadedDataset(
        dataset=dataset,
        summary=DatasetSummary(
            format=format,
            source_name=source.name,
            source_sha256=hashlib.sha256(payload).hexdigest(),
            interactions_sha256=interaction_fingerprint(dataset),
            source_bytes=len(payload),
            source_rows=len(rows),
            retained_interactions=len(dataset),
            dropped_interactions=len(rows) - len(dataset),
            users=len(dataset.user_ids),
            items=len(dataset.item_ids),
            minimum_rating=threshold,
            rating_min=float(min(ratings)),
            rating_max=float(max(ratings)),
            timestamp_min=float(min(timestamps)),
            timestamp_max=float(max(timestamps)),
        ),
    )


def load_json_dataset(path: str | Path) -> LoadedDataset:
    """Load OrchidRec JSON while recording byte and interaction hashes."""

    source = _source_file(path, "orchidrec-json")
    payload = _read_source(source)
    try:
        decoded = strict_json_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DatasetError(f"invalid interaction JSON in {source}: {exc}") from exc
    if not isinstance(decoded, list):
        raise DatasetError("interaction JSON must contain an array")
    try:
        dataset = InteractionDataset.from_records(decoded)
    except ValidationError as exc:
        raise DatasetError(f"invalid interaction data in {source}: {exc}") from exc
    if not dataset:
        raise DatasetError("interaction dataset contains no rows")
    timestamps = [event.timestamp for event in dataset if event.timestamp is not None]
    values = [event.value for event in dataset]
    return LoadedDataset(
        dataset=dataset,
        summary=DatasetSummary(
            format="orchidrec-json",
            source_name=source.name,
            source_sha256=hashlib.sha256(payload).hexdigest(),
            interactions_sha256=interaction_fingerprint(dataset),
            source_bytes=len(payload),
            source_rows=len(dataset),
            retained_interactions=len(dataset),
            dropped_interactions=0,
            users=len(dataset.user_ids),
            items=len(dataset.item_ids),
            minimum_rating=None,
            rating_min=min(values),
            rating_max=max(values),
            timestamp_min=min(timestamps) if timestamps else None,
            timestamp_max=max(timestamps) if timestamps else None,
        ),
    )


def load_dataset(
    path: str | Path,
    *,
    format: DatasetFormat,
    minimum_rating: int | float | None = None,
) -> LoadedDataset:
    """Load one supported local dataset format without network access."""

    if format == "orchidrec-json":
        if minimum_rating is not None:
            raise DatasetError("minimum_rating is only valid for MovieLens formats")
        return load_json_dataset(path)
    if format in {"movielens-100k", "movielens-1m"}:
        return load_movielens(path, format=format, minimum_rating=minimum_rating)
    raise DatasetError(f"unsupported dataset format: {format!r}")
