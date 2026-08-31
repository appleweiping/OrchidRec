"""Strict JSON decoding shared by persisted OrchidRec inputs."""

from __future__ import annotations

import json


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def strict_json_loads(text: str) -> object:
    """Decode standards-compliant JSON while rejecting duplicate object keys."""

    payload: object = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    return payload
