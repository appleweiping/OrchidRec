"""Strict JSON decoding shared by persisted OrchidRec inputs."""

from __future__ import annotations

import json
import math

MAX_JSON_DEPTH = 128
MAX_JSON_INTEGER_DIGITS = 4_096


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON floating-point number is outside the finite range")
    return result


def _bounded_int(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > MAX_JSON_INTEGER_DIGITS:
        raise ValueError(f"JSON integer exceeds the {MAX_JSON_INTEGER_DIGITS}-digit safety limit")
    return int(value)


def _decode_utf8(text: str | bytes) -> str:
    if isinstance(text, bytes):
        try:
            return text.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("JSON input is not valid UTF-8") from error
    if type(text) is not str:
        raise ValueError("JSON input must be text or UTF-8 bytes")
    return text


def _check_lexical_depth(text: str) -> None:
    """Reject excessive container nesting before entering the recursive decoder."""

    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError(f"JSON exceeds the maximum depth of {MAX_JSON_DEPTH}")
        elif character in "]}":
            depth -= 1


def _validate_decoded_tree(payload: object) -> None:
    stack: list[tuple[object, int]] = [(payload, 0)]
    while stack:
        value, depth = stack.pop()
        if isinstance(value, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
                raise ValueError("JSON strings must contain only Unicode scalar values")
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("JSON numbers must be finite")
        elif type(value) is list:
            container_depth = depth + 1
            if container_depth > MAX_JSON_DEPTH:
                raise ValueError(f"JSON exceeds the maximum depth of {MAX_JSON_DEPTH}")
            stack.extend((entry, container_depth) for entry in value)
        elif type(value) is dict:
            container_depth = depth + 1
            if container_depth > MAX_JSON_DEPTH:
                raise ValueError(f"JSON exceeds the maximum depth of {MAX_JSON_DEPTH}")
            for key, entry in value.items():
                if any(0xD800 <= ord(character) <= 0xDFFF for character in key):
                    raise ValueError("JSON strings must contain only Unicode scalar values")
                stack.append((entry, container_depth))


def strict_json_loads(text: str | bytes) -> object:
    """Decode bounded standards-compliant JSON with portable strictness."""

    decoded = _decode_utf8(text)
    _check_lexical_depth(decoded)
    try:
        payload: object = json.loads(
            decoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            parse_int=_bounded_int,
        )
    except RecursionError as error:
        raise ValueError(f"JSON exceeds the maximum depth of {MAX_JSON_DEPTH}") from error
    _validate_decoded_tree(payload)
    return payload
