"""Safe output paths and atomic text writes shared by public workflows."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from orchidrec.errors import ValidationError


def require_distinct_paths(paths: Mapping[str, Path]) -> None:
    """Reject equal, symlinked, and hard-linked input/output paths."""

    names = tuple(paths)
    try:
        resolved = {name: path.resolve() for name, path in paths.items()}
    except (OSError, RuntimeError) as exc:
        raise ValidationError(f"could not resolve input/output path: {exc}") from exc
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            same_file = False
            with contextlib.suppress(OSError):
                same_file = paths[left].samefile(paths[right])
            if resolved[left] == resolved[right] or same_file:
                raise ValidationError(f"{left} and {right} must refer to different files")


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a destination only after a complete, flushed same-directory write."""

    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
