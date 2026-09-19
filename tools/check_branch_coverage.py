"""Enforce branch-only coverage from coverage.py's JSON summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def branch_counts(payload: object) -> tuple[int, int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("totals"), dict):
        raise ValueError("coverage JSON has no totals object")
    totals: dict[str, Any] = payload["totals"]
    branches = totals.get("num_branches")
    covered = totals.get("covered_branches")
    if (
        type(branches) is not int
        or type(covered) is not int
        or branches <= 0
        or not 0 <= covered <= branches
    ):
        raise ValueError("coverage JSON has invalid branch counts")
    return covered, branches


def main(path: Path, *, minimum_percent: int = 90) -> int:
    try:
        covered, branches = branch_counts(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        print(f"invalid branch coverage report: {error}", file=sys.stderr)
        return 2
    percentage = 100 * covered / branches
    print(f"Branch-only coverage: {covered}/{branches} = {percentage:.2f}%")
    if 100 * covered < minimum_percent * branches:
        print(f"required branch-only coverage: {minimum_percent}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: check_branch_coverage.py COVERAGE_JSON", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
