from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.check_branch_coverage import branch_counts, main


class BranchCoverageGateTests(unittest.TestCase):
    def test_uses_branch_counts_not_combined_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "coverage.json"
            report.write_text(
                json.dumps(
                    {
                        "totals": {
                            "num_branches": 1_662,
                            "covered_branches": 1_493,
                            "percent_covered": 93.49,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(report), 1)
            report.write_text(
                json.dumps({"totals": {"num_branches": 100, "covered_branches": 90}}),
                encoding="utf-8",
            )
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(report), 0)

    def test_missing_or_boolean_branch_counts_fail_closed(self) -> None:
        for payload in ({}, {"totals": {}}, {"totals": {"num_branches": True}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                branch_counts(payload)
