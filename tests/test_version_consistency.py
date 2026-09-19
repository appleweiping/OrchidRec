from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

from orchidrec import __version__


class VersionConsistencyTests(unittest.TestCase):
    def test_package_lock_and_citation_versions_match(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as stream:
            project_version = tomllib.load(stream)["project"]["version"]
        with (root / "uv.lock").open("rb") as stream:
            packages = tomllib.load(stream)["package"]
        locked_version = next(
            package["version"] for package in packages if package["name"] == "orchidrec"
        )
        citation = (root / "CITATION.cff").read_text(encoding="utf-8")
        citation_version = re.search(r"(?m)^version: (\S+)$", citation)
        self.assertIsNotNone(citation_version)
        self.assertEqual(
            (project_version, locked_version, citation_version.group(1)),
            (__version__, __version__, __version__),
        )


if __name__ == "__main__":
    unittest.main()
