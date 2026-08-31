from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from orchidrec import __version__
from orchidrec.cli import main
from orchidrec.demo import run_demo

FIXTURES = Path(__file__).parent / "fixtures"


class CliTests(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_version_uses_package_version(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as stopped:
            main(["--version"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(stdout.getvalue(), f"orchidrec {__version__}\n")

    def test_demo_command_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, stdout, stderr = self.invoke("demo", "--output-dir", directory)
            payload = json.loads(stdout)
            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertTrue(Path(payload["model"]).is_file())

    def test_run_command_prints_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_demo(directory)
            code, stdout, _ = self.invoke("run", str(artifacts.config_path))
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout)["model"]["type"], "item_knn")

    def test_inspect_command_prints_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_demo(directory)
            code, stdout, _ = self.invoke("inspect", str(artifacts.model_path))
            payload = json.loads(stdout)
            self.assertEqual(code, 0)
            self.assertEqual(payload["model_type"], "item_knn")
            self.assertGreater(payload["catalog_size"], 0)

    def test_recommend_command_accepts_unquoted_string_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_demo(directory)
            code, stdout, _ = self.invoke("recommend", str(artifacts.model_path), "new-user", "--k", "2")
            self.assertEqual(code, 0)
            self.assertEqual(len(json.loads(stdout)), 2)

    def test_recommend_command_can_include_seen_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_demo(directory)
            _, excluded, _ = self.invoke("recommend", str(artifacts.model_path), "u1", "--k", "6")
            _, included, _ = self.invoke(
                "recommend", str(artifacts.model_path), "u1", "--k", "6", "--include-seen"
            )
            self.assertGreater(len(json.loads(included)), len(json.loads(excluded)))

    def test_expected_domain_error_returns_two(self) -> None:
        code, stdout, stderr = self.invoke("inspect", "missing-model.json")
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("error:", stderr)

    def test_invalid_k_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_demo(directory)
            code, _, stderr = self.invoke("recommend", str(artifacts.model_path), "u", "--k", "0")
            self.assertEqual(code, 2)
            self.assertIn("positive integer", stderr)

    def test_boolean_user_id_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = run_demo(directory)
            code, stdout, stderr = self.invoke(
                "recommend", str(artifacts.model_path), "true", "--k", "1"
            )
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("string or integer", stderr)

    def test_invalid_model_parameter_in_config_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "bad.json"
            config_path.write_text(
                json.dumps(
                    {
                        "data": {"path": "events.json"},
                        "model": {"name": "item_knn", "params": {"neighbors": 0}},
                    }
                ),
                encoding="utf-8",
            )
            code, stdout, stderr = self.invoke("run", str(config_path))
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("model.params", stderr)

    def test_huge_numeric_config_value_returns_two_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "huge.json"
            config_path.write_text(
                json.dumps(
                    {
                        "data": {"path": "events.json"},
                        "split": {"method": "random", "test_ratio": 10**400},
                        "model": {"name": "popularity"},
                    }
                ),
                encoding="utf-8",
            )
            code, stdout, stderr = self.invoke("run", str(config_path))
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("finite number", stderr)

    def test_dataset_summary_validates_local_movielens_file(self) -> None:
        code, stdout, stderr = self.invoke(
            "dataset-summary",
            str(FIXTURES / "movielens-100k"),
            "--format",
            "movielens-100k",
            "--minimum-rating",
            "4",
        )
        payload = json.loads(stdout)
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(payload["retained_interactions"], 17)
        self.assertEqual(len(payload["source_sha256"]), 64)

    def test_benchmark_command_writes_three_report_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "benchmark-config.json"
            output_dir = Path(directory) / "reports"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "seed": 3,
                        "data": {
                            "path": str(FIXTURES / "movielens-100k"),
                            "format": "movielens-100k",
                            "minimum_rating": 4,
                        },
                        "evaluation": {"k": 3, "bootstrap_samples": 3},
                        "models": [
                            {"label": "pop", "name": "popularity"},
                            {"label": "knn", "name": "item_knn", "params": {"neighbors": 2}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            code, stdout, stderr = self.invoke(
                "benchmark", str(config_path), "--output-dir", str(output_dir)
            )
            payload = json.loads(stdout)
            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertEqual(len(payload["models"]), 2)
            for path in payload["reports"].values():
                self.assertTrue(Path(path).is_file())

    def test_dataset_summary_error_is_a_domain_exit(self) -> None:
        code, stdout, stderr = self.invoke(
            "dataset-summary", "missing.data", "--format", "movielens-1m"
        )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("does not exist", stderr)


if __name__ == "__main__":
    unittest.main()
