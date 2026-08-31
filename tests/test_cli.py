from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from orchidrec.cli import main
from orchidrec.demo import run_demo


class CliTests(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

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


if __name__ == "__main__":
    unittest.main()
