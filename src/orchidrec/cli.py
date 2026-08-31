"""Command-line interface for experiments and saved models."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from orchidrec import __version__
from orchidrec.benchmark import run_benchmark
from orchidrec.benchmark_config import load_benchmark_config
from orchidrec.config import load_config
from orchidrec.data import EntityId, validate_entity_id
from orchidrec.datasets import load_dataset
from orchidrec.demo import run_demo
from orchidrec.errors import OrchidRecError, ValidationError
from orchidrec.experiment import run_experiment
from orchidrec.models import load_model
from orchidrec.reporting import save_benchmark_reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchidrec",
        description="Run dependency-free, reproducible recommendation experiments.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a JSON experiment configuration")
    run.add_argument("config", type=Path)

    demo = subparsers.add_parser("demo", help="write and run the built-in example")
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))

    inspect = subparsers.add_parser("inspect", help="show metadata for a saved model")
    inspect.add_argument("model", type=Path)

    recommend = subparsers.add_parser("recommend", help="recommend from a saved model")
    recommend.add_argument("model", type=Path)
    recommend.add_argument("user_id", help="JSON scalar ID, for example 42 or \"alice\"")
    recommend.add_argument("--k", type=int, default=10)
    recommend.add_argument("--include-seen", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="compare configured models on one shared split"
    )
    benchmark.add_argument("config", type=Path)
    benchmark.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmark"))

    dataset_summary = subparsers.add_parser(
        "dataset-summary", help="validate and fingerprint a local dataset"
    )
    dataset_summary.add_argument("path", type=Path)
    dataset_summary.add_argument(
        "--format",
        required=True,
        choices=("orchidrec-json", "movielens-100k", "movielens-1m"),
    )
    dataset_summary.add_argument("--minimum-rating", type=float)
    return parser


def _parse_cli_id(raw: str) -> EntityId:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return validate_entity_id(value, "user_id")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_experiment(load_config(args.config))
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        if args.command == "demo":
            artifacts = run_demo(args.output_dir)
            demo_summary: dict[str, object] = {
                "config": str(artifacts.config_path),
                "model": str(artifacts.model_path),
                "report": str(artifacts.report_path),
                "metrics": artifacts.result.metrics.to_dict(),
            }
            print(json.dumps(demo_summary, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        if args.command == "inspect":
            model = load_model(args.model)
            state = model.to_state()
            model_summary: dict[str, object] = {
                "model_type": model.model_type,
                "schema_version": state["schema_version"],
                "catalog_size": len(model.catalog),
                "parameters": state["parameters"],
            }
            print(json.dumps(model_summary, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        if args.command == "recommend":
            if args.k <= 0:
                raise ValidationError("k must be a positive integer")
            model = load_model(args.model)
            user_id = _parse_cli_id(args.user_id)
            recommendations = model.recommend(user_id, args.k, exclude_seen=not args.include_seen)
            print(
                json.dumps(
                    [entry.to_dict() for entry in recommendations],
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "benchmark":
            benchmark_result = run_benchmark(load_benchmark_config(args.config))
            report_paths = save_benchmark_reports(benchmark_result, args.output_dir)
            summary: dict[str, object] = {
                "reports": report_paths.to_dict(),
                "dataset": benchmark_result.dataset.to_dict(),
                "models": [
                    {
                        "label": model.label,
                        "metrics": model.metrics.to_dict(),
                        "timing": model.timing.to_dict(),
                    }
                    for model in benchmark_result.models
                ],
            }
            print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        if args.command == "dataset-summary":
            loaded = load_dataset(
                args.path,
                format=args.format,
                minimum_rating=args.minimum_rating,
            )
            print(json.dumps(loaded.summary.to_dict(), indent=2, sort_keys=True))
            return 0
        parser.error(f"unknown command: {args.command}")
    except OrchidRecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2
