"""Command-line interface for experiments and saved models."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from orchidrec import __version__
from orchidrec._files import require_distinct_paths
from orchidrec.benchmark import run_benchmark
from orchidrec.benchmark_config import load_benchmark_config
from orchidrec.config import load_config
from orchidrec.data import EntityId, validate_entity_id
from orchidrec.datasets import load_dataset
from orchidrec.demo import run_demo
from orchidrec.errors import OrchidRecError, ValidationError
from orchidrec.experiment import run_experiment
from orchidrec.features import (
    DEFAULT_FEATURE_LIMITS,
    FeatureLimits,
    FittedFeaturePipeline,
    load_feature_dataset,
    save_encoded_features,
)
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
    recommend.add_argument("user_id", help='JSON scalar ID, for example 42 or "alice"')
    recommend.add_argument("--k", type=int, default=10)
    recommend.add_argument("--include-seen", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="compare and optionally tune models on a sealed test split"
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

    fit_features = subparsers.add_parser(
        "fit-features",
        help="fit a typed preprocessing pipeline from training-only feature rows",
    )
    fit_features.add_argument("--input", type=Path, required=True)
    fit_features.add_argument("--output", type=Path, required=True)
    fit_features.add_argument("--max-rows", type=int, default=DEFAULT_FEATURE_LIMITS.max_rows)
    fit_features.add_argument(
        "--max-total-values",
        type=int,
        default=DEFAULT_FEATURE_LIMITS.max_total_values,
    )
    fit_features.add_argument(
        "--max-vocab-values",
        type=int,
        default=DEFAULT_FEATURE_LIMITS.max_vocab_values,
    )
    fit_features.add_argument(
        "--max-vocab-token-bytes",
        type=int,
        default=DEFAULT_FEATURE_LIMITS.max_vocab_token_bytes,
    )
    fit_features.add_argument(
        "--max-state-bytes",
        type=int,
        default=DEFAULT_FEATURE_LIMITS.max_state_bytes,
    )

    transform_features = subparsers.add_parser(
        "transform-features",
        help="encode feature rows with an already-fitted immutable pipeline",
    )
    transform_features.add_argument("--pipeline", type=Path, required=True)
    transform_features.add_argument("--input", type=Path, required=True)
    transform_features.add_argument("--output", type=Path, required=True)
    transform_features.add_argument(
        "--max-pipeline-bytes",
        type=int,
        default=DEFAULT_FEATURE_LIMITS.max_state_bytes,
    )
    return parser


def _require_distinct_paths(paths: dict[str, Path]) -> None:
    require_distinct_paths({f"--{name}": path for name, path in paths.items()})


def _feature_limits(args: argparse.Namespace) -> FeatureLimits:
    return FeatureLimits(
        max_rows=args.max_rows,
        max_total_values=args.max_total_values,
        max_vocab_values=args.max_vocab_values,
        max_vocab_token_bytes=args.max_vocab_token_bytes,
        max_state_bytes=args.max_state_bytes,
    )


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
            config = load_config(args.config)
            paths = {"config": args.config, "data.path": config.data.path}
            if config.output.model_path is not None:
                paths["output.model_path"] = config.output.model_path
            if config.output.report_path is not None:
                paths["output.report_path"] = config.output.report_path
            require_distinct_paths(paths)
            result = run_experiment(config)
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
            benchmark_config = load_benchmark_config(args.config)
            require_distinct_paths(
                {
                    "config": args.config,
                    "data.path": benchmark_config.data.path,
                    "benchmark.json": args.output_dir / "benchmark.json",
                    "benchmark.csv": args.output_dir / "benchmark.csv",
                    "benchmark.html": args.output_dir / "benchmark.html",
                }
            )
            benchmark_result = run_benchmark(benchmark_config)
            report_paths = save_benchmark_reports(
                benchmark_result, args.output_dir, protected_paths={"config": args.config}
            )
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
            if benchmark_result.tuning is not None:
                summary["tuning"] = {
                    "selection_metric": benchmark_result.tuning.selection_metric,
                    "direction": benchmark_result.tuning.direction,
                    "three_way_split_sha256": (benchmark_result.tuning.three_way_split_fingerprint),
                    "selected_models": [
                        {
                            "label": model.label,
                            "validation_score": model.selected_validation_score,
                            "final_parameters": dict(model.final_parameters),
                        }
                        for model in benchmark_result.tuning.models
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
        if args.command == "fit-features":
            _require_distinct_paths({"input": args.input, "output": args.output})
            limits = _feature_limits(args)
            pipeline = FittedFeaturePipeline.fit(
                load_feature_dataset(args.input, limits=limits),
                limits=limits,
            )
            pipeline.save(args.output)
            print(
                json.dumps(
                    {
                        "features": len(pipeline.schema),
                        "output": str(args.output),
                        "state_sha256": pipeline.state_sha256,
                        "training_rows": pipeline.training_rows,
                        "training_sha256": pipeline.training_sha256,
                        "training_values": pipeline.training_values,
                        "vocabulary_values": sum(
                            len(values) for values in pipeline.token_vocabularies.values()
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "transform-features":
            _require_distinct_paths(
                {"input": args.input, "output": args.output, "pipeline": args.pipeline}
            )
            pipeline = FittedFeaturePipeline.load(
                args.pipeline,
                max_state_bytes=args.max_pipeline_bytes,
            )
            encoded = pipeline.transform(load_feature_dataset(args.input, limits=pipeline.limits))
            save_encoded_features(encoded, args.output, limits=pipeline.limits)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "pipeline_sha256": encoded.pipeline_sha256,
                        "rows": len(encoded),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        parser.error(f"unknown command: {args.command}")
    except OrchidRecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2
