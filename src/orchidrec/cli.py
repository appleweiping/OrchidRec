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
    FeatureDataset,
    FeatureLimits,
    FittedFeaturePipeline,
    load_feature_dataset,
    save_encoded_features,
)
from orchidrec.models import load_model
from orchidrec.recbole_knowledge import (
    KnowledgeLimits,
    import_recbole_knowledge,
    save_recbole_knowledge,
)
from orchidrec.recbole_network import NetworkLimits, import_recbole_network, save_recbole_network
from orchidrec.recbole_registry import (
    NamedAtomicDataset,
    RegistryLimits,
    register_atomic_datasets,
    save_atomic_registry,
    verify_atomic_registry,
)
from orchidrec.recbole_side import (
    RecBoleSideLimits,
    import_recbole_side_features,
    load_recbole_side_features,
    save_recbole_side_features,
)
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
        choices=("orchidrec-json", "movielens-100k", "movielens-1m", "recbole-inter"),
    )
    dataset_summary.add_argument("--minimum-rating", type=float)

    side = subparsers.add_parser(
        "import-recbole-features", help="import local .user/.item atomic side-feature tables"
    )
    side.add_argument("--user", type=Path)
    side.add_argument("--item", type=Path)
    side.add_argument("--schema-from", type=Path)
    side.add_argument("--output", type=Path, required=True)
    for name, default in RecBoleSideLimits().to_state().items():
        side.add_argument(f"--{name.replace('_', '-')}", type=int, default=default)

    knowledge = subparsers.add_parser(
        "import-recbole-knowledge", help="import local .kg/.link knowledge bindings"
    )
    knowledge.add_argument("--kg", type=Path, required=True)
    knowledge.add_argument("--link", type=Path, required=True)
    knowledge.add_argument("--inter", type=Path)
    knowledge.add_argument("--minimum-rating", type=float)
    knowledge.add_argument("--side-features", type=Path)
    knowledge.add_argument("--output", type=Path, required=True)
    for name, default in KnowledgeLimits().to_state().items():
        knowledge.add_argument(f"--{name.replace('_', '-')}", type=int, default=default)

    network = subparsers.add_parser(
        "import-recbole-network", help="import a local directed .net social edge table"
    )
    network.add_argument("--net", type=Path, required=True)
    network.add_argument("--inter", type=Path)
    network.add_argument("--minimum-rating", type=float)
    network.add_argument("--output", type=Path, required=True)
    for name, default in NetworkLimits().to_state().items():
        network.add_argument(f"--{name.replace('_', '-')}", type=int, default=default)

    for command, help_text in (
        ("register-recbole-datasets", "snapshot named local atomic dataset families"),
        ("verify-recbole-registry", "recompute and verify an atomic registry record"),
    ):
        registry = subparsers.add_parser(command, help=help_text)
        registry.add_argument("--dataset", action="append", required=True, metavar="NAME=DIRECTORY")
        registry.add_argument(
            "--minimum-rating", action="append", default=[], metavar="NAME=THRESHOLD"
        )
        registry.add_argument(
            "--registry" if command == "register-recbole-datasets" else "--record",
            type=Path,
            required=True,
        )
        for name, default in RegistryLimits().to_state().items():
            registry.add_argument(f"--{name.replace('_', '-')}", type=int, default=default)

    fit_features = subparsers.add_parser(
        "fit-features",
        help="fit a typed preprocessing pipeline from training-only feature rows",
    )
    fit_features.add_argument("--input", type=Path, required=True)
    fit_features.add_argument(
        "--input-format", choices=("feature-dataset", "recbole-side"), default="feature-dataset"
    )
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
    transform_features.add_argument(
        "--input-format", choices=("feature-dataset", "recbole-side"), default="feature-dataset"
    )
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


def _load_feature_input(path: Path, input_format: str, limits: FeatureLimits) -> FeatureDataset:
    if input_format == "recbole-side":
        dataset = load_recbole_side_features(path, max_output_bytes=limits.max_state_bytes).dataset
        return FeatureDataset.from_state(dataset.to_state(), limits=limits)
    return load_feature_dataset(path, limits=limits)


def _parse_cli_id(raw: str) -> EntityId:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return validate_entity_id(value, "user_id")


def _registry_inputs(args: argparse.Namespace) -> tuple[list[NamedAtomicDataset], RegistryLimits]:
    limits = RegistryLimits(**{name: getattr(args, name) for name in RegistryLimits().to_state()})
    thresholds: dict[str, float] = {}
    for raw in args.minimum_rating:
        if "=" not in raw:
            raise ValidationError("--minimum-rating must be NAME=THRESHOLD")
        name, value = raw.split("=", 1)
        if name in thresholds:
            raise ValidationError(f"duplicate --minimum-rating for {name}")
        try:
            thresholds[name] = float(value)
        except ValueError as error:
            raise ValidationError(f"invalid --minimum-rating for {name}") from error
    specs = []
    for raw in args.dataset:
        if "=" not in raw:
            raise ValidationError("--dataset must be NAME=DIRECTORY")
        name, directory = raw.split("=", 1)
        specs.append(NamedAtomicDataset(name, Path(directory), thresholds.get(name)))
    if set(thresholds) - {spec.name for spec in specs}:
        raise ValidationError("--minimum-rating names must refer to declared datasets")
    return specs, limits


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config = load_config(args.config)
            paths = {"config": args.config, "data.path": config.data.path}
            if config.data.knowledge_path is not None:
                paths["data.knowledge_path"] = config.data.knowledge_path
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
            benchmark_paths = {
                "config": args.config,
                "data.path": benchmark_config.data.path,
                "benchmark.json": args.output_dir / "benchmark.json",
                "benchmark.csv": args.output_dir / "benchmark.csv",
                "benchmark.html": args.output_dir / "benchmark.html",
            }
            if benchmark_config.data.knowledge_path is not None:
                benchmark_paths["data.knowledge_path"] = benchmark_config.data.knowledge_path
            require_distinct_paths(benchmark_paths)
            benchmark_result = run_benchmark(benchmark_config)
            report_paths = save_benchmark_reports(
                benchmark_result,
                args.output_dir,
                protected_paths={
                    "config": args.config,
                    **(
                        {"data.knowledge_path": benchmark_config.data.knowledge_path}
                        if benchmark_config.data.knowledge_path is not None
                        else {}
                    ),
                },
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
            if benchmark_result.candidate_plan is not None:
                summary["evaluation"] = benchmark_result.to_dict()["evaluation"]
            if benchmark_result.tuning is not None:
                summary["tuning"] = {
                    "selection_metric": benchmark_result.tuning.selection_metric,
                    "direction": benchmark_result.tuning.direction,
                    "three_way_split_sha256": (benchmark_result.tuning.three_way_split_fingerprint),
                    **(
                        {
                            "validation_candidate_sha256": benchmark_result.tuning.validation_candidate_fingerprint
                        }
                        if benchmark_result.tuning.validation_candidate_fingerprint is not None
                        else {}
                    ),
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
        if args.command == "import-recbole-features":
            if args.user is None and args.item is None:
                raise ValidationError("--user or --item is required")
            _require_distinct_paths(
                {
                    **({"user": args.user} if args.user is not None else {}),
                    **({"item": args.item} if args.item is not None else {}),
                    **({"schema-from": args.schema_from} if args.schema_from is not None else {}),
                    "output": args.output,
                }
            )
            side_limits = RecBoleSideLimits(
                **{name: getattr(args, name) for name in RecBoleSideLimits().to_state()}
            )
            side = import_recbole_side_features(
                user_path=args.user,
                item_path=args.item,
                limits=side_limits,
                schema_from=(
                    load_recbole_side_features(
                        args.schema_from, max_output_bytes=side_limits.max_output_bytes
                    )
                    if args.schema_from is not None
                    else None
                ),
            )
            save_recbole_side_features(side, args.output)
            print(
                json.dumps(
                    {
                        "dataset_sha256": side.to_state()["dataset_sha256"],
                        "features": len(side.dataset.schema),
                        "output": str(args.output),
                        "rows": len(side.dataset),
                        "schema_reference_sha256": side.schema_reference_sha256,
                        "sources": [source.to_state() for source in side.sources],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "import-recbole-knowledge":
            _require_distinct_paths(
                {
                    "kg": args.kg,
                    "link": args.link,
                    **({"inter": args.inter} if args.inter is not None else {}),
                    **(
                        {"side-features": args.side_features}
                        if args.side_features is not None
                        else {}
                    ),
                    "output": args.output,
                }
            )
            knowledge_limits = KnowledgeLimits(
                **{name: getattr(args, name) for name in KnowledgeLimits().to_state()}
            )
            graph = import_recbole_knowledge(
                kg_path=args.kg,
                link_path=args.link,
                inter_path=args.inter,
                minimum_rating=args.minimum_rating,
                side_features_path=args.side_features,
                limits=knowledge_limits,
            )
            save_recbole_knowledge(graph, args.output)
            print(
                json.dumps(
                    {
                        "fingerprint_sha256": graph.fingerprint,
                        "triples": len(graph.triples),
                        "links": len(graph.links),
                        "linked_entities_in_kg": graph.linked_entities_in_kg,
                        "references": [reference.to_state() for reference in graph.references],
                        "sources": [source.to_state() for source in graph.sources],
                        "output": str(args.output),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "import-recbole-network":
            _require_distinct_paths(
                {
                    "net": args.net,
                    **({"inter": args.inter} if args.inter is not None else {}),
                    "output": args.output,
                }
            )
            network_limits = NetworkLimits(
                **{name: getattr(args, name) for name in NetworkLimits().to_state()}
            )
            network = import_recbole_network(
                net_path=args.net,
                inter_path=args.inter,
                minimum_rating=args.minimum_rating,
                limits=network_limits,
            )
            save_recbole_network(network, args.output)
            print(
                json.dumps(
                    {
                        "fingerprint_sha256": network.fingerprint,
                        "edges": len(network.edges),
                        "users": len(network.users),
                        "source": network.source.to_state(),
                        "reference": (
                            network.reference.to_state() if network.reference is not None else None
                        ),
                        "output": str(args.output),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command in {"register-recbole-datasets", "verify-recbole-registry"}:
            specs, registry_limits = _registry_inputs(args)
            if args.command == "register-recbole-datasets":
                manifest = register_atomic_datasets(specs, limits=registry_limits)
                output = save_atomic_registry(manifest, args.registry)
                print(
                    json.dumps(
                        {
                            "registry_id": manifest.registry_id,
                            "datasets": [dataset.name for dataset in manifest.datasets],
                            "output": str(output),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            valid = verify_atomic_registry(args.record, specs, limits=registry_limits)
            print(json.dumps({"valid": valid, "record": str(args.record)}, sort_keys=True))
            return 0 if valid else 2
        if args.command == "fit-features":
            _require_distinct_paths({"input": args.input, "output": args.output})
            limits = _feature_limits(args)
            pipeline = FittedFeaturePipeline.fit(
                _load_feature_input(args.input, args.input_format, limits),
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
            encoded = pipeline.transform(
                _load_feature_input(args.input, args.input_format, pipeline.limits)
            )
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
