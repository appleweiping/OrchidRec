"""Strict, versioned configuration for multi-model benchmarks."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from orchidrec._json import strict_json_loads
from orchidrec._numeric import safe_float
from orchidrec.datasets import DatasetFormat
from orchidrec.errors import ConfigurationError, ValidationError
from orchidrec.models import ImplicitMF, ItemKNN, Popularity

BENCHMARK_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BenchmarkDataConfig:
    path: Path
    format: DatasetFormat
    minimum_rating: float | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSplitConfig:
    method: str = "leave_one_out"
    test_ratio: float = 0.2


@dataclass(frozen=True, slots=True)
class BenchmarkEvaluationConfig:
    k: int = 10
    exclude_seen: bool = True
    bootstrap_samples: int = 1_000
    confidence: float = 0.95


@dataclass(frozen=True, slots=True)
class BenchmarkModelSpec:
    label: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "name": self.name, "params": dict(self.params)}


def default_benchmark_models() -> tuple[BenchmarkModelSpec, ...]:
    """Return lightweight, deterministic defaults for the three built-ins."""

    return (
        BenchmarkModelSpec("popularity", "popularity", {"weighted": False}),
        BenchmarkModelSpec("item-knn", "item_knn", {"neighbors": 40, "shrinkage": 10.0}),
        BenchmarkModelSpec(
            "bpr-mf",
            "implicit_mf",
            {"factors": 16, "epochs": 5, "negative_samples": 1},
        ),
    )


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """All deterministic inputs to a shared-split benchmark."""

    seed: int
    data: BenchmarkDataConfig
    split: BenchmarkSplitConfig
    evaluation: BenchmarkEvaluationConfig
    models: tuple[BenchmarkModelSpec, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": BENCHMARK_CONFIG_SCHEMA_VERSION,
            "seed": self.seed,
            "data": {
                "path": str(self.data.path),
                "format": self.data.format,
                "minimum_rating": self.data.minimum_rating,
            },
            "split": {
                "method": self.split.method,
                "test_ratio": self.split.test_ratio,
            },
            "evaluation": {
                "k": self.evaluation.k,
                "exclude_seen": self.evaluation.exclude_seen,
                "bootstrap_samples": self.evaluation.bootstrap_samples,
                "confidence": self.evaluation.confidence,
            },
            "models": [model.to_dict() for model in self.models],
        }


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ConfigurationError(f"{name} must be a JSON object")
    return value


def _unknown(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    extras = set(mapping) - allowed
    if extras:
        raise ConfigurationError(f"unknown {name} fields: {', '.join(sorted(extras))}")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a finite number")
    numeric = safe_float(value)
    if not math.isfinite(numeric):
        raise ConfigurationError(f"{name} must be a finite number")
    return numeric


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _parse_model(entry: object, index: int, seed: int) -> BenchmarkModelSpec:
    model = _object(entry, f"models[{index}]")
    _unknown(model, {"label", "name", "params"}, f"models[{index}]")
    label = model.get("label")
    name = model.get("name")
    if (
        not isinstance(label, str)
        or not label.strip()
        or label != label.strip()
        or any(ord(character) < 32 for character in label)
        or len(label) > 100
    ):
        raise ConfigurationError(f"models[{index}].label must be a trimmed printable string")
    if not isinstance(name, str) or name not in {"popularity", "item_knn", "implicit_mf"}:
        raise ConfigurationError(
            f"models[{index}].name must be popularity, item_knn, or implicit_mf"
        )
    params = _object(model.get("params", {}), f"models[{index}].params")
    allowed = {
        "popularity": {"weighted"},
        "item_knn": {"neighbors", "shrinkage"},
        "implicit_mf": {
            "factors",
            "epochs",
            "learning_rate",
            "regularization",
            "negative_samples",
            "seed",
        },
    }[name]
    _unknown(params, allowed, f"models[{index}].params")
    model_types = {
        "popularity": Popularity,
        "item_knn": ItemKNN,
        "implicit_mf": ImplicitMF,
    }
    validated_params = dict(params)
    if name == "implicit_mf":
        validated_params.setdefault("seed", seed)
    try:
        model_types[name](**validated_params)
    except (TypeError, ValidationError) as exc:
        raise ConfigurationError(f"invalid models[{index}].params: {exc}") from exc
    return BenchmarkModelSpec(label=label, name=name, params=dict(params))


def benchmark_config_from_dict(
    payload: Mapping[str, Any], *, base_dir: str | Path = "."
) -> BenchmarkConfig:
    """Validate a decoded benchmark configuration and resolve its data path."""

    root = _object(payload, "benchmark configuration")
    _unknown(
        root,
        {"schema_version", "seed", "data", "split", "evaluation", "models"},
        "benchmark configuration",
    )
    version = root.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ConfigurationError("benchmark schema_version must be 1")
    seed = root.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("seed must be an integer")
    if "data" not in root:
        raise ConfigurationError("benchmark configuration requires a data object")
    data = _object(root["data"], "data")
    _unknown(data, {"path", "format", "minimum_rating"}, "data")
    raw_path = data.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ConfigurationError("data.path must be a non-empty string")
    raw_format = data.get("format")
    supported_formats = {"orchidrec-json", "movielens-100k", "movielens-1m"}
    if not isinstance(raw_format, str) or raw_format not in supported_formats:
        raise ConfigurationError(
            "data.format must be orchidrec-json, movielens-100k, or movielens-1m"
        )
    dataset_format = cast(DatasetFormat, raw_format)
    raw_minimum = data.get("minimum_rating")
    if dataset_format == "orchidrec-json":
        if raw_minimum is not None:
            raise ConfigurationError("data.minimum_rating is only valid for MovieLens formats")
        minimum_rating = None
    else:
        minimum_rating = 4.0 if raw_minimum is None else _finite_number(raw_minimum, "data.minimum_rating")
        if not 1.0 <= minimum_rating <= 5.0:
            raise ConfigurationError("data.minimum_rating must be between 1 and 5")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(base_dir).resolve() / path
    path = path.resolve()

    split = _object(root.get("split", {}), "split")
    _unknown(split, {"method", "test_ratio"}, "split")
    split_method = split.get("method", "leave_one_out")
    if not isinstance(split_method, str) or split_method not in {
        "random",
        "temporal",
        "leave_one_out",
    }:
        raise ConfigurationError("split.method must be random, temporal, or leave_one_out")
    test_ratio = _finite_number(split.get("test_ratio", 0.2), "split.test_ratio")
    if not 0.0 < test_ratio < 1.0:
        raise ConfigurationError("split.test_ratio must be between 0 and 1")

    evaluation = _object(root.get("evaluation", {}), "evaluation")
    _unknown(evaluation, {"k", "exclude_seen", "bootstrap_samples", "confidence"}, "evaluation")
    k = _positive_int(evaluation.get("k", 10), "evaluation.k")
    exclude_seen = evaluation.get("exclude_seen", True)
    if not isinstance(exclude_seen, bool):
        raise ConfigurationError("evaluation.exclude_seen must be a boolean")
    bootstrap_samples = _positive_int(
        evaluation.get("bootstrap_samples", 1_000), "evaluation.bootstrap_samples"
    )
    confidence = _finite_number(evaluation.get("confidence", 0.95), "evaluation.confidence")
    if not 0.0 < confidence < 1.0:
        raise ConfigurationError("evaluation.confidence must be between 0 and 1")

    raw_models = root.get("models")
    if raw_models is None:
        models = default_benchmark_models()
    else:
        if not isinstance(raw_models, list) or not raw_models:
            raise ConfigurationError("models must be a non-empty array")
        models = tuple(_parse_model(entry, index, seed) for index, entry in enumerate(raw_models))
    labels = [model.label for model in models]
    if len(set(labels)) != len(labels):
        raise ConfigurationError("model labels must be unique")
    if len(models) < 2:
        raise ConfigurationError("a benchmark requires at least two models")

    return BenchmarkConfig(
        seed=seed,
        data=BenchmarkDataConfig(path=path, format=dataset_format, minimum_rating=minimum_rating),
        split=BenchmarkSplitConfig(method=split_method, test_ratio=test_ratio),
        evaluation=BenchmarkEvaluationConfig(
            k=k,
            exclude_seen=exclude_seen,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
        ),
        models=models,
    )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Read a strict JSON benchmark configuration."""

    source = Path(path)
    try:
        payload = strict_json_loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"could not read benchmark configuration from {source}: {exc}") from exc
    except ValueError as exc:
        raise ConfigurationError(f"invalid benchmark JSON in {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigurationError("benchmark configuration must contain a JSON object")
    return benchmark_config_from_dict(payload, base_dir=source.parent)


def save_benchmark_config(config: BenchmarkConfig, path: str | Path) -> None:
    """Write a normalized benchmark configuration as strict JSON."""

    if not isinstance(config, BenchmarkConfig):
        raise ConfigurationError("config must be a BenchmarkConfig")
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"could not write benchmark configuration to {destination}: {exc}") from exc
