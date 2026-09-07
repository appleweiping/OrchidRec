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
from orchidrec.models import (
    ConfidenceALS,
    ImplicitMF,
    ItemKNN,
    Popularity,
    SequentialMarkov,
    UserKNN,
)

BENCHMARK_CONFIG_SCHEMA_VERSION = 1
BENCHMARK_METRIC_NAMES = ("precision", "recall", "ndcg", "mrr", "coverage", "novelty")
MAX_GRID_CANDIDATES = 128
MAX_IMPLICIT_MF_SEEDS = 16


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
class BenchmarkValidationSplitConfig:
    """The inner split used exclusively for hyperparameter selection."""

    method: str = "leave_one_out"
    validation_ratio: float = 0.2


@dataclass(frozen=True, slots=True)
class BenchmarkTuningConfig:
    """Leakage-safe model-selection settings for an optional benchmark grid."""

    selection_metric: str
    direction: str
    validation_split: BenchmarkValidationSplitConfig
    implicit_mf_seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_metric": self.selection_metric,
            "direction": self.direction,
            "validation_split": {
                "method": self.validation_split.method,
                "validation_ratio": self.validation_split.validation_ratio,
            },
            "implicit_mf_seeds": list(self.implicit_mf_seeds),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkModelSpec:
    label: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, tuple[Any, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "label": self.label,
            "name": self.name,
            "params": dict(self.params),
        }
        if self.grid:
            payload["grid"] = {name: list(values) for name, values in self.grid.items()}
        return payload


def default_benchmark_models() -> tuple[BenchmarkModelSpec, ...]:
    """Return lightweight defaults across six built-in models."""

    return (
        BenchmarkModelSpec("popularity", "popularity", {"weighted": False}),
        BenchmarkModelSpec("item-knn", "item_knn", {"neighbors": 40, "shrinkage": 10.0}),
        BenchmarkModelSpec(
            "bpr-mf",
            "implicit_mf",
            {"factors": 16, "epochs": 5, "negative_samples": 1},
        ),
        BenchmarkModelSpec(
            "confidence-als",
            "confidence_als",
            {"factors": 16, "epochs": 3, "alpha": 40.0, "regularization": 0.1},
        ),
        BenchmarkModelSpec("user-knn", "user_knn", {"neighbors": 40, "shrinkage": 10.0}),
        BenchmarkModelSpec(
            "sequential-markov",
            "sequential_markov",
            {"weighted": True, "popularity_mix": 0.05},
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
    tuning: BenchmarkTuningConfig | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
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
        if self.tuning is not None:
            payload["tuning"] = self.tuning.to_dict()
        return payload


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


def _validated_model_parameters(
    name: str,
    params: Mapping[str, Any],
    *,
    seed: int,
    location: str,
) -> None:
    model_types = {
        "popularity": Popularity,
        "confidence_als": ConfidenceALS,
        "item_knn": ItemKNN,
        "implicit_mf": ImplicitMF,
        "user_knn": UserKNN,
        "sequential_markov": SequentialMarkov,
    }
    validated_params = dict(params)
    if name in {"implicit_mf", "confidence_als"}:
        validated_params.setdefault("seed", seed)
    try:
        model_types[name](**validated_params)
    except (TypeError, ValidationError) as exc:
        raise ConfigurationError(f"invalid {location}: {exc}") from exc


def _parse_model(
    entry: object,
    index: int,
    seed: int,
    *,
    tuning_enabled: bool,
) -> BenchmarkModelSpec:
    model = _object(entry, f"models[{index}]")
    _unknown(model, {"label", "name", "params", "grid"}, f"models[{index}]")
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
    if not isinstance(name, str) or name not in {
        "popularity",
        "item_knn",
        "implicit_mf",
        "confidence_als",
        "user_knn",
        "sequential_markov",
    }:
        raise ConfigurationError(
            f"models[{index}].name must be popularity, item_knn, implicit_mf, confidence_als, "
            "user_knn, or sequential_markov"
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
        "confidence_als": {"factors", "epochs", "alpha", "regularization", "seed"},
        "user_knn": {"neighbors", "shrinkage"},
        "sequential_markov": {"weighted", "popularity_mix"},
    }[name]
    _unknown(params, allowed, f"models[{index}].params")
    _validated_model_parameters(name, params, seed=seed, location=f"models[{index}].params")

    raw_grid = model.get("grid")
    if raw_grid is None:
        grid: dict[str, tuple[Any, ...]] = {}
    else:
        if not tuning_enabled:
            raise ConfigurationError(f"models[{index}].grid requires a tuning object")
        grid_object = _object(raw_grid, f"models[{index}].grid")
        if not grid_object:
            raise ConfigurationError(f"models[{index}].grid must not be empty")
        _unknown(grid_object, allowed, f"models[{index}].grid")
        overlap = set(params) & set(grid_object)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ConfigurationError(
                f"models[{index}] parameters cannot appear in both params and grid: {names}"
            )
        grid = {}
        candidate_count = 1
        for parameter in sorted(grid_object):
            values = grid_object[parameter]
            if not isinstance(values, list) or not values:
                raise ConfigurationError(
                    f"models[{index}].grid.{parameter} must be a non-empty array"
                )
            try:
                encoded_values = [
                    json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    for value in values
                ]
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"models[{index}].grid.{parameter} must contain strict JSON values"
                ) from exc
            if len(set(encoded_values)) != len(encoded_values):
                raise ConfigurationError(
                    f"models[{index}].grid.{parameter} must not contain duplicate values"
                )
            candidate_count *= len(values)
            if candidate_count > MAX_GRID_CANDIDATES:
                raise ConfigurationError(
                    f"models[{index}].grid expands beyond {MAX_GRID_CANDIDATES} candidates"
                )
            grid[parameter] = tuple(values)
        keys = tuple(grid)
        combinations: list[dict[str, Any]] = [dict(params)]
        for parameter in keys:
            combinations = [
                {**combination, parameter: value}
                for combination in combinations
                for value in grid[parameter]
            ]
        for candidate_index, candidate in enumerate(combinations):
            _validated_model_parameters(
                name,
                candidate,
                seed=seed,
                location=f"models[{index}].grid candidate {candidate_index}",
            )
        floating_parameters = {
            ("item_knn", "shrinkage"),
            ("implicit_mf", "learning_rate"),
            ("implicit_mf", "regularization"),
            ("confidence_als", "alpha"),
            ("confidence_als", "regularization"),
            ("user_knn", "shrinkage"),
            ("sequential_markov", "popularity_mix"),
        }
        for parameter, values in grid.items():
            semantic_values = [
                float(value) if (name, parameter) in floating_parameters else value
                for value in values
            ]
            if len(set(semantic_values)) != len(semantic_values):
                raise ConfigurationError(
                    f"models[{index}].grid.{parameter} must not contain "
                    "semantically duplicate values"
                )
    if (
        tuning_enabled
        and name in {"implicit_mf", "confidence_als"}
        and ("seed" in params or "seed" in grid)
    ):
        raise ConfigurationError(
            f"models[{index}] cannot tune or fix {name}.seed; use tuning.implicit_mf_seeds "
            "for seeded latent-model validation repeats and top-level seed for final fit"
        )
    return BenchmarkModelSpec(label=label, name=name, params=dict(params), grid=grid)


def _parse_tuning(
    raw: object,
    *,
    seed: int,
    split_method: str,
    split_ratio: float,
) -> BenchmarkTuningConfig:
    tuning = _object(raw, "tuning")
    _unknown(
        tuning,
        {"selection_metric", "direction", "validation_split", "implicit_mf_seeds"},
        "tuning",
    )
    selection_metric = tuning.get("selection_metric", "ndcg")
    if not isinstance(selection_metric, str) or selection_metric not in BENCHMARK_METRIC_NAMES:
        raise ConfigurationError(
            "tuning.selection_metric must be precision, recall, ndcg, mrr, coverage, or novelty"
        )
    direction = tuning.get("direction", "maximize")
    if not isinstance(direction, str) or direction not in {"maximize", "minimize"}:
        raise ConfigurationError("tuning.direction must be maximize or minimize")
    validation = _object(tuning.get("validation_split", {}), "tuning.validation_split")
    _unknown(validation, {"method", "validation_ratio"}, "tuning.validation_split")
    method = validation.get("method", split_method)
    if not isinstance(method, str) or method not in {"random", "temporal", "leave_one_out"}:
        raise ConfigurationError(
            "tuning.validation_split.method must be random, temporal, or leave_one_out"
        )
    validation_ratio = _finite_number(
        validation.get("validation_ratio", split_ratio),
        "tuning.validation_split.validation_ratio",
    )
    if not 0.0 < validation_ratio < 1.0:
        raise ConfigurationError("tuning.validation_split.validation_ratio must be between 0 and 1")
    raw_seeds = tuning.get("implicit_mf_seeds", [seed])
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ConfigurationError("tuning.implicit_mf_seeds must be a non-empty array")
    if len(raw_seeds) > MAX_IMPLICIT_MF_SEEDS:
        raise ConfigurationError(
            f"tuning.implicit_mf_seeds must contain at most {MAX_IMPLICIT_MF_SEEDS} seeds"
        )
    seeds: list[int] = []
    for index, value in enumerate(raw_seeds):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(f"tuning.implicit_mf_seeds[{index}] must be an integer")
        seeds.append(value)
    if len(set(seeds)) != len(seeds):
        raise ConfigurationError("tuning.implicit_mf_seeds must not contain duplicates")
    return BenchmarkTuningConfig(
        selection_metric=selection_metric,
        direction=direction,
        validation_split=BenchmarkValidationSplitConfig(
            method=method,
            validation_ratio=validation_ratio,
        ),
        implicit_mf_seeds=tuple(seeds),
    )


def benchmark_config_from_dict(
    payload: Mapping[str, Any], *, base_dir: str | Path = "."
) -> BenchmarkConfig:
    """Validate a decoded benchmark configuration and resolve its data path."""

    root = _object(payload, "benchmark configuration")
    _unknown(
        root,
        {"schema_version", "seed", "data", "split", "evaluation", "models", "tuning"},
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
        minimum_rating = (
            4.0 if raw_minimum is None else _finite_number(raw_minimum, "data.minimum_rating")
        )
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

    raw_tuning = root.get("tuning")
    tuning = (
        None
        if raw_tuning is None
        else _parse_tuning(
            raw_tuning,
            seed=seed,
            split_method=split_method,
            split_ratio=test_ratio,
        )
    )

    raw_models = root.get("models")
    if raw_models is None:
        models = default_benchmark_models()
    else:
        if not isinstance(raw_models, list) or not raw_models:
            raise ConfigurationError("models must be a non-empty array")
        models = tuple(
            _parse_model(entry, index, seed, tuning_enabled=tuning is not None)
            for index, entry in enumerate(raw_models)
        )
    labels = [model.label for model in models]
    if len(set(labels)) != len(labels):
        raise ConfigurationError("model labels must be unique")
    if len(models) < 2:
        raise ConfigurationError("a benchmark requires at least two models")
    if tuning is not None and not any(model.grid for model in models):
        raise ConfigurationError("tuning requires at least one model grid")

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
        tuning=tuning,
    )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Read a strict JSON benchmark configuration."""

    source = Path(path)
    try:
        payload = strict_json_loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(
            f"could not read benchmark configuration from {source}: {exc}"
        ) from exc
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
            json.dumps(
                config.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"could not write benchmark configuration to {destination}: {exc}"
        ) from exc
