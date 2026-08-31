"""Strict JSON configuration for reproducible experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orchidrec._json import strict_json_loads
from orchidrec._numeric import safe_float
from orchidrec.errors import ConfigurationError, ValidationError


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ConfigurationError(f"{name} must be a JSON object")
    return value


def _unknown(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    extras = set(mapping) - allowed
    if extras:
        raise ConfigurationError(f"unknown {name} fields: {', '.join(sorted(extras))}")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class DataConfig:
    path: Path


@dataclass(frozen=True, slots=True)
class SplitConfig:
    method: str = "leave_one_out"
    test_ratio: float = 0.2


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    k: int = 10
    exclude_seen: bool = True


@dataclass(frozen=True, slots=True)
class OutputConfig:
    report_path: Path | None = None
    model_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """All inputs needed for one deterministic offline experiment."""

    seed: int
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration."""

        return {
            "seed": self.seed,
            "data": {"path": str(self.data.path)},
            "split": {"method": self.split.method, "test_ratio": self.split.test_ratio},
            "model": {"name": self.model.name, "params": dict(self.model.params)},
            "evaluation": {"k": self.evaluation.k, "exclude_seen": self.evaluation.exclude_seen},
            "output": {
                "report_path": str(self.output.report_path) if self.output.report_path else None,
                "model_path": str(self.output.model_path) if self.output.model_path else None,
            },
        }


def config_from_dict(payload: Mapping[str, Any], *, base_dir: str | Path = ".") -> ExperimentConfig:
    """Validate a decoded configuration and resolve its paths."""

    root = _object(payload, "configuration")
    _unknown(root, {"seed", "data", "split", "model", "evaluation", "output"}, "configuration")
    if "data" not in root or "model" not in root:
        raise ConfigurationError("configuration requires 'data' and 'model' objects")
    seed = root.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("seed must be an integer")
    directory = Path(base_dir).resolve()

    data = _object(root["data"], "data")
    _unknown(data, {"path"}, "data")
    data_path = data.get("path")
    if not isinstance(data_path, str) or not data_path:
        raise ConfigurationError("data.path must be a non-empty string")
    resolved_data = Path(data_path)
    if not resolved_data.is_absolute():
        resolved_data = directory / resolved_data
    resolved_data = resolved_data.resolve()

    split = _object(root.get("split", {}), "split")
    _unknown(split, {"method", "test_ratio"}, "split")
    split_method = split.get("method", "leave_one_out")
    if not isinstance(split_method, str) or split_method not in {"random", "temporal", "leave_one_out"}:
        raise ConfigurationError("split.method must be random, temporal, or leave_one_out")
    test_ratio = split.get("test_ratio", 0.2)
    if isinstance(test_ratio, bool) or not isinstance(test_ratio, (int, float)):
        raise ConfigurationError("split.test_ratio must be a finite number between 0 and 1")
    numeric_ratio = safe_float(test_ratio)
    if not math.isfinite(numeric_ratio) or not 0 < numeric_ratio < 1:
        raise ConfigurationError("split.test_ratio must be a finite number between 0 and 1")

    model = _object(root["model"], "model")
    _unknown(model, {"name", "params"}, "model")
    model_name = model.get("name")
    if not isinstance(model_name, str) or model_name not in {"popularity", "item_knn", "implicit_mf"}:
        raise ConfigurationError("model.name must be popularity, item_knn, or implicit_mf")
    params = _object(model.get("params", {}), "model.params")
    allowed_params = {
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
    }
    _unknown(params, allowed_params[model_name], "model.params")
    from orchidrec.models import ImplicitMF, ItemKNN, Popularity

    model_types = {
        "popularity": Popularity,
        "item_knn": ItemKNN,
        "implicit_mf": ImplicitMF,
    }
    try:
        model_types[model_name](**dict(params))
    except (TypeError, ValidationError) as exc:
        raise ConfigurationError(f"invalid model.params for {model_name}: {exc}") from exc

    evaluation = _object(root.get("evaluation", {}), "evaluation")
    _unknown(evaluation, {"k", "exclude_seen"}, "evaluation")
    k = _positive_int(evaluation.get("k", 10), "evaluation.k")
    exclude_seen = evaluation.get("exclude_seen", True)
    if not isinstance(exclude_seen, bool):
        raise ConfigurationError("evaluation.exclude_seen must be a boolean")

    output = _object(root.get("output", {}), "output")
    _unknown(output, {"report_path", "model_path"}, "output")

    def optional_path(name: str) -> Path | None:
        raw = output.get(name)
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw:
            raise ConfigurationError(f"output.{name} must be a non-empty string or null")
        path = Path(raw)
        return path.resolve() if path.is_absolute() else (directory / path).resolve()

    return ExperimentConfig(
        seed=seed,
        data=DataConfig(path=resolved_data),
        split=SplitConfig(method=split_method, test_ratio=numeric_ratio),
        model=ModelConfig(name=model_name, params=dict(params)),
        evaluation=EvaluationConfig(k=k, exclude_seen=exclude_seen),
        output=OutputConfig(report_path=optional_path("report_path"), model_path=optional_path("model_path")),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate an experiment JSON file."""

    source = Path(path)
    try:
        payload = strict_json_loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"could not read configuration from {source}: {exc}") from exc
    except ValueError as exc:
        raise ConfigurationError(f"invalid configuration JSON in {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigurationError("configuration file must contain a JSON object")
    return config_from_dict(payload, base_dir=source.parent)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Write a normalized configuration as deterministic JSON."""

    if not isinstance(config, ExperimentConfig):
        raise ConfigurationError("config must be an ExperimentConfig")
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                config.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"could not write configuration to {destination}: {exc}") from exc
