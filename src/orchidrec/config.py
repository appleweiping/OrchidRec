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
from orchidrec.propensity import DEFAULT_EXPONENT, DEFAULT_MINIMUM_PROPENSITY
from orchidrec.sampling import SamplingConfig, parse_sampling


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ConfigurationError(f"{name} must be a JSON object")
    return value


def _unknown(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    extras = set(mapping) - allowed
    if extras:
        raise ConfigurationError(f"unknown {name} fields: {', '.join(sorted(extras))}")


def _unit_float(value: object, name: str, *, minimum: float, maximum: float) -> float:
    """Validate a bounded real configuration number without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a real number")
    number = safe_float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return number


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
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
class ExposureConfig:
    """Parameters of the popularity exposure model used for correction.

    Declaring this asks for exposure-corrected metrics alongside the ordinary
    ones. Leaving it out asks for no exposure model at all, which is a claim
    about the data rather than a default worth hiding.
    """

    exponent: float = DEFAULT_EXPONENT
    minimum: float = DEFAULT_MINIMUM_PROPENSITY


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    k: int = 10
    exclude_seen: bool = True
    exposure: ExposureConfig | None = None
    sampling: SamplingConfig | None = None


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
            "evaluation": {
                "k": self.evaluation.k,
                "exclude_seen": self.evaluation.exclude_seen,
                "exposure": (
                    None
                    if self.evaluation.exposure is None
                    else {
                        "exponent": self.evaluation.exposure.exponent,
                        "minimum": self.evaluation.exposure.minimum,
                    }
                ),
                **(
                    {"sampling": self.evaluation.sampling.to_dict()}
                    if self.evaluation.sampling
                    else {}
                ),
            },
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
    if type(seed) is not int:
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
    if not isinstance(split_method, str) or split_method not in {
        "random",
        "temporal",
        "leave_one_out",
    }:
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
    if not isinstance(model_name, str) or model_name not in {
        "popularity",
        "confidence_als",
        "ease",
        "slim_elastic",
        "item_knn",
        "implicit_mf",
        "user_knn",
        "sequential_markov",
    }:
        raise ConfigurationError(
            "model.name must be popularity, item_knn, implicit_mf, confidence_als, ease, "
            "slim_elastic, user_knn, or sequential_markov"
        )
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
        "confidence_als": {"factors", "epochs", "alpha", "regularization", "seed"},
        "ease": {"regularization", "max_items", "max_interactions", "max_work_units"},
        "slim_elastic": {
            "l1",
            "l2",
            "max_sweeps",
            "tolerance",
            "max_items",
            "max_interactions",
            "max_work_units",
        },
        "user_knn": {"neighbors", "shrinkage"},
        "sequential_markov": {"weighted", "popularity_mix"},
    }
    _unknown(params, allowed_params[model_name], "model.params")
    from orchidrec.models import (
        EASE,
        ConfidenceALS,
        ImplicitMF,
        ItemKNN,
        Popularity,
        SequentialMarkov,
        SLIMElastic,
        UserKNN,
    )

    model_types = {
        "popularity": Popularity,
        "confidence_als": ConfidenceALS,
        "ease": EASE,
        "slim_elastic": SLIMElastic,
        "item_knn": ItemKNN,
        "implicit_mf": ImplicitMF,
        "user_knn": UserKNN,
        "sequential_markov": SequentialMarkov,
    }
    try:
        model_types[model_name](**dict(params))
    except (TypeError, ValidationError) as exc:
        raise ConfigurationError(f"invalid model.params for {model_name}: {exc}") from exc

    evaluation = _object(root.get("evaluation", {}), "evaluation")
    _unknown(evaluation, {"k", "exclude_seen", "exposure", "sampling"}, "evaluation")
    k = _positive_int(evaluation.get("k", 10), "evaluation.k")
    exclude_seen = evaluation.get("exclude_seen", True)
    if not isinstance(exclude_seen, bool):
        raise ConfigurationError("evaluation.exclude_seen must be a boolean")
    sampling = parse_sampling(evaluation.get("sampling"))
    if sampling is not None and not exclude_seen:
        raise ConfigurationError("sampled evaluation requires evaluation.exclude_seen=true")
    raw_exposure = evaluation.get("exposure")
    if raw_exposure is None:
        exposure = None
    else:
        exposure_fields = _object(raw_exposure, "evaluation.exposure")
        _unknown(exposure_fields, {"exponent", "minimum"}, "evaluation.exposure")
        exposure = ExposureConfig(
            exponent=_unit_float(
                exposure_fields.get("exponent", DEFAULT_EXPONENT),
                "evaluation.exposure.exponent",
                minimum=0.0,
                maximum=1.0,
            ),
            minimum=_unit_float(
                exposure_fields.get("minimum", DEFAULT_MINIMUM_PROPENSITY),
                "evaluation.exposure.minimum",
                minimum=1e-6,
                maximum=1.0,
            ),
        )

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
        evaluation=EvaluationConfig(
            k=k, exclude_seen=exclude_seen, exposure=exposure, sampling=sampling
        ),
        output=OutputConfig(
            report_path=optional_path("report_path"), model_path=optional_path("model_path")
        ),
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
