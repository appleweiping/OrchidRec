"""End-to-end reproducible experiment orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchidrec.config import ExperimentConfig
from orchidrec.data import EntityId, InteractionDataset, stable_id_key
from orchidrec.errors import ConfigurationError, SerializationError, ValidationError
from orchidrec.metrics import MetricReport, evaluate_ranking
from orchidrec.models import (
    BaseRecommender,
    ImplicitMF,
    ItemKNN,
    Popularity,
    Recommendation,
    save_model,
)
from orchidrec.split import SplitResult, leave_one_out, random_split, temporal_split


@dataclass(frozen=True, slots=True)
class UserEvaluation:
    """Ground truth and recommendations for one evaluated user."""

    user_id: EntityId
    relevant: tuple[EntityId, ...]
    recommendations: tuple[Recommendation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "relevant": list(self.relevant),
            "recommendations": [entry.to_dict() for entry in self.recommendations],
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Stable, JSON-compatible output of one experiment run."""

    seed: int
    split_method: str
    model_type: str
    model_parameters: dict[str, Any]
    train_size: int
    test_size: int
    evaluated_test_size: int
    cold_start_test_size: int
    catalog_size: int
    metrics: MetricReport
    users: tuple[UserEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "split": self.split_method,
            "model": {"type": self.model_type, "parameters": dict(self.model_parameters)},
            "dataset": {
                "train_interactions": self.train_size,
                "test_interactions": self.test_size,
                "evaluated_test_interactions": self.evaluated_test_size,
                "cold_start_test_interactions": self.cold_start_test_size,
                "catalog_items": self.catalog_size,
                "evaluated_users": len(self.users),
            },
            "metrics": self.metrics.to_dict(),
            "users": [user.to_dict() for user in self.users],
        }

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SerializationError(f"could not write experiment report to {destination}: {exc}") from exc


def build_model(name: str, parameters: dict[str, Any], *, experiment_seed: int) -> BaseRecommender:
    """Construct a configured model without fitting it."""

    params = dict(parameters)
    if name == "implicit_mf":
        params.setdefault("seed", experiment_seed)
    registry: dict[str, type[BaseRecommender]] = {
        "popularity": Popularity,
        "item_knn": ItemKNN,
        "implicit_mf": ImplicitMF,
    }
    model_class = registry.get(name)
    if model_class is None:
        raise ConfigurationError(f"unknown model: {name!r}")
    try:
        return model_class(**params)
    except (TypeError, ValidationError) as exc:
        raise ConfigurationError(f"invalid parameters for {name}: {exc}") from exc


def split_dataset(config: ExperimentConfig, dataset: InteractionDataset) -> SplitResult:
    """Apply the configured split strategy."""

    if config.split.method == "random":
        return random_split(dataset, config.split.test_ratio, config.seed)
    if config.split.method == "temporal":
        return temporal_split(dataset, config.split.test_ratio)
    if config.split.method == "leave_one_out":
        return leave_one_out(dataset)
    raise ConfigurationError(f"unknown split method: {config.split.method!r}")


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Load, split, fit, recommend, evaluate, and optionally persist outputs."""

    if not isinstance(config, ExperimentConfig):
        raise ConfigurationError("config must be an ExperimentConfig")
    dataset = InteractionDataset.load_json(config.data.path)
    split = split_dataset(config, dataset)
    if not split.train:
        raise ConfigurationError("split produced an empty training set")
    if not split.test:
        raise ConfigurationError("split produced no evaluable test interactions")
    train_pairs = {(event.user_id, event.item_id) for event in split.train}
    test_pairs = {(event.user_id, event.item_id) for event in split.test}
    if train_pairs & test_pairs:
        raise ConfigurationError("split leaked a user-item pair across train and test")
    model = build_model(config.model.name, config.model.params, experiment_seed=config.seed)
    model.fit(split.train)
    catalog = set(model.catalog)
    relevant_sets: dict[EntityId, set[EntityId]] = {}
    evaluated_test_size = 0
    cold_start_test_size = 0
    for event in split.test:
        if event.item_id not in catalog:
            cold_start_test_size += 1
            continue
        evaluated_test_size += 1
        relevant_sets.setdefault(event.user_id, set()).add(event.item_id)
    if not relevant_sets:
        raise ConfigurationError(
            "test data contains no items present in the training catalog"
        )
    recommendation_ids: dict[EntityId, list[EntityId]] = {}
    user_rows: list[UserEvaluation] = []
    for user_id in sorted(relevant_sets, key=stable_id_key):
        recommendations = tuple(
            model.recommend(
                user_id,
                config.evaluation.k,
                exclude_seen=config.evaluation.exclude_seen,
            )
        )
        recommendation_ids[user_id] = [entry.item_id for entry in recommendations]
        user_rows.append(
            UserEvaluation(
                user_id=user_id,
                relevant=tuple(sorted(relevant_sets[user_id], key=stable_id_key)),
                recommendations=recommendations,
            )
        )
    relevant = {user_id: frozenset(items) for user_id, items in relevant_sets.items()}
    metrics = evaluate_ranking(
        recommendation_ids,
        relevant,
        model.catalog,
        split.train.item_counts(),
        config.evaluation.k,
    )
    model_parameters = dict(model.to_state()["parameters"])
    result = ExperimentResult(
        seed=config.seed,
        split_method=config.split.method,
        model_type=config.model.name,
        model_parameters=model_parameters,
        train_size=len(split.train),
        test_size=len(split.test),
        evaluated_test_size=evaluated_test_size,
        cold_start_test_size=cold_start_test_size,
        catalog_size=len(model.catalog),
        metrics=metrics,
        users=tuple(user_rows),
    )
    if config.output.model_path is not None:
        save_model(model, config.output.model_path)
    if config.output.report_path is not None:
        result.save_json(config.output.report_path)
    return result
