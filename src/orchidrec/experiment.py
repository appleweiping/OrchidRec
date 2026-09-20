"""End-to-end reproducible experiment orchestration."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchidrec._files import atomic_write_text, require_distinct_paths
from orchidrec.config import ExperimentConfig
from orchidrec.data import EntityId, InteractionDataset, stable_id_key
from orchidrec.errors import ConfigurationError, SerializationError, ValidationError
from orchidrec.metrics import MetricReport, evaluate_ranking
from orchidrec.models import (
    EASE,
    BaseRecommender,
    ConfidenceALS,
    ImplicitMF,
    ItemKNN,
    Popularity,
    Recommendation,
    SequentialMarkov,
    SLIMElastic,
    UserKNN,
    save_model,
)
from orchidrec.propensity import popularity_exposure
from orchidrec.sampling import CandidatePlan, sample_candidates
from orchidrec.split import SplitResult, leave_one_out, random_split, temporal_split
from orchidrec.unbiased import UnbiasedMetricReport, evaluate_unbiased_ranking


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
    unbiased_metrics: UnbiasedMetricReport | None = None
    candidate_plan: CandidatePlan | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        # Additive in the strict sense: the section appears only for a run that
        # configured an exposure model, so a configuration that does not ask for
        # one keeps exactly the report it produced before.
        if self.unbiased_metrics is not None:
            payload["unbiased_metrics"] = self.unbiased_metrics.to_dict()
        if self.candidate_plan is not None:
            payload["evaluation"] = {
                "mode": "sampled",
                "sampling": self.candidate_plan.sampling.to_dict(),
                "candidate_sha256": self.candidate_plan.fingerprint,
                "candidate_pairs": self.candidate_plan.candidate_pairs,
                "positive_pairs": self.candidate_plan.positive_pairs,
                "negative_pairs": self.candidate_plan.negative_pairs,
            }
        return payload

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        try:
            atomic_write_text(
                destination,
                json.dumps(
                    self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
                )
                + "\n",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SerializationError(
                f"could not write experiment report to {destination}: {exc}"
            ) from exc


def build_model(name: str, parameters: dict[str, Any], *, experiment_seed: int) -> BaseRecommender:
    """Construct a configured model without fitting it."""

    params = dict(parameters)
    if name in {"implicit_mf", "confidence_als"}:
        params.setdefault("seed", experiment_seed)
    registry: dict[str, type[BaseRecommender]] = {
        "popularity": Popularity,
        "confidence_als": ConfidenceALS,
        "ease": EASE,
        "slim_elastic": SLIMElastic,
        "item_knn": ItemKNN,
        "implicit_mf": ImplicitMF,
        "user_knn": UserKNN,
        "sequential_markov": SequentialMarkov,
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
    paths = {"data.path": config.data.path}
    if config.output.model_path is not None:
        paths["output.model_path"] = config.output.model_path
    if config.output.report_path is not None:
        paths["output.report_path"] = config.output.report_path
    require_distinct_paths(paths)
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
        raise ConfigurationError("test data contains no items present in the training catalog")
    recommendation_ids: dict[EntityId, list[EntityId]] = {}
    user_rows: list[UserEvaluation] = []
    candidate_plan = None
    if config.evaluation.sampling is not None:
        candidate_plan = sample_candidates(
            config.evaluation.sampling,
            seed=config.seed,
            catalog=model.catalog,
            train_counts=Counter(event.item_id for event in split.train),
            relevant=relevant_sets,
            seen={user: model.seen_items(user) for user in relevant_sets},
        )
    for user_id in sorted(relevant_sets, key=stable_id_key):
        recommendations = tuple(
            model.recommend(
                user_id,
                config.evaluation.k,
                exclude_seen=config.evaluation.exclude_seen,
                candidates=(
                    candidate_plan.candidates[user_id] if candidate_plan is not None else None
                ),
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
    unbiased_metrics = None
    if config.evaluation.exposure is not None:
        # Popularity comes from the training split alone. Estimating exposure
        # from the test split would let the holdout explain its own sampling.
        exposure = popularity_exposure(
            split.train.item_counts(),
            exponent=config.evaluation.exposure.exponent,
            minimum=config.evaluation.exposure.minimum,
        )
        unbiased_metrics = evaluate_unbiased_ranking(
            recommendation_ids, relevant, exposure, config.evaluation.k
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
        unbiased_metrics=unbiased_metrics,
        candidate_plan=candidate_plan,
    )
    if config.output.model_path is not None:
        save_model(model, config.output.model_path)
    if config.output.report_path is not None:
        result.save_json(config.output.report_path)
    return result
