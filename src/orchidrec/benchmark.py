"""Shared-split, multi-model recommendation benchmarks."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from orchidrec.benchmark_config import (
    BENCHMARK_METRIC_NAMES,
    BenchmarkConfig,
    BenchmarkModelSpec,
    BenchmarkTuningConfig,
    benchmark_config_from_dict,
)
from orchidrec.data import EntityId, InteractionDataset, stable_id_key
from orchidrec.datasets import DatasetSummary, interaction_fingerprint, load_dataset
from orchidrec.errors import ConfigurationError
from orchidrec.experiment import build_model
from orchidrec.metrics import MetricReport, evaluate_ranking
from orchidrec.models import KGWalkRec
from orchidrec.recbole_knowledge import LoadedKnowledgeLinks, load_recbole_knowledge
from orchidrec.sampling import CandidatePlan, sample_candidates
from orchidrec.split import SplitResult, leave_one_out, random_split, temporal_split
from orchidrec.statistics import (
    BootstrapInterval,
    PairedBootstrapResult,
    interval_from_draws,
    paired_comparison_from_draws,
)

BENCHMARK_FORMAT = "orchidrec.benchmark"
BENCHMARK_SCHEMA_VERSION = 1
METRIC_NAMES = BENCHMARK_METRIC_NAMES


@dataclass(frozen=True, slots=True)
class BenchmarkTiming:
    """Observed wall-clock time for a benchmark phase."""

    fit_seconds: float
    recommend_seconds: float

    def to_dict(self) -> dict[str, float]:
        return {
            "fit_seconds": self.fit_seconds,
            "recommend_seconds": self.recommend_seconds,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkModelResult:
    """Metrics, uncertainty, and timing for one model specification."""

    label: str
    model_type: str
    parameters: dict[str, Any]
    timing: BenchmarkTiming
    metrics: MetricReport
    confidence_intervals: Mapping[str, BootstrapInterval]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "model_type": self.model_type,
            "parameters": dict(self.parameters),
            "timing": self.timing.to_dict(),
            "metrics": self.metrics.to_dict(),
            "confidence_intervals": {
                name: self.confidence_intervals[name].to_dict() for name in METRIC_NAMES
            },
        }


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Paired bootstrap results; positive differences favor ``right_label``."""

    left_label: str
    right_label: str
    metrics: Mapping[str, PairedBootstrapResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "left_label": self.left_label,
            "right_label": self.right_label,
            "difference_direction": "right_minus_left",
            "metrics": {name: self.metrics[name].to_dict() for name in METRIC_NAMES},
        }


@dataclass(frozen=True, slots=True)
class TuningTrial:
    """One fit-and-evaluate run on the inner validation partition."""

    candidate_index: int
    seed: int | None
    parameters: dict[str, Any]
    selection_value: float
    timing: BenchmarkTiming
    metrics: MetricReport

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_index": self.candidate_index,
            "seed": self.seed,
            "parameters": dict(self.parameters),
            "selection_value": self.selection_value,
            "timing": self.timing.to_dict(),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TuningCandidateResult:
    """Validation aggregate for one explicit Cartesian-product candidate."""

    candidate_index: int
    parameters: dict[str, Any]
    mean_selection_value: float
    trial_indices: tuple[int, ...]
    selected: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_index": self.candidate_index,
            "parameters": dict(self.parameters),
            "mean_selection_value": self.mean_selection_value,
            "trial_indices": list(self.trial_indices),
            "selected": self.selected,
        }


@dataclass(frozen=True, slots=True)
class ModelTuningResult:
    """Full validation-only search record for one benchmark model."""

    label: str
    model_type: str
    search_space: Mapping[str, tuple[Any, ...]]
    candidates: tuple[TuningCandidateResult, ...]
    trials: tuple[TuningTrial, ...]
    selected_candidate_index: int
    selected_validation_score: float
    final_parameters: dict[str, Any]
    final_seed: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "model_type": self.model_type,
            "search_space": {name: list(values) for name, values in self.search_space.items()},
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "trials": [trial.to_dict() for trial in self.trials],
            "selected_candidate_index": self.selected_candidate_index,
            "selected_validation_score": self.selected_validation_score,
            "final_parameters": dict(self.final_parameters),
            "final_seed": self.final_seed,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkTuningResult:
    """Leakage-auditable inner-split model-selection evidence."""

    selection_metric: str
    direction: str
    validation_method: str
    validation_ratio: float
    implicit_mf_seeds: tuple[int, ...]
    final_seed: int
    development_interactions: int
    training_interactions: int
    validation_interactions: int
    evaluated_validation_interactions: int
    cold_start_validation_interactions: int
    evaluated_validation_users: int
    development_fingerprint: str
    training_fingerprint: str
    validation_fingerprint: str
    test_fingerprint: str
    three_way_split_fingerprint: str
    models: tuple[ModelTuningResult, ...]
    validation_candidate_fingerprint: str | None = None

    def to_dict(self) -> dict[str, object]:
        fingerprints = {
            "development_sha256": self.development_fingerprint,
            "training_sha256": self.training_fingerprint,
            "validation_sha256": self.validation_fingerprint,
            "test_sha256": self.test_fingerprint,
            "three_way_split_sha256": self.three_way_split_fingerprint,
        }
        if self.validation_candidate_fingerprint is not None:
            fingerprints["validation_candidate_sha256"] = self.validation_candidate_fingerprint
        return {
            "selection_metric": self.selection_metric,
            "direction": self.direction,
            "validation_split": {
                "method": self.validation_method,
                "validation_ratio": self.validation_ratio,
                "development_interactions": self.development_interactions,
                "training_interactions": self.training_interactions,
                "validation_interactions": self.validation_interactions,
                "evaluated_validation_interactions": self.evaluated_validation_interactions,
                "cold_start_validation_interactions": self.cold_start_validation_interactions,
                "evaluated_validation_users": self.evaluated_validation_users,
            },
            "seed_policy": {
                "implicit_mf_validation_seeds": list(self.implicit_mf_seeds),
                "confidence_als_validation_seeds": list(self.implicit_mf_seeds),
                "final_fit_seed": self.final_seed,
                "deterministic_models_are_not_repeated": True,
            },
            "fingerprints": fingerprints,
            "models": [model.to_dict() for model in self.models],
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Complete versioned result for a reproducible multi-model benchmark."""

    seed: int
    config_fingerprint: str
    split_fingerprint: str
    dataset: DatasetSummary
    split_method: str
    test_ratio: float
    train_interactions: int
    test_interactions: int
    evaluated_test_interactions: int
    cold_start_test_interactions: int
    evaluated_users: int
    k: int
    exclude_seen: bool
    bootstrap_samples: int
    confidence: float
    models: tuple[BenchmarkModelResult, ...]
    comparisons: tuple[BenchmarkComparison, ...]
    tuning: BenchmarkTuningResult | None = None
    candidate_plan: CandidatePlan | None = field(default=None, compare=False, repr=False)
    source_path: Path | None = field(default=None, compare=False, repr=False)
    knowledge_path: Path | None = field(default=None, compare=False, repr=False)
    knowledge: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": BENCHMARK_FORMAT,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "seed": self.seed,
            "fingerprints": {
                "config_sha256": self.config_fingerprint,
                "split_sha256": self.split_fingerprint,
            },
            "dataset": self.dataset.to_dict(),
            "split": {
                "method": self.split_method,
                "test_ratio": self.test_ratio,
                "train_interactions": self.train_interactions,
                "test_interactions": self.test_interactions,
                "evaluated_test_interactions": self.evaluated_test_interactions,
                "cold_start_test_interactions": self.cold_start_test_interactions,
                "evaluated_users": self.evaluated_users,
            },
            "evaluation": {
                "k": self.k,
                "exclude_seen": self.exclude_seen,
                "bootstrap_samples": self.bootstrap_samples,
                "confidence": self.confidence,
            },
            "models": [model.to_dict() for model in self.models],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
        }
        if self.tuning is not None:
            payload["tuning"] = self.tuning.to_dict()
        if self.candidate_plan is not None:
            evaluation = payload["evaluation"]
            if not isinstance(evaluation, dict):
                raise ConfigurationError("malformed evaluation report")
            evaluation.update(
                {
                    "mode": "sampled",
                    "sampling": self.candidate_plan.sampling.to_dict(),
                    "candidate_sha256": self.candidate_plan.fingerprint,
                    "candidate_pairs": self.candidate_plan.candidate_pairs,
                    "positive_pairs": self.candidate_plan.positive_pairs,
                    "negative_pairs": self.candidate_plan.negative_pairs,
                }
            )
        if self.knowledge is not None:
            payload["knowledge"] = dict(self.knowledge)
        return payload


@dataclass(frozen=True, slots=True)
class _TargetSet:
    users: tuple[EntityId, ...]
    relevant: Mapping[EntityId, frozenset[EntityId]]
    evaluated_interactions: int
    cold_start_interactions: int


@dataclass(frozen=True, slots=True)
class _RankingRow:
    precision: float
    recall: float
    ndcg: float
    mrr: float
    exposure_mask: int
    novelty_sum: float
    novelty_count: int


@dataclass(frozen=True, slots=True)
class _MetricVector:
    precision: float
    recall: float
    ndcg: float
    mrr: float
    coverage: float
    novelty: float

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "ndcg": self.ndcg,
            "mrr": self.mrr,
            "coverage": self.coverage,
            "novelty": self.novelty,
        }


@dataclass(frozen=True, slots=True)
class _EvaluatedModel:
    spec: BenchmarkModelSpec
    parameters: dict[str, Any]
    timing: BenchmarkTiming
    metrics: MetricReport
    rows: tuple[_RankingRow, ...]


def _split(config: BenchmarkConfig, dataset: InteractionDataset) -> SplitResult:
    if config.split.method == "random":
        return random_split(dataset, config.split.test_ratio, config.seed)
    if config.split.method == "temporal":
        return temporal_split(dataset, config.split.test_ratio)
    if config.split.method == "leave_one_out":
        return leave_one_out(dataset)
    raise ConfigurationError(f"unknown split method: {config.split.method!r}")


def _targets(split: SplitResult) -> _TargetSet:
    catalog = set(split.train.item_ids)
    relevant: dict[EntityId, set[EntityId]] = {}
    evaluated = 0
    cold_start = 0
    for event in split.test:
        if event.item_id not in catalog:
            cold_start += 1
            continue
        relevant.setdefault(event.user_id, set()).add(event.item_id)
        evaluated += 1
    if not relevant:
        raise ConfigurationError("split produced no test targets present in the training catalog")
    frozen = {user_id: frozenset(items) for user_id, items in relevant.items()}
    return _TargetSet(
        users=tuple(sorted(frozen, key=stable_id_key)),
        relevant=frozen,
        evaluated_interactions=evaluated,
        cold_start_interactions=cold_start,
    )


def _ranking_rows(
    users: Sequence[EntityId],
    relevant: Mapping[EntityId, frozenset[EntityId]],
    recommendations: Mapping[EntityId, tuple[EntityId, ...]],
    catalog: Sequence[EntityId],
    item_counts: Mapping[EntityId, float],
    k: int,
) -> tuple[_RankingRow, ...]:
    catalog_indices = {item_id: index for index, item_id in enumerate(catalog)}
    denominator = math.fsum(item_counts.values()) + len(catalog)
    novelty = {
        item_id: -math.log2((item_counts.get(item_id, 0.0) + 1.0) / denominator)
        for item_id in catalog
    }
    rows: list[_RankingRow] = []
    for user_id in users:
        target = relevant[user_id]
        ranking = recommendations[user_id][:k]
        hits = tuple(index for index, item_id in enumerate(ranking) if item_id in target)
        precision = len(hits) / k
        recall = len(hits) / len(target)
        dcg = math.fsum(1.0 / math.log2(index + 2) for index in hits)
        ideal = math.fsum(1.0 / math.log2(index + 2) for index in range(min(k, len(target))))
        reciprocal_rank = 1.0 / (hits[0] + 1) if hits else 0.0
        mask = 0
        for item_id in ranking:
            mask |= 1 << catalog_indices[item_id]
        rows.append(
            _RankingRow(
                precision=precision,
                recall=recall,
                ndcg=dcg / ideal,
                mrr=reciprocal_rank,
                exposure_mask=mask,
                novelty_sum=math.fsum(novelty[item_id] for item_id in ranking),
                novelty_count=len(ranking),
            )
        )
    return tuple(rows)


def _aggregate(
    rows: Sequence[_RankingRow], indices: Sequence[int], catalog_size: int
) -> _MetricVector:
    count = len(indices)
    exposure = 0
    novelty_count = 0
    novelty_sum = 0.0
    precision = 0.0
    recall = 0.0
    ndcg = 0.0
    mrr = 0.0
    for index in indices:
        row = rows[index]
        precision += row.precision / count
        recall += row.recall / count
        ndcg += row.ndcg / count
        mrr += row.mrr / count
        exposure |= row.exposure_mask
        novelty_sum += row.novelty_sum
        novelty_count += row.novelty_count
    return _MetricVector(
        precision=precision,
        recall=recall,
        ndcg=ndcg,
        mrr=mrr,
        coverage=exposure.bit_count() / catalog_size,
        novelty=novelty_sum / novelty_count if novelty_count else 0.0,
    )


def _evaluate_model(
    config: BenchmarkConfig,
    spec: BenchmarkModelSpec,
    split: SplitResult,
    targets: _TargetSet,
    *,
    parameters: Mapping[str, Any] | None = None,
    experiment_seed: int | None = None,
    candidate_plan: CandidatePlan | None = None,
    knowledge: LoadedKnowledgeLinks | None = None,
) -> _EvaluatedModel:
    model = build_model(
        spec.name,
        dict(spec.params if parameters is None else parameters),
        experiment_seed=config.seed if experiment_seed is None else experiment_seed,
    )
    fit_started = time.perf_counter_ns()
    if isinstance(model, KGWalkRec):
        if knowledge is None:
            raise ConfigurationError("kg_walk_rec requires a knowledge artifact")
        model.fit(split.train, knowledge)
    else:
        model.fit(split.train)
    fit_seconds = (time.perf_counter_ns() - fit_started) / 1_000_000_000.0
    recommendation_started = time.perf_counter_ns()
    recommendations = {
        user_id: tuple(
            entry.item_id
            for entry in model.recommend(
                user_id,
                config.evaluation.k,
                exclude_seen=config.evaluation.exclude_seen,
                candidates=(
                    candidate_plan.candidates[user_id] if candidate_plan is not None else None
                ),
            )
        )
        for user_id in targets.users
    }
    recommend_seconds = (time.perf_counter_ns() - recommendation_started) / 1_000_000_000.0
    item_counts = split.train.item_counts()
    metrics = evaluate_ranking(
        recommendations,
        targets.relevant,
        model.catalog,
        item_counts,
        config.evaluation.k,
    )
    rows = _ranking_rows(
        targets.users,
        targets.relevant,
        recommendations,
        model.catalog,
        item_counts,
        config.evaluation.k,
    )
    effective_parameters = model.to_state()["parameters"]
    if not isinstance(effective_parameters, dict):
        raise ConfigurationError("model returned malformed effective parameters")
    return _EvaluatedModel(
        spec=spec,
        parameters=dict(effective_parameters),
        timing=BenchmarkTiming(fit_seconds=fit_seconds, recommend_seconds=recommend_seconds),
        metrics=metrics,
        rows=rows,
    )


def _candidate_plan(
    config: BenchmarkConfig, split: SplitResult, targets: _TargetSet
) -> CandidatePlan | None:
    sampling = config.evaluation.sampling
    if sampling is None:
        return None
    seen_by_user = split.train.by_user()
    return sample_candidates(
        sampling,
        seed=config.seed,
        catalog=split.train.item_ids,
        train_counts=Counter(event.item_id for event in split.train),
        relevant=targets.relevant,
        seen={
            user: frozenset(event.item_id for event in seen_by_user.get(user, ()))
            for user in targets.users
        },
    )


def _validation_split(
    tuning: BenchmarkTuningConfig,
    dataset: InteractionDataset,
    seed: int,
) -> SplitResult:
    validation = tuning.validation_split
    if validation.method == "random":
        return random_split(dataset, validation.validation_ratio, seed)
    if validation.method == "temporal":
        return temporal_split(dataset, validation.validation_ratio)
    if validation.method == "leave_one_out":
        return leave_one_out(dataset)
    raise ConfigurationError(f"unknown validation split method: {validation.method!r}")


def _parameter_candidates(spec: BenchmarkModelSpec) -> tuple[dict[str, Any], ...]:
    """Expand a validated grid in a canonical, configuration-order-independent order."""

    candidates: list[dict[str, Any]] = [dict(spec.params)]
    for parameter in sorted(spec.grid):
        candidates = [
            {**candidate, parameter: value}
            for candidate in candidates
            for value in spec.grid[parameter]
        ]
    return tuple(candidates)


def _selection_value(metrics: MetricReport, metric: str) -> float:
    try:
        return _point_metrics(metrics)[metric]
    except KeyError as exc:
        raise ConfigurationError(f"unknown tuning selection metric: {metric!r}") from exc


@dataclass(frozen=True, slots=True)
class _SelectedModel:
    spec: BenchmarkModelSpec
    parameters: dict[str, Any]
    candidates: tuple[TuningCandidateResult, ...]
    trials: tuple[TuningTrial, ...]
    selected_candidate_index: int
    selected_validation_score: float


def _tune_model(
    config: BenchmarkConfig,
    spec: BenchmarkModelSpec,
    validation_split: SplitResult,
    targets: _TargetSet,
    candidate_plan: CandidatePlan | None,
    knowledge: LoadedKnowledgeLinks | None = None,
) -> _SelectedModel:
    tuning = config.tuning
    if tuning is None:
        raise ConfigurationError("internal tuning configuration is missing")
    candidates = _parameter_candidates(spec)
    aggregate_scores: list[float] = []
    trial_rows: list[TuningTrial] = []
    trial_indices_by_candidate: list[tuple[int, ...]] = []
    candidate_parameters: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        seeds: tuple[int | None, ...] = (
            tuple(tuning.implicit_mf_seeds)
            if spec.name in {"implicit_mf", "confidence_als"}
            else (None,)
        )
        scores: list[float] = []
        indices: list[int] = []
        canonical_parameters: dict[str, Any] | None = None
        for trial_seed in seeds:
            trial_parameters = dict(candidate)
            if trial_seed is not None:
                trial_parameters["seed"] = trial_seed
            evaluated = _evaluate_model(
                config,
                spec,
                validation_split,
                targets,
                parameters=trial_parameters,
                experiment_seed=config.seed if trial_seed is None else trial_seed,
                candidate_plan=candidate_plan,
                knowledge=knowledge,
            )
            value = _selection_value(evaluated.metrics, tuning.selection_metric)
            if canonical_parameters is None:
                canonical_parameters = dict(evaluated.parameters)
                canonical_parameters.pop("seed", None)
            trial_index = len(trial_rows)
            indices.append(trial_index)
            scores.append(value)
            trial_rows.append(
                TuningTrial(
                    candidate_index=candidate_index,
                    seed=trial_seed,
                    parameters=evaluated.parameters,
                    selection_value=value,
                    timing=evaluated.timing,
                    metrics=evaluated.metrics,
                )
            )
        if canonical_parameters is None:
            raise ConfigurationError("a tuning candidate produced no validation trials")
        candidate_parameters.append(canonical_parameters)
        aggregate_scores.append(math.fsum(scores) / len(scores))
        trial_indices_by_candidate.append(tuple(indices))
    if tuning.direction == "maximize":
        selected_index = max(
            range(len(candidates)),
            key=lambda index: (aggregate_scores[index], -index),
        )
    else:
        selected_index = min(
            range(len(candidates)),
            key=lambda index: (aggregate_scores[index], index),
        )
    candidate_rows = tuple(
        TuningCandidateResult(
            candidate_index=index,
            parameters=candidate_parameters[index],
            mean_selection_value=aggregate_scores[index],
            trial_indices=trial_indices_by_candidate[index],
            selected=index == selected_index,
        )
        for index in range(len(candidates))
    )
    return _SelectedModel(
        spec=spec,
        parameters=dict(candidates[selected_index]),
        candidates=candidate_rows,
        trials=tuple(trial_rows),
        selected_candidate_index=selected_index,
        selected_validation_score=aggregate_scores[selected_index],
    )


def _semantic_config_fingerprint(
    config: BenchmarkConfig, data_sha256: str, knowledge_sha256: str | None = None
) -> str:
    payload = config.to_dict()
    data = payload["data"]
    if not isinstance(data, dict):
        raise ConfigurationError("benchmark data configuration is malformed")
    data = dict(data)
    data.pop("path")
    data["interactions_sha256"] = data_sha256
    if knowledge_sha256 is not None:
        data.pop("knowledge_path")
        data["knowledge_state_sha256"] = knowledge_sha256
    payload["data"] = data
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_fingerprint(split: SplitResult) -> str:
    payload = {
        "train": interaction_fingerprint(split.train),
        "test": interaction_fingerprint(split.test),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _three_way_split_fingerprint(
    training: InteractionDataset,
    validation: InteractionDataset,
    test: InteractionDataset,
) -> str:
    payload = {
        "training": interaction_fingerprint(training),
        "validation": interaction_fingerprint(validation),
        "test": interaction_fingerprint(test),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _point_metrics(metrics: MetricReport) -> dict[str, float]:
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "ndcg": metrics.ndcg,
        "mrr": metrics.mrr,
        "coverage": metrics.coverage,
        "novelty": metrics.novelty,
    }


def _bootstrap(
    config: BenchmarkConfig,
    evaluated: Sequence[_EvaluatedModel],
    catalog_size: int,
) -> tuple[tuple[BenchmarkModelResult, ...], tuple[BenchmarkComparison, ...]]:
    sample_count = config.evaluation.bootstrap_samples
    user_count = len(evaluated[0].rows)
    generator = random.Random(config.seed)
    distributions: dict[str, dict[str, list[float]]] = {
        model.spec.label: {name: [] for name in METRIC_NAMES} for model in evaluated
    }
    for _ in range(sample_count):
        indices = tuple(generator.randrange(user_count) for _ in range(user_count))
        for model in evaluated:
            values = _aggregate(model.rows, indices, catalog_size).to_dict()
            for name in METRIC_NAMES:
                distributions[model.spec.label][name].append(values[name])
    model_results: list[BenchmarkModelResult] = []
    for model in evaluated:
        points = _point_metrics(model.metrics)
        intervals = {
            name: interval_from_draws(
                points[name],
                distributions[model.spec.label][name],
                confidence=config.evaluation.confidence,
            )
            for name in METRIC_NAMES
        }
        model_results.append(
            BenchmarkModelResult(
                label=model.spec.label,
                model_type=model.spec.name,
                parameters=model.parameters,
                timing=model.timing,
                metrics=model.metrics,
                confidence_intervals=intervals,
            )
        )
    comparisons: list[BenchmarkComparison] = []
    for left, right in itertools.combinations(evaluated, 2):
        left_points = _point_metrics(left.metrics)
        right_points = _point_metrics(right.metrics)
        comparisons.append(
            BenchmarkComparison(
                left_label=left.spec.label,
                right_label=right.spec.label,
                metrics={
                    name: paired_comparison_from_draws(
                        left_points[name],
                        right_points[name],
                        distributions[left.spec.label][name],
                        distributions[right.spec.label][name],
                        confidence=config.evaluation.confidence,
                    )
                    for name in METRIC_NAMES
                },
            )
        )
    return tuple(model_results), tuple(comparisons)


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run every configured model with an optional validation-only grid search."""

    if not isinstance(config, BenchmarkConfig):
        raise ConfigurationError("config must be a BenchmarkConfig")
    try:
        config = benchmark_config_from_dict(config.to_dict())
    except ConfigurationError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"malformed BenchmarkConfig: {exc}") from exc
    loaded = load_dataset(
        config.data.path,
        format=config.data.format,
        minimum_rating=config.data.minimum_rating,
    )
    knowledge = (
        load_recbole_knowledge(config.data.knowledge_path)
        if config.data.knowledge_path is not None
        else None
    )
    if knowledge is not None and config.data.format == "recbole-inter":
        for reference in knowledge.references:
            if reference.kind == "recbole-inter" and (
                reference.source_sha256 != loaded.summary.source_sha256
                or reference.normalized_sha256 != loaded.summary.interactions_sha256
            ):
                raise ConfigurationError("knowledge artifact references a different .inter source")
    split = _split(config, loaded.dataset)
    if not split.train or not split.test:
        raise ConfigurationError("split must produce non-empty train and test partitions")
    tuning_result: BenchmarkTuningResult | None = None
    selected_models: tuple[_SelectedModel, ...] | None = None
    validation_split: SplitResult | None = None
    validation_targets: _TargetSet | None = None
    if config.tuning is not None:
        # The search receives only the outer training partition. The outer test
        # partition is not converted to targets or passed to an evaluator until
        # every model has selected its hyperparameters.
        validation_split = _validation_split(config.tuning, split.train, config.seed)
        if not validation_split.train or not validation_split.test:
            raise ConfigurationError(
                "validation split must produce non-empty training and validation partitions"
            )
        validation_targets = _targets(validation_split)
        validation_candidate_plan = _candidate_plan(config, validation_split, validation_targets)
        selected_models = tuple(
            _tune_model(
                config,
                spec,
                validation_split,
                validation_targets,
                validation_candidate_plan,
                knowledge=knowledge,
            )
            for spec in config.models
        )
    targets = _targets(split)
    candidate_plan = _candidate_plan(config, split, targets)
    if selected_models is None:
        evaluated = tuple(
            _evaluate_model(
                config, spec, split, targets, candidate_plan=candidate_plan, knowledge=knowledge
            )
            for spec in config.models
        )
    else:
        evaluated = tuple(
            _evaluate_model(
                config,
                selected.spec,
                split,
                targets,
                parameters=selected.parameters,
                experiment_seed=config.seed,
                candidate_plan=candidate_plan,
                knowledge=knowledge,
            )
            for selected in selected_models
        )
        if config.tuning is None or validation_split is None or validation_targets is None:
            raise ConfigurationError("internal tuning state is incomplete")
        tuning_models = tuple(
            ModelTuningResult(
                label=selected.spec.label,
                model_type=selected.spec.name,
                search_space={name: tuple(values) for name, values in selected.spec.grid.items()},
                candidates=selected.candidates,
                trials=selected.trials,
                selected_candidate_index=selected.selected_candidate_index,
                selected_validation_score=selected.selected_validation_score,
                final_parameters=evaluated[index].parameters,
                final_seed=(
                    config.seed if selected.spec.name in {"implicit_mf", "confidence_als"} else None
                ),
            )
            for index, selected in enumerate(selected_models)
        )
        tuning_result = BenchmarkTuningResult(
            selection_metric=config.tuning.selection_metric,
            direction=config.tuning.direction,
            validation_method=config.tuning.validation_split.method,
            validation_ratio=config.tuning.validation_split.validation_ratio,
            implicit_mf_seeds=config.tuning.implicit_mf_seeds,
            final_seed=config.seed,
            development_interactions=len(split.train),
            training_interactions=len(validation_split.train),
            validation_interactions=len(validation_split.test),
            evaluated_validation_interactions=validation_targets.evaluated_interactions,
            cold_start_validation_interactions=validation_targets.cold_start_interactions,
            evaluated_validation_users=len(validation_targets.users),
            development_fingerprint=interaction_fingerprint(split.train),
            training_fingerprint=interaction_fingerprint(validation_split.train),
            validation_fingerprint=interaction_fingerprint(validation_split.test),
            test_fingerprint=interaction_fingerprint(split.test),
            three_way_split_fingerprint=_three_way_split_fingerprint(
                validation_split.train,
                validation_split.test,
                split.test,
            ),
            models=tuning_models,
            validation_candidate_fingerprint=(
                validation_candidate_plan.fingerprint
                if validation_candidate_plan is not None
                else None
            ),
        )
    model_results, comparisons = _bootstrap(config, evaluated, len(split.train.item_ids))
    return BenchmarkResult(
        seed=config.seed,
        config_fingerprint=_semantic_config_fingerprint(
            config,
            loaded.summary.interactions_sha256,
            cast(str, knowledge.to_state()["state_sha256"]) if knowledge is not None else None,
        ),
        split_fingerprint=_split_fingerprint(split),
        dataset=loaded.summary,
        split_method=config.split.method,
        test_ratio=config.split.test_ratio,
        train_interactions=len(split.train),
        test_interactions=len(split.test),
        evaluated_test_interactions=targets.evaluated_interactions,
        cold_start_test_interactions=targets.cold_start_interactions,
        evaluated_users=len(targets.users),
        k=config.evaluation.k,
        exclude_seen=config.evaluation.exclude_seen,
        bootstrap_samples=config.evaluation.bootstrap_samples,
        confidence=config.evaluation.confidence,
        models=model_results,
        comparisons=comparisons,
        tuning=tuning_result,
        candidate_plan=candidate_plan,
        source_path=config.data.path,
        knowledge_path=config.data.knowledge_path,
        knowledge=(
            {
                "fingerprint_sha256": knowledge.fingerprint,
                "artifact_state_sha256": knowledge.to_state()["state_sha256"],
                "kg_source_sha256": knowledge.sources[0].sha256,
                "link_source_sha256": knowledge.sources[1].sha256,
                "linked_training_items": len(
                    set(split.train.item_ids) & {row.item_id for row in knowledge.links}
                ),
            }
            if knowledge is not None
            else None
        ),
    )
