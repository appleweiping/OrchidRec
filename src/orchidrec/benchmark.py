"""Shared-split, multi-model recommendation benchmarks."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orchidrec.benchmark_config import (
    BenchmarkConfig,
    BenchmarkModelSpec,
    benchmark_config_from_dict,
)
from orchidrec.data import EntityId, InteractionDataset, stable_id_key
from orchidrec.datasets import DatasetSummary, interaction_fingerprint, load_dataset
from orchidrec.errors import ConfigurationError
from orchidrec.experiment import build_model
from orchidrec.metrics import MetricReport, evaluate_ranking
from orchidrec.split import SplitResult, leave_one_out, random_split, temporal_split
from orchidrec.statistics import (
    BootstrapInterval,
    PairedBootstrapResult,
    interval_from_draws,
    paired_comparison_from_draws,
)

BENCHMARK_FORMAT = "orchidrec.benchmark"
BENCHMARK_SCHEMA_VERSION = 1
METRIC_NAMES = ("precision", "recall", "ndcg", "mrr", "coverage", "novelty")


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

    def to_dict(self) -> dict[str, object]:
        return {
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


def _aggregate(rows: Sequence[_RankingRow], indices: Sequence[int], catalog_size: int) -> _MetricVector:
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
) -> _EvaluatedModel:
    model = build_model(spec.name, spec.params, experiment_seed=config.seed)
    fit_started = time.perf_counter_ns()
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


def _semantic_config_fingerprint(config: BenchmarkConfig, data_sha256: str) -> str:
    payload = config.to_dict()
    data = payload["data"]
    if not isinstance(data, dict):
        raise ConfigurationError("benchmark data configuration is malformed")
    data = dict(data)
    data.pop("path")
    data["interactions_sha256"] = data_sha256
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
    """Run every configured model on one shared split and bootstrap plan."""

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
    split = _split(config, loaded.dataset)
    if not split.train or not split.test:
        raise ConfigurationError("split must produce non-empty train and test partitions")
    targets = _targets(split)
    evaluated = tuple(_evaluate_model(config, spec, split, targets) for spec in config.models)
    model_results, comparisons = _bootstrap(config, evaluated, len(split.train.item_ids))
    return BenchmarkResult(
        seed=config.seed,
        config_fingerprint=_semantic_config_fingerprint(
            config, loaded.summary.interactions_sha256
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
    )
