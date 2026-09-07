"""OrchidRec: reproducible recommendation experiments and benchmarks."""

from orchidrec.benchmark import (
    BenchmarkResult,
    BenchmarkTuningResult,
    ModelTuningResult,
    TuningCandidateResult,
    TuningTrial,
    run_benchmark,
)
from orchidrec.benchmark_config import (
    BenchmarkConfig,
    BenchmarkDataConfig,
    BenchmarkEvaluationConfig,
    BenchmarkModelSpec,
    BenchmarkSplitConfig,
    BenchmarkTuningConfig,
    BenchmarkValidationSplitConfig,
    benchmark_config_from_dict,
    load_benchmark_config,
    save_benchmark_config,
)
from orchidrec.config import ExperimentConfig, load_config
from orchidrec.data import EntityId, Interaction, InteractionDataset, StableIdMap
from orchidrec.datasets import (
    DatasetFormat,
    DatasetSummary,
    LoadedDataset,
    interaction_fingerprint,
    load_dataset,
    load_movielens,
)
from orchidrec.experiment import ExperimentResult, run_experiment
from orchidrec.metrics import MetricReport, evaluate_ranking
from orchidrec.models import (
    ImplicitMF,
    ItemKNN,
    Popularity,
    Recommendation,
    SequentialMarkov,
    UserKNN,
)
from orchidrec.propensity import (
    DEFAULT_EXPONENT,
    DEFAULT_MINIMUM_PROPENSITY,
    ExposureModel,
    popularity_exposure,
    uniform_exposure,
)
from orchidrec.reporting import BenchmarkReportPaths, save_benchmark_reports
from orchidrec.split import SplitResult, leave_one_out, random_split, temporal_split
from orchidrec.statistics import (
    BootstrapInterval,
    PairedBootstrapResult,
    bootstrap_mean,
    interval_from_draws,
    paired_bootstrap_mean,
    paired_comparison_from_draws,
)
from orchidrec.unbiased import (
    UnbiasedMetricReport,
    evaluate_unbiased_ranking,
    ips_ndcg_at_k,
    ips_recall_at_k,
)

__all__ = [
    "DEFAULT_EXPONENT",
    "DEFAULT_MINIMUM_PROPENSITY",
    "BenchmarkConfig",
    "BenchmarkDataConfig",
    "BenchmarkEvaluationConfig",
    "BenchmarkModelSpec",
    "BenchmarkReportPaths",
    "BenchmarkResult",
    "BenchmarkSplitConfig",
    "BenchmarkTuningConfig",
    "BenchmarkTuningResult",
    "BenchmarkValidationSplitConfig",
    "BootstrapInterval",
    "DatasetFormat",
    "DatasetSummary",
    "EntityId",
    "ExperimentConfig",
    "ExperimentResult",
    "ExposureModel",
    "ImplicitMF",
    "Interaction",
    "InteractionDataset",
    "ItemKNN",
    "LoadedDataset",
    "MetricReport",
    "ModelTuningResult",
    "PairedBootstrapResult",
    "Popularity",
    "Recommendation",
    "SequentialMarkov",
    "SplitResult",
    "StableIdMap",
    "TuningCandidateResult",
    "TuningTrial",
    "UnbiasedMetricReport",
    "UserKNN",
    "benchmark_config_from_dict",
    "bootstrap_mean",
    "evaluate_ranking",
    "evaluate_unbiased_ranking",
    "interaction_fingerprint",
    "interval_from_draws",
    "ips_ndcg_at_k",
    "ips_recall_at_k",
    "leave_one_out",
    "load_benchmark_config",
    "load_config",
    "load_dataset",
    "load_movielens",
    "paired_bootstrap_mean",
    "paired_comparison_from_draws",
    "popularity_exposure",
    "random_split",
    "run_benchmark",
    "run_experiment",
    "save_benchmark_config",
    "save_benchmark_reports",
    "temporal_split",
    "uniform_exposure",
]

__version__ = "0.4.0"
