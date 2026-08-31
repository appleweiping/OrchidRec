"""OrchidRec: reproducible recommendation experiments and benchmarks."""

from orchidrec.benchmark import BenchmarkResult, run_benchmark
from orchidrec.benchmark_config import (
    BenchmarkConfig,
    BenchmarkDataConfig,
    BenchmarkEvaluationConfig,
    BenchmarkModelSpec,
    BenchmarkSplitConfig,
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
from orchidrec.models import ImplicitMF, ItemKNN, Popularity, Recommendation
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

__all__ = [
    "BenchmarkConfig",
    "BenchmarkDataConfig",
    "BenchmarkEvaluationConfig",
    "BenchmarkModelSpec",
    "BenchmarkReportPaths",
    "BenchmarkResult",
    "BenchmarkSplitConfig",
    "BootstrapInterval",
    "DatasetFormat",
    "DatasetSummary",
    "EntityId",
    "ExperimentConfig",
    "ExperimentResult",
    "ImplicitMF",
    "Interaction",
    "InteractionDataset",
    "ItemKNN",
    "LoadedDataset",
    "MetricReport",
    "PairedBootstrapResult",
    "Popularity",
    "Recommendation",
    "SplitResult",
    "StableIdMap",
    "benchmark_config_from_dict",
    "bootstrap_mean",
    "evaluate_ranking",
    "interaction_fingerprint",
    "interval_from_draws",
    "leave_one_out",
    "load_benchmark_config",
    "load_config",
    "load_dataset",
    "load_movielens",
    "paired_bootstrap_mean",
    "paired_comparison_from_draws",
    "random_split",
    "run_benchmark",
    "run_experiment",
    "save_benchmark_config",
    "save_benchmark_reports",
    "temporal_split",
]

__version__ = "0.2.0"
