"""Portable JSON, tidy CSV, and standalone HTML benchmark reports."""

from __future__ import annotations

import contextlib
import csv
import html
import io
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from orchidrec._files import require_distinct_paths
from orchidrec.benchmark import METRIC_NAMES, BenchmarkResult
from orchidrec.errors import SerializationError


@dataclass(frozen=True, slots=True)
class BenchmarkReportPaths:
    """Paths written by :func:`save_benchmark_reports`."""

    json_path: Path
    csv_path: Path
    html_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "json": str(self.json_path),
            "csv": str(self.csv_path),
            "html": str(self.html_path),
        }


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(path)
    except OSError as exc:
        if temporary_name is not None:
            with contextlib.suppress(OSError):
                Path(temporary_name).unlink(missing_ok=True)
        raise SerializationError(f"could not write benchmark report {path}: {exc}") from exc


def benchmark_csv(result: BenchmarkResult) -> str:
    """Return a tidy CSV containing estimates, intervals, and comparisons."""

    if not isinstance(result, BenchmarkResult):
        raise SerializationError("result must be a BenchmarkResult")
    output = io.StringIO(newline="")
    base_fields = (
        "row_type",
        "left_or_model",
        "right_model",
        "metric",
        "estimate",
        "lower",
        "upper",
        "confidence",
        "bootstrap_samples",
        "probability_right_better",
        "two_sided_p_value",
        "fit_seconds",
        "recommend_seconds",
    )
    tuning_fields = (
        "candidate_index",
        "trial_seed",
        "parameters_json",
        "search_space_json",
        "selected",
        "direction",
    )
    fields: tuple[str, ...] = (
        base_fields + tuning_fields if result.tuning is not None else base_fields
    )
    if result.candidate_plan is not None:
        fields += (
            "evaluation_mode",
            "sampling_strategy",
            "requested_negatives",
            "candidate_partition",
            "candidate_sha256",
        )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()

    def write_row(row: dict[str, object], *, partition: str = "test") -> None:
        if result.candidate_plan is not None:
            if partition == "validation":
                candidate_sha256 = (
                    result.tuning.validation_candidate_fingerprint
                    if result.tuning is not None
                    else None
                )
                if candidate_sha256 is None:
                    raise SerializationError("sampled tuning report lacks validation candidates")
            else:
                candidate_sha256 = result.candidate_plan.fingerprint
            row = {
                **row,
                "evaluation_mode": "sampled",
                "sampling_strategy": result.candidate_plan.sampling.strategy,
                "requested_negatives": result.candidate_plan.sampling.negatives,
                "candidate_partition": partition,
                "candidate_sha256": candidate_sha256,
            }
        writer.writerow(row)

    for model in result.models:
        for metric in METRIC_NAMES:
            interval = model.confidence_intervals[metric]
            write_row(
                {
                    "row_type": "model",
                    "left_or_model": _spreadsheet_text(model.label),
                    "metric": metric,
                    "estimate": interval.estimate,
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "confidence": interval.confidence,
                    "bootstrap_samples": interval.samples,
                    "fit_seconds": model.timing.fit_seconds,
                    "recommend_seconds": model.timing.recommend_seconds,
                }
            )
    for comparison in result.comparisons:
        for metric in METRIC_NAMES:
            statistic = comparison.metrics[metric]
            interval = statistic.difference
            write_row(
                {
                    "row_type": "comparison_right_minus_left",
                    "left_or_model": _spreadsheet_text(comparison.left_label),
                    "right_model": _spreadsheet_text(comparison.right_label),
                    "metric": metric,
                    "estimate": interval.estimate,
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "confidence": interval.confidence,
                    "bootstrap_samples": interval.samples,
                    "probability_right_better": statistic.probability_right_better,
                    "two_sided_p_value": statistic.two_sided_p_value,
                }
            )
    if result.tuning is not None:
        for tuning_model in result.tuning.models:
            search_space = json.dumps(
                {name: list(values) for name, values in tuning_model.search_space.items()},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for candidate in tuning_model.candidates:
                write_row(
                    {
                        "row_type": "tuning_candidate",
                        "left_or_model": _spreadsheet_text(tuning_model.label),
                        "metric": result.tuning.selection_metric,
                        "estimate": candidate.mean_selection_value,
                        "candidate_index": candidate.candidate_index,
                        "parameters_json": json.dumps(
                            candidate.parameters,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                        "search_space_json": search_space,
                        "selected": candidate.selected,
                        "direction": result.tuning.direction,
                    },
                    partition="validation",
                )
            for trial in tuning_model.trials:
                write_row(
                    {
                        "row_type": "tuning_trial",
                        "left_or_model": _spreadsheet_text(tuning_model.label),
                        "metric": result.tuning.selection_metric,
                        "estimate": trial.selection_value,
                        "fit_seconds": trial.timing.fit_seconds,
                        "recommend_seconds": trial.timing.recommend_seconds,
                        "candidate_index": trial.candidate_index,
                        "trial_seed": trial.seed,
                        "parameters_json": json.dumps(
                            trial.parameters,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                        "direction": result.tuning.direction,
                    },
                    partition="validation",
                )
    return output.getvalue()


def _spreadsheet_text(value: str) -> str:
    """Prevent user-defined labels from becoming spreadsheet formulas."""

    return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def _number(value: float) -> str:
    return f"{value:.6f}"


def benchmark_html(result: BenchmarkResult) -> str:
    """Return a self-contained, offline-viewable benchmark dashboard."""

    if not isinstance(result, BenchmarkResult):
        raise SerializationError("result must be a BenchmarkResult")
    model_rows: list[str] = []
    for model in result.models:
        cells = [
            f"<td><strong>{html.escape(model.label)}</strong><br><code>{html.escape(model.model_type)}</code></td>"
        ]
        for metric in METRIC_NAMES:
            interval = model.confidence_intervals[metric]
            cells.append(
                "<td>"
                f"{_number(interval.estimate)}"
                f"<small>[{_number(interval.lower)}, {_number(interval.upper)}]</small>"
                "</td>"
            )
        cells.append(
            "<td>"
            f"{_number(model.timing.fit_seconds)} / {_number(model.timing.recommend_seconds)}"
            "</td>"
        )
        model_rows.append("<tr>" + "".join(cells) + "</tr>")
    comparison_rows: list[str] = []
    for comparison in result.comparisons:
        pair = f"{comparison.right_label} - {comparison.left_label}"
        for metric in METRIC_NAMES:
            statistic = comparison.metrics[metric]
            interval = statistic.difference
            comparison_rows.append(
                "<tr>"
                f"<td>{html.escape(pair)}</td>"
                f"<td>{html.escape(metric)}</td>"
                f"<td>{_number(interval.estimate)}</td>"
                f"<td>[{_number(interval.lower)}, {_number(interval.upper)}]</td>"
                f"<td>{_number(statistic.probability_right_better)}</td>"
                f"<td>{_number(statistic.two_sided_p_value)}</td>"
                "</tr>"
            )
    headers = "".join(f"<th>{html.escape(name.title())}</th>" for name in METRIC_NAMES)
    dataset = result.dataset
    confidence_percent = result.confidence * 100.0
    sampling_html = "<p><strong>Evaluation mode:</strong> full-sort over the training catalog.</p>"
    if result.candidate_plan is not None:
        sampling_html = (
            "<p><strong>Evaluation mode:</strong> sampled "
            f"({html.escape(result.candidate_plan.sampling.strategy)}, "
            f"{result.candidate_plan.sampling.negatives} requested negatives per user). "
            "Scores are conditional on these candidates and are not directly comparable "
            "with full-sort scores.</p>"
            f"<p><span class='label'>Outer test candidate SHA-256</span><code>{result.candidate_plan.fingerprint}</code></p>"
        )
    tuning_html = ""
    if result.tuning is not None:
        validation_candidate_html = ""
        if result.candidate_plan is not None:
            validation_candidate = result.tuning.validation_candidate_fingerprint
            if validation_candidate is None:
                raise SerializationError("sampled tuning report lacks validation candidates")
            validation_candidate_html = (
                "<p><span class='label'>Inner validation candidate SHA-256</span>"
                f"<code>{validation_candidate}</code></p>"
            )
        tuning_rows: list[str] = []
        tuning_trial_rows: list[str] = []
        for tuning_model in result.tuning.models:
            for candidate in tuning_model.candidates:
                seeds = ", ".join(
                    "deterministic"
                    if tuning_model.trials[index].seed is None
                    else str(tuning_model.trials[index].seed)
                    for index in candidate.trial_indices
                )
                parameters = json.dumps(
                    candidate.parameters,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                tuning_rows.append(
                    "<tr>"
                    f"<td>{html.escape(tuning_model.label)}</td>"
                    f"<td>{candidate.candidate_index}</td>"
                    f"<td><code>{html.escape(parameters)}</code></td>"
                    f"<td>{html.escape(seeds)}</td>"
                    f"<td>{_number(candidate.mean_selection_value)}</td>"
                    f"<td>{'yes' if candidate.selected else 'no'}</td>"
                    "</tr>"
                )
            for trial_index, trial in enumerate(tuning_model.trials):
                trial_parameters = json.dumps(
                    trial.parameters,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                tuning_trial_rows.append(
                    "<tr>"
                    f"<td>{html.escape(tuning_model.label)}</td>"
                    f"<td>{trial_index}</td>"
                    f"<td>{trial.candidate_index}</td>"
                    f"<td>{'deterministic' if trial.seed is None else trial.seed}</td>"
                    f"<td><code>{html.escape(trial_parameters)}</code></td>"
                    f"<td>{_number(trial.selection_value)}</td>"
                    f"<td>{_number(trial.timing.fit_seconds)} / "
                    f"{_number(trial.timing.recommend_seconds)}</td>"
                    "</tr>"
                )
        tuning = result.tuning
        tuning_html = f"""
<h2>Validation-only model selection</h2>
{validation_candidate_html}
<p>Outer test data was evaluated only after all searches completed. Candidates use a
<strong>{html.escape(tuning.validation_method)}</strong> validation split and select
<strong>{html.escape(tuning.selection_metric)}</strong> by
<strong>{html.escape(tuning.direction)}</strong>. Equal scores keep the first canonical candidate.</p>
<div class="scroll"><table>
<thead><tr><th>Model</th><th>Candidate</th><th>Effective parameters</th><th>Validation seeds</th><th>Mean score</th><th>Selected</th></tr></thead>
<tbody>{"".join(tuning_rows)}</tbody>
</table></div>
<h3>All validation trials</h3>
<div class="scroll"><table>
<thead><tr><th>Model</th><th>Trial</th><th>Candidate</th><th>Seed</th><th>Effective parameters</th><th>Selection value</th><th>Fit / recommend seconds</th></tr></thead>
<tbody>{"".join(tuning_trial_rows)}</tbody>
</table></div>
<p><small>Development / training / validation interactions: {tuning.development_interactions:,} /
{tuning.training_interactions:,} / {tuning.validation_interactions:,}. ImplicitMF validation seeds:
{html.escape(", ".join(str(seed) for seed in tuning.implicit_mf_seeds))}; final fit seed:
{tuning.final_seed}.</small></p>
<p><span class="label">Training SHA-256</span><code>{tuning.training_fingerprint}</code></p>
<p><span class="label">Validation SHA-256</span><code>{tuning.validation_fingerprint}</code></p>
<p><span class="label">Test SHA-256</span><code>{tuning.test_fingerprint}</code></p>
<p><span class="label">Three-way split SHA-256</span><code>{tuning.three_way_split_fingerprint}</code></p>
"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OrchidRec benchmark</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ max-width: 1200px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }}
h1, h2 {{ letter-spacing: -.02em; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: .75rem; }}
.card {{ border: 1px solid #8886; border-radius: .6rem; padding: .8rem; }}
.label, small {{ display: block; color: #777; font-size: .78rem; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ border-bottom: 1px solid #8885; padding: .55rem; text-align: right; vertical-align: top; }}
th:first-child, td:first-child, td:nth-child(2) {{ text-align: left; }}
.scroll {{ overflow-x: auto; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>OrchidRec benchmark</h1>
<p>Shared <strong>{html.escape(result.split_method)}</strong> split, K={result.k},
{result.bootstrap_samples} paired bootstrap samples, {confidence_percent:.1f}% intervals.</p>
{sampling_html}
<div class="cards">
  <div class="card"><span class="label">Dataset</span>{html.escape(dataset.format)} / {html.escape(dataset.source_name)}</div>
  <div class="card"><span class="label">Retained events</span>{dataset.retained_interactions:,}</div>
  <div class="card"><span class="label">Users / items</span>{dataset.users:,} / {dataset.items:,}</div>
  <div class="card"><span class="label">Train / test</span>{result.train_interactions:,} / {result.test_interactions:,}</div>
</div>
<h2>Model results</h2>
<div class="scroll"><table>
<thead><tr><th>Model</th>{headers}<th>Fit / recommend seconds</th></tr></thead>
<tbody>{"".join(model_rows)}</tbody>
</table></div>
<p><small>Intervals resample evaluated users with replacement. Timing is observational and is not a deterministic fingerprint.</small></p>
<h2>Paired comparisons</h2>
<div class="scroll"><table>
<thead><tr><th>Right - left</th><th>Metric</th><th>Difference</th><th>Interval</th><th>P(right better)</th><th>Two-sided p</th></tr></thead>
<tbody>{"".join(comparison_rows)}</tbody>
</table></div>
{tuning_html}
<h2>Reproducibility</h2>
<p><span class="label">Normalized interactions SHA-256</span><code>{dataset.interactions_sha256}</code></p>
<p><span class="label">Source bytes SHA-256</span><code>{dataset.source_sha256}</code></p>
<p><span class="label">Configuration SHA-256</span><code>{result.config_fingerprint}</code></p>
<p><span class="label">Split SHA-256</span><code>{result.split_fingerprint}</code></p>
</body>
</html>
"""


def save_benchmark_reports(
    result: BenchmarkResult,
    output_dir: str | Path,
    *,
    protected_paths: Mapping[str, Path] | None = None,
) -> BenchmarkReportPaths:
    """Atomically write JSON, CSV, and standalone HTML artifacts."""

    if not isinstance(result, BenchmarkResult):
        raise SerializationError("result must be a BenchmarkResult")
    destination = Path(output_dir)
    paths = BenchmarkReportPaths(
        json_path=destination / "benchmark.json",
        csv_path=destination / "benchmark.csv",
        html_path=destination / "benchmark.html",
    )
    inputs = dict(protected_paths or {})
    if result.source_path is not None:
        inputs["data.path"] = result.source_path
    require_distinct_paths(
        {
            **inputs,
            "benchmark.json": paths.json_path,
            "benchmark.csv": paths.csv_path,
            "benchmark.html": paths.html_path,
        }
    )
    try:
        json_report = (
            json.dumps(
                result.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"benchmark result is not strict JSON: {exc}") from exc
    _atomic_write_text(paths.json_path, json_report)
    _atomic_write_text(paths.csv_path, benchmark_csv(result))
    _atomic_write_text(paths.html_path, benchmark_html(result))
    return paths
