"""Portable JSON, tidy CSV, and standalone HTML benchmark reports."""

from __future__ import annotations

import contextlib
import csv
import html
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
    fields = (
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
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for model in result.models:
        for metric in METRIC_NAMES:
            interval = model.confidence_intervals[metric]
            writer.writerow(
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
            writer.writerow(
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
<div class="cards">
  <div class="card"><span class="label">Dataset</span>{html.escape(dataset.format)} / {html.escape(dataset.source_name)}</div>
  <div class="card"><span class="label">Retained events</span>{dataset.retained_interactions:,}</div>
  <div class="card"><span class="label">Users / items</span>{dataset.users:,} / {dataset.items:,}</div>
  <div class="card"><span class="label">Train / test</span>{result.train_interactions:,} / {result.test_interactions:,}</div>
</div>
<h2>Model results</h2>
<div class="scroll"><table>
<thead><tr><th>Model</th>{headers}<th>Fit / recommend seconds</th></tr></thead>
<tbody>{''.join(model_rows)}</tbody>
</table></div>
<p><small>Intervals resample evaluated users with replacement. Timing is observational and is not a deterministic fingerprint.</small></p>
<h2>Paired comparisons</h2>
<div class="scroll"><table>
<thead><tr><th>Right - left</th><th>Metric</th><th>Difference</th><th>Interval</th><th>P(right better)</th><th>Two-sided p</th></tr></thead>
<tbody>{''.join(comparison_rows)}</tbody>
</table></div>
<h2>Reproducibility</h2>
<p><span class="label">Normalized interactions SHA-256</span><code>{dataset.interactions_sha256}</code></p>
<p><span class="label">Source bytes SHA-256</span><code>{dataset.source_sha256}</code></p>
<p><span class="label">Configuration SHA-256</span><code>{result.config_fingerprint}</code></p>
<p><span class="label">Split SHA-256</span><code>{result.split_fingerprint}</code></p>
</body>
</html>
"""


def save_benchmark_reports(
    result: BenchmarkResult, output_dir: str | Path
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
