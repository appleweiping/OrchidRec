"""Self-contained end-to-end demonstration data and runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from orchidrec.config import load_config
from orchidrec.data import Interaction, InteractionDataset
from orchidrec.errors import SerializationError
from orchidrec.experiment import ExperimentResult, run_experiment


@dataclass(frozen=True, slots=True)
class DemoArtifacts:
    """Paths and result created by :func:`run_demo`."""

    directory: Path
    data_path: Path
    config_path: Path
    model_path: Path
    report_path: Path
    result: ExperimentResult


def demo_dataset() -> InteractionDataset:
    """Return a small collaborative dataset with reproducible timestamps."""

    rows = [
        ("u1", "a"), ("u1", "b"), ("u1", "c"),
        ("u2", "a"), ("u2", "c"), ("u2", "d"),
        ("u3", "b"), ("u3", "c"), ("u3", "e"),
        ("u4", "a"), ("u4", "d"), ("u4", "e"),
        ("u5", "c"), ("u5", "e"), ("u5", "f"),
        ("u6", "d"), ("u6", "f"), ("u6", "a"),
    ]
    return InteractionDataset(
        Interaction(user_id=user_id, item_id=item_id, timestamp=float(index))
        for index, (user_id, item_id) in enumerate(rows, start=1)
    )


def run_demo(output_dir: str | Path) -> DemoArtifacts:
    """Create example files, execute the experiment, and persist its outputs."""

    directory = Path(output_dir).resolve()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SerializationError(f"could not create demo directory {directory}: {exc}") from exc
    data_path = directory / "interactions.json"
    config_path = directory / "experiment.json"
    model_path = directory / "model.json"
    report_path = directory / "report.json"
    demo_dataset().save_json(data_path)
    config_payload = {
        "seed": 2026,
        "data": {"path": "interactions.json"},
        "split": {"method": "leave_one_out", "test_ratio": 0.2},
        "model": {"name": "item_knn", "params": {"neighbors": 4, "shrinkage": 1.0}},
        "evaluation": {"k": 3, "exclude_seen": True},
        "output": {"model_path": "model.json", "report_path": "report.json"},
    }
    try:
        config_path.write_text(
            json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise SerializationError(f"could not write demo configuration {config_path}: {exc}") from exc
    result = run_experiment(load_config(config_path))
    return DemoArtifacts(
        directory=directory,
        data_path=data_path,
        config_path=config_path,
        model_path=model_path,
        report_path=report_path,
        result=result,
    )
