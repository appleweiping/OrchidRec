"""Safe, portable JSON serialization for OrchidRec models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from orchidrec._json import strict_json_loads
from orchidrec.errors import SerializationError
from orchidrec.models.base import BaseRecommender


def model_from_state(state: Mapping[str, Any]) -> BaseRecommender:
    """Construct the registered model declared by a state envelope."""

    from orchidrec.models.implicit_mf import ImplicitMF
    from orchidrec.models.item_knn import ItemKNN
    from orchidrec.models.popularity import Popularity

    if not isinstance(state, Mapping):
        raise SerializationError("model state must be a JSON object")
    model_type = state.get("model_type")
    registry: dict[str, type[BaseRecommender]] = {
        Popularity.model_type: Popularity,
        ItemKNN.model_type: ItemKNN,
        ImplicitMF.model_type: ImplicitMF,
    }
    model_class = registry.get(model_type) if isinstance(model_type, str) else None
    if model_class is None:
        raise SerializationError(f"unknown model type: {model_type!r}")
    return model_class.from_state(state)


def save_model(model: BaseRecommender, path: str | Path) -> None:
    """Write a fitted model as deterministic UTF-8 JSON."""

    if not isinstance(model, BaseRecommender):
        raise SerializationError("model must be an OrchidRec recommender")
    destination = Path(path)
    try:
        payload = model.to_state()
        text = json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    except SerializationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise SerializationError(f"could not write model to {destination}: {exc}") from exc


def load_model(path: str | Path) -> BaseRecommender:
    """Load and validate an OrchidRec JSON model."""

    source = Path(path)
    try:
        payload = strict_json_loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SerializationError(f"could not read model from {source}: {exc}") from exc
    except ValueError as exc:
        raise SerializationError(f"invalid model JSON in {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SerializationError("model file must contain a JSON object")
    return model_from_state(payload)
