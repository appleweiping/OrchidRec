"""Bounded full-batch BPR over degree-normalized training bipartite graphs.

This original CPU baseline shares a graph-propagation idea with LightGCN, but
does not implement the reference frameworks' trainer, optimizer or sampling.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any, Self

from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, InteractionDataset, stable_id_key, validate_entity_id
from orchidrec.errors import SerializationError, ValidationError
from orchidrec.models.base import BaseRecommender, make_envelope, parse_envelope
from orchidrec.training_sampling import TrainingNegativeSampler, validate_training_sampling

MAX_USERS = 256
MAX_ITEMS = 256
MAX_EDGES = 4_096
MAX_INTERACTIONS = 20_000
MAX_FACTORS = 16
MAX_EPOCHS = 30
MAX_WORK = 20_000_000
MAX_TOTAL_VALUE = 1_000_000_000.0
MAX_ABS_COORDINATE = 1_000_000.0

Edge = tuple[int, int, float]
Pair = tuple[int, int, int]
Matrix = list[list[float]]


def _real(value: object, name: str, *, positive: bool) -> float:
    if not isinstance(value, (int, float)) or type(value) not in (int, float):
        raise ValidationError(f"{name} must be a finite number in [0, 1]")
    converted = safe_float(value)
    if not math.isfinite(converted) or not 0 <= converted <= 1 or (positive and converted == 0):
        raise ValidationError(f"{name} must be a finite number in [0, 1]")
    return converted


def _identifier(value: object) -> EntityId:
    ident = validate_entity_id(value)
    if isinstance(ident, int):
        if ident.bit_length() > 512:
            raise ValidationError("graph model identifier exceeds 512 bits")
    else:
        try:
            size = len(ident.encode("utf-8", "strict"))
        except UnicodeError as error:
            raise ValidationError("graph model identifier must be UTF-8") from error
        if size > 2_048:
            raise ValidationError("graph model identifier exceeds 2048 bytes")
    return ident


def _edges(
    users: tuple[EntityId, ...],
    items: tuple[EntityId, ...],
    seen: Mapping[EntityId, frozenset[EntityId]],
) -> tuple[Edge, ...]:
    item_index = {item: len(users) + index for index, item in enumerate(items)}
    pairs = tuple(
        (user_index, item_index[item])
        for user_index, user in enumerate(users)
        for item in sorted(seen[user], key=stable_id_key)
    )
    degrees = [0] * (len(users) + len(items))
    for user_index, item_index_value in pairs:
        degrees[user_index] += 1
        degrees[item_index_value] += 1
    if any(degree == 0 for degree in degrees):
        raise ValidationError("training graph has an isolated catalog node")
    return tuple(
        (
            user_index,
            item_index_value,
            1.0 / math.sqrt(degrees[user_index] * degrees[item_index_value]),
        )
        for user_index, item_index_value in pairs
    )


def _propagate(
    ego: Sequence[Sequence[float]], edges: tuple[Edge, ...], layers: int
) -> tuple[list[Matrix], Matrix]:
    """Return every layer and their arithmetic mean on a symmetric graph."""
    width = len(ego[0])
    states: list[Matrix] = [[list(row) for row in ego]]
    for _ in range(layers):
        previous = states[-1]
        advanced = [[0.0] * width for _ in previous]
        for user, item, weight in edges:
            for axis in range(width):
                advanced[user][axis] += weight * previous[item][axis]
                advanced[item][axis] += weight * previous[user][axis]
        states.append(advanced)
    average = [
        [math.fsum(state[node][axis] for state in states) / (layers + 1) for axis in range(width)]
        for node in range(len(ego))
    ]
    return states, average


def _batch_loss_gradient(
    ego: Matrix,
    edges: tuple[Edge, ...],
    pairs: tuple[Pair, ...],
    *,
    layers: int,
    regularization: float,
) -> tuple[float, Matrix]:
    """Exact reverse-mode gradient of mean softplus(-margin) plus L2/2."""
    if not pairs:
        raise ValidationError("graph BPR requires at least one negative pair")
    states, average = _propagate(ego, edges, layers)
    width = len(ego[0])
    graph_gradient = [[0.0] * width for _ in ego]
    losses: list[float] = []
    for user, positive, negative in pairs:
        margin = math.fsum(
            average[user][axis] * (average[positive][axis] - average[negative][axis])
            for axis in range(width)
        )
        if margin >= 0:
            factor = math.exp(-margin)
            coefficient = factor / (1 + factor)
            losses.append(math.log1p(factor))
        else:
            factor = math.exp(margin)
            coefficient = 1 / (1 + factor)
            losses.append(-margin + math.log1p(factor))
        coefficient /= len(pairs)
        for axis in range(width):
            query = average[user][axis]
            graph_gradient[user][axis] -= coefficient * (
                average[positive][axis] - average[negative][axis]
            )
            graph_gradient[positive][axis] -= coefficient * query
            graph_gradient[negative][axis] += coefficient * query
    # Mean over layers gives each layer the same direct upstream gradient.
    layer_gradient = [
        [[value / (layers + 1) for value in row] for row in graph_gradient] for _ in states
    ]
    for layer in range(layers, 0, -1):
        current = layer_gradient[layer]
        previous = layer_gradient[layer - 1]
        for user, item, weight in edges:
            for axis in range(width):
                previous[user][axis] += weight * current[item][axis]
                previous[item][axis] += weight * current[user][axis]
    gradient = layer_gradient[0]
    for node, row in enumerate(ego):
        for axis, value in enumerate(row):
            gradient[node][axis] += regularization * value
    penalty = regularization * math.fsum(value * value for row in ego for value in row) / 2
    return math.fsum(losses) / len(losses) + penalty, gradient


def _work_upper(epochs: int, factors: int, layers: int, edges: int, pairs: int, nodes: int) -> int:
    """Conservative coordinate-work proxy checked before parameter allocation."""
    return epochs * factors * (8 * edges * layers + 24 * pairs + 8 * nodes * (layers + 1))


class BipartiteGraphBPR(BaseRecommender):
    """Train-only bipartite propagation with deterministic full-batch BPR."""

    model_type = "bipartite_graph_bpr"

    def __init__(
        self,
        *,
        factors: int = 8,
        layers: int = 1,
        epochs: int = 8,
        learning_rate: float = 0.05,
        regularization: float = 0.001,
        seed: int = 42,
        max_work_units: int = MAX_WORK,
        negative_strategy: str = "uniform",
        popularity_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        for name, value, maximum in (
            ("factors", factors, MAX_FACTORS),
            ("layers", layers, 2),
            ("epochs", epochs, MAX_EPOCHS),
            ("max_work_units", max_work_units, MAX_WORK),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValidationError(f"{name} must be an integer in [1, {maximum}]")
        if type(seed) is not int or not -(1 << 63) <= seed < (1 << 63):
            raise ValidationError("seed must be a signed 64-bit integer")
        self.factors = factors
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = _real(learning_rate, "learning_rate", positive=True)
        self.regularization = _real(regularization, "regularization", positive=False)
        self.seed = seed
        self.max_work_units = max_work_units
        self.negative_strategy, self.popularity_alpha = validate_training_sampling(
            negative_strategy, popularity_alpha
        )
        self._users: tuple[EntityId, ...] = ()
        self._edges: tuple[Edge, ...] = ()
        self._ego: Matrix = []
        self._average: Matrix = []
        self._history: tuple[float, ...] = ()
        self._work_upper = 0

    @property
    def work_upper(self) -> int:
        self._require_fitted()
        return self._work_upper

    @property
    def loss_history(self) -> tuple[float, ...]:
        self._require_fitted()
        return self._history

    def fit(self, dataset: InteractionDataset) -> Self:
        if type(dataset) is not InteractionDataset or not dataset:
            raise ValidationError("graph BPR needs a nonempty InteractionDataset")
        users, items = dataset.user_ids, dataset.item_ids
        if len(dataset) > MAX_INTERACTIONS or len(users) > MAX_USERS or len(items) > MAX_ITEMS:
            raise ValidationError("graph BPR interaction/user/item limit exceeded")
        for ident in (*users, *items):
            _identifier(ident)
        total = 0.0
        for event in dataset:
            total += event.value
            if not math.isfinite(total) or total > MAX_TOTAL_VALUE:
                raise ValidationError("graph BPR total interaction value exceeds bound")
        seen = dataset.by_user()
        unique_edges = sum(len({event.item_id for event in events}) for events in seen.values())
        if unique_edges > MAX_EDGES:
            raise ValidationError("graph BPR edge limit exceeded")
        available_pairs = sum(
            len({event.item_id for event in events})
            for events in seen.values()
            if len({event.item_id for event in events}) < len(items)
        )
        if not available_pairs:
            raise ValidationError("graph BPR needs at least one unseen training-catalog item")
        work = _work_upper(
            self.epochs,
            self.factors,
            self.layers,
            unique_edges,
            available_pairs,
            len(users) + len(items),
        )
        if work > self.max_work_units:
            raise ValidationError("graph BPR coordinate work exceeds max_work_units")
        self._work_upper = work
        return super().fit(dataset)

    def _fit_model(self, dataset: InteractionDataset) -> None:
        self._users = tuple(sorted(self._seen, key=stable_id_key))
        self._edges = _edges(self._users, self._catalog, self._seen)
        sampler = (
            TrainingNegativeSampler(
                dataset,
                seen=self._seen,
                catalog=self._catalog,
                strategy=self.negative_strategy,
                alpha=self.popularity_alpha,
                epochs=self.epochs,
                draws_per_positive=1,
            )
            if self.negative_strategy == "popularity"
            else None
        )
        item_index = {item: len(self._users) + offset for offset, item in enumerate(self._catalog)}
        negative_pools = {
            user_index: tuple(
                len(self._users) + item_index
                for item_index, item in enumerate(self._catalog)
                if item not in self._seen[user]
            )
            for user_index, user in enumerate(self._users)
        }
        positives = tuple((user, item) for user, item, _ in self._edges if negative_pools[user])
        rng = random.Random(self.seed)  # nosec B311
        self._ego = [
            [rng.uniform(-0.1, 0.1) for _ in range(self.factors)]
            for _ in range(len(self._users) + len(self._catalog))
        ]
        history: list[float] = []
        for _ in range(self.epochs):
            pairs = tuple(
                (
                    user,
                    positive,
                    negative_pools[user][rng.randrange(len(negative_pools[user]))]
                    if sampler is None
                    else item_index[sampler.sample(self._users[user], rng)],
                )
                for user, positive in positives
            )
            loss, gradient = _batch_loss_gradient(
                self._ego,
                self._edges,
                pairs,
                layers=self.layers,
                regularization=self.regularization,
            )
            if not math.isfinite(loss):
                raise ValidationError("graph BPR training loss diverged")
            history.append(loss)
            for row, grad in zip(self._ego, gradient, strict=True):
                for axis in range(self.factors):
                    updated = row[axis] - self.learning_rate * grad[axis]
                    if not math.isfinite(updated) or abs(updated) > MAX_ABS_COORDINATE:
                        raise ValidationError("graph BPR training coordinate diverged")
                    row[axis] = updated
        self._history = tuple(history)
        _, self._average = _propagate(self._ego, self._edges, self.layers)

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        try:
            user_index = self._users.index(user_id)
        except ValueError:
            return self._popularity_fallback(item_id)
        item_index = len(self._users) + self._catalog.index(item_id)
        return math.fsum(
            left * right
            for left, right in zip(
                self._average[user_index], self._average[item_index], strict=True
            )
        )

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        return make_envelope(
            self.model_type,
            {
                "factors": self.factors,
                "layers": self.layers,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "seed": self.seed,
                "max_work_units": self.max_work_units,
                **(
                    {
                        "negative_strategy": self.negative_strategy,
                        "popularity_alpha": self.popularity_alpha,
                    }
                    if self.negative_strategy == "popularity"
                    else {}
                ),
            },
            self._base_state(),
            {
                "edges": [[user, item] for user, item, _ in self._edges],
                "ego_embeddings": [list(row) for row in self._ego],
                "loss_history": list(self._history),
                "work_upper": self._work_upper,
            },
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        old_parameters = {
            "factors",
            "layers",
            "epochs",
            "learning_rate",
            "regularization",
            "seed",
            "max_work_units",
        }
        if set(parameters) not in (
            old_parameters,
            old_parameters | {"negative_strategy", "popularity_alpha"},
        ):
            raise SerializationError("graph BPR parameters are malformed")
        try:
            instance = cls(**parameters)
        except (TypeError, ValidationError) as error:
            raise SerializationError(f"invalid graph BPR parameters: {error}") from error
        if set(parameters) != old_parameters and instance.negative_strategy != "popularity":
            raise SerializationError("graph BPR augmented sampler state must use popularity")
        if (
            not isinstance(base.get("catalog"), list)
            or not isinstance(base.get("users"), list)
            or not isinstance(base.get("popularity"), list)
            or not 0 < len(base["catalog"]) <= MAX_ITEMS
            or not 0 < len(base["users"]) <= MAX_USERS
            or len(base["popularity"]) != len(base["catalog"])
            or any(
                not isinstance(entry, Mapping)
                or not isinstance(entry.get("seen"), list)
                or len(entry["seen"]) > MAX_ITEMS
                for entry in base["users"]
            )
        ):
            raise SerializationError("graph BPR base state exceeds limits")
        instance._restore_base_state(base)
        try:
            for ident in (*instance._catalog, *instance._seen):
                _identifier(ident)
        except ValidationError as error:
            raise SerializationError(str(error)) from error
        total = 0.0
        for value in instance._popularity.values():
            total += value
            if not math.isfinite(total) or total > MAX_TOTAL_VALUE:
                raise SerializationError("graph BPR popularity total exceeds bound")
        if set(model) != {"edges", "ego_embeddings", "loss_history", "work_upper"}:
            raise SerializationError("graph BPR model state is malformed")
        instance._users = tuple(sorted(instance._seen, key=stable_id_key))
        if sum(map(len, instance._seen.values())) > MAX_EDGES:
            raise SerializationError("graph BPR edge limit exceeded")
        try:
            instance._edges = _edges(instance._users, instance._catalog, instance._seen)
        except ValidationError as error:
            raise SerializationError(str(error)) from error
        expected_edges = [[user, item] for user, item, _ in instance._edges]
        if model["edges"] != expected_edges:
            raise SerializationError("graph BPR edges do not match fitted seen-item state")
        pairs = sum(
            len(seen) for seen in instance._seen.values() if len(seen) < len(instance._catalog)
        )
        if not pairs:
            raise SerializationError("graph BPR has no eligible negative pair")
        expected_work = _work_upper(
            instance.epochs,
            instance.factors,
            instance.layers,
            len(instance._edges),
            pairs,
            len(instance._users) + len(instance._catalog),
        )
        if (
            type(model["work_upper"]) is not int
            or model["work_upper"] != expected_work
            or expected_work > instance.max_work_units
        ):
            raise SerializationError("graph BPR work budget is inconsistent")
        instance._work_upper = expected_work
        raw_ego = model["ego_embeddings"]
        if not isinstance(raw_ego, list) or len(raw_ego) != len(instance._users) + len(
            instance._catalog
        ):
            raise SerializationError("graph BPR embedding row count is malformed")
        ego: Matrix = []
        for row in raw_ego:
            if not isinstance(row, list) or len(row) != instance.factors:
                raise SerializationError("graph BPR embedding width is malformed")
            converted: list[float] = []
            for value in row:
                if type(value) not in (int, float):
                    raise SerializationError("graph BPR embedding value is malformed")
                coordinate = safe_float(value)
                if not math.isfinite(coordinate) or abs(coordinate) > MAX_ABS_COORDINATE:
                    raise SerializationError("graph BPR embedding coordinate exceeds bound")
                converted.append(coordinate)
            ego.append(converted)
        history = model["loss_history"]
        if not isinstance(history, list) or len(history) != instance.epochs:
            raise SerializationError("graph BPR loss history is malformed")
        for value in history:
            if type(value) not in (int, float) or not math.isfinite(safe_float(value)) or value < 0:
                raise SerializationError("graph BPR loss history is invalid")
        instance._ego = ego
        instance._history = tuple(safe_float(value) for value in history)
        _, instance._average = _propagate(ego, instance._edges, instance.layers)
        if any(not math.isfinite(value) for row in instance._average for value in row):
            raise SerializationError("graph BPR propagation is non-finite")
        return instance
