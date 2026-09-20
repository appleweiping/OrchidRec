"""Bounded deterministic relation-weighted graph-walk recommendation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Self

from orchidrec._numeric import safe_float
from orchidrec.data import EntityId, InteractionDataset, stable_id_key, validate_entity_id
from orchidrec.errors import DatasetError, SerializationError, ValidationError
from orchidrec.models.base import BaseRecommender, make_envelope, parse_envelope
from orchidrec.recbole_knowledge import (
    ItemEntityLink,
    KnowledgeLimits,
    KnowledgeTriple,
    LoadedKnowledgeLinks,
    _digest,
    _token,
    _validate_semantics,
)

MAX_TRIPLES = 20_000
MAX_LINKS = 5_000
MAX_USERS = 2_000
MAX_ITEMS = 2_000
MAX_INTERACTIONS = 50_000
MAX_TOTAL_VALUE = 1e12
MAX_WORK = 20_000_000
MAX_STATE_CELLS = 1_000_000


def _number(value: object, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be finite in [{minimum}, {maximum}]")
    result = safe_float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValidationError(f"{name} must be finite in [{minimum}, {maximum}]")
    return result


def _id(value: object, name: str) -> EntityId:
    result = validate_entity_id(value, name)
    if isinstance(result, int):
        if result.bit_length() > 512:
            raise ValidationError(f"{name} exceeds 512 integer bits")
    else:
        try:
            encoded_length = len(result.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ValidationError(f"{name} must be valid UTF-8") from error
        if encoded_length > 2_048:
            raise ValidationError(f"{name} exceeds 2048 UTF-8 bytes")
    return result


class KGWalkRec(BaseRecommender):
    """Rank linked items by exact-H-step graph-walk mass from training positives.

    Each knowledge triple contributes one arc in each direction, retaining its
    relation weight. Dangling entities have a deterministic self-loop.
    """

    model_type = "kg_walk_rec"

    def __init__(
        self,
        *,
        hops: int = 2,
        relation_weights: Mapping[str, float] | None = None,
        weighted: bool = True,
        popularity_mix: float = 0.05,
        max_work_units: int = MAX_WORK,
    ) -> None:
        super().__init__()
        if type(hops) is not int or not 1 <= hops <= 3:
            raise ValidationError("hops must be an integer in [1, 3]")
        if type(weighted) is not bool:
            raise ValidationError("weighted must be a boolean")
        if type(max_work_units) is not int or not 1 <= max_work_units <= MAX_WORK:
            raise ValidationError(f"max_work_units must be in [1, {MAX_WORK}]")
        if relation_weights is not None and (
            not isinstance(relation_weights, Mapping) or len(relation_weights) > 128
        ):
            raise ValidationError("relation_weights must be a bounded mapping")
        weights: dict[str, float] = {}
        for relation, value in (relation_weights or {}).items():
            if type(relation) is not str or not relation:
                raise ValidationError("relation_weights keys must be bounded relation IDs")
            try:
                encoded_length = len(relation.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValidationError("relation_weights keys must be valid UTF-8") from error
            if encoded_length > 2_048:
                raise ValidationError("relation_weights keys must be bounded relation IDs")
            weights[relation] = _number(value, "relation weight", minimum=1e-9, maximum=1e6)
        self.hops = hops
        self.relation_weights = dict(sorted(weights.items()))
        self.weighted = weighted
        self.popularity_mix = _number(popularity_mix, "popularity_mix", minimum=0, maximum=1)
        self.max_work_units = max_work_units
        self._triples: tuple[KnowledgeTriple, ...] = ()
        self._links: tuple[ItemEntityLink, ...] = ()
        self._knowledge_fingerprint = ""
        self._sources: tuple[str, str] = ("", "")
        self._seeds: dict[EntityId, dict[str, float]] = {}
        self._walks: dict[EntityId, dict[str, float]] = {}
        self._item_entities: dict[str, str] = {}
        self._max_popularity = 0.0
        self._work_units = 0

    @property
    def knowledge_fingerprint(self) -> str:
        self._require_fitted()
        return self._knowledge_fingerprint

    @property
    def work_units(self) -> int:
        self._require_fitted()
        return self._work_units

    def fit(
        self, dataset: InteractionDataset, knowledge: LoadedKnowledgeLinks | None = None
    ) -> Self:
        if type(dataset) is not InteractionDataset or not dataset:
            raise ValidationError("dataset must be a non-empty InteractionDataset")
        if type(knowledge) is not LoadedKnowledgeLinks:
            raise ValidationError("knowledge must be a LoadedKnowledgeLinks artifact")
        if len(dataset) > MAX_INTERACTIONS or len(dataset.user_ids) > MAX_USERS:
            raise ValidationError("KGWalkRec interaction or user limit exceeded")
        if len(dataset.item_ids) > MAX_ITEMS:
            raise ValidationError("KGWalkRec catalog limit exceeded")
        oversized_triples = (
            type(knowledge.triples) is tuple and len(knowledge.triples) > MAX_TRIPLES
        )
        oversized_links = type(knowledge.links) is tuple and len(knowledge.links) > MAX_LINKS
        if oversized_triples or oversized_links:
            raise ValidationError("KGWalkRec graph or link limit exceeded")
        _validate_semantics(knowledge)
        if any(type(item) is not str for item in dataset.item_ids):
            raise ValidationError("KGWalkRec item IDs must be strings matching .link tokens")
        for ident in (*dataset.user_ids, *dataset.item_ids):
            _id(ident, "interaction ID")
        total = 0.0
        for event in dataset:
            total += event.value
            if not math.isfinite(total) or total > MAX_TOTAL_VALUE:
                raise ValidationError("KGWalkRec total interaction value exceeds bound")
        self._triples = knowledge.triples
        self._links = knowledge.links
        self._knowledge_fingerprint = knowledge.fingerprint
        self._sources = (knowledge.sources[0].sha256, knowledge.sources[1].sha256)
        self._fitted = False
        try:
            super().fit(dataset)
            links = {link.item_id: link.entity_id for link in knowledge.links}
            self._seeds = {}
            for event in dataset:
                if event.item_id in links:
                    user = self._seeds.setdefault(event.user_id, {})
                    item = event.item_id
                    user[item] = user.get(item, 0.0) + (event.value if self.weighted else 1.0)
            if not self._seeds:
                raise ValidationError("KGWalkRec requires at least one linked training positive")
            self._prepare_walks()
        except BaseException:
            self._fitted = False
            raise
        self._fitted = True
        return self

    def _fit_model(self, dataset: InteractionDataset) -> None:
        del dataset  # Side graph preparation follows the shared catalog/seen setup.

    def _prepare_walks(self) -> None:
        links = {link.item_id: link.entity_id for link in self._links}
        self._item_entities = {item: links[item] for item in self._catalog if item in links}
        relations = {triple.relation_id for triple in self._triples}
        if set(self.relation_weights) - relations:
            raise ValidationError("relation_weights refers to an unknown relation")
        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for triple in self._triples:
            weight = self.relation_weights.get(triple.relation_id, 1.0)
            adjacency[triple.head_id].append((triple.tail_id, weight))
            adjacency[triple.tail_id].append((triple.head_id, weight))
        totals = {node: math.fsum(weight for _, weight in arcs) for node, arcs in adjacency.items()}
        if any(not math.isfinite(total) or total <= 0 for total in totals.values()):
            raise ValidationError("KGWalkRec adjacency weight is not finite")
        work = 0
        cells = 0
        walks: dict[EntityId, dict[str, float]] = {}
        for user in sorted(self._seeds, key=stable_id_key):
            seed = self._seeds[user]
            seed_total = math.fsum(seed.values())
            if not math.isfinite(seed_total) or not 0 < seed_total <= MAX_TOTAL_VALUE:
                raise ValidationError("KGWalkRec user seed total exceeds bound")
            frontier: dict[str, float] = {}
            for item, weight in seed.items():
                entity = self._item_entities[item]
                frontier[entity] = frontier.get(entity, 0.0) + weight / seed_total
            for _ in range(self.hops):
                advanced: dict[str, float] = {}
                for entity, mass in sorted(frontier.items()):
                    arcs = adjacency.get(entity)
                    if not arcs:
                        work += 1
                        if work > self.max_work_units:
                            raise ValidationError("KGWalkRec propagation work limit exceeded")
                        advanced[entity] = advanced.get(entity, 0.0) + mass
                    else:
                        for neighbor, weight in arcs:
                            work += 1
                            if work > self.max_work_units:
                                raise ValidationError("KGWalkRec propagation work limit exceeded")
                            advanced[neighbor] = (
                                advanced.get(neighbor, 0.0) + mass * weight / totals[entity]
                            )
                    if len(advanced) + cells > MAX_STATE_CELLS:
                        raise ValidationError("KGWalkRec sparse state limit exceeded")
                frontier = advanced
            cells += len(frontier)
            if cells > MAX_STATE_CELLS:
                raise ValidationError("KGWalkRec sparse state limit exceeded")
            if any(not math.isfinite(mass) or mass < 0 for mass in frontier.values()):
                raise ValidationError("KGWalkRec propagation produced non-finite mass")
            walks[user] = frontier
        self._walks = walks
        self._max_popularity = max(self._popularity.values())
        self._work_units = work

    def _score(self, user_id: EntityId, item_id: EntityId) -> float:
        fallback = self._popularity[item_id] / self._max_popularity
        walk = self._walks.get(user_id)
        if walk is None:
            return fallback
        entity = self._item_entities.get(item_id) if isinstance(item_id, str) else None
        graph_score = walk.get(entity, 0.0) if entity is not None else 0.0
        return (1 - self.popularity_mix) * graph_score + self.popularity_mix * fallback

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        return make_envelope(
            self.model_type,
            {
                "hops": self.hops,
                "relation_weights": self.relation_weights,
                "weighted": self.weighted,
                "popularity_mix": self.popularity_mix,
                "max_work_units": self.max_work_units,
            },
            self._base_state(),
            {
                "triples": [triple.to_state() for triple in self._triples],
                "links": [link.to_state() for link in self._links],
                "knowledge_fingerprint": self._knowledge_fingerprint,
                "source_sha256": list(self._sources),
                "user_seeds": [
                    {
                        "user_id": user,
                        "items": [
                            {"item_id": item, "weight": value}
                            for item, value in sorted(self._seeds[user].items())
                        ],
                    }
                    for user in sorted(self._seeds, key=stable_id_key)
                ],
            },
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        parameters, base, model = parse_envelope(state, cls.model_type)
        if set(parameters) != {
            "hops",
            "relation_weights",
            "weighted",
            "popularity_mix",
            "max_work_units",
        }:
            raise SerializationError("KGWalkRec parameters are malformed")
        try:
            instance = cls(**parameters)
        except (TypeError, ValidationError) as error:
            raise SerializationError(f"invalid KGWalkRec parameters: {error}") from error
        if (
            set(base) != {"catalog", "popularity", "users"}
            or any(not isinstance(base[key], list) for key in ("catalog", "popularity", "users"))
            or len(base["catalog"]) > MAX_ITEMS
            or len(base["popularity"]) > MAX_ITEMS
            or len(base["users"]) > MAX_USERS
        ):
            raise SerializationError("KGWalkRec base arrays exceed limits")
        for entry in base["users"]:
            if (
                not isinstance(entry, Mapping)
                or not isinstance(entry.get("seen"), list)
                or len(entry["seen"]) > MAX_ITEMS
            ):
                raise SerializationError("KGWalkRec user seen array exceeds limits")
        instance._restore_base_state(base)
        if (
            not instance._catalog
            or len(instance._catalog) > MAX_ITEMS
            or len(instance._seen) > MAX_USERS
        ):
            raise SerializationError("KGWalkRec base state exceeds limits")
        if any(type(item) is not str for item in instance._catalog):
            raise SerializationError("KGWalkRec catalog items must be strings")
        try:
            popularity_total = math.fsum(instance._popularity.values())
        except (OverflowError, ValueError) as error:
            raise SerializationError("KGWalkRec popularity total is not finite") from error
        if popularity_total > MAX_TOTAL_VALUE:
            raise SerializationError("KGWalkRec popularity total exceeds bound")
        for ident in (*instance._catalog, *instance._seen):
            try:
                _id(ident, "state ID")
            except ValidationError as error:
                raise SerializationError(str(error)) from error
        if set(model) != {
            "triples",
            "links",
            "knowledge_fingerprint",
            "source_sha256",
            "user_seeds",
        }:
            raise SerializationError("KGWalkRec state fields are malformed")
        raw_triples, raw_links = model["triples"], model["links"]
        if (
            not isinstance(raw_triples, list)
            or not 0 < len(raw_triples) <= MAX_TRIPLES
            or not isinstance(raw_links, list)
            or not 0 < len(raw_links) <= MAX_LINKS
        ):
            raise SerializationError("KGWalkRec graph arrays exceed limits")

        token_limits = KnowledgeLimits()

        def tokens(raw: object, width: int) -> tuple[str, ...]:
            if (
                not isinstance(raw, list)
                or len(raw) != width
                or any(type(value) is not str or not value for value in raw)
            ):
                raise SerializationError("KGWalkRec graph row is malformed")
            try:
                for value in raw:
                    _token(value, "graph token", token_limits)
            except (DatasetError, UnicodeEncodeError) as error:
                raise SerializationError(f"KGWalkRec graph token is invalid: {error}") from error
            return tuple(raw)

        triples = tuple(KnowledgeTriple(*tokens(row, 3)) for row in raw_triples)
        links = tuple(ItemEntityLink(*tokens(row, 2)) for row in raw_links)
        if triples != tuple(sorted(set(triples))) or links != tuple(sorted(set(links))):
            raise SerializationError("KGWalkRec graph is not unique and ordered")
        if len({link.item_id for link in links}) != len(links) or len(
            {link.entity_id for link in links}
        ) != len(links):
            raise SerializationError("KGWalkRec links are not one-to-one")
        fingerprint = _digest(
            {
                "triples": [row.to_state() for row in triples],
                "links": [row.to_state() for row in links],
            }
        )
        if model["knowledge_fingerprint"] != fingerprint:
            raise SerializationError("KGWalkRec knowledge fingerprint mismatch")
        source = model["source_sha256"]
        if (
            not isinstance(source, list)
            or len(source) != 2
            or any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in source
            )
        ):
            raise SerializationError("KGWalkRec source digests are malformed")
        instance._triples = triples
        instance._links = links
        instance._knowledge_fingerprint = fingerprint
        instance._sources = (source[0], source[1])
        raw_seeds = model["user_seeds"]
        if not isinstance(raw_seeds, list) or not 0 < len(raw_seeds) <= MAX_USERS:
            raise SerializationError("KGWalkRec user seeds are malformed")
        linked = {link.item_id for link in links}
        seeds: dict[EntityId, dict[str, float]] = {}
        total_cells = 0
        for entry in raw_seeds:
            if not isinstance(entry, Mapping) or set(entry) != {"user_id", "items"}:
                raise SerializationError("KGWalkRec user seed entry is malformed")
            try:
                user = _id(entry["user_id"], "seed user")
            except ValidationError as error:
                raise SerializationError(str(error)) from error
            rows = entry["items"]
            if (
                user in seeds
                or user not in instance._seen
                or not isinstance(rows, list)
                or not rows
            ):
                raise SerializationError("KGWalkRec user seed rows are invalid")
            row: dict[str, float] = {}
            for item_entry in rows:
                if not isinstance(item_entry, Mapping) or set(item_entry) != {"item_id", "weight"}:
                    raise SerializationError("KGWalkRec item seed entry is malformed")
                item = item_entry["item_id"]
                try:
                    weight = _number(
                        item_entry["weight"], "seed weight", minimum=0, maximum=MAX_TOTAL_VALUE
                    )
                except ValidationError as error:
                    raise SerializationError(str(error)) from error
                if weight <= 0:
                    raise SerializationError("KGWalkRec seed weights must be positive")
                if (
                    type(item) is not str
                    or item in row
                    or item not in linked
                    or item not in instance._seen[user]
                ):
                    raise SerializationError("KGWalkRec seed item is invalid")
                row[item] = weight
                total_cells += 1
                if total_cells > MAX_INTERACTIONS:
                    raise SerializationError("KGWalkRec user seeds exceed cell limit")
            try:
                row_total = math.fsum(row.values())
            except (OverflowError, ValueError) as error:
                raise SerializationError("KGWalkRec user seed total is not finite") from error
            if list(row) != sorted(row) or row_total > MAX_TOTAL_VALUE:
                raise SerializationError("KGWalkRec user seeds are not canonical or bounded")
            seeds[user] = row
        if list(seeds) != sorted(seeds, key=stable_id_key):
            raise SerializationError("KGWalkRec seed users are not stably ordered")
        try:
            seed_total = math.fsum(weight for row in seeds.values() for weight in row.values())
        except (OverflowError, ValueError) as error:
            raise SerializationError("KGWalkRec aggregate seed total is not finite") from error
        if seed_total > MAX_TOTAL_VALUE:
            raise SerializationError("KGWalkRec aggregate seed total exceeds bound")
        instance._seeds = seeds
        try:
            instance._prepare_walks()
        except ValidationError as error:
            raise SerializationError(f"invalid KGWalkRec graph state: {error}") from error
        return instance
