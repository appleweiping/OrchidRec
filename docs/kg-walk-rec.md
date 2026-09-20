# KGWalkRec: bounded knowledge-graph walks

`KGWalkRec` is a deterministic, relation-weighted multi-hop recommender that
actually consumes OrchidRec's local `.kg`/`.link` interchange. It is an
original, small graph-walk baseline, **not** RecBole KGCN, CKE, KGAT, or a
trained graph neural network. It does not establish model-family or whole
RecBole/LensKit parity.

## Reproduce the synthetic workflow

```bash
orchidrec import-recbole-knowledge \
  --kg examples/kg_walk_synthetic.kg \
  --link examples/kg_walk_synthetic.link \
  --inter examples/kg_walk_synthetic.inter --minimum-rating 1 \
  --output artifacts/kg-walk/knowledge.json
orchidrec run examples/kg_walk_config.json
orchidrec recommend artifacts/kg-walk/model.json u --k 3
orchidrec benchmark examples/kg_walk_benchmark_config.json \
  --output-dir artifacts/kg-walk-benchmark
```

The importer is create-only: use a new output path for a repeated run. The
single-model example uses a native JSON interaction table; the benchmark uses
an equivalent RecBole-style `.inter` table. Both use leave-one-out splits and
the same synthetic KG artifact. The four training catalog items are present
even when each user's latest item is held out. Matching synthetic benchmark
metrics are an integration check, not a quality or speed comparison.

## Exact score

Each triple `(head, relation, tail)` adds two graph arcs, one per direction.
An arc has the configured positive relation weight, or weight 1 if omitted.
Outgoing weights at an entity are normalized to one. An entity with no arcs
has a self-loop. A user's positive **training** events seed their linked
entities; with `weighted=true`, the normalized seed uses interaction values,
and otherwise each event contributes one. Interactions with unlinked items
do not enter this seed. The graph distribution is propagated for exactly
`hops` steps (1–3); a candidate's graph score is its linked entity's final
probability mass. The output blends that mass with normalized **training**
popularity using `popularity_mix`. An unlinked candidate has zero graph mass;
an unknown user or one without linked training positives uses full popularity
fallback. Shared top-K code filters seen items unless requested otherwise.

For the checked-in KG, item A's entity reaches genre `g` via relation `r`
and attribute `h` via `q`. Item B also touches `g`; C also touches `h`.
With `r:q = 2:1`, two steps from A put `1/3` mass on B and `1/6` on C.
The independent test asserts these exact values with popularity mixing off.

## Scope, provenance, and bounds

The model accepts only string item IDs with an exact case-sensitive `.link`
join; integer `42` never silently matches string `"42"`. User IDs may be
bounded strings or integers. The model is in-memory and caps training at
50,000 interactions, 2,000 users, 2,000 catalog items, 20,000 triples, 5,000
links, 20 million propagation arc visits, and one million cached sparse walk
cells. Parameter and state limits are checked; saving and loading refuse model
files above the common 256 MiB file limit. Saved models embed the
canonical graph and training user seeds, so inference does not reread the KG
artifact. The persisted fingerprint is recomputed from the embedded graph.

The source `.kg`/`.link` hashes and checked artifact-state hash in reports
bind local bytes and normalized content; they are **not** provenance
signatures, authenticity proof, or point-in-time certification. The caller
must ensure KG edges and links were available at prediction time; a graph
containing future-derived facts can leak information even when interaction
seeds come only from the training partition. A benchmark using `.inter`
rejects a KG artifact whose optional `.inter` source reference names a
different source. Native JSON experiments cannot make that `.inter` source
identity assertion. No licensed or official external dataset/result is
claimed. A random interaction split can also place later user events in
training; choose a temporal or per-user leave-one-out split for temporal
questions.
