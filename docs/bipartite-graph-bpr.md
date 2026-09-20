# Bounded bipartite graph BPR

`bipartite_graph_bpr` is an original dependency-free CPU recommender. It
uses the training interactions to learn embeddings through a normalized
user–item graph. The graph propagation is LightGCN-style, but the local
full-batch BPR optimizer, negative sampler, initialization, bounds, and
state format are different from frozen RecBole and LensKit implementations.
This is not a reproduction of their scores, trainer, benchmark, or repository.

## Synthetic workflow

From the repository root:

```bash
orchidrec run examples/bipartite_graph_config.json
orchidrec recommend artifacts/bipartite-graph/model.json u1 --k 2
orchidrec benchmark examples/bipartite_graph_benchmark_config.json \
  --output-dir artifacts/bipartite-graph-benchmark
```

The six fictional events give each of three users one training event and one
held-out event under `leave_one_out`. The training catalog still contains all
three items. The saved graph has exactly three user–item edges, not the three
held-out edges. The benchmark evaluates this model and popularity on one
shared split, not an official dataset or metric target.

## Objective and exact boundary

Each unique positive user–item pair in the **training** split creates an
undirected bipartite edge. Event values and duplicate events do not multiply
edges or BPR examples; weighted training popularity is retained only for an
unknown-user fallback. Write `P = D⁻¹ᐟ² A D⁻¹ᐟ²`. Seeded ego embeddings are
propagated one or two layers, `Eˡ = P Eˡ⁻¹`, and the scoring embedding is the
arithmetic mean of all layers `0..L`. A known user's candidate score is the
user/item dot product; an unknown user falls back to normalized **training**
popularity. The common ranker removes seen items by default.

Each epoch samples one unseen **training-catalog** negative for each eligible
unique positive using a local seeded generator. Users who have seen every
training item contribute no BPR pair, but their graph edges remain. One
full-batch step minimizes mean `log(1+exp(-(s_positive-s_negative)))` plus
`regularization * ||E⁰||² / 2`. Backpropagation goes through every propagation
layer, not just the final dot product. The independent tests hand-compute
a two-user/two-item one-hop graph and gradient and compare two-layer gradients
to central finite differences. Validation and test events are never used for
graph edges, negatives, degrees, gradients, or popularity.

## Limits, state, and interpretation

At most 256 users, 256 items, 4,096 unique edges, and 20,000 interaction
records are admitted. Factors are 1–16, layers 1–2, epochs 1–30, and a
conservative coordinate-work proxy (propagation, reverse pass, BPR pairs,
and node updates) must not exceed the configured maximum of 20 million.
Coordinates and total interaction weight have separate finite caps. Invalid
IDs, parameters, work, non-finite gradients, or no eligible negative fail
closed before publishing a model. Input sizes are checked before allocating
the embedding matrix.

The versioned saved state embeds training graph edges, final ego embeddings,
loss history, and a shared base catalog/seen/popularity state. Reload checks
edge equality against seen items, row shapes, numerical bounds, and the work
proxy before recomputing scores. Structural state checks do **not** prove the
training source or authenticate a checkpoint: a party able to rewrite JSON
can also change otherwise valid embeddings. Preserve and govern input files
separately. Temporal validity of the split is the caller's responsibility;
for chronological questions choose a temporal/per-user chronology-aware
split and avoid future-derived features.
