# Candidate sampling for offline evaluation

OrchidRec defaults to **full-sort evaluation**: every item in the training
catalog is eligible for ranking, subject to the usual seen-item filter. An
opt-in `evaluation.sampling` object changes the estimand. It ranks every
held-out positive alongside a fixed number of sampled *unobserved* items per
user. A sampled score must not be described as a full-catalog score or
compared directly with one.

```json
"evaluation": {
  "k": 10,
  "exclude_seen": true,
  "sampling": {"strategy": "uniform", "negatives": 100}
}
```

`strategy` is `uniform` or `popularity`; `negatives` is an integer from 1 to
10,000. The immutable built-in sampler registry is also available through the
Python API (`SAMPLER_REGISTRY`). Uniform sampling is without replacement.
Popularity sampling is without replacement, weighted by **training-only
interaction counts** using an exponential race. The holdout never changes the
sampling weight. If fewer than the requested negatives are eligible, all
eligible negatives are used. A user with no eligible negatives is evaluated
with their positives alone.

For each user, OrchidRec excludes both the training history and **all**
held-out positives before sampling negatives. It then includes every
catalog-present held-out positive. A train/test positive overlap fails closed;
`exclude_seen: false` cannot be combined with sampling. Cold-start positives
outside the training catalog remain excluded, as in full-sort mode. The
candidate universe is fixed for each `(seed, user)` and sorted by typed ID,
independent of input order. In a shared-split benchmark, all models and all
trials on the same split see the same candidates; inner validation candidates
are built from the inner training split only, never the outer test partition.

The report's `evaluation` section is additive in sampled mode. It records
`mode: "sampled"`, strategy/requested count, the SHA-256 of actual per-user
candidates, and positive/negative/total candidate-pair counts. A default
full-sort report remains backward compatible. Ranking metrics still use the
original full training catalog for catalog coverage and training counts for
novelty. Precision@K, Recall@K, NDCG@K, and MRR@K are conditional on the
sampled candidate sets; unlike full-sort values, they generally become easier
as the candidate pool shrinks. Neither mode corrects unobserved-item labeling
bias. Exposure-corrected metrics retain their existing interpretation and
should be interpreted alongside the candidate mode.

For a tuned benchmark, JSON records the **outer test** candidate digest in
`evaluation.candidate_sha256` and the distinct **inner validation** digest in
`tuning.fingerprints.validation_candidate_sha256`. CSV model/comparison rows
carry `candidate_partition=test` and the outer digest; tuning candidate/trial
rows carry `candidate_partition=validation` and the inner digest. HTML and CLI
summaries label both. Never attribute validation selection scores to the outer
test candidate pool.

```bash
orchidrec run examples/sampled_config.json
orchidrec benchmark examples/sampled_benchmark_config.json --output-dir artifacts/sampled-benchmark
orchidrec benchmark examples/sampled_tuned_benchmark_config.json --output-dir artifacts/sampled-tuned-benchmark
```

`sample_candidates()` is the standalone Python entry point for constructing
and auditing a `CandidatePlan`. It accepts a catalog, positive training
counts, held-out relevant sets, and training seen sets. The `seen` mapping
must contain **every** evaluated user; callers pass an explicit empty set for
a cold user. Omitting a user's history is rejected, not interpreted as empty.
Resource ceilings are
100,000 catalog items, 100,000 users, 20 million user-item checks, and 5
million materialized candidate pairs. Inputs beyond a ceiling fail before
unbounded work. This is an in-memory sampler, not RecBole's complete training
sampler system or a distributed candidate service.

The public sampler also bounds seeds to signed 64-bit integers, training item
counts to one billion, and IDs to 4,096 UTF-8 bytes or integer bits. These
limits keep direct Python API inputs within the same finite evaluation
contract as file-backed experiments.
