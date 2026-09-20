# Bounded training-time negative sampling

`ImplicitMF`, `SideFeatureFM`, and `BipartiteGraphBPR` support an opt-in
`negative_strategy: "popularity"` with `popularity_alpha` in `(0, 2]`. The
default remains the original seeded, uniform-with-replacement `randrange`
path. This is an original local policy, **not** RecBole's sampler registry or
training protocol. Evaluation candidate sampling in [`sampling.md`](sampling.md)
is a separate estimand and is unchanged.

For each training user, eligible negatives are the training catalog minus all
items observed by that user in the training split. A popularity draw chooses
item `i` with probability `count_train(i)^alpha / sum_j count_train(j)^alpha`
over that user's eligible pool. Counts are *event counts*, not event values.
Duplicate events affect popularity but create no extra positive graph edge.
Each draw uses the model-local seeded generator and samples with replacement;
users with no eligible item are skipped. Validation/test labels and counts are
not passed to the sampler. As with all implicit BPR, an item withheld from one
user may still be an eligible training-catalog negative for that user; using
holdout labels to remove it would leak evaluation information.

The optional popularity policy preflights at most 20,000 training events,
512 users, 512 catalog items, 262,144 user-item eligibility checks, and one
million planned draws before building pools or model parameters. Its bounded
CDF draws have a deterministic catalog-order tie rule. Impossible or
precision-collapsed weights fail closed. Existing uniform model fits and
schema-1 checkpoint parameter shapes are unchanged; popularity checkpoints
explicitly record strategy and alpha. The checkpoint does not prove the
training source or historical distribution: preserve training bytes and split
reports separately.

For `ImplicitMF` popularity training only, a further pre-allocation cap allows
at most 64 factors, 65,536 latent coordinates and 50 million planned
pair-coordinate updates. The two other models retain their existing factor and
training-work caps.

Synthetic experiment and shared-split benchmark:

The fixture has four training clicks on `b` but one on `a` and `c`, so
user `u1`'s two eligible negatives have probabilities `4/5` and `1/5` at
`alpha=1`; the held-out click is never counted.

```bash
orchidrec run examples/training_sampling_config.json
orchidrec recommend artifacts/training-sampling/model.json u1 --k 2
orchidrec benchmark examples/training_sampling_benchmark_config.json \
  --output-dir artifacts/training-sampling-benchmark
```

The benchmark compares two local sampling policies, not official RecBole or
LensKit results. Train/test chronology and source-feature availability remain
the caller's responsibility.
