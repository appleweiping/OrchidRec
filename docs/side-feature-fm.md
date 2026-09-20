# Side-feature factorization machine

`SideFeatureFM` is the first OrchidRec recommender that consumes the typed
feature pipeline. It is a compact, deterministic CPU implementation of a
pairwise Bayesian-personalized-ranking factorization machine. It is **not**
a claim of model-family or benchmark equivalence with RecBole.

## Run it

```bash
orchidrec run examples/side_feature_fm_config.json
orchidrec inspect artifacts/side-feature-fm/model.json
orchidrec recommend artifacts/side-feature-fm/model.json u1 --k 3
```

The example uses checked-in synthetic interactions and separate user/item
metadata. The experiment splits interactions first, selects only feature rows
for users/items present in the training split, fits `FittedFeaturePipeline` on
that selection, and never refits vocabulary or normalization on the holdout.
This guards against split-derived feature-statistic leakage; it does **not**
prove that the source metadata itself was available at the prediction time.
Callers must exclude future-derived or otherwise point-in-time-unsafe values.
`data.features_path` is required only for this model. Saved models embed the
fitted pipeline, encoded training-side rows, their checksums, and FM weights;
serving therefore does not need the original feature file.

## Model and objective

Each `(user, item)` pair becomes a sparse vector `x`. It has a one-hot user
ID coordinate, a one-hot item ID coordinate, and coordinates derived from
their side-feature rows. Token features use fitted vocabulary indices;
token sequences contribute `count(token) / retained_length` to each token
coordinate; normalized floats use one coordinate; numeric sequences use one
coordinate per retained position. Padding is zero. User/item feature names
are globally unique in the schema. Interaction-context features are rejected:
`Interaction` has no stable event ID for a defensible join.

The score is

```text
s(x) = sum_i w_i x_i + 1/2 sum_f [ (sum_i V_if x_i)^2 - sum_i (V_if x_i)^2 ]
```

For each unique positive user/item pair, each epoch samples one unseen item
from that user's training catalog and performs an SGD ascent step on
`log sigmoid(s(positive) - s(negative))` with L2 shrinkage on coordinates
active in either pair. Pair order, negative sampling, and initialization use
a model-local seed. Repeated positive events collapse to one pair; event
`value` is not a confidence weight. A user with no unseen item is skipped;
at least one update must be possible. Unknown users use the existing
popularity fallback. Unknown items remain outside the fitted catalog.

This is pairwise implicit-feedback ranking, not a calibrated probability,
explicit-rating regression, causal effect estimate, or cold-item model.
The overall score is not constrained to be positive.

## Failure and resource boundaries

Training enforces at most 2,048 users, 2,048 items, 100,000 events, 8,192
sparse dimensions, 128 active coordinates per entity, 64 latent factors,
100 epochs, and a configurable upper bound on pair-update work. Feature
vectors must be finite and have magnitude at most 1,000 per coordinate.
User/item IDs are limited to 512 integer bits or 2,048 UTF-8 bytes; fitted
weights are limited to magnitude 1,000,000.
The reproducibility seed must fit a signed 64-bit integer.
Constructor parameters, feature ID joins, schema, serialized dimensions,
padding, pipeline identity, and encoded-row digest are checked. The model
file reader also has a 256 MiB byte ceiling. In-memory computation is not
distributed, and the global feature loader's own ceilings still apply.

The loss is non-convex and this bounded implementation does not certify a
global optimum. Learning rate and data scaling affect convergence; a
non-finite update fails rather than emitting a misleading model.

## Verification

Tests use hand-computed two-coordinate FM scores and one-step BPR gradients,
including L2 shrinkage. They also check actual token/sequence/float effects,
repeatability, training-only pipeline selection, saved-model round trips,
malformed state, invalid joins, unsupported context, and work ceilings.
