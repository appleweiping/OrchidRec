# Sequential backoff recommender

`SequentialBackoff` is a bounded second-order item-transition model. It uses
only fitted training interactions, orders each user's events by timestamp
(original input position breaks ties), and rejects missing timestamps. The
saved model retains each fitted user's last two items. For an unseen user, it
uses the existing normalized training-popularity fallback.

## Run it

```bash
orchidrec run examples/sequential_backoff_config.json
orchidrec inspect artifacts/sequential-backoff/model.json
orchidrec recommend artifacts/sequential-backoff/model.json u1 --k 3
orchidrec benchmark examples/sequential_backoff_benchmark_config.json \
  --output-dir artifacts/sequential-backoff-benchmark
```

The benchmark runs both first-order `SequentialMarkov` and this model on the
same synthetic split. It demonstrates integration, not superiority or parity
with a published benchmark. The example has too few events for meaningful
performance estimates.

## Score and interpretation

Let `C1[b,c]` be the sum of target-event weights for transitions `b -> c`;
`C2[a,b,c]` is the same for triples `a,b -> c`. With `weighted=false`, each
transition/triple instead contributes one. For the user's last two items
`(a,b)`, define empirical conditional distributions `P1(c|b)` and
`P2(c|a,b)`. The second-order support is `s = sum_c C2[a,b,c]` and its
confidence is `lambda = s / (s + backoff_strength)`. The sequence component
is `lambda * P2 + (1 - lambda) * P1`. If there is no second-order context,
it uses `P1`; if no first-order context exists, it uses normalized training
popularity. The final score is `(1 - popularity_mix) * sequence +
popularity_mix * popularity`. Zero `backoff_strength` uses a known second-order
context without shrinkage.

These are ranking scores, not calibrated future-event probabilities. They do
not model time gaps, repeated-event sessions, exposure, sequence embeddings,
or higher-order neural architectures such as GRU4Rec/SASRec. Holdout selection
is still the caller's responsibility: use a chronological or per-user
leave-one-out split for a temporal evaluation, and avoid future-derived source
data. A random split may place future user events into training, even though
the model orders its fitted events internally.

## Bounds and verification

Training accepts at most 50,000 interactions, 5,000 catalog items, 50,000
users, and total positive event weight at most `1e15`. All events need finite
timestamps and positive finite weights. Constructor parameters and loaded
JSON state are validated; the state uses canonical stable ID ordering and
checks that aggregated second-order counts do not exceed their corresponding
first-order transitions. This is structural validation only: a model file
cannot reconstruct the original training-event order or prove dataset origin.
It contains no source-data hash or provenance signature. The generic model
file loader enforces its byte
ceiling. This is an in-memory Python model, not a streaming or native-kernel
implementation.

Tests include hand-counted conditional probabilities (including weighted and
unweighted targets), unseen-context backoff, timestamp ties, malformed state,
resource bounds, saved-model round trips, and actual experiment/benchmark
consumers.
