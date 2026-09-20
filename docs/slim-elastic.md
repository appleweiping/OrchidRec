# SLIM Elastic-Net model contract

`SLIMElastic` is an original, dependency-free, small-catalog implementation of
nonnegative sparse linear item-to-item recommendation. It supports a bounded
subset of the general SLIM family, not the full RecBole training stack or its
scikit-learn implementation. There is no GPU, intercept, negative coefficient,
or weighted-regression mode.

## Data and objective

`X[u,i]` is `1` if the training partition contains any interaction for user
`u` and item `i`, otherwise `0`. Duplicate events and their positive values do
not change `X`. They do affect the shared weighted-popularity fallback for an
unknown user. IDs are ordered with OrchidRec's stable integer-before-string
order, so input-row order does not change the fitted state.

For each target item `j`, training finds a nonnegative vector `w_j` with
`w_j[j] = 0` to minimize

```text
0.5 * ||X[:,j] - X @ w_j||_2^2 + l1 * ||w_j||_1 + 0.5 * l2 * ||w_j||_2^2.
```

`l1 >= 0`; `l2 >= 1e-8` makes the permitted predictor subproblem strictly
convex. All coefficients are nonnegative; `w_j[j]=0` prevents self-copying.
For a known user, the candidate score is the sum of weights from their binary
history to that target. Seen-item exclusion, tie-breaking, and candidate
validation are the same as the other OrchidRec recommenders.

The implementation first forms `G=X.T X`. Each cyclic coordinate update is

```text
w_i <- max(0, (G[i,j] - sum_{k!=i} G[i,k] w_k - l1) / (G[i,i] + l2)).
```

It maintains the Gram residual incrementally, recomputes that residual from
the original matrix after each sweep, and checks the nonnegative KKT
conditions after each sweep. For an active coordinate, the violation is the
absolute gradient; for a zero coordinate it is the amount by which the
gradient is negative. The reported `largest_kkt` is the maximum of these
violations divided by `max(1, G[i,j])`. Thus `tolerance` is a *relative-to-
cross-product* bound for positive co-occurrences and an absolute bound where
the cross product is zero. A target that does not meet this declared check
within `max_sweeps` fails; there is no silent "converged" label.

## Resource and persistence policy

| Parameter | Default | Maximum | Meaning |
| --- | ---: | ---: | --- |
| `l1` | `0.1` | `1e12` | Nonnegative L1 penalty. |
| `l2` | `0.1` | `1e12` | Positive L2 penalty, minimum `1e-8`. |
| `max_sweeps` | `100` | `1,000` | Cyclic sweeps per target. |
| `tolerance` | `1e-7` | `0.1` | KKT bound, minimum `1e-8`. |
| `max_items` | `96` | `256` | Dense catalog width. |
| `max_interactions` | `2,000,000` | `2,000,000` | Raw events, including repeats. |
| `max_work_units` | `200,000,000` | `1,000,000,000` | Conservative deterministic work budget. |

Before allocating the Gram matrix, both fit and load enforce
`sum_u |history(u)|^2 + 2 * max_sweeps * items^3 <= max_work_units`.
This counts all permitted sweep iterations, not only iterations actually
used; it also bounds a loader's canonical re-solve. The base state permits at
most 32,768 users, 16 KiB per UTF-8 identifier, and 32 MiB across identifier
occurrences. The pretty-printed model must fit the common 256 MiB model-file
ceiling. This is a correctness-oriented CPU baseline, not a large-catalog
production solver.

The versioned JSON model stores the effective parameters, shared catalog and
history, dense zero-diagonal coefficients, sweeps per target, KKT diagnostic,
raw event count, and conservative work estimate. Loading validates sizes,
types, finite/nonnegative values, item references, and exact field sets before
reconstructing a canonical solution from the binary histories. The stored
coefficients and diagnostics must exactly match that solution. Model files are
data only; no executable deserialization is used. The weighted-popularity
base state is validated by the shared loader. Since duplicate raw events are
not retained in the binary history, loading can only check that the declared
raw event count lies between the unique-pair count and the configured raw
event ceiling; that field is not an independently verified source-provenance
claim.

## Usage and evidence

Run the included synthetic experiment with
`orchidrec run examples/slim_elastic_config.json`. Benchmarks accept an
explicit `"name": "slim_elastic"` specification and can tune `l1` and `l2`
using the existing inner split; it is not silently added to the default
benchmark suite. The test suite compares a two-item closed form and a coupled
three-item rational solution derived independently of coordinate descent. It
also checks deterministic row order, binary duplicates, nonconvergence,
resource limits, corrupted state, and save/load scoring parity. These are
algorithmic correctness tests, not an external RecBole performance claim.
