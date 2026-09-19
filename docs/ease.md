# EASE model contract

OrchidRec's `EASE` model is a deterministic, dependency-free implementation of
the closed-form shallow autoencoder for binary implicit feedback. This guide
states exactly what is optimized, how input events are interpreted, which
resource limits apply, and which checks make a fitted artifact acceptable.

## Data semantics

For each user `u` and catalog item `i`, OrchidRec constructs the binary matrix

```text
X[u, i] = 1  if at least one training event exists for (u, i)
          0  otherwise.
```

Event values and duplicate events do not increase `X[u, i]`. They still feed
the shared weighted-popularity state used for unknown-user fallback. This
distinction is intentional and is recorded in the public class documentation.
Weighted popularity uses a single accurately rounded sum per item, so
permuting finite positive events does not change the persisted fallback.
Summation holds at most one additional float reference per input event; EASE's
interaction limit bounds that extra memory.
The catalog and users use OrchidRec's stable integer-before-string ID order, so
row order does not change the learned matrix.

## Closed-form solution

For a positive regularization value `lambda`, training constructs

```text
G = X.T X + lambda I
P = inverse(G)
B[i, j] = -P[i, j] / P[j, j]  when i != j
B[i, i] = 0
```

The zero diagonal prevents an item from trivially reconstructing itself. For a
known user's binary history vector `x_u`, the candidate score is

```text
score(u, j) = sum_i x_u[i] B[i, j].
```

The common ranking layer validates the candidate set, optionally excludes
seen items, rejects non-finite scores, and breaks ties by stable item ID.
Unknown users receive normalized weighted-popularity scores.

## Checked linear algebra

The implementation does not import a native numerical package. It factors the
symmetric positive-definite system once with Cholesky decomposition, solves
one right-hand side per item, and verifies the result against the original
system. Training fails instead of publishing a model when it observes:

- a nonsymmetric, singular, non-positive, or relatively tiny pivot;
- a non-finite intermediate, inverse entry, coefficient, or norm;
- an inverse residual or symmetry error beyond a scale-aware tolerance; or
- a non-positive inverse diagonal used by the coefficient formula.

`inverse_residual` persists the maximum absolute entry of `G P - I`. It is a
diagnostic, not a claim that floating-point inversion is exact. The unit suite
checks the inverse of a hand-computed 2-by-2 system and separately checks the
closed-form coefficients and ranking of a tiny interaction matrix.

## Resource policy

Dense closed-form EASE is quadratic in fitted state and cubic in training time.
OrchidRec applies all configured limits before allocating `G`:

| Parameter | Default | Supported maximum | Meaning |
| --- | ---: | ---: | --- |
| `regularization` | `100.0` | `1e12` | Positive diagonal regularizer; minimum `1e-8`. |
| `max_items` | `256` | `512` | Maximum catalog width and coefficient dimension. |
| `max_interactions` | `2,000,000` | `2,000,000` | Maximum raw training events, including duplicates. |
| `max_work_units` | `100,000,000` | `1,000,000,000` | Conservative profile-enumeration plus cubic solve budget. |

The deterministic work estimate is

```text
sum_u |history(u)|^2 + 3 * items^3.
```

It is deliberately conservative and machine-independent; it is not elapsed
time. The pure-Python model is aimed at correctness experiments and compact
catalogs. Use an optimized sparse/native implementation when a production
catalog exceeds these declared boundaries.

Persisted EASE base state is also capped at 32,768 users, 16 KiB for one
UTF-8 identifier, and 32 MiB for all identifier occurrences. These hard
ceilings are checked on the raw arrays, together with interaction and work
budgets, before the shared base-state restorer allocates normalized copies.
Fitting applies the same limits before constructing the dense Gram matrix.
It also rejects identifiers outside the strict JSON integer/Unicode envelope
and counts the exact pretty-printed model bytes before reporting a successful
fit, so a fitted model remains within the 256 MiB loader ceiling.

## Configuration

A single experiment can select EASE directly:

```json
{
  "data": {"path": "interactions.json"},
  "model": {
    "name": "ease",
    "params": {
      "regularization": 100.0,
      "max_items": 256,
      "max_interactions": 2000000,
      "max_work_units": 100000000
    }
  }
}
```

A benchmark grid can tune the regularizer on its inner validation split:

```json
{
  "label": "ease",
  "name": "ease",
  "params": {"max_items": 256},
  "grid": {"regularization": [10.0, 100.0, 1000.0]}
}
```

EASE is deterministic, so each candidate runs once. Unlike the seeded latent
models, it does not consume `implicit_mf_seeds`. The selected candidate is
refitted on the complete outer development partition before the one final test
evaluation, using the same leakage-safe path as every other model.

## Persistence invariants

The versioned JSON envelope stores:

- exact effective hyperparameters;
- stable catalog, popularity, and user history state;
- a square finite coefficient matrix with an exactly zero diagonal;
- raw training interaction count;
- deterministic work estimate; and
- inverse residual.

Loading reconstructs no executable object graph. It rejects unknown fields,
wrong types, booleans used as integers, invalid dimensions, unknown seen items,
non-finite coefficients, a nonzero diagonal, inconsistent counts or work, and
resource limits smaller than the state they enclose. A successful load
reproduces the same score and ranking path as the original fitted model. The
loader derives the Gram matrix, inverse, coefficient matrix, work diagnostic,
and inverse residual once from the canonical catalog and binary histories. The
persisted coefficient matrix and residual must match that derivation exactly;
they are never trusted as an independent source of model behavior.

All model files have a 256 MiB read ceiling. Before decoding, strict JSON
validation enforces a fixed nesting limit, valid UTF-8 and Unicode scalar text,
bounded integers, finite floating-point values, and unique object keys. CLI
model commands report violations as domain errors with exit status 2.

## Reproducibility boundary

For the same validated training events on supported CPython runtimes, catalog,
binary histories, Gram matrix construction order, factorization order,
coefficient layout, and JSON state are deterministic. Exact reconstruction on
load is part of the artifact integrity boundary; an artifact produced by an
incompatible floating-point implementation must be retrained rather than
silently accepted through an attacker-expandable tolerance.

## Reference

The algorithm follows Harald Steck, “Embarrassingly Shallow Autoencoders for
Sparse Data,” WWW 2019, DOI `10.1145/3308558.3313710`. OrchidRec's code,
validation, persistence format, public API, tests, and resource policy are an
independent implementation; no external project source is included.
