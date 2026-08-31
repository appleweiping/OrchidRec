# OrchidRec

[![CI](https://github.com/appleweiping/OrchidRec/actions/workflows/ci.yml/badge.svg)](https://github.com/appleweiping/OrchidRec/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](https://www.python.org/)
[![MIT](https://img.shields.io/badge/license-MIT-2ea44f.svg)](LICENSE)

OrchidRec is a compact, independent toolkit for reproducible offline
recommendation experiments. It runs on Python 3.11 or newer and has **no
runtime dependencies outside the Python standard library**.

It is intentionally inspectable while providing a complete experimental path:
strict interaction validation, local MovieLens adapters, content fingerprints,
deterministic ID mapping, three train/test split strategies, three
recommenders, six ranking metrics, user bootstrap intervals, paired model
comparisons, portable JSON model state, and JSON/CSV/standalone-HTML reports.

## Quick start

```bash
python -m pip install -e .
orchidrec demo --output-dir artifacts/demo
orchidrec inspect artifacts/demo/model.json
orchidrec recommend artifacts/demo/model.json u1 --k 3
```

The demo writes its input data, experiment configuration, fitted
model, and evaluation report into one directory. To run the checked-in
example instead:

```bash
orchidrec run examples/config.json
```

To exercise the shared-split benchmark runner on the checked-in synthetic
example:

```bash
orchidrec benchmark examples/benchmark_config.json --output-dir artifacts/benchmark
```

Open `artifacts/benchmark/benchmark.html` directly in a browser; it has no
external scripts, fonts, or network requests.

## Real MovieLens benchmark

OrchidRec does not download or redistribute MovieLens. Download either
MovieLens 100K or MovieLens 1M from the official
[GroupLens dataset page](https://grouplens.org/datasets/movielens/) yourself,
read its license/readme, and extract it locally. The adapters accept either the
ratings file or its containing directory:

```bash
orchidrec dataset-summary /datasets/ml-100k --format movielens-100k --minimum-rating 4
orchidrec dataset-summary /datasets/ml-1m --format movielens-1m --minimum-rating 4
```

The 100K adapter reads `u.data` (`user<TAB>item<TAB>rating<TAB>timestamp`);
the 1M adapter reads `ratings.dat` (`user::item::rating::timestamp`). Parsing
is strict: malformed fields, out-of-range ratings, duplicate user-item rows,
blank records, and non-ASCII ratings data fail with a line-specific error.
Ratings at or above the configured threshold become unit-valued positive
events; ratings below it are counted as dropped, not interpreted as explicit
negatives.

Create `movielens-benchmark.json`:

```json
{
  "schema_version": 1,
  "seed": 2026,
  "data": {
    "path": "/datasets/ml-100k",
    "format": "movielens-100k",
    "minimum_rating": 4
  },
  "split": {"method": "leave_one_out", "test_ratio": 0.2},
  "evaluation": {
    "k": 10,
    "exclude_seen": true,
    "bootstrap_samples": 1000,
    "confidence": 0.95
  },
  "models": [
    {"label": "popularity", "name": "popularity", "params": {"weighted": false}},
    {"label": "item-knn", "name": "item_knn", "params": {"neighbors": 40, "shrinkage": 10.0}},
    {"label": "bpr-mf", "name": "implicit_mf", "params": {"factors": 16, "epochs": 5}}
  ]
}
```

Then run:

```bash
orchidrec benchmark movielens-benchmark.json --output-dir artifacts/ml-100k
```

The output directory contains:

- `benchmark.json`: the complete versioned result, effective model parameters,
  dataset/split/config SHA-256 fingerprints, timings, intervals, and pairwise
  statistics;
- `benchmark.csv`: tidy rows suitable for a spreadsheet or downstream
  analysis; and
- `benchmark.html`: an offline dashboard showing estimates, uncertainty, and
  every pairwise comparison.

Absolute data paths are deliberately excluded from the semantic configuration
fingerprint. Identical bytes at two locations therefore identify the same
experiment. The report retains only the source filename, byte hash, normalized
interaction hash, counts, rating range, and timestamp range.

## Architecture

```mermaid
flowchart LR
    A[JSON or local MovieLens ratings] --> B[Strict adapter and immutable dataset]
    B --> B2[Source and normalized SHA-256]
    B --> C[Stable user and item ID maps]
    C --> D{Split strategy}
    D -->|random| E[Train / test]
    D -->|temporal| E
    D -->|leave-one-out| E
    E --> F{Model}
    F -->|Popularity| G[Candidate scores]
    F -->|ItemKNN| G
    F -->|ImplicitMF / BPR| G
    G --> H[Seen-item filter and stable Top-K]
    H --> I[Precision Recall NDCG MRR]
    H --> J[Coverage Novelty]
    I --> K[User bootstrap and paired comparisons]
    J --> K
    K --> M[JSON / tidy CSV / standalone HTML]
    F --> L[Versioned JSON model]
```

The modules have deliberately narrow responsibilities:

- `data.py` validates events and maps string/integer IDs to deterministic
  contiguous indices.
- `datasets.py` strictly parses local datasets and records content-addressed
  provenance without downloading data.
- `split.py` partitions events without dropping or duplicating records.
- `models/` owns fitting, scoring, Top-K ranking, cold-start fallback, and
  versioned serialization.
- `metrics.py` evaluates plain recommendation/relevance mappings and can be
  used independently of the models.
- `config.py` parses strict JSON and resolves relative paths from the config
  file location.
- `experiment.py` joins the pieces without adding wall-clock timestamps or
  other nondeterministic report fields.
- `benchmark.py` fits multiple models on exactly one split and one shared
  bootstrap resampling plan.
- `statistics.py` provides deterministic percentile intervals and paired mean
  comparisons independently of the recommender classes.
- `reporting.py` atomically writes strict JSON, tidy CSV, and self-contained
  HTML artifacts.
- `cli.py` exposes the runner and saved models to shell workflows.

## Interaction data

An input file is a JSON array. Each object has two required fields and two
optional fields:

```json
[
  {"user_id": "alice", "item_id": "article-1", "value": 1.0, "timestamp": 1710000000},
  {"user_id": 42, "item_id": 1007, "value": 2.0, "timestamp": 1710000100}
]
```

| Field | Type | Meaning |
| --- | --- | --- |
| `user_id` | non-empty string or integer | Stable user identifier; booleans are rejected. |
| `item_id` | non-empty string or integer | Stable item identifier; integers and numeric strings remain distinct. |
| `value` | finite number greater than zero | Positive implicit-feedback strength; default `1.0`. |
| `timestamp` | finite number or `null` | Ordering value used by temporal splits. |

Unknown fields are rejected. Repeated user-item events are allowed: Popularity
and ItemKNN aggregate their values, while ImplicitMF treats the pair as one
positive preference. A user-item pair is never split across training and test,
so repeated events cannot leak the evaluation target into model fitting.

`StableIdMap` sorts integer IDs numerically before string IDs, which are sorted
lexicographically. The mapping is therefore independent of input row order and
can be serialized without relying on hash iteration order.

## Split strategies

- `random` shuffles user-item groups with a local `random.Random(seed)` and
  chooses the group boundary closest to the requested ratio. Original event
  order is preserved inside both partitions and at least one pair remains on
  each side.
- `temporal` holds out the newest suffix at the closest boundary that does not
  cut a repeated user-item pair. Every event must have a timestamp; input
  position resolves equal timestamps. It fails explicitly if no safe boundary
  exists.
- `leave_one_out` selects each multi-item user's latest item and holds out all
  events for that item. If any event for that user lacks a timestamp, original
  input order is the stable fallback. Users with only one distinct item stay
  in training.

Random and temporal splits use `test_ratio`. Leave-one-out ignores the ratio.

## Models

### Popularity

Ranks by summed event values (`weighted: true`) or event count. It is a useful
sanity baseline and also supplies cold-start rankings to the personalized
models.

### ItemKNN

Builds weighted user vectors, computes cosine similarities for co-observed
items, adds non-negative shrinkage to the cosine denominator, and retains the
strongest configured neighbors per item. A known user's score is a
similarity-weighted history sum. Unknown users receive the popularity fallback.

Training cost is approximately `sum_u |history(u)|²`; this implementation is
designed for learning and small/medium experiments rather than large catalogs.

### ImplicitMF

Learns user and item latent factors with Bayesian Personalized Ranking (BPR):
each positive user-item pair is contrasted with seeded negative samples, then
updated by pairwise stochastic gradient descent. Users who have seen every
catalog item are skipped safely during negative sampling. Unknown users receive
the popularity fallback.

Training cost is roughly
`epochs × positives × negative_samples × factors`. All initialization,
shuffling, and negative sampling use a model-local seeded generator.

## Top-K behavior

All models share the same ranking implementation:

1. validate that candidates belong to the fitted catalog;
2. compute finite scores;
3. remove already-seen items by default;
4. sort by descending score, then stable item ID;
5. return one-based `Recommendation(item_id, score, rank)` objects.

Pass `exclude_seen=False` in Python or `--include-seen` to the CLI when a
diagnostic needs the full catalog.

## Metrics

Metrics are macro-averaged across users with at least one relevant test item.
Test interactions for items absent from the training catalog are counted as
`cold_start_test_interactions` in the report and excluded from ranking metrics;
the models cannot score an item they have never fitted. The evaluated and raw
test sizes are both reported, and an all-cold-start test set is rejected.

- **Precision@K**: relevant recommendations divided by `K`; short lists are
  not given a smaller denominator.
- **Recall@K**: recovered relevant items divided by all relevant test items.
- **NDCG@K**: binary discounted gain normalized by the ideal ranking.
- **MRR@K**: reciprocal rank of the first relevant item.
- **Coverage@K**: fraction of the training catalog exposed across all lists.
- **Novelty@K**: mean `-log2(p(item))`, with Laplace-smoothed training event
  frequencies so unseen catalog items have a finite value.

Duplicate items anywhere in a ranking, invalid ID types, duplicate catalog
entries, and recommendations outside the declared catalog are validation
errors instead of being silently counted—even when the invalid entry appears
beyond the requested cutoff.

### Uncertainty and paired comparisons

The benchmark runner resamples evaluated users with replacement using a local
seeded generator. For every resample it recomputes all six aggregates,
including the union-based catalog coverage and recommendation-weighted
novelty. The JSON report records percentile confidence intervals around each
point estimate.

Every model pair uses the same resampled user indices. Differences are
reported as `right - left`, together with their percentile interval, the
bootstrap probability that the right model is better, and a finite-sample
corrected exploratory two-sided p-value. These values quantify uncertainty in
this particular offline sample; they are not a substitute for multiple-test
correction, online experiments, or a causal claim.

## Experiment configuration

```json
{
  "seed": 2026,
  "data": {"path": "interactions.json"},
  "split": {"method": "leave_one_out", "test_ratio": 0.2},
  "model": {
    "name": "item_knn",
    "params": {"neighbors": 40, "shrinkage": 10.0}
  },
  "evaluation": {"k": 10, "exclude_seen": true},
  "output": {
    "model_path": "artifacts/model.json",
    "report_path": "artifacts/report.json"
  }
}
```

Paths are relative to the configuration file, not the caller's current
directory. Unknown sections, keys, model names, parameters, and invalid numeric
values fail early. Supported model parameters are:

| Model | Parameters |
| --- | --- |
| `popularity` | `weighted` |
| `item_knn` | `neighbors`, `shrinkage` |
| `implicit_mf` | `factors`, `epochs`, `learning_rate`, `regularization`, `negative_samples`, `seed` |

When `implicit_mf.seed` is absent, the experiment-level seed is used.

## CLI

```text
orchidrec run CONFIG
orchidrec demo [--output-dir DIRECTORY]
orchidrec inspect MODEL
orchidrec recommend MODEL USER_ID [--k K] [--include-seen]
orchidrec dataset-summary DATA --format FORMAT [--minimum-rating RATING]
orchidrec benchmark CONFIG [--output-dir DIRECTORY]
```

`USER_ID` accepts a JSON scalar. `42` is an integer ID, `"42"` is a string ID,
and a plain non-JSON token such as `alice` is treated as a string. Expected
data/config/model errors are printed to standard error with exit code `2`.

## Python API

```python
from orchidrec import Interaction, InteractionDataset, ItemKNN

events = InteractionDataset([
    Interaction("u1", "a"),
    Interaction("u1", "b"),
    Interaction("u2", "a"),
    Interaction("u2", "c"),
])

model = ItemKNN(neighbors=10, shrinkage=1.0).fit(events)
for recommendation in model.recommend("u1", k=5):
    print(recommendation.rank, recommendation.item_id, recommendation.score)
```

For portable persistence:

```python
from orchidrec.models import load_model, save_model

save_model(model, "model.json")
restored = load_model("model.json")
```

The JSON envelope records a format marker, schema version, model type,
hyperparameters, catalog, seen-item state, and fitted numeric state. Loading is
strict: unknown fields, invalid IDs, bad dimensions, non-finite numbers, and an
unsupported version are rejected. It does not execute serialized code.

For a programmatic real-data benchmark:

```python
from orchidrec import load_benchmark_config, run_benchmark
from orchidrec.reporting import save_benchmark_reports

config = load_benchmark_config("movielens-benchmark.json")
result = run_benchmark(config)
paths = save_benchmark_reports(result, "artifacts/ml-100k")
print(result.dataset.interactions_sha256, paths.html_path)
```

## Reproducibility contract

For the same validated input order, configuration, and supported Python
runtime, OrchidRec guarantees deterministic partitions, ID maps, model-local
sampling, bootstrap resamples, tie-breaking, fingerprints, metric estimates,
and serialized key/order layout. Floating-point values can still differ at the
final bits across unusual hardware or Python math implementations; comparisons
across platforms should use tolerances.

The single-model runner deliberately excludes elapsed time, current time, host
names, and random run IDs. The benchmark runner records measured fit and
recommendation seconds because performance comparison requires them; timings
are observational and are the only intentionally nondeterministic result
fields. Neither runner changes Python's process-global random state.

## Scope and limitations

- Data and model state are held in memory; the adapters are not streaming ETL.
- Feedback is positive/implicit; zero and negative values are rejected.
- MovieLens rating thresholding discards lower ratings rather than learning
  from them, and this toolkit does not predict explicit star ratings.
- ItemKNN uses dense per-user pair enumeration and ImplicitMF uses simple SGD,
  not optimized native kernels. Full MovieLens 1M runs can therefore be slow.
- There is no feature store, distributed execution, online serving layer, or
  hyperparameter search.
- Offline holdout metrics assume unobserved items are candidates and do not
  correct for exposure/selection bias. Bootstrap intervals treat users as the
  resampling unit and do not model temporal or social dependence.
- Recorded wall-clock timing depends on the host, Python build, background
  load, and filesystem cache. Compare timing only under a controlled protocol.
- Saved JSON models can be large. Validate file provenance and apply ordinary
  resource limits when loading untrusted inputs.

These boundaries keep the implementation inspectable and dependency-free.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check src tests examples
mypy src
python -m unittest discover -s tests -v
coverage run -m unittest discover -s tests && coverage report
python -m compileall -q src tests examples
orchidrec demo --output-dir artifacts/smoke
orchidrec benchmark examples/benchmark_config.json --output-dir artifacts/benchmark-smoke
python -m build
```

The test suite covers validation failures, deterministic splitting and
training, ranking semantics, all metrics, strict configuration, serialization
tampering, MovieLens format failures, content hashes, deterministic bootstrap
statistics, paired comparisons, portable report formats, CLI exit behavior,
and end-to-end runs for every included model.
See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes.

## Algorithm reference

OrchidRec's implementation and public interfaces are independent. The
`ImplicitMF` pairwise objective follows Rendle et al., “BPR: Bayesian
Personalized Ranking from Implicit Feedback” (UAI 2009, arXiv:1205.2618).
The citation identifies the published algorithm; no external project code is
included.

## License

OrchidRec is available under the [MIT License](LICENSE).
See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CHANGELOG.md](CHANGELOG.md) for project policies and release history.
