# Typed feature pipeline

OrchidRec's feature pipeline turns typed user, item, and interaction metadata
into deterministic numeric rows. Fitting and transformation are separate on
purpose: vocabularies and numeric statistics are learned from the training
split, persisted with a checksum, and then reused unchanged for validation,
test, or serving data.

The pipeline is independent from the built-in collaborative recommenders. It
prepares model-ready features and records their provenance; it does not yet
define a feature-aware ranking model or join feature rows to interactions.

## Data model

A `FeatureSchema` is an ordered collection of globally unique `FeatureSpec`
objects. Every feature declares an entity namespace and one of four physical
types:

| Kind | Accepted raw value | Encoded value |
| --- | --- | --- |
| `token` | one non-empty string or bounded integer | one vocabulary index |
| `float` | one finite integer or float | one normalized float |
| `token-sequence` | a bounded string/integer sequence | fixed-width index sequence |
| `float-sequence` | a bounded finite numeric sequence | fixed-width normalized sequence |

The entity namespace is `user`, `item`, or `interaction`. A row contains
exactly the schema fields declared for its namespace. Missing fields, unknown
fields, duplicate `(source, key)` identities, booleans, non-finite numbers,
oversized tokens, and mixed sequence types are rejected before fitting.

Sequence specifications also declare:

- `sequence_length`: the fixed encoded width; and
- `keep`: `head` or `tail`, selecting which values survive truncation.

Short sequences are padded on the right. Token sequences use index `0` as
padding. Numeric sequences use `0.0`, while a separate per-feature length map
distinguishes real zeroes from padding.

## Fit and transform contract

`FittedFeaturePipeline.fit(training)` performs the only learning step:

1. It snapshots and validates the complete training dataset.
2. For each token feature it collects unique training values and sorts them by
   a stable, type-aware order. Integer `7` and string `"7"` remain distinct.
3. It reserves index `0` for padding and index `1` for unknown values, then
   assigns learned values from index `2` onward.
4. For each numeric feature it computes population mean and population
   standard deviation with a scaled algorithm that remains finite for values
   near the floating-point range limits. Constant and empty numeric features
   use scale `1.0`.
5. It records the canonical SHA-256 identity of the training feature dataset.

`pipeline.transform(dataset)` requires the exact fitted schema and never
changes pipeline state. Tokens absent from training map to the unknown index;
numeric values use `(value - mean) / scale`; and sequences follow their
declared truncation and padding rules. The encoded dataset records the exact
pipeline-state SHA-256 that produced it.

This separation prevents a common offline-evaluation error: learning a token
vocabulary or normalization statistic from validation or test rows.

## Command line

The repository includes a training example and a validation example whose
country and topic values were not seen during fitting:

```bash
orchidrec fit-features \
  --input examples/features-train.json \
  --output artifacts/feature-pipeline.json

orchidrec transform-features \
  --pipeline artifacts/feature-pipeline.json \
  --input examples/features-validation.json \
  --output artifacts/features-validation-encoded.json
```

The first command prints the training-row/value counts, vocabulary size,
training digest, and fitted-state digest. The second prints the encoded row
count and producing pipeline digest. Input, pipeline, and output paths must be
different files, including hard-link aliases.

The fit command exposes the main work ceilings:

```text
--max-rows INTEGER
--max-total-values INTEGER
--max-vocab-values INTEGER
--max-vocab-token-bytes INTEGER
--max-state-bytes INTEGER
```

These ceilings are persisted in the pipeline. Transformation therefore uses
the same schema, token, row, sequence, vocabulary, and state limits that were
active during fitting. `--max-pipeline-bytes` independently bounds the initial
pipeline read.

## Python API

```python
from orchidrec import (
    FeatureDataset,
    FeatureKind,
    FeatureRow,
    FeatureSchema,
    FeatureSource,
    FeatureSpec,
    FittedFeaturePipeline,
    SequenceKeep,
)

schema = FeatureSchema(
    (
        FeatureSpec("country", FeatureKind.TOKEN, FeatureSource.USER),
        FeatureSpec("age", FeatureKind.FLOAT, FeatureSource.USER),
        FeatureSpec(
            "topics",
            FeatureKind.TOKEN_SEQUENCE,
            FeatureSource.ITEM,
            sequence_length=3,
            keep=SequenceKeep.HEAD,
        ),
    )
)

training = FeatureDataset(
    schema,
    (
        FeatureRow("user", "alice", {"country": "US", "age": 20}),
        FeatureRow("user", "bob", {"country": "CA", "age": 40}),
        FeatureRow("item", "story-1", {"topics": ["grid", "storage"]}),
    ),
)

pipeline = FittedFeaturePipeline.fit(training)
encoded_training = pipeline.transform(training)
pipeline.save("artifacts/feature-pipeline.json")
```

Load and apply the immutable fitted state later:

```python
from orchidrec import FittedFeaturePipeline, load_feature_dataset

pipeline = FittedFeaturePipeline.load("artifacts/feature-pipeline.json")
validation = load_feature_dataset(
    "examples/features-validation.json",
    limits=pipeline.limits,
)
encoded_validation = pipeline.transform(validation)
assert encoded_validation.pipeline_sha256 == pipeline.state_sha256
```

`save_feature_dataset`, `load_feature_dataset`, `save_encoded_features`, and
`load_encoded_features` provide strict versioned JSON interchange for the two
dataset forms.

## Persistence and provenance

All three formats have a fixed marker and `schema_version: 1`:

- `orchidrec.feature-dataset`
- `orchidrec.feature-pipeline`
- `orchidrec.encoded-features`

Pipeline state contains its schema, every active limit, learned vocabularies,
numeric statistics, training counts, training dataset digest, and a checksum
over the complete state. Unknown or missing JSON fields, duplicate object
keys, unsupported versions, invalid dimensions, and checksum mismatches fail
closed.

Raw and encoded datasets are serialized row by row through a byte-counted
staging file. A save that exceeds `max_state_bytes` never replaces the previous
target, and interrupted writes remove their staging file. The staging file is
flushed and synchronized before atomic replacement; POSIX directory entries
are synchronized after replacement. Reads consume at most one byte beyond the
declared limit before rejecting an oversized file.

The training dataset digest uses canonical compact UTF-8 JSON. Rows are sorted
by source and stable entity ID, object keys are sorted, strings are emitted
without ASCII escaping, and non-finite values are forbidden. Reordering valid
input rows therefore does not change a fitted pipeline.

## Resource boundaries

`FeatureLimits` applies explicit ceilings to schema fields, rows, total values,
values per sequence, token characters and UTF-8 bytes, integer bit width,
vocabulary count, cumulative vocabulary bytes, and serialized state bytes.
The implementation reads iterables only through `limit + 1`, so rejection does
not require consuming an unbounded source. Fitted constructors snapshot mutable
collections, and public state is exposed through tuples and read-only mappings.

The pipeline is intentionally in-memory. Row-wise serialization prevents a
second full JSON copy during saving, but the validated dataset and fitted
vocabularies themselves must fit in memory. It is not a distributed feature
store, categorical hasher, text tokenizer, missing-value imputer, or online
schema registry.

## Reproducibility checks

For a given supported Python runtime, identical validated training rows and
schema produce the same:

- canonical training digest;
- token indices;
- numeric statistics;
- fitted-state checksum; and
- encoded row order and values.

Tests include hand-computed vocabulary and normalization oracles, unseen-token
behavior, head/tail truncation, padding lengths, extreme finite numeric values,
order invariance, strict state tamper detection, bounded reads and writes,
interrupted replacement, path alias rejection, CLI round trips, and
wheel/sdist clean-install smoke runs.
