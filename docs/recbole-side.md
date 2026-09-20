# Local `.user` / `.item` side-feature interchange

OrchidRec imports locally supplied tab-separated RecBole-style atomic user
and item tables into its own typed `FeatureDataset`. This is a strict interchange
adapter, not the RecBole dataset registry. It does not download or redistribute
datasets and does not implement `.kg`, `.link`, `.net`, interaction joins, or a
feature-aware recommender.

```bash
orchidrec import-recbole-features \
  --user examples/recbole-side-synthetic.user \
  --item examples/recbole-side-synthetic.item \
  --output artifacts/side-features.json

orchidrec fit-features --input artifacts/side-features.json \
  --input-format recbole-side --output artifacts/side-pipeline.json
orchidrec transform-features --input artifacts/side-features.json \
  --input-format recbole-side --pipeline artifacts/side-pipeline.json \
  --output artifacts/side-encoded.json
```

The sample files are original, synthetic data. `--user` and `--item` are
independent optional inputs; at least one is required. Each must be an actual
local file with the matching suffix. The first physical line is a tab-separated
header `name:type`; user rows require `user_id:token` and item rows require
`item_id:token`, each exactly once. At least one additional feature is required
per file. Feature names are at most 64 ASCII letters, digits, and underscores,
starting with a letter. Other columns may use `token`, `float`, `token_seq`, or
`float_seq`; unsupported types and duplicate names fail. IDs and token fields
stay strings: `001` and `1` are distinct. Floats must be finite decimal
numbers. Sequence cells are space-delimited with no repeated/leading/trailing
spaces; an empty cell is an empty sequence. Scalar cells cannot be empty.

Imported feature names are prefixed with `user.` or `item.` to preserve both
namespaces. For the training import, sequence width is the maximum observed
length (minimum one); the fitted pipeline uses `head` truncation. Therefore,
**import only training-side rows before fitting**. Import held-out rows
separately with `--schema-from artifacts/side-features.json`. This reuses the
training schema, binds its artifact checksum into held-out provenance, and
rejects fields with different names/types or sequences wider than the trained
width; it never infers a new width from held-out data or silently truncates.
Use that held-out artifact as the `transform-features --input`. Importing all
rows before a train/test split can leak vocabulary, normalization statistics,
and sequence widths. This adapter does not infer a split or silently join IDs
to interactions.

The importer accepts LF or CRLF and rejects bare CR, blank rows, malformed
UTF-8, wrong tab counts, duplicate IDs, and invalid values. Default caps are
16 MiB per source file, 64 KiB per physical line, 100,000 combined data rows,
128 combined feature fields, 4,096 values per sequence, 1,000,000 total values,
and 64 MiB per output artifact. Expanded row keys are preflighted against the
output cap before a complete JSON state is materialized. Matching `--max-*`
flags can tighten these limits within hard ceilings. The entire bounded source
snapshot is read once before parsing; nothing is downloaded.

The JSON artifact contains the existing `FeatureDataset` state, source byte
counts and exact-byte SHA-256 hashes, a canonical normalized dataset SHA-256,
resource limits, and a checksum over the envelope. It does not contain local
source paths. Renaming a source preserves the artifact; changing LF to CRLF
changes its raw hash but not the normalized dataset hash. Checksums detect
accidental corruption, not malicious provenance forgery. Output publication
is same-directory, atomic, and no-overwrite; existing output files are not
replaced, including aliases. The original feature-dataset JSON remains the
default input for `fit-features` and `transform-features`; select this artifact
explicitly with `--input-format recbole-side`.
