# Changelog

Notable changes are recorded here. Versions follow semantic versioning.

## [0.2.0] - 2026-08-31

- Added strict, local-only MovieLens 100K and 1M adapters with source and
  normalized-interaction SHA-256 fingerprints.
- Added a shared-split benchmark runner for Popularity, ItemKNN, and BPR-MF,
  including fit/recommendation timing and effective parameter capture.
- Added deterministic user bootstrap confidence intervals and paired
  comparisons for all six ranking metrics.
- Added versioned JSON, tidy CSV, and standalone HTML benchmark reports plus
  `benchmark` and `dataset-summary` CLI workflows.

## [0.1.0] - 2026-08-31

- Added deterministic interaction validation, ID mapping, and three split strategies.
- Added Popularity, ItemKNN, and BPR implicit-matrix-factorization recommenders.
- Added six ranking metrics, strict JSON configuration, versioned model state, CLI workflows, and examples.
