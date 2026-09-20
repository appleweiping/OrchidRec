# Changelog

Notable changes are recorded here. Versions follow semantic versioning.

## [Unreleased]

_No changes yet._

## [0.18.0] - 2026-09-20

### Added

- Added an original bounded full-batch bipartite graph BPR baseline with
  train-only normalized user–item propagation, exact reverse-mode gradients,
  deterministic excluded-positive negative sampling, strict model reload,
  independent hand and finite-difference oracles, and real experiment/shared
  benchmark consumers. It is LightGCN-style, not frozen framework/model parity.

## [0.17.0] - 2026-09-20

### Added

- Added a bounded deterministic knowledge-graph walk recommender that consumes
  local `.kg`/`.link` interchange, uses training-only user seeds, and supports
  declared relation weights and exact 1–3-hop propagation. Added strict
  standalone model state, importer-to-experiment and shared benchmark paths,
  hand-computed probability oracles, path/size guards, source-boundary checks,
  report digests, and runnable synthetic fixtures. This is not a learned
  KGCN/CKE/KGAT implementation or whole-comparator parity.

## [0.16.0] - 2026-09-20

### Added

- Added `SequentialBackoff`, a bounded deterministic second-order transition
  recommender with support-weighted first-order fallback, strict timestamp
  semantics and persisted state. Added hand-counted tests, experiment and
  shared-split benchmark integration, runnable synthetic examples and model
  documentation. This is not neural sequential recommendation or whole
  RecBole-family parity.

## [0.15.0] - 2026-09-20

### Added

- Added `SideFeatureFM`, a bounded deterministic pairwise factorization machine
  that uses the existing train-only typed user/item feature pipeline. Added
  experiment configuration, strict model persistence, a synthetic example,
  hand-gradient tests, and explicit join/leakage/resource contracts. This is
  one feature-aware model, not context-aware or full RecBole model parity.

## [0.14.0] - 2026-09-20

### Added

- Added a bounded deterministic nonnegative SLIM Elastic-Net item-item model
  with checked cyclic coordinate descent, strict canonical persistence,
  experiment/benchmark configuration, synthetic example, and hand-solved
  numerical regression tests. This is one compact-catalog model, not full
  RecBole model-stack parity.

## [0.13.0] - 2026-09-19

### Added

- Bounded local registry for multiple named RecBole-style atomic dataset
  directories. It composes `.inter`, `.user`, `.item`, `.kg`, `.link`, and `.net`
  adapters into a staged, checksummed manifest with deterministic dataset
  names, exact source-byte provenance, namespace-specific overlap reports,
  no-overwrite publication, and reimport verification.
- Added a two-dataset synthetic example, register/verify CLI commands, and
  independent/adversarial tests. This remains a narrow local interchange
  registry, not the full RecBole loading, filtering, model, or dataloader stack.

## [0.12.0] - 2026-09-19

### Added

- Bounded local directed `.net` source/target graph interchange with optional
  `.inter` user-overlap audit, exact source-byte and normalized-edge hashes,
  atomic no-overwrite artifacts, and strict CLI/package regressions.
- This is an OrchidRec-specific social-edge interchange, not a social
  recommender, graph-training model, or full RecBole dataset registry.

## [0.11.0] - 2026-09-19

### Added

- Added strict, bounded local RecBole-style `.kg` triple and `.link` item/entity
  interchange, preserving token IDs and one-to-one links without remapping or
  filtering virtual entities. Optional `.inter` and `.item` references record
  read-only catalog overlap with source and normalized-state provenance.
- Added canonical checksummed, atomic no-overwrite knowledge artifacts,
  shared save/load semantic validation, public-constructor corruption tests,
  synthetic CLI fixtures, and wheel/sdist smoke coverage. This does not train
  a knowledge-aware model, implement `.net`, or claim full RecBole registry
  parity.

## [0.10.0] - 2026-09-19

- Added strict local RecBole-style `.user` and `.item` side-feature import
  for token, float, token-sequence, and float-sequence columns, preserving
  string IDs and separate namespaces without joining interactions.
- Added bounded, checksummed, atomic no-overwrite artifacts and an explicit
  training-schema reference for held-out transforms; oversized expanded
  feature state is rejected before whole-state materialization.
- Added synthetic interchange examples, installed CLI smoke, and adversarial
  parser, provenance, schema-lock, and output-bound tests.

This is not RecBole's dataset registry or feature-aware model training;
`.kg`, `.link`, and `.net` remain unsupported.

## [0.9.0] - 2026-09-19

- Added a strict, bounded adapter for user-supplied RecBole `.inter` atomic
  interaction files, preserving token IDs including leading zeros and declared
  rating/timestamp semantics. Rated files require an explicit threshold;
  unrated files retain unit-valued implicit events without one.
- Added content-addressed source and normalized-event fingerprints, unambiguous
  directory lookup, exact tab-field validation, duplicate-pair checks, UTF-8
  and LF/CRLF handling, physical-row limits, and clear rejection of unsupported
  columns or bare CR line endings.
- Integrated the adapter into dataset-summary and benchmark configuration, with
  pure synthetic examples, independent MovieLens-layout cross-checks,
  hardlink/path-alias and malformed-input regressions, packaging manifest
  coverage, and Linux/Windows CLI smoke tests.
- Scope remains limited to `.inter`; RecBole `.user`, `.item`, `.kg`, `.link`,
  and `.net` atomic families are not yet supported.

## [0.8.0] - 2026-09-19

- Added deterministic, bounded uniform and training-popularity negative
  sampling without replacement, with a public sampler registry and auditable
  per-user candidate-set fingerprint.
- Added opt-in sampled evaluation to experiments and shared-split benchmarks,
  including validation-only tuning. All models on a split share candidates;
  training history and held-out positives cannot be mislabeled as negatives.
- Kept full-sort evaluation as the backward-compatible default. Sampled JSON,
  CSV, HTML, and CLI output explicitly identify the different candidate
  universe and warn against direct full-sort comparison.
- Tuned sampled benchmarks separately fingerprint the inner validation and
  outer test candidate pools, and label every CSV result by its actual pool.
- Require an explicit seen-history entry for every evaluated user, including
  an empty set for cold users, so omitted histories cannot leak positives
  into negative samples.
- Added hand-computed candidate and no-leakage tests, weighted-sampling
  frequency checks, resource and malformed-config tests, CLI examples,
  documentation, and CI smoke runs.

## [0.7.0] - 2026-09-19

- Added a genuine EASE closed-form recommender over deduplicated binary
  histories, with zero-diagonal item regression, popularity cold start, and
  deterministic scoring and persistence.
- Added a checked standard-library Cholesky inverse with symmetry,
  conditioning, finiteness, and residual verification, plus explicit catalog,
  interaction, state, and cubic-work limits.
- Integrated EASE into strict experiment and benchmark configuration,
  validation-only grid search, default seven-model comparisons, public imports,
  examples, and clean-install model loading.
- Added hand-computed coefficient and inverse oracles, duplicate/order
  semantics, malformed-state checks, resource-boundary tests, and end-to-end
  experiment and benchmark coverage.
- Hardened persisted models with a 256 MiB read ceiling, interpreter-independent
  JSON depth and integer limits, finite-number and Unicode-scalar validation,
  exact built-in integer boundaries, and allocation-first EASE base-state
  preflight. EASE loads now rebuild the closed-form solution once and reject
  even single-ULP coefficient or residual changes.
- Made weighted item popularity accurately rounded and input-order independent,
  including for extreme finite event weights, while bounding extra summation
  storage by the interaction count.
- Added input/output alias guards, atomic model and report replacement, and an
  independent branch-only coverage gate for CI and release.

## [0.6.0] - 2026-09-08

- Added immutable, typed user, item, and interaction feature schemas for token,
  numeric, token-sequence, and numeric-sequence data with strict shape, type,
  identity, and resource validation.
- Added a deterministic training-only preprocessing pipeline with type-aware
  token vocabularies, reserved padding/unknown indices, stable numeric
  normalization, configurable head/tail sequence truncation, fixed-width
  padding, and explicit real sequence lengths.
- Added canonical training provenance, checksummed/versioned fitted state,
  pipeline-bound encoded datasets, bounded row-wise atomic JSON persistence,
  and strict tamper, duplicate-field, byte-limit, and interruption handling.
- Added `fit-features` and `transform-features` workflows, checked-in train and
  validation examples, a complete feature contract guide, hand-computed
  correctness oracles, and clean-install feature smoke tests.

## [0.5.0] - 2026-09-07

- Added an independent confidence-weighted implicit-feedback ALS model with
  user/item alternating normal equations, deterministic local initialization,
  repeated-event confidence aggregation, and weighted-popularity cold start.
- Added a checked Cholesky solver with symmetry, conditioning, finiteness,
  overflow, and residual safeguards, plus explicit factor, epoch, confidence,
  state-size, interaction, entity, and work limits.
- Persisted a validated objective trace and added independent dense-objective,
  normal-equation-residual, and hand-solved-system oracles alongside order,
  seed, tamper, overflow, resource, experiment, CLI, and benchmark tests.
- Integrated ConfidenceALS into strict configuration, shared-split evaluation,
  validation-only grid search with multi-seed repeats, reports, public imports,
  portable model state, examples, and default six-model benchmarks.

## [0.4.0] - 2026-09-07

- Added a genuine user-user cosine KNN recommender with deterministic nearest-
  neighbor selection, shrinkage, popularity cold-start fallback, strict model
  state validation, and hand-computed similarity/ranking tests.
- Added a genuine first-order sequential Markov recommender over timestamped
  user histories. Weighted and unweighted transition probabilities, chronology
  requirements, equal-time ordering, cold start, tamper-resistant persistence,
  configuration, experiment, and benchmark integration are independently tested.
- Expanded the default shared-split benchmark from three to five models while
  retaining the same sealed-test, bootstrap, reporting, and release gates.

## [0.3.0] - 2026-09-07

- Added a tag-gated release pipeline with locked builds, clean wheel and sdist
  installation checks, CycloneDX SBOM, SHA-256 manifest, and GitHub provenance.
- Added optional, deterministic validation-only grid search for every built-in
  model. The outer test partition remains sealed while all candidates are fit
  on an inner training partition and selected on validation; selected
  parameters are then refit on training plus validation and evaluated on test
  once per model.
- Added explicit multi-seed validation repeats for ImplicitMF. Candidate scores
  are arithmetic means across the configured validation seeds, while the one
  final refit uses the benchmark's top-level seed. Deterministic models are not
  needlessly repeated.
- Tuned benchmark reports now retain the complete search space, every trial,
  selection metric and direction, timings, effective and final parameters, and
  content hashes for development, training, validation, test, configuration,
  and source/normalized data. Untuned configurations and reports retain their
  prior shape and behavior.
- Added exposure-corrected offline evaluation. `orchidrec.propensity` supplies an
  explicit exposure model -- a popularity model following Yang et al. (RecSys 2018),
  or `uniform_exposure` to state that none is assumed -- and `orchidrec.unbiased`
  supplies inverse-propensity recall and NDCG estimated under it. Declaring
  `evaluation.exposure` in a configuration adds an `unbiased_metrics` section to the
  experiment report; a configuration that does not declare one keeps exactly the
  report it produced before.
- The corrected estimators are normalized over the whole population rather than per
  user. A per-user ratio cannot correct anything: for a user with one observed
  interaction it is `w / w` on a hit and `0 / w` on a miss, so the weight cancels
  exactly, and strong exposure bias is the regime where most users have exactly one.
  The population-wide form therefore estimates a micro-averaged quantity, and
  evaluating under `uniform_exposure` reproduces the uncorrected micro metric exactly,
  which is the like-for-like baseline and is asserted in the tests.
- Measured against a known ground truth rather than argued: with relevance drawn
  independently of popularity and exposure made popularity-biased, the uncorrected
  recall of a popularity ranker overstates its true recall by 0.15 at moderate bias
  and 0.52 at strong bias; the corrected estimate removes 62% and 73% of that error.
  With unbiased logging, correction and no correction agree to twelve decimal places.
- Every corrected report carries the Kish effective sample size of its weights and the
  number of observations that sat on the propensity floor, because inverse weighting
  concentrates an estimate on rarely-observed items and an estimate resting on a few
  heavily weighted observations is differently untrustworthy rather than more
  trustworthy.
- Propensities are estimated from training popularity only. Estimating them from the
  holdout would let it explain its own sampling.

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
