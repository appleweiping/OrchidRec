# Changelog

Notable changes are recorded here. Versions follow semantic versioning.

## [Unreleased]

_No changes yet._

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
