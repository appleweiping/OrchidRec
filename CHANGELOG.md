# Changelog

Notable changes are recorded here. Versions follow semantic versioning.

## [Unreleased]

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
