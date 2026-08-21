# Existing-800 SLAM-risk predictability audit

Date: 2026-08-21

## Decision

The frozen decision is **FAIL** for the proposed joint 0.5 s hazard plus 1.0 s recovery
predictor.  This stops that exact new-C route before Isaac/PPO training; it does not invalidate
the existing confidence estimator, deterministic supervisor B, or the formal 800-run result.

The useful result is narrower.  Deployable motion history contains repeatable information about
near-term tracking loss, while the existing artifacts do not support a reliable learned recovery
probability.  The next architecture must therefore keep deterministic invalid/stale/recovery
safety and must not train the originally proposed joint hazard/recovery student unchanged.

## Data and leakage boundary

The audit reads only accepted `publication_run_record.json`, `policy_diagnostics.json`, and
`offline_usability.json` artifacts.  It does not read bags, run ROS or Isaac, use ground truth as
a runtime feature, train PPO, or modify a policy.

- A is excluded because its 48-D diagnostics do not contain the three SLAM-state inputs.
- B/C/D provide 600 eligible scheduled runs; 596 yield at least one complete causal prediction
  window.  Four runs have no eligible window after the 1 s history and future-label censoring
  rules.
- The final dataset has 108,866 prediction ticks sampled at 10 Hz.  Ticks may train the model,
  but inference for model differences is clustered by complete `(paired_block_id, profile)`;
  ticks are not treated as independent experiments.
- Histories contain only data at or before the prediction tick.  Duplicate policy-clock records
  keep the last record at the tick.  A backwards clock is a hard error.
- Existing 800 outcomes are development data after this audit and cannot be reused as a blind
  confirmatory test for a future policy.

The three fixed input comparisons are:

1. SLAM-state history: confidence, tracking-valid, and normalized age;
2. SLAM state plus angular velocity, gravity, command, joint position/velocity, and previous
   action, excluding estimator linear velocity;
3. the same motion history plus estimator15 linear velocity.

Each channel is summarized over a causal 1 s window by current value, mean, standard deviation,
minimum, maximum, and slope.  The predictor is L2-regularized logistic regression (`C=0.01`),
with standardization fit on the training split only and equal total training weight per run.
Evaluation uses five-fold complete-block holdout and leave-one-arm-out B/C/D generalization.

## Frozen gate

The core comparison excludes estimator linear velocity.  PASS required lower Brier score and
log loss, no ECE worsening above 0.02, and a cluster-bootstrap Brier interval below zero in every
backend, endpoint, and split scheme.  Credible Brier worsening or ECE worsening above 0.02 in any
cell is FAIL; all other outcomes are INCONCLUSIVE.  The rule was not relaxed after seeing results.

## Results

Values below are equal-run mean Brier differences, motion history minus SLAM-state history;
negative values favor motion history.  Intervals use 10,000 `(block, profile)` cluster bootstrap
resamples.

| split | backend | endpoint | Brier difference [95% CI] | result |
|---|---|---|---:|---|
| block 5-fold | FAST-LIO2 | hazard 0.5 s | -0.01273 [-0.02411, -0.00149] | credible improvement |
| block 5-fold | FAST-LIO2 | recovery 1.0 s | +0.02456 [+0.00545, +0.04395] | credible worsening |
| block 5-fold | LIO-SAM | hazard 0.5 s | -0.02373 [-0.05212, +0.00491] | calibration fails gate |
| block 5-fold | LIO-SAM | recovery 1.0 s | +0.00461 [-0.01306, +0.02354] | calibration fails gate |
| leave one arm out | FAST-LIO2 | hazard 0.5 s | -0.02560 [-0.03417, -0.01702] | credible improvement |
| leave one arm out | FAST-LIO2 | recovery 1.0 s | -0.00910 [-0.02200, +0.00410] | calibration fails gate |
| leave one arm out | LIO-SAM | hazard 0.5 s | -0.08866 [-0.11079, -0.06834] | credible improvement |
| leave one arm out | LIO-SAM | recovery 1.0 s | -0.01258 [-0.02634, +0.00149] | calibration fails gate |

Hazard discrimination improves materially: block-holdout AUROC changes from `0.8763` to
`0.9112` for FAST and from `0.7048` to `0.8248` for LIO.  Recovery is not stable: FAST block
holdout AUROC falls from `0.6523` to `0.6306`, and its Brier score worsens credibly.  Adding
estimator15 linear velocity changes results only slightly and does not repair recovery; therefore
the evidence does not justify making the new risk model depend on estimator velocity.

## Consequence for the next method

- Do not implement or train the proposed joint hazard/recovery temporal student as new C.
- Keep the existing confidence state and deterministic supervisor for invalid, stale, and
  recovery behavior.
- Preserve the 0.5 s motion-conditioned hazard signal as a development finding, not a causal or
  publication claim.  A redesigned method may use a hazard-only auxiliary, but it needs a new
  frozen test—especially LIO block-level calibration—before any policy training.
- Do not delete the existing confidence artifacts or validation logs.  They are the evidence that
  stopped this route before another long training/live campaign.
- No model promotion, default switch, 3,200 replay, commit, or push is authorized by this audit.

The machine-readable report and feature cache are ignored runtime artifacts under
`outputs/slam_confidence_publication_v1/formal_analysis_v1/risk_predictability_audit_v1/`.
The reproducible entry point is `scripts/validation/audit_slam_risk_predictability.py`.
