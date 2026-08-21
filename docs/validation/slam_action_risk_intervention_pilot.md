# Action-conditioned SLAM-risk intervention pilot v1

Date: 2026-08-21

## Why this pilot exists

The existing confidence estimator already predicts 0.5 s future localization usability from
SLAM diagnostics.  The existing-800 audit found that locomotion history is associated with
future hazard, but passive prediction cannot establish that changing an action changes SLAM
risk.  Training another passive confidence model would duplicate the original objective.

This pilot therefore asks a smaller causal question before any new student or PPO training:

> With command, simulator seed, terrain, friction, LiDAR support schedule, and backend held
> fixed, does a bounded signed change to the locomotion action change body motion and the next
> 0.5 s localization hazard?

The machine-readable frozen design is
`configs/slam_action_risk_intervention_pilot.yaml`.

## Real SLAM boundary

This is not an Isaac-only synthetic-confidence experiment.  Isaac supplies the simulated dog,
contacts, IMU, and LiDAR; FAST-LIO2 or LIO-SAM must run through the existing ROS 2 Bridge live
chain.  Confidence and offline usability use the existing backend artifacts and GT labeling
boundary.  Isaac Python still does not import `rclpy`, and GT/future labels are never policy
inputs.

The formal model1450 remains frozen.  Every arm starts from publication Arm B, including its
deterministic invalid/stale behavior and estimator15.  No policy is trained in this pilot.

The `smooth`, `zero`, and `antismooth` counterbalanced experimental policies have been exported under
`exported/slam_action_risk_intervention_pilot_v1/`.  All three passed TorchScript/ONNX parity,
the zero arm is exactly Arm B on the parity sample, invalid inputs return exact Arm B for every
arm, and the realized residual respects the 0.05 raw-action bound.  This proves artifact wiring
only; no simulator or SLAM outcome has been collected yet.

## Intervention

At 50 Hz, while tracking is valid:

```text
delta(t) = clip(alpha * (previous_applied_action - ArmB_action(t)), -0.05, +0.05)
```

The three paired policy regimes are `smooth` (`alpha=+0.10`), exact `zero`, and
`antismooth` (`alpha=-0.10`).  The residual bound corresponds to at most 0.025 rad at the joint
target under the frozen action scale.  When tracking is invalid or stale, every arm returns the
exact Arm-B action and the residual is exactly zero.

The signed pair is a local action-rate intervention, not a proposed final gait controller.  Its
purpose is to determine whether a measurable action→body-motion→SLAM-risk path exists under the
actual backends.  Intervention identity is baked into the assigned arm and is not a runtime
policy feature.

## Timing and schedule

The previous formal challenge placed its nominal healthy window before the 5 s command warmup.
This pilot fixes that alias: support stays healthy for 7 s, giving an active-motion healthy
window from 5–7 s, followed by a 3 s ramp, 2 s low-support hold, and 2 s recovery.

Stages are deliberately small:

- wiring smoke: 6 runs (two backends × one profile × three arms × one block);
- pilot: 24 new runs (two backends × two profiles × three arms × two blocks);
- expanded stage: 72 runs only after an inconclusive pilot and without parameter changes
  (`expanded_only_after_pilot_inconclusive`).

Blocks 542–550 are disjoint from all calibration, excluded pilot, and formal blocks.  The smoke
cannot support an efficacy decision.  The 24-run pilot is a futility decision, not publication
evidence, and the existing 800 runs are not reused as a blind test.

## Decision boundary

The hazard endpoint uses only causal ticks where tracking is currently valid, motion is
requested, and the existing offline 0.5 s future-usability label is known.  Already-invalid
ticks are excluded; otherwise recovery persistence would be mislabeled as action-conditioned
onset risk.

PASS requires verified intervention separation, ordered action-rate and body-rate response,
non-worse smooth-arm tracking direction in both backends, non-higher hazard direction, no safety
failure, and moving-speed differences within 0.05 m/s of zero.  Realized speed is explicitly a
post-treatment variable; the speed check is an eligibility/interpretability guard, not a new
randomized estimand.

FAIL stops the action-conditioned new-C route if there is no realized action separation or if
the smooth regime makes both body rate and tracking survival worse in either backend.  All other
results are INCONCLUSIVE and permit only the predeclared 72-run expansion—no alpha, checkpoint,
or profile sweep.

Only a PASS permits design of an action-conditioned transition model.  It still would not prove
the final policy improves SLAM; that would require a separately frozen zero/shuffle/delay
closed-loop experiment.

## Reproducible entry points

Validate and render every predeclared schedule without starting ROS:

```bash
python3 scripts/validation/validate_slam_action_risk_intervention_protocol.py
python3 scripts/validation/run_slam_action_risk_intervention_pilot.py \
  --stage wiring_smoke \
  --output-root outputs/slam_action_risk_intervention_pilot_v1
```

After a stage has complete run records, the frozen automated decision is reproduced with:

```bash
python3 scripts/validation/analyze_slam_action_risk_intervention_pilot.py \
  --stage pilot \
  --input-root outputs/slam_action_risk_intervention_pilot_v1/pilot \
  --output outputs/slam_action_risk_intervention_pilot_v1/pilot/decision.json
```

The runner invokes the same analyzer automatically after an executed stage.  Incomplete,
duplicate, unexpected, non-finite, or contract-invalid run records produce FAIL rather than an
ad hoc partial comparison.

After a clean hash-locked commit is explicitly authorized, the six-run smoke uses the same
second command with `--execute`.  Execution also requires both
`boundaries.live_execution_authorized: true` and the requested stage in
`boundaries.authorized_stages` in the committed release config.  Plan mode never creates or
overwrites a `run_manifest.json`.  The runner refuses a dirty tracked worktree, project-owned
untracked source/config/docs, or an output stage containing earlier execution artifacts.

`pilot` recomputes the raw smoke records and requires `WIRING_PASS`; the 72-run expansion
recomputes the raw pilot records and requires exactly `INCONCLUSIVE`.  A stored decision string
alone cannot unlock the next stage.  Each completed cell must pass policy-formula trace parity,
estimator replay, offline 0.5 s usability labeling, and map/data-integrity gates; policy or SLAM
safety failures remain outcomes rather than replacement-run exclusions.  A wiring/data-integrity
failure, non-finite required metric, intervention bound violation, or smooth-arm fall/base contact
stops the schedule immediately and leaves the remaining cells unexecuted.
