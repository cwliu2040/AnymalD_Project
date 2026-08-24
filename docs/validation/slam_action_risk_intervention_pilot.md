# Action-conditioned SLAM-risk intervention pilots v1 and v2

Date: 2026-08-22

## Why this pilot exists

The existing confidence estimator already predicts 0.5 s future localization usability from
SLAM diagnostics.  The existing-800 audit found that locomotion history is associated with
future hazard, but passive prediction cannot establish that changing an action changes SLAM
risk.  Training another passive confidence model would duplicate the original objective.

This pilot therefore asks a smaller causal question before any new student or PPO training:

> With command, simulator seed, terrain, friction, LiDAR support schedule, and backend held
> fixed, does a bounded signed change to the locomotion action change body motion and the next
> 0.5 s localization hazard?

The original machine-readable frozen design is
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
`boundaries.authorized_stages` in the release config.  Plan mode never creates or overwrites a
`run_manifest.json`.  A live run no longer requires an intermediate commit: its manifest records
the branch, HEAD, and SHA-256/size/existence of every project-owned tracked or untracked dirty
file, so `HEAD + worktree_snapshot` identifies the exact executed source.  Runtime output trees
are excluded from that snapshot.  The runner still refuses a wrong branch or an output stage
containing earlier execution artifacts.  Verbose ROS/Isaac and post-processing stdout is retained
per cell as `launch.log` and `<step>.log`; controller stdout contains only the final compact gate
summary, so long runs do not repeatedly load logs into conversation context.

`pilot` recomputes the raw smoke records and requires `WIRING_PASS`; the 72-run expansion
recomputes the raw pilot records and requires exactly `INCONCLUSIVE`.  A stored decision string
alone cannot unlock the next stage.  Each completed cell must pass policy-formula trace parity,
estimator replay, offline 0.5 s usability labeling, and map/data-integrity gates; policy or SLAM
safety failures remain outcomes rather than replacement-run exclusions.  A wiring/data-integrity
failure, non-finite required metric, intervention bound violation, or smooth-arm fall/base contact
stops the schedule immediately and leaves the remaining cells unexecuted.

## First wiring-smoke attempt

The first authorized smoke at commit `0f6ebb6` stopped after cell 1/6
(`fastlio2/curve_1_5_right_1_0/block_542/smooth`) with `WIRING_FAIL`; no zero,
antismooth, or LIO-SAM cell ran.  The ROS launch itself exited successfully and the existing
cell showed a valid degradation/recovery transition, stable locomotion, valid offline labels,
and valid map registration.  Two post-processing wiring defects prevented a run record:

- independently re-evaluating the ONNX Arm-B baseline yielded a realized residual of
  `0.05000042915344238` for a `0.05` clip, while formula parity error was only
  `5.364418029785156e-07`; the bound check had used a tighter tolerance than the frozen
  `1e-5` formula-parity check;
- the new runner omitted the repository's existing `deployment/python_vendor` path for the
  ROS-Python estimator replay subprocess, so `onnx` could not be imported.

The runner, analyzer, and trace validator now use the recorded formula-parity tolerance
consistently, and estimator replay follows the already-qualified publication-runner vendor-path
pattern.  Offline repair validation on the same retained bag passes all five intervention checks
and reproduces 837/837 estimator timestamps exactly with `0.0 m/s` maximum error.  This is wiring
repair evidence only: the original attempt remains `WIRING_FAIL`, live authorization was disabled
again, and no rerun or causal claim is authorized.

## Wiring-smoke attempt 2

The repaired rerun used
`outputs/slam_action_risk_intervention_pilot_v1_attempt2/wiring_smoke/` and completed all six
scheduled cells.  Every cell passed intervention trace, stability, offline usability, map
integrity, estimator replay, and run-record gates; the frozen decision is `WIRING_PASS` with
`claim_allowed=false`.  The manifest records HEAD `0f6ebb6` plus SHA-256 snapshots of all seven
project-owned dirty repair files.  Live authorization was disabled immediately afterward, and
pilot remains unauthorized.

The six records permit only an exploratory mechanism check.  LIO-SAM had the intended
smooth < zero < antismooth ordering for both action rate and stable roll/pitch rate.  FAST-LIO2
did not have the intended action-rate ordering, although smooth body rate was still slightly
below zero.  Smooth-minus-zero moving speed was about `-0.031 m/s` for FAST-LIO2 and
`-0.053 m/s` for LIO-SAM; the latter is just beyond the frozen `0.05 m/s` pilot guard.  There
were no falls or base contacts.  Therefore wiring is proven, but the smoke does not prove that
the mechanism generalizes or that risk changes independently of slowing.  The frozen 24-run
pilot remains the minimum next causal gate; failure to obtain three of four ordered strata or
the speed guard stops the route.

## Pilot stop decision

The authorized frozen pilot in
`outputs/slam_action_risk_intervention_pilot_v1_attempt2/pilot/` stopped after its first scheduled
cell: `fastlio2/curve_1_5_right_1_0/block_543/smooth`.  The launch returned zero and intervention
trace, offline usability, map integrity, estimator replay, and run-record post-processing all
completed.  The locomotion outcome nevertheless had `fall=true` and `base_contact=true`; the
stability gate reported `terminated_count=1` and `event_order=body_instability_first` while the
driver itself passed.  The runner emitted `smooth_arm_safety_failure`, executed 1/24 cells, and
the frozen decision was `FAIL` with next step `stop_action_conditioned_new_C_route`.

This is the v1 protocol's explicit operational safety stop, not a wiring failure.  It remains a
retained outcome and cannot be excluded.  It also cannot establish smooth-specific harm because
the matched zero and antismooth cells in block 543 were never run.

After reviewing the stop semantics, a single fall in simulation is not a sufficient route-level
futility decision: simulated training/probing normally terminates and records the episode, whereas
physical deployment must stop immediately.  Therefore the stored v1 `FAIL` is authoritative for
that interrupted matrix but does **not** prove that the research hypothesis failed.  Before any
new collection, a v2 protocol must separately freeze:

- immediate matrix stop for wiring, non-finite values, contract failure, or intervention-bound
  violation;
- simulated fall/base contact as a retained cell outcome that ends that episode but permits the
  minimum matched arm set to complete;
- a predeclared paired/repeated criterion for route-level smooth-specific safety harm;
- immediate physical-hardware stop as a separate deployment rule.

Live authorization is disabled.  No remaining v1 cell, v2 collection, parameter sweep, PPO, or
72-run expansion is currently authorized.  Formal model1450 and deterministic Arm-B safety
behavior remain unchanged.

## Frozen v2 safety semantics and fresh-block decision

The prospective replacement is
`configs/slam_action_risk_intervention_pilot_v2.yaml`, with the hash-locked artifact release in
`configs/slam_action_risk_intervention_release_v2.yaml`. V2 changes safety disposition and block
allocation only; the intervention formula, alpha values, exported policies, SLAM backends,
profiles, outcomes, and causal-development role remain unchanged.

V1 block 543 will not be completed as a new decision gate. Its smooth outcome was already known
when the safety semantics were revised, so completing zero and antismooth afterward would be a
useful diagnostic but not a clean prospective v2 test. V2 therefore forbids every v1 intervention
block 542--550 and allocates fresh block 551 to a contingency smoke, 552--553 to the 24-run pilot,
and 554--559 to the predeclared expansion. The completed v1 attempt2 smoke at block 542 remains
valid wiring evidence because the runtime artifacts and wiring are unchanged. The v2 release
hash-locks its manifest and the runner recomputes `WIRING_PASS` from all six raw run records; it
fails closed if that ignored evidence is absent or changed. No new smoke run is required unless
that evidence can no longer be verified.

Simulation and hardware now have separate rules:

- wiring/data-integrity, contract/non-finite, and intervention-bound failures stop the matrix
  immediately;
- a simulation fall or base contact terminates and retains that episode but does not by itself
  stop the matrix, so the scheduled matched arms remain observable;
- within each `(backend, profile, block)`, a paired smooth-specific safety excess means smooth has
  a fall or base contact while exact-zero has neither;
- one such paired excess blocks PASS but is only `INCONCLUSIVE`; route-level safety `FAIL` requires
  the excess in two distinct blocks of the same backend/profile (the two-of-two pilot repeats);
- after the second complete matched triplet establishes that repeated harm, the remaining matrix
  stops and the partial inventory is a valid safety-terminated `FAIL`, not an integrity failure;
- physical ANYmal-D retains immediate stop on any fall risk, and no simulation result can relax
  that deployment rule.

Thus a common fall in smooth and zero is not called smooth-specific harm, and isolated smooth harm
is not silently accepted: it prevents PASS and can only lead to the already predeclared expansion.
The analyzer reports the exact excess block IDs per backend/profile. V1 analysis remains on its
legacy semantics so the stored interrupted result is reproducible.

V2 plan rendering is read-only:

```bash
python3 scripts/validation/validate_slam_action_risk_intervention_protocol.py \
  --protocol configs/slam_action_risk_intervention_pilot_v2.yaml
python3 scripts/validation/run_slam_action_risk_intervention_pilot.py \
  --stage pilot \
  --protocol configs/slam_action_risk_intervention_pilot_v2.yaml \
  --release configs/slam_action_risk_intervention_release_v2.yaml \
  --output-root outputs/slam_action_risk_intervention_pilot_v2
```

The second command has no `--execute`; it does not start ROS or write a live manifest. The v2
release has `live_execution_authorized: false` and an empty authorized-stage list.

## V2 pilot execution and decision

The first execution attempt used
`outputs/slam_action_risk_intervention_pilot_v2/pilot/` inside the restricted tool sandbox.
Isaac Sim could not enumerate the host GPU (`No CUDA GPUs are available`) and could not write its
own cache outside the repository. The launch wrapper returned zero but produced no policy or
locomotion diagnostics, so the runner correctly stopped after the first scheduled cell with a
wiring/data-integrity failure and zero run records. `simulation_safety_event=false`. This is an
execution-environment failure, not a v2 intervention outcome, and it is retained separately rather
than overwritten or included in the causal pilot.

The authorized GPU-visible rerun used
`outputs/slam_action_risk_intervention_pilot_v2_attempt2/pilot/`. It completed all 24/24 fresh
scheduled cells for blocks 552--553. Inventory, intervention trace, zero/invalid Arm-B parity,
finite metrics, residual bounds, offline usability, map integrity, estimator replay, and run-record
gates all passed. There were no falls or base contacts in any smooth cell, no paired smooth-specific
safety excess, and no safety-terminated early stop.

The frozen gate returned `INCONCLUSIVE`:

- action rate had the intended smooth < zero < antismooth order in 2/4 strata, below the required
  3/4;
- stable roll/pitch body rate had the intended order in 1/4 strata, also below 3/4;
- smooth-minus-zero tracking survival was nonnegative for FAST-LIO2 (`+0.975 s`) and LIO-SAM
  (`+0.025 s`), but hazard direction was slightly positive rather than nonpositive in both backends
  (`+0.00553` FAST-LIO2 and `+0.000316` LIO-SAM);
- the `0.05 m/s` speed guard failed. The largest discrepancies were lateral motion:
  `-0.136 m/s` for FAST-LIO2 and `-0.364 m/s` for LIO-SAM. Curve differences were
  `-0.0174 m/s` and `+0.0486 m/s`, respectively;
- none of the frozen route-FAIL conditions occurred: action separation was nonzero, neither
  backend had joint body-rate and survival harm, and there was no repeated safety harm.

This result supports neither an action-conditioned transition-model design nor termination of the
research hypothesis. Its only permitted next experimental step is the already predeclared 72-run
fresh-block expansion, without alpha/checkpoint/profile changes and only after separate live
authorization. It does not authorize PPO or a new model. The release was returned immediately to
`live_execution_authorized: false` with no authorized stages.

## Post-pilot expansion value diagnosis

After the frozen decision, a post-pilot exploratory diagnosis was performed only to decide whether
spending 72 additional live runs on the unchanged smoothing intervention had useful information
value. It does not revise the `INCONCLUSIVE` decision or add a retrospective causal gate. The unit
remains each of the eight complete `(backend, profile, block)` matched arm sets.

Smooth-minus-zero block effects were:

| Backend/profile/block | action rate | body rate | hazard fraction | speed | progress |
|---|---:|---:|---:|---:|---:|
| FAST curve 552 | -7.64/s | +0.00225 rad/s | +0.0369 | -0.0116 m/s | -0.0136 |
| FAST curve 553 | -6.96/s | -0.0126 rad/s | 0 | -0.0232 m/s | -0.0367 |
| FAST lateral 552 | -4.84/s | -0.0661 rad/s | -0.0148 | -0.0638 m/s | -0.0602 |
| FAST lateral 553 | +3.41/s | +0.0152 rad/s | 0 | -0.209 m/s | -0.116 |
| LIO curve 552 | -9.50/s | +0.0156 rad/s | -0.150 | +0.113 m/s | +0.130 |
| LIO curve 553 | -4.24/s | -0.00560 rad/s | +0.0805 | -0.0161 m/s | -0.0200 |
| LIO lateral 552 | +6.93/s | -0.0531 rad/s | +0.0712 | -0.322 m/s | -0.241 |
| LIO lateral 553 | +4.80/s | -0.154 rad/s | 0 | -0.407 m/s | -0.345 |

Only 5/8 blocks reduced smooth action rate, 5/8 reduced body rate, 5/8 had nonpositive hazard
difference, and only 3/8 met the absolute `0.05 m/s` speed guard. Both FAST lateral blocks and both
LIO lateral blocks violated the speed guard, with normalized progress lower in all four. Most
importantly, LIO lateral repeated the wrong action mechanism direction in both blocks: smooth
action rate increased while speed and progress fell sharply. Thus the aggregate lower body rate
cannot be separated from slower locomotion in the lateral route.

All eight smooth and all eight antismooth records realized a nonzero residual on every valid tick
and touched the `0.05` bound at least once; mean residual L2 ranged `0.0136--0.0238` for smooth and
`0.0127--0.0232` for antismooth. This confirms that the assigned interventions were active, but the
run-record summary cannot determine the fraction of individual joint-ticks that clipped. The
problem is therefore not absent intervention; it is inconsistent realized mechanism and repeated
speed/progress confounding.

The resource recommendation is **do not exercise the optional 72-run expansion for this unchanged
previous-action smoothing intervention**. More blocks could narrow uncertainty but cannot repair a
repeated violation of the command-preserving interpretation, and the frozen expansion forbids the
parameter or intervention-family changes that would be required. This is a decision to stop the
current intervention, not a claim that all action-conditioned localization-risk hypotheses fail.

Any successor must be a separately frozen, disjoint development route. Before real-SLAM efficacy
collection it should demonstrate a repeatable action/body mechanism while preserving speed and
progress under both curve and lateral commands; it must not tune alpha or select a candidate using
the current 24 runs and then reuse these blocks as validation. PPO and transition-model design
remain unauthorized.
