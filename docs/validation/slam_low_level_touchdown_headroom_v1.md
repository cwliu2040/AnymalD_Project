# Matched-speed low-level touchdown headroom v1

Date: 2026-08-24

## Role in the original new C

This is the unfinished low-level residual branch of the original localization-aware C, not a new
velocity governor and not a replacement research topic. The frozen backbone remains Recovery v0.4.0
model1450. Every valid experimental arm evaluates model1450 with the exact original requested
command and differs only by a bounded 12-D joint-action residual.

The earlier C2 experiments exercised command-scale candidates and therefore returned to limiter-like
behavior. The retired smooth/zero/antismooth experiment did apply joint residuals, but it failed to
preserve realized speed and progress reliably. This protocol does not reuse or retune its previous-
action smoothing coefficient.

## Prespecified mechanism

The first mechanism is late-swing touchdown displacement shaping. For each leg, a phase estimator
using only joint-position, joint-velocity, and previous-action history identifies a confident
late-swing interval. The mechanism predicts the foot's downward displacement over the next 20 ms
control tick and asks a damped minimum-norm foot-Jacobian solve to cancel a fixed fraction of that
vertical displacement. The same solve constrains the first-order horizontal foot-position target to
exact zero.

The intended causal chain is:

```text
bounded late-swing residual
  -> lower downward foot velocity and touchdown impulse
  -> lower unwanted roll/pitch impulse and LiDAR scan-time rotation
  -> changed future 0.5 s SLAM usability
```

The core is
`deployment/ros2_ws/src/anymal_locomotion_ros2/anymal_locomotion_ros2/
touchdown_shaping_intervention_core.py`. It accepts a foot-position Jacobian already expressed per
normalized policy-action unit. For eligible feet it solves

```text
min ||r||
subject approximately to J_xyz r = [0, 0, -a v_z dt]
```

with damped least squares, then uniformly rescales the complete residual vector if required to keep
`L-inf <= 0.05`. Uniform rescaling preserves the solved direction; it does not independently clip
joint coordinates into a different mechanism.

The three arms are exact zero, 25% one-tick downward-displacement attenuation, and the primary 50%
attenuation. These values and the mechanism may not be changed using SLAM outcomes from the pilot.
There is no sign-reversed or deliberately destabilizing arm.

## Deployable-input and evaluation boundary

The foot Jacobian and vertical velocity must be derived from canonical joint position/velocity
history and a frozen ANYmal-D kinematic model; simulator foot velocity is not a runtime input. The
intervention does not receive confidence-scaled commands, backend ID, profile, route phase,
intervention ID, future usability, simulator contact, or GT state. The experiment scheduler may
assign an arm and collection window, but neither is a runtime-policy feature.

Simulator contact truth has two offline roles only:

1. label and validate the history-only gait-phase estimator before any effect run;
2. evaluate touchdown timing, contact impulse, duty factor, contact switching, and slip.

The first phase-estimator artifact now exists only as a rejected artifact (`frozen=false`). Its
prespecified calibration and one-shot holdout precision gates failed, so no wiring smoke or effect
execution is authorized. A low phase-confidence leg receives exact-zero residual. Malformed phase,
Jacobian, or numeric input is a wiring failure rather than an invitation to clip or guess. Invalid or
stale SLAM is handled outside the intervention by the existing localization-aware exact-zero hard-stop
contract.

## Baseline envelope before treatment

Blocks 585--596 were used only for fresh original-command model1450 plus exact-zero residual
baseline collection. They cover bounded mixed translation/yaw, free yaw rotation, and lateral
translation. Their only role is to freeze the matched-command normal envelopes for:

- stride length, cadence, duty factor, contact-switch rate, and swing duration;
- body height, stance width, joint margin, and foot clearance;
- torque, energy per progress, stance-foot slip, and stopped fraction.

Bounds are formed hierarchically: each complete run contributes its own 1st/99th-percentile run or
complete-stride summary, and the profile envelope takes the minimum lower and maximum upper summary
across the 12 runs. Frames are not independent replicates and a run with more frames cannot dominate
the envelope. Candidate outcomes cannot expand these bounds. The builder is
`scripts/validation/build_slam_low_level_model1450_envelope.py`; it rejects incomplete identities,
nonfinite metrics, frame-level pseudoreplication, or any non-zero arm. The GPU-visible retry2 completed
all 36 identities, and the envelope is frozen at
`outputs/slam_low_level_touchdown_headroom_v1_retry2/baseline_envelope/
model1450_anti_collapse_envelope.json` with SHA-256
`c9c987497fe990367388b1308fd1967dbd6ddca4e9d691fc737da81e125aad88`. This envelope and a separately
passing phase-estimator artifact are prerequisites for wiring smoke. The baseline cannot support a
SLAM effect claim.

## Effect stages and disjoint data roles

Blocks 581--584 remain reserved and untouched. No stage in this protocol schedules them.

| Stage | Blocks | Runs | Role |
| --- | --- | ---: | --- |
| baseline envelope | 585--596 | 36 | anti-collapse calibration only |
| wiring smoke | 597 | 6 | excluded wiring evidence |
| pilot | 598--601 | 72 | causal mechanism development |
| confirmation | 602--605 | 72 | unchanged causal headroom confirmation |

Effect stages use FAST-LIO2 and LIO-SAM separately, three command families, three arms, and balanced
arm order. Pilot data can decide whether the frozen mechanism is worth one unchanged confirmation
stage, but it cannot establish the final headroom claim. Confirmation is usable only if the mechanism
config and baseline-envelope hashes remain unchanged after a pilot pass.

Every effect stage requires separate authorization. The completed baseline-only release authorized
only baseline collection; intervention wiring, block 597, training, default switching, and physical
use remain unauthorized.

## Matched-command and matched-motion gate

Command equality is a trace-integrity requirement, not a statistical adjustment:

- the requested command must be exact across arms;
- the command in model1450's observation must be the original request;
- the zero residual arm must be bit-exact model1450;
- no intervention output may contain a command scale.

Realized motion is post-treatment, so individual runs are never discarded afterward for being too
slow. The entire matched group fails residual attribution unless the primary treatment versus zero
has:

- absolute moving-linear-speed difference at most `0.05 m/s`;
- absolute realized-yaw-rate difference at most `0.05 rad/s`;
- normalized progress ratio at least `0.98`;
- no stopped-fraction excess.

The yaw gate is essential: reducing total angular motion by refusing a commanded turn is
`LIMITER_COLLAPSE`, not body stabilization.

## Required mechanism and SLAM chain

The primary treatment must reduce absolute touchdown vertical speed. At least two of touchdown
contact impulse, touchdown roll/pitch-rate impulse, and LiDAR scan-time rotation must change in the
same improving direction. The low-dose arm must lie in the same touchdown-speed direction between
zero and the primary arm.

Within a prespecified `(backend, profile)` stratum, at least three of four distinct blocks must pass
the complete matched-motion, mechanism, SLAM-direction, and safety chain. The stratum mean must have
nonpositive future-unusable-fraction difference and nonnegative tracking-survival difference. At
least one stratum is required; FAST-LIO2 and LIO-SAM are always reported separately and cannot rescue
each other through pooling.

This gate establishes controllable low-level headroom only. It does not establish a learned runtime
policy, universal superiority to B, or physical-robot safety.

## Four collapse gates

- `STOP_COLLAPSE`: no progress loss or stopped-fraction excess is allowed.
- `SHUFFLE_COLLAPSE`: stride, cadence, duty factor, switching, and swing duration must remain inside
  the independently frozen model1450 envelope.
- `POSTURE_COLLAPSE`: body height, stance width, joint margin, clearance, torque, energy, and slip
  must remain inside safety and baseline envelopes; fall and base contact remain explicit outcomes.
- `LIMITER_COLLAPSE`: command equality, linear/yaw realized-motion equivalence, progress, and the
  required touchdown-to-body/LiDAR mechanism must all pass.

A simulation fall or base contact terminates and retains that episode. One treatment-specific excess
is `INCONCLUSIVE`; the same `(backend, profile)` showing treatment-specific excess in two distinct
blocks is `FAIL`. Physical use would retain immediate-stop semantics regardless of simulation data.

## Current implementation status

Completed:

- mechanism and amplitude contract;
- ROS-independent damped-Jacobian residual core;
- balanced baseline/effect schedule validator;
- candidate-blind hierarchical baseline-envelope builder;
- matched-motion/mechanism/SLAM/anti-collapse decision analyzer;
- static random-command preservation and focused unit tests;
- canonical simulator joint/torque/foot instrumentation, zero-residual trace validator, run reducer,
  baseline-only release, and fail-closed runner;
- 36/36 baseline runs and the frozen model1450 anti-collapse envelope.

The prespecified history-only linear phase-estimator v1 did not pass its gate. It used train blocks
585--592, report-only calibration blocks 593--594, and one-shot holdout blocks 595--596. Calibration
late-swing precision was `0.6924` with false-trigger fraction `0.01822`, versus required `>=0.90` and
`<=0.01`; holdout precision was `0.8386`. Its recall and progress MAE passed, but that cannot override
unsafe false activation. The rejected artifact is retained at
`exported/slam_low_level_touchdown_phase_estimator_v1/phase_estimator.json`; its validation report is
`docs/validation/slam_low_level_touchdown_phase_estimator_v1.json`.

Still incomplete and unauthorized:

- a phase estimator that passes fresh disjoint validation;
- deployable frozen-model Jacobian wiring and block 597 intervention wiring smoke;
- every scheduled wiring, pilot, and confirmation run;
- privileged teacher, history adaptation module, PPO, ROS policy-node wiring, default switching, and
  physical robot use.

The next legitimate research step is a prospectively frozen temporal phase-estimator design with
fresh disjoint validation data. The failed holdout may not tune its model, threshold, or gates. Block
597 cannot run until such an artifact is frozen PASS and separately authorized.

That temporal v2 implementation now exists as a nondeployable development candidate. Its causal state
machine requires confirmed stance-to-early progression before one late-swing observation can activate
the gate; every active tick must retain late evidence, progress must not regress materially, and the
late state has a fixed maximum duration. On retired development data it reports precision `0.9095`,
recall `0.5728`, false-trigger fraction `0.00405`, and progress MAE `0.0921`. These are development
numbers only. The exact candidate/config are hash-locked by
`configs/slam_low_level_touchdown_phase_tracker_v2_release.yaml`, which remains execution-disabled.
Fresh zero-residual blocks 606--609 across all three profiles independently passed the unchanged
gate: 12/12 integrity/safety runs passed, aggregate precision/recall/false-trigger/progress-MAE were
`0.9138/0.5883/0.00394/0.1046`, and all profile/foot coverage gates passed. The resulting artifact is
`frozen=true` with SHA-256 `943a04fbe2964fb86ec61c43e5675c469150e162913dc36c2ef4dd94767fd3cb`.
This phase gate permits block 597 as the next research stage, but effect execution remains disabled.

The remaining non-ROS runtime mathematics are complete. The frozen ANYmal-D kinematics core was
validated against all 36 baseline traces and central finite differences, then composed with the
temporal phase tracker and horizontal-nullspace touchdown solve. The complete development replay
preserved commands and zero-residual ineligible ticks exactly, bounded every active residual, and
reduced maximum horizontal first-order leakage from the earlier damped XYZ solve to `5.91e-10 m`.
The machine-readable reports are `docs/validation/anymal_d_frozen_kinematics_v1.json` and
`docs/validation/slam_low_level_touchdown_pipeline_v1.json`. Neither report tests realized motion,
body/LiDAR response, or future SLAM.

Block 597 then completed the excluded wiring smoke on both real SLAM backends and all three arms.
An initial no-evidence attempt exposed and fixed a field-name wiring error; the unchanged second
attempt completed 6/6 runs and received `WIRING_PASS` with `claim_allowed=false`. The zero arm stayed
bit-exact, the 25% and 50% arms produced 40 and 35--42 nonzero residual ticks per run, commands stayed
exact, and mean speed/yaw differences remained far inside their matched-motion limits. However,
50%-vs-zero normalized-progress ratios were only `0.934/0.950` for FAST-LIO2/LIO-SAM, a mandatory
warning for the pilot gate. Touchdown metrics showed preliminary reductions while LiDAR/SLAM
directions were mixed; one excluded block cannot
support an effect claim. Blocks 598--601 remain unexecuted and require separate pilot authorization.

Blocks 598--601 subsequently completed all 72 pilot simulations. Two reducer-only integrity
revisions were required and preserved: pure-yaw progress/stop semantics replaced a linear-only null,
then temporary absolute shuffle thresholds were replaced by the pre-candidate hash-locked model1450
envelope. The final v3 inventory is complete and integrity-clean, but the decision is `INCONCLUSIVE`
with zero supported strata. Only 3/24 block comparisons passed the mechanism gate, only 6/24 primary
comparisons passed every anti-collapse check, and no block jointly passed matched motion, mechanism,
SLAM direction, and anti-collapse. Three comparisons failed matched motion; LIO-SAM mixed block 600
had a progress ratio of only `0.107`. Mean touchdown-speed direction improved only in the two mixed
strata and worsened in the other four, while SLAM directions were inconsistent. Confirmation blocks
602--605 must not run, and this touchdown family cannot advance to teacher/adaptation/PPO.

## Post-pilot failure audit and route retirement

The frozen v3 decision remains unchanged and may not be relabeled. A later read-only audit of the
72 raw traces separated measurement failure from mechanism failure:

- All three formal matched-motion failures came from endpoint nearest-segment aliasing on the curved
  route, not from stopping. In the extreme LIO-SAM mixed block 600 primary run, endpoint route
  progress was `0.1066`, but maximum attained route progress was `0.9972` and realized planar path
  length divided by requested translational distance was `1.0107`. Future protocols must freeze a
  monotone or accumulated progress definition before collection.
- The exact-zero arms passed the aggregate absolute shuffle envelope in only 14/24 comparisons and
  the aggregate posture envelope in only 10/24. Future gates must combine paired treatment-minus-zero
  noninferiority with independent absolute physical limits. This does not erase real primary-specific
  warnings: cadence newly exceeded its envelope in 4/24 comparisons, contact-switch rate in 3/24,
  and stance width in 3/24.
- The formal reducer averaged all simulator touchdowns, but exploratory event alignment did not
  rescue the targeted mechanism. Across 532 primary foot-specific pulse groups, 527 matched the next
  primary and same-block zero touchdown. Only 236/527 (`44.8%`) had lower primary touchdown vertical
  speed; mean and median primary-minus-zero differences were `+0.0146` and `+0.0023 m/s`. Sensitivity
  over `0.05--0.30 s` matching tolerances retained the same non-improving direction.
- Primary active-foot tick shares were approximately RH `65.0%`, LF `20.5%`, RF `7.6%`, and LH
  `7.0%`. About `40.6%` of pulse groups were no more than 20 ms before detected touchdown and `72.2%`
  were no more than 40 ms before it. Timing/actuator latency is a supported explanation but remains
  an inference rather than a separately identified actuator model.
- Primary-minus-zero touchdown speed improved in only 10/24 comparisons; LiDAR scan-time rotation
  split 12/24 improving and 12/24 worsening. Future-unusable fraction improved in 4/24, was exactly
  unchanged in 15/24, and worsened in 5/24. The paired transmission correlations were near zero.

The audit does not establish general 12-D residual impossibility, but it retires this one-tick
touchdown mechanism and all hand-designed successors. The next formal target is original-command
confidence-conditioned full locomotion-policy fine-tuning with frozen J0/J1/J2 comparators and
prospective anti-collapse gates.
