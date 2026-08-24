# Matched-prefix SLAM speed-pulse causal pilot v2

Date: 2026-08-22

## Why v2 exists

The whole-episode fixed-scale pilot completed 32/32 valid runs but failed its frozen hazard
dose-response gate. Its arms accumulated different route progress before the endpoint, so matched
seed did not guarantee matched state. Those runs remain valid development evidence but cannot
unlock the action-conditioned risk model.

V2 changes only the intervention timing. Every arm uses the same hash-locked Arm B, unmodified
command profile, terrain, seed, sensor degradation, and real SLAM backend through an untreated
prefix. At profile time 7.25 s the benchmark driver applies one uniform XYZ command-scale pulse for
0.75 s, then restores the original profile exactly. The future-usability endpoint remains 0.5 s.

## Runtime and trace contract

`CommandScalePulse` is a ROS-independent deterministic primitive. The stability benchmark owns the
pulse because it already owns `/cmd_vel`; the policy and confidence estimator are not modified.
The driver records profile-start clock, scale, start, duration, end, and pulse publication count.

Post-processing must verify all of the following from recorded artifacts:

- untreated-prefix commands equal the common profile;
- pulse commands equal the assigned uniform scale times the common profile;
- post-pulse commands return to the common profile;
- the policy observation contains the effective command;
- recorded actions equal the hash-locked Arm-B ONNX output;
- the four arms in every backend/profile/block set meet frozen pre-pulse state tolerances for body
  velocity, attitude proxy, joints, previous action, and SLAM confidence.

The 60 ms boundary grace covers the publish/inference ordering around the two pulse edges; it does
not relax samples inside any verified phase. A trace, non-finite, missing-label, or pre-pulse-match
failure stops the matrix. A single simulation fall remains an episode outcome; repeated paired
reduced-scale-specific harm is required for route-level safety failure. Physical use remains
unauthorized and would retain immediate-stop semantics.

## Frozen schedule and authorization

The current post-development schedule reserves block 566 for an eight-cell wiring smoke across
FAST-LIO2 and LIO-SAM, curve profile, and the four scales. Blocks 567--568 are a separate 32-cell
curve/lateral pilot. All earlier blocks through 565 are forbidden. Both stages use balanced arm
order and report backends separately.

The headroom decision is ceiling-aware and frozen before any retry. FAST-LIO2 is the safeguard:
when full-scale Arm B already has pulse-window hazard at or below 0.05, the reduced-scale arms do
not need to manufacture a superiority result; otherwise their mean excess hazard may be at most
0.02. LIO-SAM is the prespecified improvement backend. At least one of its curve or lateral strata
must be non-ceiling, hazard must be ordered with scale, and one reduced scale must beat full scale
in both blocks with a mean hazard benefit of at least 0.05. Backend pooling cannot rescue a failure.
This prevents a strong FAST-LIO2 baseline from making the experiment impossible while still
requiring evidence that locomotion action has useful SLAM headroom somewhere defined in advance.

The ROS package builds, protocol and plan modes validate, and offline tests cover pulse boundaries,
disabled control, malformed configuration, synthetic prefix/pulse/recovery traces, release hashes,
pre-pulse mismatch detection, and monotonic analyzer behavior. A live manifest will lock HEAD plus
every project-owned dirty file hash before collection. Live authorization, model training,
PPO, default switching, and physical execution are all false. The next possible action is only a
separately authorized block-563 wiring smoke.

## Wiring smoke attempt 1

The first authorized block-563 attempt stopped after FAST-LIO2 curve scale 1.00 with
`WIRING_FAIL`; the other seven cells were not run. Launch, stability, offline usability, map
consistency, estimator replay, pulse-window command, recovery command, policy observation, and
Arm-B action parity all passed. Two instrumentation checks failed:

- the driver counted pulse publications only when the numerical scale differed from 1.00, so the
  scale-1.00 control incorrectly reported zero pulse publications;
- prefix validation included the profile's normal acceleration ramp, where publisher and policy
  sampling can differ by one 20 ms tick, producing a 0.0225 command discrepancy.

The repair counts every enabled pulse-window publication including scale 1.00, and verifies the
untreated prefix only after the predefined profile ramp is complete. Reprocessing the attempt-1
artifacts with the repaired validator confirms prefix, pulse, recovery, observation-command, and
action parity all pass; the retained old driver count correctly remains the sole failure. Attempt 1
is not relabeled. ROS rebuild and 15 focused tests pass, and live authorization remains closed.

## Downstream C2 boundary

Even if the short-pulse gate later passes, it authorizes only offline action-conditioned modeling,
not PPO. Candidate C2 is separately specified in
`configs/slam_constrained_residual_c2_v1.yaml`. It keeps model1450 frozen, receives the original
requested command before Arm-B soft scaling, and may produce anisotropic command scales plus a
small joint-action residual. Healthy behavior must return exactly to model1450; missing risk falls
back exactly to B, and invalid or stale SLAM remains the hard zero stop.

C2 reuses the existing locomotion reward stack and continues tracking the original requested
velocity. It explicitly forbids the old C reward functions that changed the tracking target to a
confidence-scaled command; those terms structurally made slowing or stopping easier to reward.
FAST-LIO2 only requires noninferiority to B, while at least one prespecified degraded LIO-SAM
stratum must improve risk and progress. Repeated behavioral equivalence to B or old C is recorded as
`COLLAPSED_TO_B_OR_OLD_C`, not counted as a successful new policy.

## Wiring smoke attempts 2 and 3

Attempt 2 completed the first three FAST-LIO2 arms and stopped on scale 0.25 because its 50 Hz
policy trace missed the single 7.18 s sample. The original pre-pulse selector had only a 20 ms
effective window. All other trace, stability, SLAM usability, map, estimator, and safety checks
passed. The attempt remains `WIRING_FAIL`. The repaired contract selects the latest uncontaminated
sample no more than 0.12 s old and separately requires at most 0.04 s pairwise time skew. Reanalysis
of the retained scale-0.25 trace selected the 7.16 s sample, 0.03 s from the uncontaminated edge, and
passed every trace check.

Attempt 3 then completed all four FAST-LIO2 arms, each with a valid record and no safety event. The
complete set failed the frozen pre-state match because joint-velocity pairwise L-infinity distance
was 2.945 rad/s versus the 1.5 limit. Sample-time skew was effectively zero; base velocity, angular
velocity, gravity, joint position, previous action, and SLAM confidence all passed. This is evidence
that separate same-seed ROS/Isaac launches do not guarantee identical instantaneous gait phase. The
attempt remains `WIRING_FAIL`, live authorization is closed, and the tolerance must not be loosened
post hoc without an explicit design decision.

## Composite matching and route stop

Attempts 2 and 3 supplied 12 pairwise FAST-LIO2 pre-treatment-only development comparisons. A
composite joint-velocity rule was frozen at RMS 1.25 rad/s plus an L-infinity hard bound of 3.25
rad/s. Attempt 4 completed 8/8 records and passed FAST matching, but LIO-SAM narrowly exceeded the
old base-angular and previous-action limits. Its pre-treatment-only values were added to a
backend-neutral 24-comparison calibration; the respective limits were frozen at 0.25 rad/s and
0.45. No post-pulse outcome was used to choose a matching threshold. Attempt 5 then completed 8/8
and achieved `WIRING_PASS` with no safety event.

The first attempt-5 causal-pilot cell had already lost FAST-LIO2 tracking before the pulse, leaving
zero eligible valid-requested ticks. It is retained as an integrity `FAIL`, not an intervention
effect. The sensor-degradation onset was aligned to the pulse at 7.25 s and blocks through 565 were
retired. Fresh block-566 attempt 6 completed 8/8 valid records with no safety event. FAST matched,
but independent LIO-SAM launches again diverged before treatment: body angular velocity reached
0.357 rad/s pairwise difference, joint-velocity RMS 1.881 rad/s and L-infinity 4.429 rad/s, and
previous action 0.482. These are larger than both the original and calibrated limits.

Repeatedly expanding thresholds to chase new independent-run maxima would remove the causal value
of matching. Therefore attempt 6 remains `WIRING_FAIL`, the 32-cell effect pilot was not run, and
live authorization is closed. The evidence now rejects separate sequential same-seed launches as a
reliable exact-prestate design for this ROS/Isaac/real-SLAM path. The next method decision is between
a randomized repeated-run experiment with frozen pre-treatment covariate adjustment or a much more
expensive simulator/SLAM state-restoration design.

## Randomized repeated-run result

The route was changed before further collection to retain every randomized run rather than require
exact prestate matching. One robot executed 64 sequential runs: two backends, two profiles, four
scales, and four fresh blocks 567--570. Arm order was balanced. Prestate covariates were reported but
could not exclude a run, and pre-pulse tracking loss was defined as intention-to-treat failure 1.0.
All 64 records passed integrity; none had zero conditional eligible ticks after onset alignment.

FAST-LIO2 curve showed a clear scale response: mean risk for scales 1.00/0.75/0.50/0.25 was
0.500/0.474/0.336/0.250. FAST lateral was directionally favorable but not perfectly monotonic:
0.467/0.468/0.408/0.368. Both FAST safeguards passed.

The prespecified LIO-SAM target did not. LIO curve means were
0.530/0.507/1.000/0.664: scale 0.75 had only a small inconsistent benefit, while stronger slowdown
was worse. LIO lateral was a complete zero-risk ceiling for every arm. Supported LIO headroom was
0/1 required strata, so the frozen decision is `FAIL`. Four fall/base-contact outcomes occurred,
but none formed the required repeated scale-specific harm across blocks. The result does show that
command action can affect future FAST-LIO2 usability; it does not authorize a risk model because the
prespecified backend requirement failed. Uniform slowdown should not enter C2/PPO from this dataset.
