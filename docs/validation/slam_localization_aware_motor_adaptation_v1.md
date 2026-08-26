# Command-preserving localization-aware motor adaptation v1

Date: 2026-08-24

## Research decision

The primary candidate is no longer a learned XYZ velocity limiter. Arm B remains the deterministic
uniform confidence limiter, and the anisotropic translation/yaw governor remains an experimental
comparison baseline. The new question is whether bounded low-level motor adaptation can improve
future SLAM usability while preserving the original requested command, realized speed, progress,
and locomotion safety.

This is a candidate locomotion-as-active-sensing contribution: SLAM history informs a gait or joint
residual, the residual changes body and sensor motion, and that motion changes subsequent SLAM. It
must not claim success from command reduction alone.

## Difference from B, component scaling, and old C

Arm B maps confidence to one scalar applied uniformly to linear X, linear Y, and yaw. The component
baseline can scale translation and yaw separately. Both remain velocity limiters. Old C additionally
trained with confidence-scaled velocity targets, making slow or stopped behavior easier to reward.

The residual module's valid adapted path evaluates frozen model1450 using the exact original
requested command and adds only a bounded 12-D joint-action residual. If it predicts no beneficial
residual, it uses model1450 with the original command and exact-zero residual rather than stopping.
Missing, uncertain, malformed, or out-of-bound adaptation output falls back to B. Invalid or stale
SLAM remains an exact hard stop.

Command preservation is the attribution rule for proving the low-level residual contribution, not
an instruction to force an unsafe requested speed. The complete runtime controller is residual-first:
it tries scale 1.0 plus a residual, but may apply the smallest necessary slowdown when scale 1.0 is
not feasible. Every reduction must be logged separately and cannot be counted as residual benefit.
B is a safety/reference baseline and need not be universally outperformed; comparisons report the
risk--progress tradeoff.

## RMA-inspired boundary

The proposed architecture borrows RMA's teacher/adaptation separation, not its final task or a
hardcoded route phase. A training-only privileged teacher may use future usability, GT drift, point
support, backend diagnostics, and realized intervention outcomes to define a compact localization-
dynamics latent. A deployment adaptation module must infer that latent only from histories of
confidence, validity/age, requested XYZ command, body/joint motion, and previous action.

The latent must represent action sensitivity, degradation, or recovery dynamics. Backend ID,
intervention ID, route phase, and a hardcoded turn timer are forbidden runtime inputs. Free rotation,
mixed translation/yaw, and bounded random commands must remain representable by the same history
contract.

## Existing low-level evidence review

The prior smooth/zero/antismooth experiment is not sufficient headroom evidence for this claim. It
completed 24/24 runs and applied nonzero bounded residuals, but only 5/8 matched groups reduced action
rate, 5/8 reduced body rate, 5/8 had nonpositive hazard difference, and only 3/8 met the absolute
`0.05 m/s` speed guard. Both lateral profiles repeatedly lost speed and progress. Therefore it cannot
separate a low-level SLAM mechanism from slowing, and its optional 72-run expansion remains retired.

The component-pulse experiments do establish that motion commands can causally affect SLAM and that
uniform scaling is not always optimal. They do not establish command-preserving gait headroom and
remain baselines only.

## First implementation boundary

`localization_aware_residual_core.py` implements the ROS-independent composition contract. It does
not implement a teacher, adaptation network, trained residual policy, ROS node wiring, or a live
protocol. The initial residual L-inf bound is `0.05`, matching the previously wired low-level
intervention envelope; it is an interface safety bound, not evidence that a useful residual exists.

Before any teacher or PPO training, a separately frozen fresh intervention must demonstrate:

- exact original command on the valid path;
- repeatable joint/gait and body/sensor-motion separation;
- moving-speed difference within `0.05 m/s` and progress at least 98% of zero residual;
- repeated future-SLAM improvement in at least one prespecified backend/profile stratum, with all
  backends reported separately and no pooled rescue;
- no excess fall or base contact.

The residual is not rewarded for gait difference by itself. Four explicit failure modes prevent the
model from returning to an easy but invalid solution:

- `STOP_COLLAPSE`: requested motion without progress or excess stopped time;
- `SHUFFLE_COLLAPSE`: pathological short-stride/high-cadence/contact-switching behavior outside a
  frozen model1450 matched-command envelope;
- `POSTURE_COLLAPSE`: degraded body height, stance width, joint margin, foot clearance, torque,
  energy, or slip;
- `LIMITER_COLLAPSE`: apparent SLAM benefit explained by command/realized-speed reduction or behavior
  equivalent to B/component scaling without a low-level mechanism.

Whole-body sensing and outcomes cover commanded mean linear/yaw motion separately from unwanted
roll/pitch oscillation, yaw tracking error and high-frequency oscillation, linear/angular
acceleration and jerk, and LiDAR scan-time translation/rotation. The policy must not suppress a
commanded turn merely to reduce total angular velocity.

The first fresh mechanism and decision gate are now frozen in
`slam_low_level_touchdown_headroom_v1.md`: a history-phase late-swing foot-Jacobian residual reduces
a fixed fraction of predicted one-tick downward foot displacement while targeting exact-zero
horizontal foot displacement. It does not reuse previous-action smoothing or fixed joint offsets.
The ROS-independent mechanism core, schedule validator, decision analyzer, and tests exist. The
36-run model1450 baseline completed and its anti-collapse envelope is frozen. The first prespecified
linear history-only phase estimator failed its false-activation/precision gate and is retained with
`frozen=false`; consequently the protocol is still not intervention-ready. Blocks 581--584 remain
untouched and block 597 has not run. Every effect run, privileged teacher training, adaptation-module
training, PPO, ROS policy wiring, default switching, and physical robot use remain unauthorized.

The next candidate replaces independent frame decisions with a causal three-state tracker. It must
observe confirmed `stance -> early_swing -> late_swing` progression, monotonic swing progress, and a
bounded late-state duration; a direct stance-to-late jump is impossible. Retired blocks 585--596 are
development-only for this v2. Its development metrics justify fresh validation but are not a PASS.
Blocks 606--609 were used exactly once as the frozen 12-run fresh phase-only validation set. All
12 integrity/safety runs and the unchanged one-shot phase gate passed. Aggregate precision/recall/
false-trigger/progress-MAE were `0.9138/0.5883/0.00394/0.1046`, with every profile and foot covered.
The promoted artifact is frozen with SHA-256
`943a04fbe2964fb86ec61c43e5675c469150e162913dc36c2ef4dd94767fd3cb`; execution authorization was
closed again after collection. Block 597 is phase-gate eligible but has not been authorized or run.

The deployable kinematics and complete pure-function path are also implemented. Frozen official
ANYmal-D geometry maps canonical joint position/velocity to base-frame foot position, `3x12`
Jacobian, action-scaled Jacobian, and relative foot velocity without simulator foot state. Across the
36-run baseline its position RMSE is `9.15e-7 m`; vertical-velocity RMSE is `0.0934 m/s`, and the
analytic Jacobian agrees with finite differences to `1.54e-10 m/rad`. A 28,114-tick offline pipeline
replay produced 2,128 eligible residual ticks, retained exact-zero residual on all other ticks,
preserved every command exactly, respected the `0.05` residual bound, and limited horizontal
first-order displacement to `5.91e-10 m`. This is integrity evidence only, not locomotion or SLAM
effect evidence. The phase artifact is now deployable as a pure runtime component, but it has not
yet been used for a residual-effect simulation or wired into the ROS policy path.
