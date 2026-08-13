# SLAM confidence PPO second-round validation

Date: 2026-08-12

Status: the bounded safe-command `model_19` candidate passed the frozen
five-profile confidence-state behavior gate and 256-sample export parity, then
failed the fail-fast two-backend live locomotion screening at the LIO-SAM arm.
It is not promotable. Recovery v0.4.0 model1450 remains the formal policy and
has not been replaced.

## Starting-point contract

The second round does not resume the first-round `model_1599`. The project
training runner now supports `--actor-only-warm-start` and constructs the
starting point as follows:

- copy `std` and every `actor.*` parameter from formal model1450;
- expand only `actor.0.weight` from 48 to 51 inputs;
- initialize columns 48..50 to exact zero;
- leave the 51-D critic at the runner's fresh seeded initialization;
- leave the Adam state empty and set the learning iteration to zero.

The legacy `bootstrap_slam_confidence_checkpoint.py` is retained only to
reproduce the first round. Its report now identifies it as a full-state legacy
bootstrap that expands the critic and carries optimizer state; it is not an
authorized second-round starting point.

The 2-environment real Isaac Sim bootstrap smoke passed. A separate 256-sample
parameter/output check found:

- actor maximum absolute error versus model1450: `0` for arbitrary values in
  the three new inputs;
- nonzero entries in new columns: `0`;
- optimizer state entries: `0`;
- target iteration: `0`;
- fresh critic is not identical to the source critic.

The untrained checkpoint used by the iteration-zero behavior baseline is:

`logs/rsl_rl/anymal_d_locomotion_slam_confidence_v2/2026-08-12_12-39-20_second_ppo_actor_only/model_0.pt`

SHA-256: `e8136acbab5b0457b55ceffa8130b8fc725a1575dcf20e745ca6e066f7366200`

New training runs use the unambiguous filename `bootstrap_model_0.pt`, because
RSL-RL's normal `model_0.pt` is written after its first PPO update.

## Frozen behavior gate

The machine-readable gate is
`configs/slam_confidence_behavior_gate.yaml`, SHA-256
`298df1eb772b2f2a077af6a0eacd0e746fd4e8a0a446b740882e60a81cf8eaef`.
It fixes seed 43, 512 environments, 1,000 policy steps, command
`[vx=1.5, vy=0, wz=0]`, and a synchronized 10 s confidence cycle. It reports
separate settled windows for:

- healthy tracking;
- late degraded deceleration;
- invalid stopping;
- recovery command re-acquisition.

It also separates hard terminations from ordinary episode timeouts. A
candidate must pass every check; aggregate reward is not a behavior gate.

The iteration-zero report is:

`logs/rsl_rl/anymal_d_locomotion_slam_confidence_v2/2026-08-12_12-39-20_second_ppo_actor_only/iteration_0_behavior_gate.json`

SHA-256: `722fc9d46d860654ccc0aaa2459f0482d0053a9b7a63ccef0a4b5d2b784dd37a`

It passed healthy tracking, recovery tracking, yaw stop and hard-termination
checks, but correctly failed degraded and invalid stopping. Healthy and invalid
settled mean planar speeds were `1.494046` and `1.494028 m/s`; the degraded to
healthy speed ratio was `0.999964`. Hard terminations were zero; the 512
timeouts were the expected 20 s episode boundary.

## Controlled 50-iteration tranches

Every tranche used 4,096 environments, seed 42, 50 iterations and 4,915,200
simulation steps. Every tranche restarted from formal model1450 with a fresh
critic and optimizer. No tranche resumed `model_1599` or another failed
candidate.

| Tranche | Actor scope / reward change | Invalid settled speed | Degraded / healthy | Invalid yaw | Result |
| --- | --- | ---: | ---: | ---: | --- |
| unrestricted actor | all actor parameters trainable; speed/yaw stop curriculum | `1.501238 m/s` | `0.998959` | `0.059311 rad/s` | fail |
| input adaptation | only columns 48..50 and first-layer bias trainable; rest of actor frozen | `1.300765 m/s` | `1.003298` | `0.358839 rad/s` | fail |
| input + action stop | input adaptation plus invalid-only action L2; milder late speed/yaw weights | `1.355592 m/s` | `1.000356` | `0.274866 rad/s` | fail |

All three retained healthy/recovery linear checks and had zero hard
terminations in the fixed behavior evaluator. None learned degraded
deceleration or invalid stopping. The final two also failed invalid yaw stop.
Training-distribution base-contact episode fractions at iteration 49 were about
`6.08%`, `2.78%`, and `3.86%`, respectively; these reinforce the decision not
to continue blindly.

The three retired model49 checkpoints and gate report hashes are:

| Run | Checkpoint SHA-256 | Gate report SHA-256 |
| --- | --- | --- |
| `2026-08-12_12-44-19_second_ppo_actor_only_main_50iter` | `176b540aa3702d87c2119d02f40313ead5895991f8ece2a66f296365955bd6fe` | `60a398dc5e9c108d2eb9e057bdbf392a28857538c9cd4df84b16849b5d07c02f` |
| `2026-08-12_12-47-45_second_ppo_input_adaptation_50iter` | `27edfee24b12f446f69c68bb617fcacf14b6f93af5ab4d5c21031c4b6010ce05` | `95aedf47416c8ca95708eeb531312785d10948d92863f3cc64af3c3eec2ecd9e` |
| `2026-08-12_12-50-43_second_ppo_input_action_stop_50iter` | `e0020f71ded0e8da2d36ed38f3d754f835157d720ea2d39ef9bf63c6c95ddf32` | `dc96c282b9619884a736719543d11adbb392f35348d8434366f668a327225008` |

For the last input-adaptation checkpoint, the formal actor's original 48 input
columns, downstream actor layers and `std` remained bit-exact. The new-column
L2 norms were `0.6283/0.6205/1.0548`, and the first-layer bias changed by L2
`0.9047`. Therefore the failure is not explained by accidentally training the
wrong columns or by the policy ignoring the new inputs.

## Short-credit-path experiments

The failed reward-only tranches motivated an architecture in which confidence
can directly suppress the locomotion command seen by a frozen copy of the
formal actor. Intermediate experiments were deliberately kept separate and
retired when their own gate failed:

- a free action-residual adapter failed the lateral holdout and produced hard
  terminations in 506 of 512 environments;
- a nominal teacher target stopped in invalid state but failed command
  re-acquisition after recovery;
- an unbounded safe-command teacher passed the base profiles, but its learned
  gain had no stable deployment bound;
- forcing an exact unbounded gain target caused 48 of 512 hard terminations in
  the right-lateral holdout.

These failures were not continued. The surviving design is a frozen
model1450 backbone with a learned per-action blend between the original command
and a safe zero command. Both the trained teacher target and inference gain are
bounded to `0.8`. At gain zero, the complete actor path is exactly model1450;
the critic and optimizer are fresh. This supplies a short, auditable confidence
credit path without permitting an unconstrained residual action.

The registered task is
`Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-BoundedSafeCommand-v0`.
The runner experiment name is
`anymal_d_locomotion_slam_confidence_bounded_safe_command_v1`.

## Bounded iteration-zero baseline

The untrained actor-only checkpoint is:

`logs/rsl_rl/anymal_d_locomotion_slam_confidence_bounded_safe_command_v1/2026-08-12_14-16-51_bounded_safe_command_bootstrap/bootstrap_model_0.pt`

SHA-256: `7d827f76573ce2fa48f79d8a44add24bd33645601b5956c6898ce405a4d8deea`

Its safe-command gain is exactly zero, its backbone and action standard
deviation are bit-exact with model1450, its critic is fresh and its optimizer
has no state. The behavior report is
`docs/validation/slam_confidence_bounded_safe_command_iteration0_behavior_gate.json`
(SHA-256
`46a0a01fbbd6b4b0f2a7631612434a4d7a0137e898c53b401132f77d76875d82`).
It exactly repeats the formal-policy baseline: healthy and invalid settled
speeds are `1.494046` and `1.494028 m/s`, and the degraded/healthy ratio is
`0.999964`. It therefore preserves iteration-zero behavior and, as required,
fails the untrained degraded/invalid stop gates.

Iteration-zero TorchScript/ONNX export parity also passed on 256 samples:
TorchScript maximum absolute error was `0`, ONNX was `3.576e-6`.

## Qualified bounded candidate

The qualification run restarted from formal model1450 rather than resuming any
earlier candidate:

`logs/rsl_rl/anymal_d_locomotion_slam_confidence_bounded_safe_command_v1/2026-08-12_14-20-44_bounded_safe_command_qualification_20iter/model_19.pt`

Checkpoint SHA-256:
`73d5870de4acc5c79b0d0fa30f15cd70ed5c9539f9b7ec9a4d35d30d6377fb16`.
Training used 4,096 environments, seed 42, 20 iterations and 1,966,080
simulation steps. The final gain range was `0.792782..0.796848` against the
hard `0.8` limit; the final two teacher losses rounded to `0.0001`.

Five independent seed-43, 512-environment, 1,000-step profiles passed both the
original behavior gate and the additional distinct-environment hard-termination
overlay (`<=1%`):

| Profile | Healthy speed | Degraded / healthy | Invalid speed | Invalid tail | Recovery speed | Hard-terminated envs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | `1.493843` | `0.277624` | `0.051550` | `0` | `1.218892` | `0/512` |
| combined | `1.496417` | `0.292980` | `0.108266` | `0` | `1.222759` | `0/512` |
| lateral left | `1.516218` | `0.307405` | `0.186376` | `0.000078` | `1.222202` | `1/512` |
| lateral right | `1.454215` | `0.313544` | `0.206863` | `0.005273` | `1.179043` | `1/512` |
| reverse | `1.493464` | `0.324083` | `0.103101` | `0` | `1.212391` | `1/512` |

The machine-readable aggregate is
`docs/validation/slam_confidence_bounded_safe_command_model19_qualification.json`
(SHA-256
`b0474d52a9164da3c8529a7e50e10496e44708eba8dd114351f1de68570f81c2`).

The exported ONNX is:

`exported/anymal_d_locomotion_slam_confidence_bounded_safe_command_v1/2026-08-12_14-20-44_bounded_safe_command_qualification_20iter/policy.onnx`

SHA-256: `511383d9a6b0d4d6c26588667e195f9d66b7e703db991af59fcc8aad4aeb84df`.
Checkpoint/TorchScript/ONNX parity passed on 256 samples; TorchScript maximum
absolute error was `0`, ONNX was `4.2915e-6`. The parity report is
`docs/validation/slam_confidence_bounded_safe_command_model19_export_parity.json`.

## Conclusion and next gate

The second round produced one candidate that satisfies the simulated
confidence-state qualification. Passing that gate authorized backend
evaluation, not promotion. The backend screening result below prevents
promotion; Recovery v0.4.0 model1450 remains the formal policy.

## Two-backend live screening

The machine-readable matrix freezes the same checkpoint, ONNX, metadata and
parity evidence before launch. It requires exact backend/calibration identity,
native deskew, `/slam/odom` policy input, a real 51-D confidence observation,
and the existing locomotion stability gate:

`configs/slam_confidence_locomotion_matrix.yaml`

SHA-256:
`ada67bda2713ec6dec20e25b6cd8caa75640fc551ccc52cfa80d9df46e929921`.
The screening is configured as two backends by five profiles with one
repetition. It was intentionally run fail-fast on `forward_1_5` before the
remaining profiles.

- FAST-LIO2 passed: exact calibration
  `native-v1-1e6cf8347be1`, 0 terminations, 706/736 policy records with
  `tracking_valid=1`, and both policy-diagnostics and stability gates passed.
- LIO-SAM failed: exact calibration `native-v1-edc098b0bd98` and the 51-D
  consumption gate passed, but locomotion had one termination,
  `foot_slip_first`, `vx MAE=1.09995 m/s`, and `vy MAE=0.320096 m/s`.

The remaining eight cells were not run because the minimum backend screen had
already failed. The result is not explained solely by model19's invalid-state
suppression: divergence was visible before the policy first consumed an
invalid confidence vector at 7.73 s. Offline comparison of LIO-SAM's
pose-delta canonical twist to same-stamp GT measured mean L2 error about
`1.1066 m/s`; GT was evaluator-only and never entered runtime.

A formal model1450 control on the same LIO-SAM path also failed with three
terminations (`vx/vy MAE=0.371643/0.210254 m/s`). A causal 0.5 s pose window
still produced three terminations. A timestamp-matched native
IMU-preintegration velocity control removed hard terminations but still failed
`foot_slip_first` and `vx MAE=0.266499 m/s`. Both experimental twist arms were
retired and removed from source; neither altered LIO-SAM, the confidence
estimator, native deskew, or formal artifacts.

The machine-readable screening summary is
`docs/validation/slam_confidence_bounded_safe_command_model19_backend_screening.json`.
The blocker is now LIO-SAM canonical odometry state quality for policy
consumption. No additional profile or training run should proceed until that
contract has a separately designed and frozen qualification gate.

## LIO-SAM policy-state contract and high-rate control

The next gate was frozen in `configs/liosam_policy_state_quality.yaml`. It
separates the low-rate exact-stamp mapping authority (`/slam/odom`) from an
experimental high-rate policy-state topic (`/slam/policy_odom`). The latter is
derived from LIO-SAM's existing map-corrected IMU predictor
`/lio_sam/odometry/imu`; its upstream world-frame velocity is rotated into the
current `base_link` frame. Upstream LIO-SAM, native deskew, confidence
calibration and the formal policy were not modified. GT was added only to the
offline diagnostics/evaluator.

Against formal 48-D model1450, all three registered `forward_1_5` repetitions
passed the direct age/accuracy/outlier gate:

- policy-state age p95 was about `0.010 s`, versus about `0.44 s` on the old
  mapping pose-delta path;
- body velocity MAE ranges were `x=0.0558..0.0589`,
  `y=0.0674..0.0729`, and `z=0.0208..0.0248 m/s`;
- update rate was about `50 Hz`, with zero timestamp regressions.

That improvement did not complete locomotion qualification. All three
registered repetitions had `foot_slip_first` followed by body instability at
`14.72/14.84/14.92 s`, during the stopped tail. There were no hard
terminations, but all three stability gates failed. Offline one-step
counterfactual evaluation replaced only the recorded three velocity inputs
with same-time GT: active-motion action differences remained inside the
provisional limit, while stopped-tail mean absolute action differences were
`0.1443..0.1776` and maxima `0.4940..0.8227`, so all three action-sensitivity
gates failed. This confirms that the residual stopped-state velocity error is
small numerically but large enough to materially change model1450 actions.

A single 0.1 s causal regression over the high-rate fused pose was also tried.
Map corrections became derivative spikes; velocity MAE rose to
`1.185/0.570/1.334 m/s` and the robot terminated once. That arm was removed
from source. The retained high-rate native predictor is an isolated candidate
topic and is not the default policy input.

Machine-readable evidence is
`docs/validation/liosam_policy_state_v1_forward_control.json`. The LIO-SAM
policy-state gate remains failed, so model19's remaining backend matrix is
still blocked. The architectural next step is to source the locomotion base
velocity from a policy-grade proprioceptive estimator (IMU plus leg/contact
state, or the physical ANYmal state estimator), while SLAM supplies global pose
and confidence only. More differentiation or smoothing of mapping pose is not
an approved next arm.
