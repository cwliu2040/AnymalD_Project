# SLAM confidence structured gait head v1

Date: 2026-08-13

Status: architecture and one-iteration PPO smoke passed; the fixed 25-iteration
pilot failed the behavior gate. No gait-value, replay/live, backend-matrix, or
deployment promotion gate has passed. Recovery v0.4.0 model1450 remains formal.

## Purpose

Candidate C addresses the failed external-governor B result. The raw velocity
command remains in the 51-D policy observation. Confidence does not directly
clamp command speed. A trainable head consumes the 51-D observation and frozen
model1450 action, then emits only four bounded gait coordinates:

1. signed stride modulation, where positive values attenuate amplitude;
2. crouch;
3. stance width;
4. action smoothing toward the previous action.

These coordinates are projected through fixed canonical 12-joint bases. The
head cannot emit an arbitrary 12-D residual. The whole structured adjustment is
multiplied by confidence severity, so a healthy `[1, 1, 0]` confidence tuple is
exact model1450 even after training.

## Warm-start and smoke evidence

The 64-environment seed-43 bootstrap is:

`logs/rsl_rl/anymal_d_locomotion_slam_confidence_structured_gait_v1/2026-08-13_17-25-51_four_coordinate_gait_head/bootstrap_model_0.pt`

SHA-256: `5f1c0b48a6a7e2f6fb39c16c6865396903d5775acf83f5e4d2446f0546c6b0df`.
Its report records source model1450 SHA-256
`3bbaff6247fdb59808af42a59e2ec9477c047531a3e2aa6cd0df6208a796feee`,
fresh critic, empty optimizer, exact-zero final gait-head layer, and trainable
scope limited to the gait head plus critic.

A separate 64-environment seed-43 one-iteration run produced:

`logs/rsl_rl/anymal_d_locomotion_slam_confidence_structured_gait_v1/2026-08-13_17-26-33_four_coordinate_gait_head/model_0.pt`

SHA-256: `d3f1d599b70c1081a30806c7c4465cf63591ef3e5ae7a4ebf7c51b038a77e3e3`.
All 6 gait-head parameter tensors changed, with maximum absolute delta
`0.0019810`; all 8 frozen backbone tensors remained bit-identical. This proves
PPO gradients and checkpointing reach the intended low-dimensional head. One
iteration is not a performance result and must not be evaluated as a trained
candidate.

## Fixed 25-iteration pilot

The predeclared pilot used 512 environments, seed 43, 25 PPO iterations, and
checkpoints every 5 iterations. The final checkpoint is:

`logs/rsl_rl/anymal_d_locomotion_slam_confidence_structured_gait_v1/2026-08-13_17-34-02_four_coordinate_gait_head/model_24.pt`

SHA-256: `10fa91ba3e5fcfb09fe37ab8aabc8fbf90a988ddf21cab48c6cbcd4d4c348b1d`.
All 6 gait-head tensors changed from bootstrap and the maximum absolute
parameter delta was `0.030823`. The final synchronized 512-environment report
is `behavior_gate_model24.json`.

The candidate had zero hard terminations and retained healthy/recovery
locomotion, but did not respond sufficiently to low confidence:

- degraded speed ratio to healthy: `1.013` (required no greater than `0.65`);
- degraded-late linear RMSE: `1.303 m/s` (required no greater than `0.50`);
- invalid-settled mean planar speed: `1.500 m/s` (required no greater than
  `0.25`);
- invalid-settled samples above `0.50 m/s`: `100%` (required no greater than
  `5%`).

Therefore C-v1 fails. Gait-value evaluation, export, longer training, and
intermediate-checkpoint selection are intentionally stopped. The evidence
shows that posture, stance-width, smoothing, and bounded stride modulation
alone preserved balance but did not provide an explicit learnable locomotion
intent/step-frequency path capable of reaching the confidence-conditioned stop
target in the fixed pilot.

## Next architecture gate

Do not merely extend C-v1 training or tune reward weights. A successor should
add a low-dimensional PPO-controlled locomotion-intent coordinate that blends
the frozen raw-command and frozen safe-command model1450 actions, alongside the
four gait coordinates. This differs from B because PPO decides the blend jointly
with balance-related gait parameters; confidence is not a deterministic
external speed limiter. It must again start at exact model1450, use a fresh
critic/optimizer, and pass the same fixed behavior and distinct-env safety gate
before gait-value evaluation.
