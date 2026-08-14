# SLAM confidence gait-mode governor v1

Date: 2026-08-13

Status: architecture gate failed. This design is not an exported or deployable
policy candidate, and Recovery v0.4.0 model1450 remains formal.

## Motivation

The retired transition-gated safe-gait checkpoints exposed a trade-off between
hard termination and gait quality. Their actor used a memoryless confidence
scale and could add a bounded residual directly to all 12 actions during a
transition. The next experiment removes that residual and makes the command
transition itself stateful and auditable.

## Architecture

The governor owns four modes: `TRACK`, `DECELERATE`, `HOLD`, and `RECOVER`.
It consumes only the backend-neutral 51-D confidence tuple. SLAM validity
remains an immediate fail-closed authority; the governor shapes only the
locomotion command passed to the frozen actor.

- `TRACK`: command scale is exactly one.
- `DECELERATE`: scale follows the continuous safe scale
  `valid * clamp((confidence - 0.2) / 0.8, 0, 1)`. A `0.8/s` downward rate
  limit prevents an abrupt invalidation from becoming an instantaneous stop.
- `HOLD`: scale is exactly zero and cannot exit before both hold and healthy
  recovery dwell complete.
- `RECOVER`: after the healthy dwell, scale follows the same continuous safe
  scale with a `2.0/s` upward rate limit. Invalidity or renewed confidence loss
  returns immediately to `DECELERATE`.

The frozen contract is
`configs/slam_confidence_gait_mode_v1.yaml`. Deceleration starts as soon as the
continuous confidence-derived safe scale falls below one; it no longer waits
for confidence to cross the degrade threshold. Downward and upward rate limits
are therefore independently `0.8/s` and `2.0/s`.

State is deliberately outside the neural actor:

- Isaac Lab holds one batched state per environment and resets it when that
  environment's episode counter regresses.
- ROS deployment holds one scalar state in `PolicyRuntime` and clears it with
  the existing episode reset handshake.
- The candidate actor remains stateless and evaluates the frozen model1450
  backbone on the governed legacy 48-D prefix. Confidence remains appended at
  offsets 48..50 for the unchanged observation contract.

This avoids encoding transition history inside shuffled PPO minibatches or an
implicit stateful ONNX graph. Isaac Lab code does not import `rclpy`.

## Initial validation gates

Before training or export:

1. actor-only warm-start from formal model1450;
2. fresh critic and empty optimizer;
3. saved `bootstrap_model_0.pt` before any update;
4. arbitrary-observation healthy actor parity with model1450, maximum absolute
   error exactly zero;
5. seed 43, 512 environments, 1,000 steps and the unchanged behavior gate;
6. matched A/B against the existing model1450 supervisor using the unchanged
   gait-value gate and velocity estimator candidate 08;
7. distinct hard-terminated environment fraction no greater than 1%.

Only if the deterministic governor passes behavior/safety and approaches the
gait-value gate should a separate structured gait-parameter head be designed.
No failed safe-gait checkpoint may be resumed.

## Current verification

Pure-core tests cover healthy exact scale, immediate stale-high invalidation,
monotonic deceleration, hold/recovery dwell, confidence chatter, and per-tick
Torch/NumPy implementation parity. `PolicyRuntime` tests confirm that the
governed command occupies observation offsets 9..11 while the original
confidence tuple remains at offsets 48..50.

Isaac Sim successfully instantiated the 51-D task and created an exact
model1450 iteration-0 bootstrap at
`logs/rsl_rl/anymal_d_locomotion_slam_confidence_gait_mode_v1/2026-08-13_17-11-09_gait_mode_architecture_gate/bootstrap_model_0.pt`.

The first synchronized 512-environment behavior run used the late threshold
transition and `2.0/s` deceleration. It retained good settled-stop gait but
missed degraded target tracking and hard-terminated 9/512 distinct
environments (1.758%), above the frozen 1% safety overlay.

A single causal correction then made deceleration follow the continuous safe
scale from the first confidence decrease, with a `0.8/s` downward rate. The
report is `behavior_gait_gate_v3.json`. All velocity checks passed, but
89/512 distinct environments (17.383%) hard-terminated; recovery roll/pitch
rate RMS reached `0.925 rad/s`. The report's legacy `passed` field used
per-timestep termination frequency and is therefore not a valid safety pass.
The evaluator now applies the 1% threshold to distinct terminated environments
and records termination phase counts for future runs.

Conclusion: B demonstrated that an external command governor changes behavior,
but neither tested deterministic form is safe. No rate sweep, gait-value
promotion, export, or long training follows from B. C must learn gait adaptation
inside PPO instead of treating confidence as only a speed limiter.
