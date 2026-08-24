# Risk-aware locomotion governor v1 staged implementation

Date: 2026-08-22

## Goal and boundary

The target closed loop keeps the calibrated SLAM confidence contract unchanged and lets locomotion
choose the fastest command scale whose predicted future 0.5 s localization risk is acceptable.
It does not teach confidence that slow motion is automatically good.

Arm B is split by role:

- its normal continuous confidence throttle is the baseline that a valid risk-aware decision may
  replace and exceed;
- its invalid/stale exact-stop behavior remains a non-bypassable hard boundary;
- if the optional risk model is missing or malformed while SLAM remains valid and fresh, the
  system falls back to the exact Arm-B continuous scale.

This permits a candidate to outperform B during valid operation without losing fail-closed
behavior when the information required by the candidate is unavailable.

## Stages

1. **Interface and hard-boundary decomposition.** Freeze freshness, numeric, fallback, candidate
   scale, and risk interfaces as ROS-independent pure functions.
2. **Speed-scale causal pilot design.** Treat speed as the intervention rather than a confound;
   freeze matched candidate scales, outcomes, monotonicity/Pareto gates, and fresh blocks.
3. **Small real-SLAM pilot.** Collect only after separate live authorization. Both backends remain
   separate and all future labels remain offline.
4. **Action-conditioned risk model.** Train only if speed interventions have repeatable causal
   risk response; the runtime model receives causal history and a candidate scale, never GT or a
   future label.
5. **Simulation closed loop.** Choose the highest admissible scale, retain B fallback, and verify
   safety, rate limits, hysteresis, and model failure injection.
6. **Pareto qualification.** Compare the candidate against full B at equal risk and equal progress.
   Only a genuine Pareto improvement can unlock PPO or physical-integration planning.

## Phase-1 implementation

`risk_aware_governor_core.py` implements stateless arbitration only; it is not connected to the ROS
policy node or Isaac task. A valid risk prediction may choose scale `0.75` at confidence `0.45`,
where full B would choose `0.3125`, proving that B's soft throttle is replaceable. Invalid tracking,
source age above `0.50 s`, receipt age above `0.15 s`, or malformed SLAM values produce exact zero.
Unavailable/malformed risk predictions fall back to exact B. A valid prediction above the
development risk limit produces a stop rather than silently using B.

The configuration is `interface_skeleton_only_no_model_no_live_authorization`: no risk model,
simulation execution, policy replacement, PPO, default switch, or physical use is authorized.
The `0.10` risk limit is a development default and must be separately justified and refrozen before
any live collection or controller qualification.

## Phase-3/3b fixed-scale speed pilot implementation

`configs/slam_speed_scale_causal_pilot_v1.yaml` now freezes speed as the treatment rather than a
nuisance variable. The four assigned uniform XYZ command scales are `1.00`, `0.75`, `0.50`, and
`0.25`; uniform scaling preserves command curvature. Tracking invalid/stale always means exact-zero
command, not an assigned low-speed arm.

The plan contains an eight-cell wiring smoke at fresh block 560 and a 32-cell causal pilot at fresh
blocks 561--562 across both backends and curve/lateral profiles. Realized speed is an intended
mediator, so there is no matched-speed gate. The development gate instead requires verified scale
and speed ordering plus consistent future-hazard dose response, while reporting the full
risk--progress tradeoff. Passing may unlock only offline action-conditioned model development.

The protocol validator renders a balanced Latin-square schedule and rejects all earlier blocks.
Phase 3b now supplies four hash-locked ONNX/TorchScript wrappers around frozen Arm B. Each wrapper
uniformly changes observation offsets 9--11 before invoking B; tracking invalid or normalized age
above one sends an exact-zero command into B. Export checks compare the regenerated Arm B against
the existing hash-locked ONNX, then check eager/TorchScript/ONNX parity and the command transform.

The live trace validator does not trust the arm label alone. For every recorded 51-D observation it
reconstructs the assigned valid/fresh command or exact-zero invalid/stale command, evaluates the
hash-locked base policy, and compares the expected 12-D action with the recorded raw action. The
runner connects the existing real-backend ROS launch, stability evaluation, offline 0.5 s usability,
map consistency, estimator replay, run record, and causal analyzer. It captures the exact dirty
worktree in a live manifest and stops on wiring, contract, non-finite metric, or scale-trace failure.
A simulation fall remains an episode outcome and is not by itself a matrix stop.

Both plan schedules and the artifact release validate offline. After separate authorization, the
block-560 wiring smoke completed 8/8 across both backends and all four scales with `WIRING_PASS`.
All cells passed their data-integrity gate, maximum reconstructed-action error was
`7.75e-7`, invalid/stale exact-zero checks passed, and there were no falls or base contacts. The
manifest and decision hashes are locked in the release so a later stage can recompute the gate from
the eight raw run records rather than trusting the label.

Live authorization was closed immediately after the smoke. At that point the 32-cell causal pilot
remained a separate decision and had not been implicitly authorized or started.

## Phase-4 causal pilot result

After a second explicit authorization, blocks 561--562 completed all 32 scheduled cells. Data
integrity passed, every fixed-scale trace passed, the maximum reconstructed-action error was
`1.20e-6`, and invalid/stale exact-zero checks passed. One fall with base contact occurred in the
LIO-SAM curve `scale_100` control at block 562; no reduced-scale arm had a safety event, so the
predefined reduced-scale-specific repeated-harm condition did not occur.

The frozen decision is `FAIL`, specifically because future 0.5 s hazard was nondecreasing with
assigned scale in only one of four backend/profile strata; the gate required at least three. Speed
was ordered in three of four strata and the average reduced-minus-full hazard was favorable for
both FAST-LIO2 (`-0.1407`) and LIO-SAM (`-0.0408`), but these averages do not repair the missing
dose response. Examples of non-monotonicity include FAST-LIO2 curve hazard rising again at scale
0.25, LIO-SAM curve scale 0.75 being worse than scale 1.00, and LIO-SAM lateral scale 0.25 being
worse than scales 0.50/0.75.

Block inspection shows why simply adding a model would be unsafe. FAST-LIO2 block 561 had zero
hazard in every arm, while block 562 contained most failures; LIO-SAM curve also showed strong
non-monotonic block effects, including only 79 eligible hazard ticks for block-561 scale 0.75.
Whole-episode scaling changes spatial progress before the 0.5 s endpoint, so the arms can reach
different route states even under the same seed and time-based point-density schedule. A redesign
should therefore use short, pre-scheduled scale pulses after an identical untreated prefix (or an
equivalent state-restored design), so treatment begins from a genuinely matched state and exposure
horizon. More repetitions of the unchanged whole-episode arms would not fix that identifiability
problem.

Therefore these runs cannot unlock an action-conditioned risk model. The valid conclusion is not
that slowing can never help SLAM; it is that the current four-arm treatment does not provide a
repeatable monotonic mapping suitable for learning the proposed closed-loop governor. Live
authorization was closed after collection, and no model, PPO, or physical stage was started.
