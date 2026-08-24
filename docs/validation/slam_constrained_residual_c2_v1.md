# Ceiling-aware constrained residual C2 v1

Date: 2026-08-22

This document freezes the next candidate direction before short-pulse evidence or PPO exists. Its
ROS-independent command/action composition core and offline qualification gate are implemented; it
is not a trained policy and not authorization to run a simulation matrix.

## Why this is not B or old C

B remains the deterministic safety fallback: confidence reduces the requested command, and
invalid or stale SLAM commands exact zero. C2 does not receive B's already-soft-scaled command as
its normal input. It sees the original requested command and can choose separate XYZ scales plus a
bounded joint-action residual around frozen model1450. Therefore it can preserve useful forward or
yaw motion instead of only applying B's uniform slowdown.

Old C replaced the ordinary linear and yaw velocity tracking functions with
`confidence_track_lin_vel_xy_exp` and `confidence_track_ang_vel_z_exp`. Those functions multiply
the target command by a confidence-derived safe scale. A lower target consequently makes slow or
stopped motion easier to score well. C2 explicitly forbids both terms. It reuses
`AnymalDLocomotionRobustRewardsCfg`, tracks the original requested command, and adds only constrained
terms for future SLAM advantage over B, unnecessary reduction, healthy deviation, and invalid-state
safety. It does not duplicate the locomotion velocity reward.

## Ceiling-aware success rule

FAST-LIO2 is treated as a ceiling/noninferiority safeguard because B already performs strongly
there. C2 may equal B if risk is no more than 0.01 worse, progress is at least 98% of B, stopping is
not increased, and no safety event is added. Superiority on FAST-LIO2 is not mandatory.

LIO-SAM supplies the prespecified headroom test. At least one predefined degraded LIO-SAM stratum
must have risk no worse than B, progress at least 5% higher, no extra stopping, and no extra safety
event. Backends are reported separately; a pooled average cannot hide either failure.

In those same LIO-SAM strata, the candidate's action behavior is compared with both B and old C. If
more than 80% of samples are equivalent within the frozen action tolerance across repeated
evaluations, qualification returns `COLLAPSED_TO_B_OR_OLD_C`. Merely being different is not rewarded;
the difference must produce the predefined risk/progress improvement.

## Current gates

`constrained_residual_c2_core.py` first validates SLAM state and the entire candidate output, then
chooses the effective command. Only after that should the caller evaluate frozen model1450 with the
same effective command present in its observation; the validated residual is added last. Missing or
malformed candidate output returns B's exact soft-scaled command and zero residual. Invalid or stale
SLAM and a well-formed but risk-inadmissible candidate both use an exact-zero command and zero
residual, with distinct hard-stop and risk-stop modes. Candidate values are never silently clipped
into the valid range.

The pure core is complete. Short-pulse headroom, an action-conditioned risk signal, policy-node and
training wiring, PPO, and physical use are all false. Until the short-pulse experiment independently
demonstrates controllable headroom, no risk model or candidate training should start.
