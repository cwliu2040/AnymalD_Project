"""ROS-independent simulated SLAM confidence for PPO training."""

from __future__ import annotations

import torch


def simulated_slam_confidence(
    env,
    cycle_s: float = 10.0,
    phase_offset_mode: str = "distributed",
) -> torch.Tensor:
    """Return confidence, tracking-valid, and normalized age for each env.

    The deterministic per-environment phase offsets cover healthy, gradual
    degradation, stale-high hard loss, and recovery without importing ROS or
    using simulator ground truth as a runtime confidence signal.
    """
    if cycle_s <= 0.0:
        raise ValueError("cycle_s must be positive")
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.float32)
    if phase_offset_mode == "distributed":
        offsets = torch.frac(env_ids * 0.61803398875)
    elif phase_offset_mode == "synchronized":
        offsets = torch.zeros_like(env_ids)
    else:
        raise ValueError(
            "phase_offset_mode must be 'distributed' or 'synchronized'"
        )
    elapsed_s = env.episode_length_buf.to(torch.float32) * float(env.step_dt)
    phase = torch.frac(elapsed_s / cycle_s + offsets)

    confidence = torch.ones_like(phase)
    valid = torch.ones_like(phase)
    age = torch.zeros_like(phase)

    # 3 s healthy -> 2 s gradual deceleration -> 2 s invalid stop ->
    # 1 s invalid recovery dwell -> 2 s valid command re-acquisition.
    degrading = (phase >= 0.30) & (phase < 0.50)
    degrade_progress = torch.clamp((phase - 0.30) / 0.20, 0.0, 1.0)
    confidence = torch.where(
        degrading,
        1.0 - 0.8 * degrade_progress,
        confidence,
    )
    age = torch.where(degrading, 0.30 * degrade_progress, age)

    lost = (phase >= 0.50) & (phase < 0.70)
    stale_high = (env_ids.to(torch.int64) % 2) == 0
    lost_score = torch.where(stale_high, torch.ones_like(phase), 0.2 * torch.ones_like(phase))
    confidence = torch.where(lost, lost_score, confidence)
    valid = torch.where(lost, torch.zeros_like(valid), valid)
    lost_progress = torch.clamp((phase - 0.50) / 0.20, 0.0, 1.0)
    age = torch.where(lost, 0.3 + 0.7 * lost_progress, age)

    recovering = phase >= 0.70
    recover_progress = torch.clamp((phase - 0.70) / 0.30, 0.0, 1.0)
    confidence = torch.where(
        recovering,
        0.2 + 0.8 * recover_progress,
        confidence,
    )
    valid = torch.where(
        recovering,
        (recover_progress >= (1.0 / 3.0)).to(valid.dtype),
        valid,
    )
    age = torch.where(recovering, 0.5 * (1.0 - recover_progress), age)

    return torch.stack((confidence, valid, torch.clamp(age, 0.0, 1.0)), dim=-1)


def confidence_safe_scale(env, cycle_s: float = 10.0) -> torch.Tensor:
    """Map the simulated contract to the safe target-command scale."""
    observation = simulated_slam_confidence(env, cycle_s=cycle_s)
    confidence = observation[:, 0]
    valid = observation[:, 1]
    return valid * torch.clamp((confidence - 0.2) / 0.8, 0.0, 1.0)


def confidence_phase(env, cycle_s: float = 10.0) -> torch.Tensor:
    """Return synchronized normalized phase for deterministic evaluation."""
    elapsed_s = env.episode_length_buf.to(torch.float32) * float(env.step_dt)
    return torch.frac(elapsed_s / cycle_s)
