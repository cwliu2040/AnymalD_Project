"""ROS-independent simulated SLAM confidence for PPO training."""

from __future__ import annotations

import torch

from anymal_locomotion.gait_mode_governor import BatchedGaitModeGovernor


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


def confidence_recovery_mask(
    env,
    cycle_s: float = 10.0,
    phase_offset_mode: str = "distributed",
) -> torch.Tensor:
    """Return the privileged training label for the confidence recovery phase."""
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
    return (phase >= 0.70).to(dtype=torch.float32)


def confidence_degradation_mask(
    env,
    cycle_s: float = 10.0,
    phase_offset_mode: str = "distributed",
) -> torch.Tensor:
    """Return the privileged training label for gradual degradation."""
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.float32)
    offsets = (
        torch.frac(env_ids * 0.61803398875)
        if phase_offset_mode == "distributed"
        else torch.zeros_like(env_ids)
    )
    if phase_offset_mode not in ("distributed", "synchronized"):
        raise ValueError(
            "phase_offset_mode must be 'distributed' or 'synchronized'"
        )
    elapsed_s = env.episode_length_buf.to(torch.float32) * float(env.step_dt)
    phase = torch.frac(elapsed_s / cycle_s + offsets)
    return ((phase >= 0.30) & (phase < 0.50)).to(dtype=torch.float32)


def gait_mode_command_scale(
    env,
    cycle_s: float = 10.0,
    phase_offset_mode: str = "distributed",
) -> torch.Tensor:
    """Return the cached 50 Hz stateful gait-mode command scale.

    Observation and reward managers may request this value in either order.
    The episode-length cache guarantees exactly one state transition per
    simulation step and resets each environment independently.
    """
    confidence_observation = simulated_slam_confidence(
        env,
        cycle_s=cycle_s,
        phase_offset_mode=phase_offset_mode,
    )
    confidence = confidence_observation[:, 0]
    valid = confidence_observation[:, 1]
    episode_length = env.episode_length_buf.to(torch.int64)
    governor = getattr(env, "_slam_confidence_gait_mode_governor", None)
    last_episode_length = getattr(env, "_slam_confidence_gait_mode_last_step", None)
    if governor is None:
        governor = BatchedGaitModeGovernor(env.num_envs, env.device)
        governor.reset(confidence, valid)
        env._slam_confidence_gait_mode_governor = governor
        env._slam_confidence_gait_mode_last_step = episode_length.clone()
        return governor.command_scale

    if last_episode_length is not None and bool(torch.equal(episode_length, last_episode_length)):
        return governor.command_scale
    reset_mask = (
        torch.zeros_like(episode_length, dtype=torch.bool)
        if last_episode_length is None
        else episode_length < last_episode_length
    )
    governor.update(confidence, valid, float(env.step_dt), reset_mask=reset_mask)
    env._slam_confidence_gait_mode_last_step = episode_length.clone()
    return governor.command_scale


def gait_mode_velocity_command(
    env,
    command_name: str,
    cycle_s: float = 10.0,
    phase_offset_mode: str = "distributed",
) -> torch.Tensor:
    """Scale the original command through the stateful gait-mode governor."""
    command = env.command_manager.get_command(command_name)
    return command * gait_mode_command_scale(
        env,
        cycle_s=cycle_s,
        phase_offset_mode=phase_offset_mode,
    ).unsqueeze(-1)
