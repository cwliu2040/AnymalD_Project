"""Observation terms for confidence-conditioned full-policy fine-tuning."""

from __future__ import annotations

import torch

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg

from anymal_locomotion.joint_training_motion_core import (
    angular_acceleration_l2,
    lidar_scan_rotation_distortion_l2,
    lidar_scan_translation_distortion_l2,
    linear_jerk_l2,
    localization_vulnerability,
)

from .slam_confidence import simulated_slam_confidence


def full_policy_history_frame(
    env,
    command_name: str,
    localization_mode: str,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return one canonical 51-D causal history frame.

    The first 48 values reproduce the legacy policy term ordering.  The
    command is read directly from the command manager and is never confidence
    scaled. J1 and J2 therefore differ only in the final localization triplet.
    ObservationManager owns the 20-frame oldest-to-newest circular buffer.
    """
    command = isaac_mdp.generated_commands(env, command_name=command_name)
    legacy = torch.cat(
        (
            isaac_mdp.base_lin_vel(env, asset_cfg=asset_cfg),
            isaac_mdp.base_ang_vel(env, asset_cfg=asset_cfg),
            isaac_mdp.projected_gravity(env, asset_cfg=asset_cfg),
            command,
            isaac_mdp.joint_pos_rel(env, asset_cfg=asset_cfg),
            isaac_mdp.joint_vel_rel(env, asset_cfg=asset_cfg),
            isaac_mdp.last_action(env),
        ),
        dim=-1,
    )
    if legacy.shape[-1] != 48:
        raise RuntimeError(
            f"joint-training legacy history frame is {legacy.shape[-1]}-D, expected 48-D"
        )
    if localization_mode == "neutral":
        localization = torch.stack(
            (
                torch.ones_like(command[:, 0]),
                torch.ones_like(command[:, 0]),
                torch.zeros_like(command[:, 0]),
            ),
            dim=-1,
        )
    elif localization_mode == "actual":
        localization = simulated_slam_confidence(
            env,
            cycle_s=cycle_s,
            phase_offset_mode="distributed",
        )
    else:
        raise ValueError("localization_mode must be 'neutral' or 'actual'")
    return torch.cat((legacy, localization), dim=-1)


def _motion_derivatives(env, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, ...]:
    """Return cached body-frame linear/angular acceleration and linear jerk."""
    asset = env.scene[asset_cfg.name]
    linear_velocity = asset.data.root_lin_vel_b
    angular_velocity = asset.data.root_ang_vel_b
    step = env.episode_length_buf.to(torch.long)
    cache = getattr(env, "_joint_training_motion_derivative_cache", None)
    if cache is not None and torch.equal(step, cache["step"]):
        return cache["linear_acceleration"], cache["angular_acceleration"], cache["linear_jerk"]

    if cache is None:
        linear_acceleration = torch.zeros_like(linear_velocity)
        angular_acceleration = torch.zeros_like(angular_velocity)
        linear_jerk = torch.zeros_like(linear_velocity)
    else:
        dt = float(env.step_dt)
        if dt <= 0.0:
            raise ValueError("joint-training derivative step must be positive")
        reset = step <= cache["step"]
        linear_acceleration = (linear_velocity - cache["linear_velocity"]) / dt
        angular_acceleration = (angular_velocity - cache["angular_velocity"]) / dt
        linear_jerk = (linear_acceleration - cache["linear_acceleration"]) / dt
        linear_acceleration[reset] = 0.0
        angular_acceleration[reset] = 0.0
        linear_jerk[reset] = 0.0

    env._joint_training_motion_derivative_cache = {
        "step": step.clone(),
        "linear_velocity": linear_velocity.clone(),
        "angular_velocity": angular_velocity.clone(),
        "linear_acceleration": linear_acceleration.clone(),
        "angular_acceleration": angular_acceleration.clone(),
        "linear_jerk": linear_jerk.clone(),
    }
    return linear_acceleration, angular_acceleration, linear_jerk


def _localization_vulnerability(env, cycle_s: float, severity_gain: float) -> torch.Tensor:
    """Privileged reward weight shared by J1/J2; never exposed through J1 input."""
    state = simulated_slam_confidence(
        env,
        cycle_s=cycle_s,
        phase_offset_mode="distributed",
    )
    return localization_vulnerability(state, severity_gain)


def joint_training_angular_acceleration_l2(
    env,
    cycle_s: float = 10.0,
    severity_gain: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize angular acceleration, not requested constant yaw rate."""
    _, angular_acceleration, _ = _motion_derivatives(env, asset_cfg)
    return angular_acceleration_l2(
        angular_acceleration,
        _localization_vulnerability(env, cycle_s, severity_gain),
    )


def joint_training_linear_jerk_l2(
    env,
    cycle_s: float = 10.0,
    severity_gain: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body jerk while leaving mean requested velocity unscaled."""
    _, _, linear_jerk = _motion_derivatives(env, asset_cfg)
    return linear_jerk_l2(
        linear_jerk,
        _localization_vulnerability(env, cycle_s, severity_gain),
    )


def joint_training_lidar_scan_translation_distortion_l2(
    env,
    scan_time_s: float = 0.10,
    cycle_s: float = 10.0,
    severity_gain: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-constant scan translation caused by body acceleration."""
    linear_acceleration, _, _ = _motion_derivatives(env, asset_cfg)
    return lidar_scan_translation_distortion_l2(
        linear_acceleration,
        _localization_vulnerability(env, cycle_s, severity_gain),
        scan_time_s,
    )


def joint_training_lidar_scan_rotation_distortion_l2(
    env,
    scan_time_s: float = 0.10,
    cycle_s: float = 10.0,
    severity_gain: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch scan motion and angular acceleration, preserving yaw."""
    asset = env.scene[asset_cfg.name]
    _, angular_acceleration, _ = _motion_derivatives(env, asset_cfg)
    return lidar_scan_rotation_distortion_l2(
        asset.data.root_ang_vel_b,
        angular_acceleration,
        _localization_vulnerability(env, cycle_s, severity_gain),
        scan_time_s,
    )
