"""Command-conditioned rewards for Recovery v0.5."""

from __future__ import annotations

import torch
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide

from .slam_confidence import (
    confidence_degradation_mask,
    confidence_recovery_mask,
    confidence_safe_scale,
    simulated_slam_confidence,
)


def _high_combined_command_mask(
    env,
    command_name: str,
    min_forward_speed: float,
    min_yaw_speed: float,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return (
        (command[:, 0] >= min_forward_speed)
        & (torch.abs(command[:, 2]) >= min_yaw_speed)
    )


def _refinery_command_mask(
    env,
    command_name: str,
    target_forward_speed: float,
    target_yaw_speed: float,
    command_tolerance: float,
) -> torch.Tensor:
    """Select the moving phases in the reproduced refinery command trace."""
    command = env.command_manager.get_command(command_name)
    forward_match = (
        torch.abs(command[:, 0] - target_forward_speed)
        <= command_tolerance
    )
    yaw_match = (
        torch.abs(torch.abs(command[:, 2]) - target_yaw_speed)
        <= command_tolerance
    )
    straight_match = (
        forward_match
        & (torch.abs(command[:, 2]) <= command_tolerance)
    )
    pivot_match = (
        (torch.abs(command[:, 0]) <= command_tolerance)
        & yaw_match
    )
    combined_match = forward_match & yaw_match
    return (
        (torch.abs(command[:, 1]) <= command_tolerance)
        & (straight_match | pivot_match | combined_match)
    )


def high_combined_feet_slide(
    env,
    command_name: str,
    min_forward_speed: float,
    min_yaw_speed: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize extra foot sliding only in the high-combined failure regime."""
    penalty = feet_slide(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
    )
    return penalty * _high_combined_command_mask(
        env,
        command_name,
        min_forward_speed,
        min_yaw_speed,
    )


def lateral_stance_slip_barrier(
    env,
    command_name: str,
    min_lateral_speed: float,
    free_slip_speed: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only stance-foot slip above a fixed lateral-command allowance."""
    command = env.command_manager.get_command(command_name)
    lateral_active = torch.abs(command[:, 1]) >= min_lateral_speed
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]
    foot_speed = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    excess = torch.clamp_min(foot_speed - free_slip_speed, 0.0)
    return torch.sum(torch.square(excess) * contacts, dim=1) * lateral_active


def mixed_yaw_tracking_barrier(
    env,
    command_name: str,
    min_planar_speed: float,
    min_yaw_speed: float,
    free_yaw_error: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw tracking error beyond a fixed allowance during mixed motion."""
    command = env.command_manager.get_command(command_name)
    mixed_active = (
        torch.linalg.vector_norm(command[:, :2], dim=1) >= min_planar_speed
    ) & (torch.abs(command[:, 2]) >= min_yaw_speed)
    asset = env.scene[asset_cfg.name]
    error = torch.abs(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    excess = torch.clamp_min(error - free_yaw_error, 0.0)
    return torch.square(excess) * mixed_active


def refinery_feet_slide(
    env,
    command_name: str,
    target_forward_speed: float,
    target_yaw_speed: float,
    command_tolerance: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize slip throughout the reproduced refinery moving phases."""
    penalty = feet_slide(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
    )
    return penalty * _refinery_command_mask(
        env,
        command_name,
        target_forward_speed,
        target_yaw_speed,
        command_tolerance,
    )


def high_combined_flat_orientation_l2(
    env,
    command_name: str,
    min_forward_speed: float,
    min_yaw_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Apply extra body-tilt control only during high-combined commands."""
    penalty = isaac_mdp.flat_orientation_l2(env, asset_cfg=asset_cfg)
    return penalty * _high_combined_command_mask(
        env,
        command_name,
        min_forward_speed,
        min_yaw_speed,
    )


def refinery_flat_orientation_l2(
    env,
    command_name: str,
    target_forward_speed: float,
    target_yaw_speed: float,
    command_tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize tilt throughout the reproduced refinery moving phases."""
    penalty = isaac_mdp.flat_orientation_l2(env, asset_cfg=asset_cfg)
    return penalty * _refinery_command_mask(
        env,
        command_name,
        target_forward_speed,
        target_yaw_speed,
        command_tolerance,
    )


def low_yaw_track_ang_vel_z_exp(
    env,
    command_name: str,
    std: float,
    target_yaw_speed: float,
    yaw_tolerance: float,
    max_planar_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Strengthen tracking only for the pure low-yaw regression profiles."""
    command = env.command_manager.get_command(command_name)
    low_yaw = (
        (torch.abs(command[:, 0]) <= max_planar_speed)
        & (torch.abs(command[:, 1]) <= max_planar_speed)
        & (
            torch.abs(torch.abs(command[:, 2]) - target_yaw_speed)
            <= yaw_tolerance
        )
    )
    tracking = isaac_mdp.track_ang_vel_z_exp(
        env,
        std=std,
        command_name=command_name,
        asset_cfg=asset_cfg,
    )
    return tracking * low_yaw


def low_curve_track_lin_vel_xy_exp(
    env,
    command_name: str,
    std: float,
    target_forward_speed: float,
    target_yaw_speed: float,
    command_tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Strengthen linear tracking for the low-speed curve regressions."""
    command = env.command_manager.get_command(command_name)
    low_curve = (
        (torch.abs(command[:, 0] - target_forward_speed) <= command_tolerance)
        & (torch.abs(command[:, 1]) <= command_tolerance)
        & (
            torch.abs(torch.abs(command[:, 2]) - target_yaw_speed)
            <= command_tolerance
        )
    )
    tracking = isaac_mdp.track_lin_vel_xy_exp(
        env,
        std=std,
        command_name=command_name,
        asset_cfg=asset_cfg,
    )
    return tracking * low_curve


def high_curve_track_lin_vel_xy_exp(
    env,
    command_name: str,
    std: float,
    target_forward_speed: float,
    target_yaw_speed: float,
    command_tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Strengthen linear tracking only for the 3 m/s curve regressions."""
    command = env.command_manager.get_command(command_name)
    high_curve = (
        (torch.abs(command[:, 0] - target_forward_speed) <= command_tolerance)
        & (torch.abs(command[:, 1]) <= command_tolerance)
        & (
            torch.abs(torch.abs(command[:, 2]) - target_yaw_speed)
            <= command_tolerance
        )
    )
    tracking = isaac_mdp.track_lin_vel_xy_exp(
        env,
        std=std,
        command_name=command_name,
        asset_cfg=asset_cfg,
    )
    return tracking * high_curve


def confidence_track_lin_vel_xy_exp(
    env,
    command_name: str,
    std: float,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track a confidence-scaled planar target during simulated SLAM loss."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    target = command * confidence_safe_scale(env, cycle_s=cycle_s).unsqueeze(-1)
    error = torch.sum(torch.square(target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / (std * std))


def confidence_track_ang_vel_z_exp(
    env,
    command_name: str,
    std: float,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track a confidence-scaled yaw target during simulated SLAM loss."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, 2]
    target = command * confidence_safe_scale(env, cycle_s=cycle_s)
    error = torch.square(target - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-error / (std * std))


def confidence_invalid_planar_speed_l2(
    env,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize residual planar motion only while tracking is invalid."""
    asset = env.scene[asset_cfg.name]
    valid = simulated_slam_confidence(env, cycle_s=cycle_s)[:, 1]
    speed_squared = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    return speed_squared * (valid < 0.5)


def confidence_invalid_yaw_rate_l2(
    env,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize residual yaw motion only while tracking is invalid."""
    asset = env.scene[asset_cfg.name]
    valid = simulated_slam_confidence(env, cycle_s=cycle_s)[:, 1]
    return torch.square(asset.data.root_ang_vel_b[:, 2]) * (valid < 0.5)


def confidence_invalid_action_l2(
    env,
    cycle_s: float = 10.0,
) -> torch.Tensor:
    """Penalize non-neutral joint offsets only while tracking is invalid."""
    valid = simulated_slam_confidence(env, cycle_s=cycle_s)[:, 1]
    action_cost = torch.sum(torch.square(env.action_manager.action), dim=1)
    return action_cost * (valid < 0.5)


def _confidence_severity(env, cycle_s: float) -> torch.Tensor:
    return 1.0 - confidence_safe_scale(env, cycle_s=cycle_s)


def confidence_gait_action_rate_l2(env, cycle_s: float = 10.0) -> torch.Tensor:
    """Penalize abrupt joint-target changes as confidence deteriorates."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta), dim=1) * _confidence_severity(env, cycle_s)


def confidence_gait_lin_vel_z_l2(
    env,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical bouncing outside the healthy confidence state."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2]) * _confidence_severity(env, cycle_s)


def confidence_gait_ang_vel_xy_l2(
    env,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch motion outside the healthy confidence state."""
    asset = env.scene[asset_cfg.name]
    rate = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return rate * _confidence_severity(env, cycle_s)


def confidence_gait_flat_orientation_l2(
    env,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body tilt outside the healthy confidence state."""
    penalty = isaac_mdp.flat_orientation_l2(env, asset_cfg=asset_cfg)
    return penalty * _confidence_severity(env, cycle_s)


def confidence_gait_feet_slide(
    env,
    sensor_cfg: SceneEntityCfg,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize stance-foot slip outside the healthy confidence state."""
    penalty = feet_slide(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg)
    return penalty * _confidence_severity(env, cycle_s)


def confidence_recovery_action_rate_l2(env, cycle_s: float = 10.0) -> torch.Tensor:
    """Penalize abrupt joint targets specifically during confidence recovery."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta), dim=1) * confidence_recovery_mask(
        env, cycle_s=cycle_s
    )


def confidence_recovery_ang_vel_xy_l2(
    env,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize recovery roll/pitch rate using a privileged training label."""
    asset = env.scene[asset_cfg.name]
    rate = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return rate * confidence_recovery_mask(env, cycle_s=cycle_s)


def confidence_recovery_flat_orientation_l2(
    env,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body tilt specifically while locomotion is reacquired."""
    penalty = isaac_mdp.flat_orientation_l2(env, asset_cfg=asset_cfg)
    return penalty * confidence_recovery_mask(env, cycle_s=cycle_s)


def confidence_degradation_feet_slide(
    env,
    sensor_cfg: SceneEntityCfg,
    cycle_s: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize stance-foot slip specifically during gradual degradation."""
    penalty = feet_slide(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg)
    return penalty * confidence_degradation_mask(env, cycle_s=cycle_s)
