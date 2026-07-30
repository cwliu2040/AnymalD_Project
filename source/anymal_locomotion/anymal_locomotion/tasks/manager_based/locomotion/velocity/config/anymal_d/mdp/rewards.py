"""Command-conditioned rewards for Recovery v0.5."""

from __future__ import annotations

import torch
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide


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
