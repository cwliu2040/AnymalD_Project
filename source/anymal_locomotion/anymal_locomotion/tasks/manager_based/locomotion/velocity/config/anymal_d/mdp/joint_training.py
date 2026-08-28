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
from anymal_locomotion.causal_slam_dynamics_core import (
    causal_slam_transition,
    delayed_localization_advantage,
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
    elif localization_mode == "causal":
        localization = causal_slam_state(env)
    else:
        raise ValueError("localization_mode must be 'neutral', 'actual', or 'causal'")
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


def _causal_initial_localization_state(env) -> torch.Tensor:
    """Deterministic reset coverage without a route timer or future label."""
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.float32)
    confidence = 0.25 + 0.75 * torch.frac((env_ids + 1.0) * 0.61803398875)
    valid = (confidence >= 0.30).to(confidence.dtype)
    age = torch.where(valid >= 0.5, torch.zeros_like(confidence), torch.full_like(confidence, 0.25))
    return torch.stack((confidence, valid, age), dim=1)


def _causal_slam_update(
    env,
    *,
    horizon_steps: int = 25,
    scan_time_s: float = 0.10,
    translation_scale_m: float = 0.02,
    rotation_scale_rad: float = 0.10,
    degradation_rate_hz: float = 0.80,
    recovery_rate_hz: float = 0.40,
    invalid_enter_confidence: float = 0.20,
    valid_exit_confidence: float = 0.40,
    maximum_age_s: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Advance an uncalibrated action-dependent proxy exactly once per step."""
    if horizon_steps <= 0 or scan_time_s <= 0.0:
        raise ValueError("causal SLAM horizon and scan time must be positive")
    step = env.episode_length_buf.to(torch.long)
    cache = getattr(env, "_causal_slam_dynamics_cache", None)
    if cache is not None and torch.equal(step, cache["step"]):
        return cache

    initial_state = _causal_initial_localization_state(env)
    if cache is None:
        history = initial_state[:, None, :].repeat(1, horizon_steps + 1, 1)
        cache = {
            "step": step.clone(),
            "state": initial_state,
            "history": history,
            "point_support": 0.60
            + 0.40
            * torch.frac(
                (torch.arange(env.num_envs, device=env.device, dtype=torch.float32) + 1.0)
                * 0.41421356237
            ),
            "scan_translation_error_m": torch.zeros(env.num_envs, device=env.device),
            "scan_rotation_error_rad": torch.zeros(env.num_envs, device=env.device),
            "confidence_delta": torch.zeros(env.num_envs, device=env.device),
        }
        env._causal_slam_dynamics_cache = cache
        return cache

    reset = step < cache["step"]
    asset = env.scene[asset_cfg.name]
    linear_acceleration, angular_acceleration, _ = _motion_derivatives(env, asset_cfg)
    translation_vector = 0.5 * linear_acceleration * scan_time_s**2
    rotation_vector = 0.5 * angular_acceleration * scan_time_s**2
    rotation_vector = rotation_vector.clone()
    rotation_vector[:, :2] += asset.data.root_ang_vel_b[:, :2] * scan_time_s
    scan_translation = torch.linalg.vector_norm(translation_vector, dim=1)
    scan_rotation = torch.linalg.vector_norm(rotation_vector, dim=1)
    next_state, diagnostics = causal_slam_transition(
        cache["state"],
        scan_translation,
        scan_rotation,
        cache["point_support"],
        dt_s=float(env.step_dt),
        translation_scale_m=translation_scale_m,
        rotation_scale_rad=rotation_scale_rad,
        degradation_rate_hz=degradation_rate_hz,
        recovery_rate_hz=recovery_rate_hz,
        invalid_enter_confidence=invalid_enter_confidence,
        valid_exit_confidence=valid_exit_confidence,
        maximum_age_s=maximum_age_s,
    )
    next_state[reset] = initial_state[reset]
    history = torch.roll(cache["history"], shifts=-1, dims=1)
    history[:, -1] = next_state
    history[reset] = initial_state[reset, None, :]
    cache.update(
        {
            "step": step.clone(),
            "state": next_state,
            "history": history,
            "scan_translation_error_m": scan_translation,
            "scan_rotation_error_rad": scan_rotation,
            **diagnostics,
        }
    )
    env._causal_slam_dynamics_cache = cache
    return cache


def causal_slam_state(
    env,
    horizon_steps: int = 25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Expose only runtime-deployable confidence, validity and normalized age."""
    return _causal_slam_update(
        env, horizon_steps=horizon_steps, asset_cfg=asset_cfg
    )["state"]


def causal_slam_delayed_advantage(
    env,
    horizon_steps: int = 25,
    validity_bonus: float = 0.50,
    normalized_age_penalty: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward localization change caused by the preceding action window."""
    cache = _causal_slam_update(
        env, horizon_steps=horizon_steps, asset_cfg=asset_cfg
    )
    return delayed_localization_advantage(
        cache["state"],
        cache["history"][:, 0],
        validity_bonus=validity_bonus,
        normalized_age_penalty=normalized_age_penalty,
    )


def _localization_vulnerability(
    env, cycle_s: float, severity_gain: float, localization_mode: str = "actual"
) -> torch.Tensor:
    """Privileged reward weight shared by J1/J2; never exposed through J1 input."""
    if localization_mode == "actual":
        state = simulated_slam_confidence(
            env,
            cycle_s=cycle_s,
            phase_offset_mode="distributed",
        )
    elif localization_mode == "causal":
        state = causal_slam_state(env)
    else:
        raise ValueError("localization_mode must be 'actual' or 'causal'")
    return localization_vulnerability(state, severity_gain)


def joint_training_angular_acceleration_l2(
    env,
    cycle_s: float = 10.0,
    severity_gain: float = 1.0,
    localization_mode: str = "actual",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize angular acceleration, not requested constant yaw rate."""
    _, angular_acceleration, _ = _motion_derivatives(env, asset_cfg)
    return angular_acceleration_l2(
        angular_acceleration,
        _localization_vulnerability(env, cycle_s, severity_gain, localization_mode),
    )


def joint_training_linear_jerk_l2(
    env,
    cycle_s: float = 10.0,
    severity_gain: float = 1.0,
    localization_mode: str = "actual",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body jerk while leaving mean requested velocity unscaled."""
    _, _, linear_jerk = _motion_derivatives(env, asset_cfg)
    return linear_jerk_l2(
        linear_jerk,
        _localization_vulnerability(env, cycle_s, severity_gain, localization_mode),
    )


def joint_training_lidar_scan_translation_distortion_l2(
    env,
    scan_time_s: float = 0.10,
    cycle_s: float = 10.0,
    severity_gain: float = 1.0,
    localization_mode: str = "actual",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-constant scan translation caused by body acceleration."""
    linear_acceleration, _, _ = _motion_derivatives(env, asset_cfg)
    return lidar_scan_translation_distortion_l2(
        linear_acceleration,
        _localization_vulnerability(env, cycle_s, severity_gain, localization_mode),
        scan_time_s,
    )


def joint_training_lidar_scan_rotation_distortion_l2(
    env,
    scan_time_s: float = 0.10,
    cycle_s: float = 10.0,
    severity_gain: float = 1.0,
    localization_mode: str = "actual",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch scan motion and angular acceleration, preserving yaw."""
    asset = env.scene[asset_cfg.name]
    _, angular_acceleration, _ = _motion_derivatives(env, asset_cfg)
    return lidar_scan_rotation_distortion_l2(
        asset.data.root_ang_vel_b,
        angular_acceleration,
        _localization_vulnerability(env, cycle_s, severity_gain, localization_mode),
        scan_time_s,
    )
