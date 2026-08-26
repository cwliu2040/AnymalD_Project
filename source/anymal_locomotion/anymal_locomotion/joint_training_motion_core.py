"""Torch-only body/LiDAR motion objectives for full-policy joint training."""

from __future__ import annotations

import torch


def localization_vulnerability(
    localization_state: torch.Tensor,
    severity_gain: float,
) -> torch.Tensor:
    """Return a positive reward weight without changing a command target."""
    if localization_state.ndim != 2 or localization_state.shape[-1] != 3:
        raise ValueError("localization_state must have shape [N,3]")
    if severity_gain < 0.0:
        raise ValueError("severity_gain must be non-negative")
    if not torch.all(torch.isfinite(localization_state)):
        raise ValueError("localization_state must be finite")
    confidence = localization_state[:, 0]
    valid = localization_state[:, 1]
    age = localization_state[:, 2]
    if not bool(torch.all((confidence >= 0.0) & (confidence <= 1.0))):
        raise ValueError("confidence must be in [0,1]")
    if not bool(torch.all((valid == 0.0) | (valid == 1.0))):
        raise ValueError("validity must be binary")
    if not bool(torch.all((age >= 0.0) & (age <= 1.0))):
        raise ValueError("normalized age must be in [0,1]")
    safe = valid * torch.clamp((confidence - 0.2) / 0.8, 0.0, 1.0)
    return 1.0 + severity_gain * (1.0 - safe)


def angular_acceleration_l2(
    angular_acceleration_radps2: torch.Tensor,
    vulnerability: torch.Tensor,
) -> torch.Tensor:
    """Mean squared angular acceleration; constant yaw rate has zero cost."""
    return vulnerability * torch.mean(torch.square(angular_acceleration_radps2), dim=-1)


def linear_jerk_l2(
    linear_jerk_mps3: torch.Tensor,
    vulnerability: torch.Tensor,
) -> torch.Tensor:
    """Mean squared linear jerk; constant translation has zero cost."""
    return vulnerability * torch.mean(torch.square(linear_jerk_mps3), dim=-1)


def lidar_scan_translation_distortion_l2(
    linear_acceleration_mps2: torch.Tensor,
    vulnerability: torch.Tensor,
    scan_time_s: float,
) -> torch.Tensor:
    """Squared non-constant scan displacement from linear acceleration."""
    if scan_time_s <= 0.0:
        raise ValueError("scan_time_s must be positive")
    displacement_error = 0.5 * linear_acceleration_mps2 * scan_time_s**2
    return vulnerability * torch.sum(torch.square(displacement_error), dim=-1)


def lidar_scan_rotation_distortion_l2(
    body_angular_velocity_radps: torch.Tensor,
    angular_acceleration_radps2: torch.Tensor,
    vulnerability: torch.Tensor,
    scan_time_s: float,
) -> torch.Tensor:
    """Penalize roll/pitch scan motion and non-constant yaw, not constant yaw."""
    if scan_time_s <= 0.0:
        raise ValueError("scan_time_s must be positive")
    rotation_error = 0.5 * angular_acceleration_radps2 * scan_time_s**2
    rotation_error = rotation_error.clone()
    rotation_error[:, :2] += body_angular_velocity_radps[:, :2] * scan_time_s
    return vulnerability * torch.sum(torch.square(rotation_error), dim=-1)
