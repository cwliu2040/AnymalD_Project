"""Pure action-dependent SLAM-state dynamics for causal PPO wiring.

The functions in this module deliberately know nothing about Isaac Lab or ROS.
They provide a testable state transition whose inputs are LiDAR scan-motion
errors caused by locomotion.  Numerical parameters are an uncalibrated wiring
proxy until qualified against held-out FAST-LIO2/LIO-SAM outcomes.
"""

from __future__ import annotations

import torch


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must be finite")


def causal_slam_transition(
    state: torch.Tensor,
    scan_translation_error_m: torch.Tensor,
    scan_rotation_error_rad: torch.Tensor,
    point_support: torch.Tensor,
    *,
    dt_s: float,
    translation_scale_m: float,
    rotation_scale_rad: float,
    degradation_rate_hz: float,
    recovery_rate_hz: float,
    invalid_enter_confidence: float,
    valid_exit_confidence: float,
    maximum_age_s: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Advance confidence/validity/age from realized scan motion.

    This creates the missing causal arrow for PPO wiring:
    locomotion action -> body/LiDAR motion -> next localization state.
    It is not itself evidence that the chosen scales match a real backend.
    """
    if state.ndim != 2 or state.shape[1] != 3:
        raise ValueError("state must have shape [N,3]")
    count = state.shape[0]
    for name, value in (
        ("scan_translation_error_m", scan_translation_error_m),
        ("scan_rotation_error_rad", scan_rotation_error_rad),
        ("point_support", point_support),
    ):
        if value.shape != (count,):
            raise ValueError(f"{name} must have shape [N]")
        _finite(name, value)
    _finite("state", state)
    if dt_s <= 0.0 or translation_scale_m <= 0.0 or rotation_scale_rad <= 0.0:
        raise ValueError("time step and scan-motion scales must be positive")
    if degradation_rate_hz < 0.0 or recovery_rate_hz < 0.0 or maximum_age_s <= 0.0:
        raise ValueError("rates must be non-negative and maximum age positive")
    if not 0.0 <= invalid_enter_confidence < valid_exit_confidence <= 1.0:
        raise ValueError("validity hysteresis thresholds are invalid")
    if not bool(torch.all((state >= 0.0) & (state <= 1.0))):
        raise ValueError("state channels must be normalized to [0,1]")
    if not bool(torch.all((point_support >= 0.0) & (point_support <= 1.0))):
        raise ValueError("point support must be normalized to [0,1]")
    if not bool(torch.all(scan_translation_error_m >= 0.0)) or not bool(
        torch.all(scan_rotation_error_rad >= 0.0)
    ):
        raise ValueError("scan-motion errors must be non-negative")

    confidence, valid, normalized_age = state.unbind(dim=1)
    normalized_distortion = torch.square(
        scan_translation_error_m / translation_scale_m
    ) + torch.square(scan_rotation_error_rad / rotation_scale_rad)
    scan_quality = torch.exp(-normalized_distortion)
    usable_support = point_support * scan_quality
    degradation = degradation_rate_hz * (1.0 - usable_support) * confidence
    recovery = recovery_rate_hz * usable_support * (1.0 - confidence)
    next_confidence = torch.clamp(
        confidence + dt_s * (recovery - degradation), 0.0, 1.0
    )

    was_valid = valid >= 0.5
    next_valid = torch.where(
        was_valid,
        next_confidence >= invalid_enter_confidence,
        next_confidence >= valid_exit_confidence,
    ).to(state.dtype)
    next_age = torch.where(
        next_valid >= 0.5,
        torch.zeros_like(normalized_age),
        torch.clamp(normalized_age + dt_s / maximum_age_s, 0.0, 1.0),
    )
    next_state = torch.stack((next_confidence, next_valid, next_age), dim=1)
    diagnostics = {
        "normalized_distortion": normalized_distortion,
        "scan_quality": scan_quality,
        "usable_support": usable_support,
        "confidence_delta": next_confidence - confidence,
        "validity_lost": ((valid >= 0.5) & (next_valid < 0.5)).to(state.dtype),
        "validity_recovered": ((valid < 0.5) & (next_valid >= 0.5)).to(state.dtype),
    }
    return next_state, diagnostics


def localization_potential(
    state: torch.Tensor,
    *,
    validity_bonus: float,
    normalized_age_penalty: float,
) -> torch.Tensor:
    """Training-only scalar potential; the actor never observes this value."""
    if state.ndim != 2 or state.shape[1] != 3:
        raise ValueError("state must have shape [N,3]")
    _finite("state", state)
    if validity_bonus < 0.0 or normalized_age_penalty < 0.0:
        raise ValueError("potential weights must be non-negative")
    return (
        state[:, 0]
        + validity_bonus * state[:, 1]
        - normalized_age_penalty * state[:, 2]
    )


def delayed_localization_advantage(
    current_state: torch.Tensor,
    past_state: torch.Tensor,
    *,
    validity_bonus: float,
    normalized_age_penalty: float,
) -> torch.Tensor:
    """Reward the localization change realized over an earlier action window."""
    if current_state.shape != past_state.shape:
        raise ValueError("current and past localization states must have equal shape")
    return localization_potential(
        current_state,
        validity_bonus=validity_bonus,
        normalized_age_penalty=normalized_age_penalty,
    ) - localization_potential(
        past_state,
        validity_bonus=validity_bonus,
        normalized_age_penalty=normalized_age_penalty,
    )
