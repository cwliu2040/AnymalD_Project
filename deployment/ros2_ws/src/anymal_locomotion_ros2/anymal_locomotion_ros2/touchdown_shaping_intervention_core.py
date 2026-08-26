"""ROS-independent, command-preserving touchdown-shaping intervention."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


COMMAND_DIMENSION = 3
ACTION_DIMENSION = 12
FOOT_COUNT = 4
FOOT_CARTESIAN_DIMENSION = 3


@dataclass(frozen=True)
class TouchdownShapingConfig:
    """Frozen mechanism limits; phase estimation remains a separate component."""

    control_period_s: float = 0.02
    late_swing_start: float = 0.70
    late_swing_end: float = 1.00
    minimum_phase_confidence: float = 0.80
    residual_linf_limit: float = 0.05
    damping: float = 1.0e-4

    def validate(self) -> None:
        values = (
            self.control_period_s,
            self.minimum_phase_confidence,
            self.residual_linf_limit,
            self.damping,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("touchdown-shaping limits must be finite and positive")
        if not 0.0 <= self.late_swing_start < self.late_swing_end <= 1.0:
            raise ValueError("late-swing window must satisfy 0 <= start < end <= 1")
        if self.minimum_phase_confidence > 1.0:
            raise ValueError("minimum phase confidence must not exceed one")


@dataclass(frozen=True)
class TouchdownShapingResult:
    effective_command: np.ndarray
    applied_action: np.ndarray
    action_residual: np.ndarray
    active_feet: np.ndarray
    phase_weights: np.ndarray
    residual_scale: float
    reason: str


def _vector(values: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    if value.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), received {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _exact_baseline(command: np.ndarray, action: np.ndarray, reason: str) -> TouchdownShapingResult:
    return TouchdownShapingResult(
        effective_command=command.astype(np.float32, copy=True),
        applied_action=action.astype(np.float32, copy=True),
        action_residual=np.zeros(ACTION_DIMENSION, dtype=np.float32),
        active_feet=np.zeros(FOOT_COUNT, dtype=bool),
        phase_weights=np.zeros(FOOT_COUNT, dtype=np.float32),
        residual_scale=1.0,
        reason=reason,
    )


def apply_touchdown_shaping_intervention(
    *,
    requested_command: Sequence[float] | np.ndarray,
    backbone_action: Sequence[float] | np.ndarray,
    swing_progress: Sequence[float] | np.ndarray,
    phase_confidence: Sequence[float] | np.ndarray,
    foot_vertical_velocity_mps: Sequence[float] | np.ndarray,
    foot_position_jacobian_per_action: Sequence[float] | np.ndarray,
    tracking_valid: bool,
    attenuation_fraction: float,
    config: TouchdownShapingConfig | None = None,
) -> TouchdownShapingResult:
    """Reduce predicted late-swing downward displacement without command scaling.

    ``foot_position_jacobian_per_action`` maps a 12-D normalized action residual
    to first-order XYZ foot-position changes and has shape ``(4, 3, 12)``.  The
    minimum-norm damped solve requests zero horizontal change and cancels a
    fixed fraction of the next control tick's downward foot displacement.

    Swing phase and its confidence must be estimated from deployable joint and
    action history outside this core.  Contact truth is intentionally absent.
    """

    cfg = config or TouchdownShapingConfig()
    cfg.validate()
    command = _vector(requested_command, COMMAND_DIMENSION, "requested_command")
    action = _vector(backbone_action, ACTION_DIMENSION, "backbone_action")
    progress = _vector(swing_progress, FOOT_COUNT, "swing_progress")
    confidence = _vector(phase_confidence, FOOT_COUNT, "phase_confidence")
    vertical_velocity = _vector(
        foot_vertical_velocity_mps, FOOT_COUNT, "foot_vertical_velocity_mps"
    )
    jacobian = np.asarray(foot_position_jacobian_per_action, dtype=np.float64)
    expected_shape = (FOOT_COUNT, FOOT_CARTESIAN_DIMENSION, ACTION_DIMENSION)
    if jacobian.shape != expected_shape:
        raise ValueError(
            "foot_position_jacobian_per_action must have shape "
            f"{expected_shape}, received {jacobian.shape}"
        )
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("foot_position_jacobian_per_action must be finite")
    if not math.isfinite(attenuation_fraction) or not 0.0 <= attenuation_fraction <= 1.0:
        raise ValueError("attenuation_fraction must be finite and in [0, 1]")
    if np.any(progress < 0.0) or np.any(progress > 1.0):
        raise ValueError("swing_progress must be in [0, 1]")
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise ValueError("phase_confidence must be in [0, 1]")
    if not tracking_valid:
        return _exact_baseline(command, action, "tracking_invalid_exact_baseline")
    if attenuation_fraction == 0.0:
        return _exact_baseline(command, action, "zero_arm_exact_baseline")

    normalized_phase = np.clip(
        (progress - cfg.late_swing_start) / (cfg.late_swing_end - cfg.late_swing_start),
        0.0,
        1.0,
    )
    phase_weights = normalized_phase**2 * (3.0 - 2.0 * normalized_phase)
    active = (
        (progress >= cfg.late_swing_start)
        & (progress <= cfg.late_swing_end)
        & (confidence >= cfg.minimum_phase_confidence)
        & (vertical_velocity < 0.0)
    )
    phase_weights = np.where(active, phase_weights, 0.0)
    if not np.any(active):
        return _exact_baseline(command, action, "no_eligible_late_swing_foot")

    active_indices = np.flatnonzero(active)
    horizontal_rows: list[np.ndarray] = []
    vertical_rows: list[np.ndarray] = []
    targets: list[float] = []
    for foot_index in active_indices:
        weight = float(phase_weights[foot_index])
        foot_jacobian = jacobian[foot_index]
        downward_displacement = min(float(vertical_velocity[foot_index]), 0.0) * cfg.control_period_s
        target_vertical_correction = -attenuation_fraction * weight * downward_displacement
        horizontal_rows.extend((foot_jacobian[0], foot_jacobian[1]))
        vertical_rows.append(foot_jacobian[2])
        targets.append(target_vertical_correction)

    horizontal = np.stack(horizontal_rows, axis=0)
    vertical = np.stack(vertical_rows, axis=0)
    target = np.asarray(targets, dtype=np.float64)
    horizontal_gram = horizontal @ horizontal.T
    horizontal_projector = np.eye(ACTION_DIMENSION) - (
        horizontal.T @ np.linalg.pinv(horizontal_gram) @ horizontal
    )
    constrained_vertical = vertical @ horizontal_projector
    gram = constrained_vertical @ constrained_vertical.T
    regularized = gram + cfg.damping * np.eye(gram.shape[0], dtype=np.float64)
    try:
        raw_residual = (
            horizontal_projector
            @ constrained_vertical.T
            @ np.linalg.solve(regularized, target)
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("touchdown-shaping Jacobian solve failed") from exc
    if not np.all(np.isfinite(raw_residual)):
        raise RuntimeError("touchdown-shaping solve produced a non-finite residual")

    maximum = float(np.max(np.abs(raw_residual)))
    residual_scale = 1.0 if maximum <= cfg.residual_linf_limit else cfg.residual_linf_limit / maximum
    residual = raw_residual * residual_scale
    applied = action + residual
    if not np.all(np.isfinite(applied)):
        raise RuntimeError("touchdown-shaping composition produced a non-finite action")
    return TouchdownShapingResult(
        effective_command=command.astype(np.float32, copy=True),
        applied_action=applied.astype(np.float32),
        action_residual=residual.astype(np.float32),
        active_feet=active,
        phase_weights=phase_weights.astype(np.float32),
        residual_scale=float(residual_scale),
        reason="touchdown_shaping_applied",
    )
