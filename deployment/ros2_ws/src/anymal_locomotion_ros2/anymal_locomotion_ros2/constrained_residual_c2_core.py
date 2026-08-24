"""ROS-independent command and action composition for candidate C2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np

from .risk_aware_governor_core import legacy_b_scale


COMMAND_DIMENSION = 3
ACTION_DIMENSION = 12


class C2Mode(str, Enum):
    CANDIDATE = "candidate"
    LEGACY_B_FALLBACK = "legacy_b_fallback"
    RISK_STOP = "risk_stop"
    HARD_STOP = "hard_stop"


@dataclass(frozen=True)
class C2Config:
    minimum_command_scales: tuple[float, float, float] = (0.2, 0.2, 0.2)
    maximum_command_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)
    residual_linf_limit: float = 0.10
    maximum_confidence_age_s: float = 0.50
    maximum_receipt_age_s: float = 0.15

    def validate(self) -> None:
        minimum = _vector(self.minimum_command_scales, COMMAND_DIMENSION, "minimum scales")
        maximum = _vector(self.maximum_command_scales, COMMAND_DIMENSION, "maximum scales")
        if np.any(minimum < 0.0) or np.any(maximum > 1.0) or np.any(minimum > maximum):
            raise ValueError("command scale bounds must satisfy 0 <= minimum <= maximum <= 1")
        if not math.isfinite(self.residual_linf_limit) or self.residual_linf_limit <= 0.0:
            raise ValueError("residual_linf_limit must be finite and positive")
        if self.maximum_confidence_age_s <= 0.0 or self.maximum_receipt_age_s <= 0.0:
            raise ValueError("freshness limits must be positive")


@dataclass(frozen=True)
class C2Plan:
    """A command decision made before evaluating the frozen model1450 backbone."""

    effective_command: np.ndarray
    action_residual: np.ndarray
    mode: C2Mode
    reason: str
    legacy_b_scale: float


def _vector(values: Sequence[float] | np.ndarray, dimension: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), received {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _fallback_plan(command: np.ndarray, confidence: float, reason: str) -> C2Plan:
    scale = legacy_b_scale(confidence, True)
    return C2Plan(
        effective_command=(command * scale).astype(np.float32),
        action_residual=np.zeros(ACTION_DIMENSION, dtype=np.float32),
        mode=C2Mode.LEGACY_B_FALLBACK,
        reason=reason,
        legacy_b_scale=scale,
    )


def plan_c2_control(
    *,
    requested_command: Sequence[float] | np.ndarray,
    confidence: float,
    tracking_valid: bool,
    confidence_age_s: float,
    receipt_age_s: float,
    candidate_available: bool,
    candidate_admissible: bool,
    candidate_command_scales: Sequence[float] | np.ndarray,
    candidate_action_residual: Sequence[float] | np.ndarray,
    config: C2Config | None = None,
) -> C2Plan:
    """Choose the effective command and residual without invoking ROS or a policy.

    Invalid or stale SLAM produces an exact-zero command. Missing or malformed
    candidate output falls back to Arm B. A well-formed but risk-inadmissible
    candidate produces a risk stop. Only an admissible candidate can bypass B's
    normal soft throttle.
    """

    cfg = config or C2Config()
    cfg.validate()
    command = _vector(requested_command, COMMAND_DIMENSION, "requested_command")
    slam_finite = (
        math.isfinite(confidence)
        and 0.0 <= confidence <= 1.0
        and math.isfinite(confidence_age_s)
        and confidence_age_s >= 0.0
        and math.isfinite(receipt_age_s)
        and receipt_age_s >= 0.0
    )
    if not slam_finite:
        return C2Plan(
            np.zeros(COMMAND_DIMENSION, dtype=np.float32),
            np.zeros(ACTION_DIMENSION, dtype=np.float32),
            C2Mode.HARD_STOP,
            "slam_numeric_invalid",
            0.0,
        )
    if not tracking_valid:
        reason = "tracking_invalid"
    elif confidence_age_s > cfg.maximum_confidence_age_s:
        reason = "confidence_source_stale"
    elif receipt_age_s > cfg.maximum_receipt_age_s:
        reason = "confidence_receipt_stale"
    else:
        reason = ""
    if reason:
        return C2Plan(
            np.zeros(COMMAND_DIMENSION, dtype=np.float32),
            np.zeros(ACTION_DIMENSION, dtype=np.float32),
            C2Mode.HARD_STOP,
            reason,
            0.0,
        )
    if not candidate_available:
        return _fallback_plan(command, confidence, "candidate_unavailable")
    try:
        scales = _vector(candidate_command_scales, COMMAND_DIMENSION, "candidate scales")
        residual = _vector(candidate_action_residual, ACTION_DIMENSION, "candidate residual")
    except (TypeError, ValueError):
        return _fallback_plan(command, confidence, "candidate_numeric_or_shape_invalid")
    lower = np.asarray(cfg.minimum_command_scales, dtype=np.float32)
    upper = np.asarray(cfg.maximum_command_scales, dtype=np.float32)
    if np.any(scales < lower) or np.any(scales > upper):
        return _fallback_plan(command, confidence, "candidate_scale_out_of_bounds")
    if float(np.max(np.abs(residual))) > cfg.residual_linf_limit:
        return _fallback_plan(command, confidence, "candidate_residual_out_of_bounds")
    fallback = legacy_b_scale(confidence, True)
    if not candidate_admissible:
        return C2Plan(
            np.zeros(COMMAND_DIMENSION, dtype=np.float32),
            np.zeros(ACTION_DIMENSION, dtype=np.float32),
            C2Mode.RISK_STOP,
            "no_admissible_candidate",
            fallback,
        )
    return C2Plan(
        effective_command=(command * scales).astype(np.float32),
        action_residual=residual.copy(),
        mode=C2Mode.CANDIDATE,
        reason="candidate_accepted",
        legacy_b_scale=fallback,
    )


def compose_c2_action(
    backbone_action: Sequence[float] | np.ndarray,
    plan: C2Plan,
) -> np.ndarray:
    """Add the already-validated residual to model1450's effective-command action."""

    baseline = _vector(backbone_action, ACTION_DIMENSION, "backbone_action")
    residual = _vector(plan.action_residual, ACTION_DIMENSION, "plan.action_residual")
    if plan.mode is not C2Mode.CANDIDATE and np.count_nonzero(residual):
        raise ValueError("fallback and stop plans must carry an exact-zero residual")
    applied = baseline + residual
    if not np.all(np.isfinite(applied)):
        raise RuntimeError("C2 action composition produced a non-finite action")
    return applied.astype(np.float32)
