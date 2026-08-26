"""ROS-independent safety composition for localization-aware motor adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np

from .risk_aware_governor_core import legacy_b_scale


COMMAND_DIMENSION = 3
ACTION_DIMENSION = 12


class AdaptationMode(str, Enum):
    ADAPTED_RESIDUAL = "adapted_residual"
    BACKBONE_ONLY = "backbone_only"
    LEGACY_B_FALLBACK = "legacy_b_fallback"
    HARD_STOP = "hard_stop"


@dataclass(frozen=True)
class AdaptationConfig:
    residual_linf_limit: float = 0.05
    maximum_confidence_age_s: float = 0.50
    maximum_receipt_age_s: float = 0.15
    maximum_adaptation_uncertainty: float = 0.20

    def validate(self) -> None:
        values = (
            self.residual_linf_limit,
            self.maximum_confidence_age_s,
            self.maximum_receipt_age_s,
            self.maximum_adaptation_uncertainty,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("adaptation limits must be finite and positive")


@dataclass(frozen=True)
class AdaptationPlan:
    effective_command: np.ndarray
    action_residual: np.ndarray
    mode: AdaptationMode
    reason: str
    legacy_b_scale: float


def _vector(values: Sequence[float] | np.ndarray, dimension: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), received {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _hard_stop(reason: str) -> AdaptationPlan:
    return AdaptationPlan(
        effective_command=np.zeros(COMMAND_DIMENSION, dtype=np.float32),
        action_residual=np.zeros(ACTION_DIMENSION, dtype=np.float32),
        mode=AdaptationMode.HARD_STOP,
        reason=reason,
        legacy_b_scale=0.0,
    )


def _legacy_b_fallback(command: np.ndarray, confidence: float, reason: str) -> AdaptationPlan:
    scale = legacy_b_scale(confidence, True)
    return AdaptationPlan(
        effective_command=(command * scale).astype(np.float32),
        action_residual=np.zeros(ACTION_DIMENSION, dtype=np.float32),
        mode=AdaptationMode.LEGACY_B_FALLBACK,
        reason=reason,
        legacy_b_scale=scale,
    )


def plan_localization_aware_residual(
    *,
    requested_command: Sequence[float] | np.ndarray,
    confidence: float,
    tracking_valid: bool,
    confidence_age_s: float,
    receipt_age_s: float,
    adaptation_available: bool,
    adaptation_uncertainty: float,
    residual_beneficial: bool,
    candidate_action_residual: Sequence[float] | np.ndarray,
    config: AdaptationConfig | None = None,
) -> AdaptationPlan:
    """Plan a command-preserving residual or a fail-closed fallback.

    A valid candidate never changes the requested command. If the adaptation
    model predicts no benefit, the frozen backbone executes the original
    command with an exact-zero residual; this is intentionally not a risk stop.
    Missing, uncertain, malformed, or out-of-bound adaptation output falls back
    to legacy B. Invalid or stale SLAM remains an exact hard stop.
    """

    cfg = config or AdaptationConfig()
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
        return _hard_stop("slam_numeric_invalid")
    if not tracking_valid:
        return _hard_stop("tracking_invalid")
    if confidence_age_s > cfg.maximum_confidence_age_s:
        return _hard_stop("confidence_source_stale")
    if receipt_age_s > cfg.maximum_receipt_age_s:
        return _hard_stop("confidence_receipt_stale")
    if not adaptation_available:
        return _legacy_b_fallback(command, confidence, "adaptation_unavailable")
    if (
        not math.isfinite(adaptation_uncertainty)
        or adaptation_uncertainty < 0.0
        or adaptation_uncertainty > cfg.maximum_adaptation_uncertainty
    ):
        return _legacy_b_fallback(command, confidence, "adaptation_uncertain_or_invalid")
    try:
        residual = _vector(candidate_action_residual, ACTION_DIMENSION, "candidate_action_residual")
    except (TypeError, ValueError):
        return _legacy_b_fallback(command, confidence, "residual_numeric_or_shape_invalid")
    if float(np.max(np.abs(residual))) > cfg.residual_linf_limit:
        return _legacy_b_fallback(command, confidence, "residual_out_of_bounds")
    fallback_scale = legacy_b_scale(confidence, True)
    if not residual_beneficial:
        return AdaptationPlan(
            effective_command=command.copy(),
            action_residual=np.zeros(ACTION_DIMENSION, dtype=np.float32),
            mode=AdaptationMode.BACKBONE_ONLY,
            reason="no_beneficial_residual",
            legacy_b_scale=fallback_scale,
        )
    return AdaptationPlan(
        effective_command=command.copy(),
        action_residual=residual.copy(),
        mode=AdaptationMode.ADAPTED_RESIDUAL,
        reason="residual_accepted",
        legacy_b_scale=fallback_scale,
    )


def compose_localization_aware_action(
    backbone_action: Sequence[float] | np.ndarray,
    plan: AdaptationPlan,
) -> np.ndarray:
    """Add a validated residual to model1450's original-command action."""

    baseline = _vector(backbone_action, ACTION_DIMENSION, "backbone_action")
    residual = _vector(plan.action_residual, ACTION_DIMENSION, "plan.action_residual")
    if plan.mode is not AdaptationMode.ADAPTED_RESIDUAL and np.count_nonzero(residual):
        raise ValueError("non-adapted modes must carry an exact-zero residual")
    applied = baseline + residual
    if not np.all(np.isfinite(applied)):
        raise RuntimeError("localization-aware action composition produced non-finite output")
    return applied.astype(np.float32)
