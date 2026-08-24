"""ROS-independent arbitration between a risk-aware governor and Arm-B safety."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class GovernorMode(str, Enum):
    RISK_AWARE = "risk_aware"
    LEGACY_B_FALLBACK = "legacy_b_fallback"
    RISK_STOP = "risk_stop"
    HARD_STOP = "hard_stop"


@dataclass(frozen=True)
class RiskAwareGovernorConfig:
    """Phase-1 boundary; no trained risk model or live use is authorized."""

    confidence_floor: float = 0.2
    confidence_span: float = 0.8
    maximum_confidence_age_s: float = 0.5
    maximum_receipt_age_s: float = 0.15
    maximum_predicted_unusable_probability: float = 0.10

    def validate(self) -> None:
        if not 0.0 <= self.confidence_floor < 1.0:
            raise ValueError("confidence_floor must be in [0, 1)")
        if not 0.0 < self.confidence_span <= 1.0:
            raise ValueError("confidence_span must be in (0, 1]")
        if self.confidence_floor + self.confidence_span != 1.0:
            raise ValueError("confidence floor and span must reproduce Arm-B scaling")
        if self.maximum_confidence_age_s <= 0.0 or self.maximum_receipt_age_s <= 0.0:
            raise ValueError("freshness limits must be positive")
        if not 0.0 < self.maximum_predicted_unusable_probability < 1.0:
            raise ValueError("risk limit must be in (0, 1)")


@dataclass(frozen=True)
class GovernorDecision:
    command_scale: float
    mode: GovernorMode
    reason: str
    legacy_b_scale: float


def legacy_b_scale(confidence: float, tracking_valid: bool) -> float:
    """Return the exact publication Arm-B continuous command scale."""

    if not tracking_valid:
        return 0.0
    return min(max((confidence - 0.2) / 0.8, 0.0), 1.0)


def arbitrate_command_scale(
    *,
    confidence: float,
    tracking_valid: bool,
    confidence_age_s: float,
    receipt_age_s: float,
    prediction_available: bool,
    selected_scale: float,
    predicted_unusable_probability: float,
    config: RiskAwareGovernorConfig | None = None,
) -> GovernorDecision:
    """Fail closed on SLAM faults; otherwise arbitrate new control versus B.

    A valid risk-aware decision replaces B's normal continuous throttle.  B is
    retained as the fallback when the optional prediction path is unavailable
    or malformed.  Invalid/stale SLAM state always produces an exact stop.
    """

    cfg = config or RiskAwareGovernorConfig()
    cfg.validate()
    confidence_finite = math.isfinite(confidence) and 0.0 <= confidence <= 1.0
    ages_finite = (
        math.isfinite(confidence_age_s)
        and math.isfinite(receipt_age_s)
        and confidence_age_s >= 0.0
        and receipt_age_s >= 0.0
    )
    if not confidence_finite or not ages_finite:
        return GovernorDecision(0.0, GovernorMode.HARD_STOP, "slam_numeric_invalid", 0.0)
    if not tracking_valid:
        return GovernorDecision(0.0, GovernorMode.HARD_STOP, "tracking_invalid", 0.0)
    if confidence_age_s > cfg.maximum_confidence_age_s:
        return GovernorDecision(0.0, GovernorMode.HARD_STOP, "confidence_source_stale", 0.0)
    if receipt_age_s > cfg.maximum_receipt_age_s:
        return GovernorDecision(0.0, GovernorMode.HARD_STOP, "confidence_receipt_stale", 0.0)

    fallback = legacy_b_scale(confidence, True)
    prediction_finite = (
        math.isfinite(selected_scale)
        and 0.0 <= selected_scale <= 1.0
        and math.isfinite(predicted_unusable_probability)
        and 0.0 <= predicted_unusable_probability <= 1.0
    )
    if not prediction_available:
        return GovernorDecision(
            fallback, GovernorMode.LEGACY_B_FALLBACK,
            "risk_prediction_unavailable", fallback,
        )
    if not prediction_finite:
        return GovernorDecision(
            fallback, GovernorMode.LEGACY_B_FALLBACK,
            "risk_prediction_invalid", fallback,
        )
    if predicted_unusable_probability > cfg.maximum_predicted_unusable_probability:
        return GovernorDecision(0.0, GovernorMode.RISK_STOP, "no_admissible_safe_scale", fallback)
    return GovernorDecision(
        selected_scale, GovernorMode.RISK_AWARE, "risk_prediction_accepted", fallback,
    )
