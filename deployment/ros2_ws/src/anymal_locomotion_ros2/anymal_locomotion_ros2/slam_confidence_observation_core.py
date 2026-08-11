"""ROS-independent contract for the future locomotion observation adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PpoConfidenceObservation:
    slam_confidence: float
    slam_tracking_valid: float
    confidence_age_normalized: float
    receipt_watchdog_valid: bool


def ppo_confidence_observation(
    *,
    confidence: float,
    tracking_valid: bool,
    source_stamp_valid: bool,
    confidence_age_s: float,
    receipt_age_s: float,
    age_normalization_s: float = 0.50,
    receipt_timeout_s: float = 0.15,
) -> PpoConfidenceObservation:
    """Validate and normalize a received snapshot without policy integration."""

    values = (confidence, confidence_age_s, receipt_age_s, age_normalization_s, receipt_timeout_s)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("confidence observation inputs must be finite")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("slam confidence must be in [0, 1]")
    if confidence_age_s < 0.0 or receipt_age_s < 0.0:
        raise ValueError("confidence ages must be non-negative")
    if age_normalization_s <= 0.0 or receipt_timeout_s <= 0.0:
        raise ValueError("watchdog scales must be positive")
    receipt_valid = receipt_age_s <= receipt_timeout_s
    hard_valid = bool(tracking_valid and source_stamp_valid and receipt_valid)
    return PpoConfidenceObservation(
        slam_confidence=float(confidence if receipt_valid else 0.0),
        slam_tracking_valid=float(hard_valid),
        confidence_age_normalized=min(confidence_age_s / age_normalization_s, 1.0),
        receipt_watchdog_valid=receipt_valid,
    )
