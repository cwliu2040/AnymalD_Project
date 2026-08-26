"""ROS-independent causal history phase estimator inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TouchdownPhaseEstimate:
    swing_progress: np.ndarray
    confidence: np.ndarray
    tracking_valid: bool


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def estimate_touchdown_phase(
    joint_position_history: Any,
    joint_velocity_history: Any,
    previous_action_history: Any,
    artifact: dict[str, Any],
) -> TouchdownPhaseEstimate:
    """Estimate four foot phases; malformed/stale histories fail to invalid zeros."""
    try:
        if artifact.get("frozen") is not True or artifact.get("passed") is not True:
            raise ValueError("phase estimator artifact is not frozen PASS")
        offsets = tuple(int(value) for value in artifact["history_offsets_samples"])
        if not offsets or min(offsets) < 0:
            raise ValueError("history offsets must be nonnegative")
        required = max(offsets) + 1
        position = np.asarray(joint_position_history, dtype=np.float64)
        velocity = np.asarray(joint_velocity_history, dtype=np.float64)
        action = np.asarray(previous_action_history, dtype=np.float64)
        if any(value.ndim != 2 or value.shape[1] != 12 for value in (position, velocity, action)):
            raise ValueError("histories must have shape [time, 12]")
        if min(len(position), len(velocity), len(action)) < required:
            raise ValueError("history is too short")
        if not all(np.all(np.isfinite(value)) for value in (position, velocity, action)):
            raise ValueError("history must be finite")
        features = np.concatenate([
            np.concatenate((position[-1-offset], velocity[-1-offset], action[-1-offset]))
            for offset in offsets
        ])
        mean = _array(artifact["feature_mean"], features.shape, "feature mean")
        scale = _array(artifact["feature_scale"], features.shape, "feature scale")
        if np.any(scale <= 0.0):
            raise ValueError("feature scale must be positive")
        standardized = (features - mean) / scale
        weights = _array(artifact["classifier_weight"], (4, len(features)), "classifier")
        intercept = _array(artifact["classifier_intercept"], (4,), "classifier intercept")
        logits = np.clip(weights @ standardized + intercept, -60.0, 60.0)
        confidence = 1.0 / (1.0 + np.exp(-logits))
        progress_weight = _array(
            artifact["progress_weight"], (4, len(features)), "progress regressor"
        )
        progress_intercept = _array(
            artifact["progress_intercept"], (4,), "progress intercept"
        )
        progress = np.clip(
            progress_weight @ standardized + progress_intercept, 0.0, 1.0
        )
        return TouchdownPhaseEstimate(progress, confidence, True)
    except (KeyError, TypeError, ValueError):
        return TouchdownPhaseEstimate(np.zeros(4), np.zeros(4), False)
