"""ROS-independent bounded action intervention for causal pilot arms."""

from __future__ import annotations

import numpy as np


ACTION_DIMENSION = 12


def apply_previous_action_intervention(
    baseline_action: np.ndarray,
    previous_applied_action: np.ndarray,
    tracking_valid: np.ndarray | bool,
    *,
    alpha: float,
    residual_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a signed, clipped smoothing intervention to an Arm-B action.

    Positive ``alpha`` moves toward the previously applied action (smoothing),
    negative ``alpha`` moves away from it (antismoothing).  Invalid tracking
    always returns the exact baseline action and an exact-zero residual.
    """
    baseline = np.asarray(baseline_action, dtype=np.float32)
    previous = np.asarray(previous_applied_action, dtype=np.float32)
    if baseline.shape != previous.shape or baseline.shape[-1:] != (ACTION_DIMENSION,):
        raise ValueError("baseline and previous actions must share a final 12-D axis")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(previous)):
        raise ValueError("actions must be finite")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not np.isfinite(residual_limit) or residual_limit <= 0.0:
        raise ValueError("residual_limit must be finite and positive")
    valid = np.asarray(tracking_valid)
    if valid.shape != baseline.shape[:-1]:
        raise ValueError("tracking_valid must match the action batch shape")
    residual = np.clip(
        float(alpha) * (previous - baseline),
        -float(residual_limit),
        float(residual_limit),
    ).astype(np.float32)
    residual = np.where(np.expand_dims(valid.astype(bool), -1), residual, 0.0)
    applied = baseline + residual
    if not np.all(np.isfinite(applied)):
        raise RuntimeError("intervention produced a non-finite action")
    return applied.astype(np.float32), residual.astype(np.float32)
