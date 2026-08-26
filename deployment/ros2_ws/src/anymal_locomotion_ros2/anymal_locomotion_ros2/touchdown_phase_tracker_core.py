"""Causal three-state gait-phase tracker for touchdown residual eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TouchdownPhaseTrackerState:
    phase_state: np.ndarray
    transition_count: np.ndarray
    early_age_samples: np.ndarray
    previous_progress: np.ndarray


@dataclass(frozen=True)
class TouchdownPhaseTrackerResult:
    swing_progress: np.ndarray
    confidence: np.ndarray
    eligible_late_swing: np.ndarray
    tracking_valid: bool
    state: TouchdownPhaseTrackerState


def initial_touchdown_phase_tracker_state() -> TouchdownPhaseTrackerState:
    return TouchdownPhaseTrackerState(
        phase_state=np.zeros(4, dtype=np.int64),
        transition_count=np.zeros(4, dtype=np.int64),
        early_age_samples=np.zeros(4, dtype=np.int64),
        previous_progress=np.zeros(4, dtype=np.float64),
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -60.0, 0.0))
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _invalid() -> TouchdownPhaseTrackerResult:
    state = initial_touchdown_phase_tracker_state()
    return TouchdownPhaseTrackerResult(
        np.zeros(4), np.zeros(4), np.zeros(4, dtype=bool), False, state
    )


def update_tracker_from_probabilities(
    probabilities: Any,
    progress: Any,
    state: TouchdownPhaseTrackerState,
    tracker: dict[str, Any],
) -> TouchdownPhaseTrackerResult:
    """Advance stance->early->late without permitting a direct late-swing jump."""
    try:
        probability = np.asarray(probabilities, dtype=np.float64)
        phase_progress = np.asarray(progress, dtype=np.float64)
        if probability.shape != (4, 3) or phase_progress.shape != (4,):
            raise ValueError("phase probabilities/progress shape mismatch")
        if not np.all(np.isfinite(probability)) or not np.all(np.isfinite(phase_progress)):
            raise ValueError("phase probabilities/progress must be finite")
        if np.any(probability < 0.0) or not np.allclose(np.sum(probability, axis=1), 1.0, atol=1e-6):
            raise ValueError("phase probabilities must be normalized")
        current = np.asarray(state.phase_state, dtype=np.int64).copy()
        count = np.asarray(state.transition_count, dtype=np.int64).copy()
        age = np.asarray(state.early_age_samples, dtype=np.int64).copy()
        previous = np.asarray(state.previous_progress, dtype=np.float64).copy()
        if any(value.shape != (4,) for value in (current, count, age, previous)):
            raise ValueError("invalid phase tracker state")
        early_confirmation = int(tracker["early_transition_confirmation_samples"])
        late_confirmation = int(tracker["late_transition_confirmation_samples"])
        reset_confirmation = int(tracker["stance_reset_confirmation_samples"])
        minimum_early = int(tracker["minimum_early_swing_samples_before_late"])
        early_threshold = float(tracker["early_enter_probability"])
        early_maximum_progress = float(tracker["early_enter_maximum_progress"])
        late_threshold = float(tracker["late_enter_probability"])
        late_hold_threshold = float(tracker["late_hold_probability"])
        reset_threshold = float(tracker["stance_reset_probability"])
        max_regression = float(tracker["maximum_progress_regression_per_sample"])
        maximum_late_samples = int(tracker["maximum_late_state_samples"])
        late_minimum = float(tracker["late_swing_progress_minimum"])
        if (
            min(early_confirmation, late_confirmation, reset_confirmation) < 1
            or minimum_early < 1 or max_regression < 0.0
            or maximum_late_samples < 1
        ):
            raise ValueError("invalid phase tracker thresholds")
        for foot in range(4):
            if current[foot] == 0:
                early_evidence = (
                    probability[foot, 1] >= early_threshold
                    and phase_progress[foot] <= early_maximum_progress
                )
                count[foot] = count[foot] + 1 if early_evidence else 0
                if count[foot] >= early_confirmation:
                    current[foot] = 1
                    count[foot] = 0
                    age[foot] = 1
                    previous[foot] = phase_progress[foot]
            elif current[foot] == 1:
                age[foot] += 1
                if probability[foot, 0] >= reset_threshold:
                    count[foot] += 1
                    if count[foot] >= reset_confirmation:
                        current[foot] = 0
                        count[foot] = 0
                        age[foot] = 0
                        previous[foot] = 0.0
                        continue
                else:
                    monotonic = phase_progress[foot] >= previous[foot] - max_regression
                    late_evidence = (
                        age[foot] >= minimum_early
                        and probability[foot, 2] >= late_threshold
                        and phase_progress[foot] >= late_minimum
                        and monotonic
                    )
                    count[foot] = count[foot] + 1 if late_evidence else 0
                    if count[foot] >= late_confirmation:
                        current[foot] = 2
                        count[foot] = 0
                        age[foot] = 0
                previous[foot] = phase_progress[foot]
            elif current[foot] == 2:
                age[foot] += 1
                count[foot] = count[foot] + 1 if probability[foot, 0] >= reset_threshold else 0
                if count[foot] >= reset_confirmation or age[foot] > maximum_late_samples:
                    current[foot] = 0
                    count[foot] = 0
                    age[foot] = 0
                    previous[foot] = 0.0
                else:
                    previous[foot] = phase_progress[foot]
            else:
                raise ValueError("unknown phase state")
        eligible = (
            (current == 2)
            & (probability[:, 2] >= late_hold_threshold)
            & (phase_progress >= late_minimum)
        )
        confidence = np.where(eligible, probability[:, 2], 0.0)
        output_progress = np.where(eligible, phase_progress, 0.0)
        next_state = TouchdownPhaseTrackerState(current, count, age, previous)
        return TouchdownPhaseTrackerResult(
            output_progress, confidence, eligible, True, next_state
        )
    except (KeyError, TypeError, ValueError):
        return _invalid()


def estimate_touchdown_phase_with_tracker(
    joint_position_history: Any,
    joint_velocity_history: Any,
    previous_action_history: Any,
    artifact: dict[str, Any],
    state: TouchdownPhaseTrackerState,
    *,
    allow_unfrozen_for_validation: bool = False,
) -> TouchdownPhaseTrackerResult:
    """Run frozen three-state classifier and advance its causal tracker."""
    try:
        usable = artifact.get("frozen") is True and artifact.get("passed") is True
        if not usable and not (
            allow_unfrozen_for_validation and artifact.get("development_candidate") is True
        ):
            raise ValueError("phase tracker artifact is not frozen PASS")
        offsets = tuple(int(value) for value in artifact["history_offsets_samples"])
        required = max(offsets) + 1
        histories = tuple(np.asarray(value, dtype=np.float64) for value in (
            joint_position_history, joint_velocity_history, previous_action_history,
        ))
        if any(value.ndim != 2 or value.shape[1] != 12 or len(value) < required for value in histories):
            raise ValueError("phase histories must have shape [time,12] and sufficient length")
        if not all(np.all(np.isfinite(value)) for value in histories):
            raise ValueError("phase histories must be finite")
        features = np.concatenate([
            np.concatenate([value[-1-offset] for value in histories]) for offset in offsets
        ])
        mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
        scale = np.asarray(artifact["feature_scale"], dtype=np.float64)
        weights = np.asarray(artifact["classifier_weight"], dtype=np.float64)
        intercept = np.asarray(artifact["classifier_intercept"], dtype=np.float64)
        progress_weight = np.asarray(artifact["progress_weight"], dtype=np.float64)
        progress_intercept = np.asarray(artifact["progress_intercept"], dtype=np.float64)
        if mean.shape != features.shape or scale.shape != features.shape or np.any(scale <= 0.0):
            raise ValueError("invalid phase feature normalization")
        if weights.shape != (4, 3, len(features)) or intercept.shape != (4, 3):
            raise ValueError("invalid three-state classifier")
        if progress_weight.shape != (4, len(features)) or progress_intercept.shape != (4,):
            raise ValueError("invalid progress regressor")
        standardized = (features - mean) / scale
        logits = np.einsum("fcn,n->fc", weights, standardized) + intercept
        probability = _softmax(logits)
        progress = np.clip(progress_weight @ standardized + progress_intercept, 0.0, 1.0)
        return update_tracker_from_probabilities(
            probability, progress, state, artifact["tracker"]
        )
    except (KeyError, TypeError, ValueError):
        return _invalid()
