"""Fail-closed phase->kinematics->touchdown residual pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .anymal_d_kinematics_core import AnymalDFootKinematics, anymal_d_foot_kinematics
from .touchdown_phase_tracker_core import (
    TouchdownPhaseTrackerResult,
    TouchdownPhaseTrackerState,
    estimate_touchdown_phase_with_tracker,
)
from .touchdown_shaping_intervention_core import (
    TouchdownShapingConfig,
    TouchdownShapingResult,
    apply_touchdown_shaping_intervention,
)


@dataclass(frozen=True)
class TouchdownResidualPipelineResult:
    intervention: TouchdownShapingResult
    phase: TouchdownPhaseTrackerResult
    kinematics: AnymalDFootKinematics | None
    wiring_valid: bool
    mode: str


def apply_touchdown_residual_pipeline(
    *,
    requested_command: Any,
    backbone_action: Any,
    joint_position_history: Any,
    joint_velocity_history: Any,
    previous_action_history: Any,
    current_joint_position_rad: Any,
    current_joint_velocity_radps: Any,
    phase_artifact: dict[str, Any],
    phase_state: TouchdownPhaseTrackerState,
    attenuation_fraction: float,
    tracking_valid: bool,
    config: TouchdownShapingConfig = TouchdownShapingConfig(),
    allow_unfrozen_phase_for_validation: bool = False,
) -> TouchdownResidualPipelineResult:
    """Apply a residual only when every causal runtime prerequisite is valid."""
    phase = estimate_touchdown_phase_with_tracker(
        joint_position_history,
        joint_velocity_history,
        previous_action_history,
        phase_artifact,
        phase_state,
        allow_unfrozen_for_validation=allow_unfrozen_phase_for_validation,
    )
    kinematics: AnymalDFootKinematics | None = None
    kinematics_valid = True
    try:
        kinematics = anymal_d_foot_kinematics(
            current_joint_position_rad, current_joint_velocity_radps
        )
    except (TypeError, ValueError):
        kinematics_valid = False
    wiring_valid = bool(tracking_valid and phase.tracking_valid and kinematics_valid)
    if kinematics is None:
        foot_velocity = np.zeros(4, dtype=np.float64)
        jacobian = np.zeros((4, 3, 12), dtype=np.float64)
    else:
        foot_velocity = kinematics.foot_velocity_base_mps[:, 2]
        jacobian = kinematics.foot_jacobian_per_policy_action
    intervention = apply_touchdown_shaping_intervention(
        requested_command=requested_command,
        backbone_action=backbone_action,
        swing_progress=phase.swing_progress,
        phase_confidence=phase.confidence,
        foot_vertical_velocity_mps=foot_velocity,
        foot_position_jacobian_per_action=jacobian,
        tracking_valid=wiring_valid,
        attenuation_fraction=attenuation_fraction,
        config=config,
    )
    if not tracking_valid:
        mode = "tracking_invalid_exact_zero_residual"
    elif not phase.tracking_valid:
        mode = "phase_invalid_exact_zero_residual"
    elif not kinematics_valid:
        mode = "kinematics_invalid_exact_zero_residual"
    elif not np.any(intervention.active_feet):
        mode = "valid_no_eligible_foot_exact_zero_residual"
    else:
        mode = "valid_touchdown_residual"
    return TouchdownResidualPipelineResult(
        intervention=intervention,
        phase=phase,
        kinematics=kinematics,
        wiring_valid=wiring_valid,
        mode=mode,
    )
