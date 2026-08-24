from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))
from anymal_locomotion_ros2.constrained_residual_c2_core import (
    C2Mode,
    compose_c2_action,
    plan_c2_control,
)


def _plan(**updates):
    values = {
        "requested_command": [1.0, -0.5, 0.8],
        "confidence": 0.45,
        "tracking_valid": True,
        "confidence_age_s": 0.1,
        "receipt_age_s": 0.05,
        "candidate_available": True,
        "candidate_admissible": True,
        "candidate_command_scales": [0.8, 0.6, 1.0],
        "candidate_action_residual": [0.05] * 12,
    }
    values.update(updates)
    return plan_c2_control(**values)


def test_candidate_uses_original_command_with_anisotropic_scales_and_residual() -> None:
    plan = _plan()
    assert plan.mode is C2Mode.CANDIDATE
    assert plan.legacy_b_scale == pytest.approx(0.3125)
    np.testing.assert_allclose(plan.effective_command, [0.8, -0.3, 0.8])
    np.testing.assert_allclose(compose_c2_action(np.zeros(12), plan), [0.05] * 12)


def test_healthy_candidate_is_exact_model1450_on_original_command() -> None:
    plan = _plan(
        candidate_command_scales=[1.0, 1.0, 1.0],
        candidate_action_residual=[0.0] * 12,
    )
    baseline = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
    np.testing.assert_array_equal(
        plan.effective_command,
        np.asarray([1.0, -0.5, 0.8], dtype=np.float32),
    )
    np.testing.assert_array_equal(compose_c2_action(baseline, plan), baseline)


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"candidate_available": False}, "candidate_unavailable"),
        ({"candidate_command_scales": [float("nan"), 0.5, 0.5]}, "candidate_numeric_or_shape_invalid"),
        ({"candidate_command_scales": [0.1, 0.5, 0.5]}, "candidate_scale_out_of_bounds"),
        ({"candidate_action_residual": [0.1001] * 12}, "candidate_residual_out_of_bounds"),
    ],
)
def test_missing_or_malformed_candidate_falls_back_exactly_to_b(updates, reason) -> None:
    plan = _plan(**updates)
    assert plan.mode is C2Mode.LEGACY_B_FALLBACK
    assert plan.reason == reason
    np.testing.assert_allclose(plan.effective_command, np.asarray([1.0, -0.5, 0.8]) * 0.3125)
    np.testing.assert_array_equal(plan.action_residual, np.zeros(12))


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"tracking_valid": False}, "tracking_invalid"),
        ({"confidence_age_s": 0.500001}, "confidence_source_stale"),
        ({"receipt_age_s": 0.150001}, "confidence_receipt_stale"),
        ({"confidence": float("nan")}, "slam_numeric_invalid"),
    ],
)
def test_invalid_or_stale_slam_is_an_exact_hard_stop(updates, reason) -> None:
    plan = _plan(**updates)
    assert plan.mode is C2Mode.HARD_STOP
    assert plan.reason == reason
    np.testing.assert_array_equal(plan.effective_command, np.zeros(3))
    np.testing.assert_array_equal(plan.action_residual, np.zeros(12))


def test_well_formed_but_risk_inadmissible_candidate_stops() -> None:
    plan = _plan(candidate_admissible=False)
    assert plan.mode is C2Mode.RISK_STOP
    np.testing.assert_array_equal(plan.effective_command, np.zeros(3))


def test_original_command_contract_failure_is_not_silently_hidden() -> None:
    with pytest.raises(ValueError, match="requested_command"):
        _plan(requested_command=[1.0, float("inf"), 0.0])


def test_yaml_contract_matches_core_limits_and_keeps_wiring_closed() -> None:
    config = yaml.safe_load((ROOT / "configs/slam_constrained_residual_c2_v1.yaml").read_text())
    outputs = config["architecture"]["candidate_outputs"]
    assert outputs["command_component_scales_xyz"]["minimum"] == [0.2, 0.2, 0.2]
    assert outputs["command_component_scales_xyz"]["maximum"] == [1.0, 1.0, 1.0]
    assert outputs["bounded_joint_action_residual_linf"] == 0.10
    assert config["gates"]["c2_pure_core_complete"] is True
    assert config["gates"]["training_environment_wiring_complete"] is False
    assert config["gates"]["ppo_training_authorized"] is False
