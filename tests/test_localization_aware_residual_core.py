from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))
from anymal_locomotion_ros2.localization_aware_residual_core import (
    AdaptationMode,
    compose_localization_aware_action,
    plan_localization_aware_residual,
)


def _plan(**updates):
    values = {
        "requested_command": [1.5, -0.2, 0.8],
        "confidence": 0.6,
        "tracking_valid": True,
        "confidence_age_s": 0.1,
        "receipt_age_s": 0.05,
        "adaptation_available": True,
        "adaptation_uncertainty": 0.1,
        "residual_beneficial": True,
        "candidate_action_residual": np.full(12, 0.02, dtype=np.float32),
    }
    values.update(updates)
    return plan_localization_aware_residual(**values)


def test_adapted_path_preserves_exact_original_command() -> None:
    plan = _plan()
    assert plan.mode is AdaptationMode.ADAPTED_RESIDUAL
    np.testing.assert_array_equal(plan.effective_command, np.asarray([1.5, -0.2, 0.8], np.float32))
    np.testing.assert_array_equal(plan.action_residual, np.full(12, 0.02, np.float32))


def test_no_beneficial_residual_does_not_stop_or_limit_command() -> None:
    plan = _plan(residual_beneficial=False)
    assert plan.mode is AdaptationMode.BACKBONE_ONLY
    np.testing.assert_array_equal(plan.effective_command, np.asarray([1.5, -0.2, 0.8], np.float32))
    np.testing.assert_array_equal(plan.action_residual, np.zeros(12, np.float32))


def test_missing_or_uncertain_adaptation_falls_back_to_b() -> None:
    missing = _plan(adaptation_available=False)
    uncertain = _plan(adaptation_uncertainty=0.3)
    for plan in (missing, uncertain):
        assert plan.mode is AdaptationMode.LEGACY_B_FALLBACK
        np.testing.assert_allclose(plan.effective_command, [0.75, -0.1, 0.4], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(plan.action_residual, np.zeros(12, np.float32))


def test_invalid_or_stale_slam_is_exact_hard_stop() -> None:
    for plan in (
        _plan(tracking_valid=False),
        _plan(confidence_age_s=0.6),
        _plan(receipt_age_s=0.2),
        _plan(confidence=float("nan")),
    ):
        assert plan.mode is AdaptationMode.HARD_STOP
        np.testing.assert_array_equal(plan.effective_command, np.zeros(3, np.float32))
        np.testing.assert_array_equal(plan.action_residual, np.zeros(12, np.float32))


def test_malformed_or_out_of_bound_residual_falls_back_to_b() -> None:
    for candidate in ([0.0] * 11, [0.051] + [0.0] * 11, [float("nan")] * 12):
        plan = _plan(candidate_action_residual=candidate)
        assert plan.mode is AdaptationMode.LEGACY_B_FALLBACK
        np.testing.assert_array_equal(plan.action_residual, np.zeros(12, np.float32))


def test_action_composition_adds_residual_only_in_adapted_mode() -> None:
    plan = _plan()
    baseline = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    np.testing.assert_allclose(
        compose_localization_aware_action(baseline, plan),
        baseline + np.float32(0.02),
        rtol=0,
        atol=1e-7,
    )


def test_frozen_research_contract_excludes_limiter_as_primary_claim() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/slam_localization_aware_motor_adaptation_v1.yaml").read_text()
    )
    assert config["research_claim"]["velocity_limiter_as_primary_contribution"] is False
    assert config["architecture"]["valid_adapted_command"] == "exact_original_requested_command"
    assert config["architecture"]["residual_module_command_scaling_allowed"] is False
    assert config["architecture"]["full_system_governor"]["minimal_necessary_slowdown_allowed_when_scale_one_is_infeasible"] is True
    assert config["rma_inspiration"]["localization_dynamics_latent"]["hardcoded_turn_timer_forbidden"] is True
    assert config["low_level_headroom_gate"]["reporting_required"]["superiority_to_arm_b_required"] is False
    assert set(config["anti_collapse"]) == {
        "stop_collapse", "shuffle_collapse", "posture_collapse", "limiter_collapse", "body_sensor_motion"
    }
    assert config["boundaries"]["live_execution_authorized"] is False
    assert config["boundaries"]["ppo_training_authorized"] is False
