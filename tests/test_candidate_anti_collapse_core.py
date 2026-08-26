from __future__ import annotations

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))
from anymal_locomotion_ros2.candidate_anti_collapse_core import evaluate_candidate_strata


def _config() -> dict:
    return yaml.safe_load((ROOT / "configs/slam_constrained_residual_c2_v1.yaml").read_text())


def _fast() -> dict:
    return {"backend": "fastlio2", "profile": "curve", "candidate_minus_b_risk": 0.0, "candidate_over_b_progress_ratio": 1.0, "candidate_minus_b_stopped_fraction": 0.0, "candidate_minus_b_safety_events": 0}


def _lio(**updates) -> dict:
    value = {"backend": "liosam", "profile": "curve", "candidate_minus_b_risk": -0.02, "candidate_over_b_progress_ratio": 1.08, "candidate_minus_b_stopped_fraction": -0.01, "candidate_minus_b_safety_events": 0, "equivalence_fraction_vs_b": 0.55, "equivalence_fraction_vs_old_c": 0.50}
    value.update(updates)
    return value


def test_fast_can_equal_b_when_liosam_has_noncollapsed_improvement() -> None:
    report = evaluate_candidate_strata([_fast(), _lio()], _config())
    assert report["status"] == "PASS", report


def test_candidate_equal_to_b_and_old_c_is_explicit_collapse() -> None:
    report = evaluate_candidate_strata([
        _fast(), _lio(
            candidate_minus_b_risk=0.0,
            candidate_over_b_progress_ratio=1.0,
            equivalence_fraction_vs_b=0.98,
            equivalence_fraction_vs_old_c=0.97,
        )
    ], _config())
    assert report["status"] == "COLLAPSED_TO_B_OR_OLD_C"


def test_progress_gain_cannot_rescue_more_stopping() -> None:
    report = evaluate_candidate_strata([
        _fast(), _lio(candidate_minus_b_stopped_fraction=0.02)
    ], _config())
    assert report["status"] == "FAIL"


def test_c2_contract_reuses_requested_command_rewards_and_keeps_training_closed() -> None:
    config = _config()
    architecture = config["architecture"]
    reward = config["reward_contract"]
    evaluation = config["ceiling_aware_evaluation"]
    assert architecture["frozen_backbone"] == "model1450"
    assert architecture["initialize_from_old_c_model48"] is False
    assert architecture["valid_fresh_input_command"] == "original_requested_command"
    assert architecture["apply_arm_b_soft_scale_before_candidate"] is False
    assert reward["reuse_existing_locomotion_reward_stack"] is True
    assert reward["duplicate_velocity_reward_allowed"] is False
    assert reward["velocity_tracking_target"] == "original_requested_command"
    assert set(reward["forbidden_velocity_tracking_terms"]) == {
        "confidence_track_lin_vel_xy_exp",
        "confidence_track_ang_vel_z_exp",
    }
    assert evaluation["fastlio2"]["superiority_required"] is False
    assert evaluation["liosam"]["minimum_improved_degraded_strata"] == 1
    assert config["optimization"]["ppo_training_authorized"] is False
    assert config["gates"]["short_pulse_headroom_passed"] is True
    assert config["gates"]["richer_component_identification_passed"] is True
    assert config["gates"]["offline_component_risk_model_blockwise_gate_passed"] is False
    assert config["gates"]["action_conditioned_risk_signal_available"] is False
