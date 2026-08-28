from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/run_slam_confidence_joint_training_motion_audit.py"
PROTOCOL = ROOT / "configs/slam_confidence_joint_training_v1.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("joint_motion_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _report(multiplier: float = 1.0) -> dict:
    gait = {
        "cadence_hz": 2.5,
        "contact_switch_rate_hz": 5.0,
        "duty_factor": 0.55,
        "body_height_m": 0.55,
        "stance_width_m": 0.45,
        "minimum_joint_margin_rad": 0.25,
        "foot_clearance_m": 0.05,
        "torque_rms_nm": 20.0,
        "energy_per_progress": 100.0,
        "stance_foot_slip_rms_mps": 0.10,
        "stopped_fraction": 0.0,
    }
    body = {
        "roll_pitch_rate_rms_radps": 0.20 * multiplier,
        "yaw_tracking_error_rms_radps": 0.05,
        "linear_tracking_error_rms_mps": 0.05,
        "linear_acceleration_rms_mps2": 2.0 * multiplier,
        "angular_acceleration_rms_radps2": 10.0 * multiplier,
        "linear_jerk_rms_mps3": 100.0 * multiplier,
        "lidar_scan_translation_error_rms_m": 0.02 * multiplier,
        "lidar_scan_rotation_error_rms_rad": 0.08 * multiplier,
    }
    return {
        "command": {"vx_mps": 1.0, "vy_mps": 0.0, "wz_radps": 0.5},
        "stability": {"survival_rate": 1.0, "termination_events": 0},
        "motion_audit": {
            "body_lidar_metrics": body,
            "gait_safety_metrics": gait,
            "additional_metrics": {
                "mean_planar_speed_mps": 1.0,
                "mean_absolute_yaw_rate_radps": 0.5,
                "positive_linear_progress_m": 10.0,
                "positive_yaw_progress_rad": 5.0,
            },
        },
    }


def test_motion_audit_comparison_requires_matched_motion_and_mechanism() -> None:
    module = _load_module()
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    comparator = _report()
    candidate = _report(0.8)
    passed = module._compare(candidate, comparator, protocol)
    assert passed["passed"] is True

    candidate["motion_audit"]["additional_metrics"]["positive_linear_progress_m"] = 9.0
    failed = module._compare(candidate, comparator, protocol)
    assert failed["passed"] is False
    assert failed["matched_motion"]["checks"]["linear_progress"] is False


def test_completed_motion_audit_is_closed_before_slam() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    audit = protocol["post_training_motion_audit"]
    assert audit["status"] == "complete_failed_pre_slam_gate"
    assert audit["result"]["held_out_fast_lio2_or_lio_sam_executed"] is False
    assert audit["result"]["decision"] == "STOP_BEFORE_SLAM_EVALUATION"
