"""Deterministic FAST-LIO2 confidence hard-fault validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from anymal_locomotion_ros2.fastlio_confidence_fault_validation import (
    validate_fastlio_fault_matrix,
)
from anymal_locomotion_ros2.slam_confidence_core import DegradationReason


def _case_by_name(report: dict, name: str) -> dict:
    return next(case for case in report["cases"] if case["name"] == name)


def test_required_fast_fault_matrix_passes() -> None:
    report = validate_fastlio_fault_matrix()

    assert report["schema_version"] == 1
    assert report["backend_id"] == "fastlio2"
    assert report["deployment_calibration"] is False
    assert report["logical_tick_ns"] == 50_000_000
    assert report["case_count"] == 13
    assert report["passed_case_count"] == report["case_count"]
    assert report["passed"] is True


def test_fault_matrix_thresholds_match_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    contract = yaml.safe_load(
        (project_root / "configs/slam_confidence_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    report = validate_fastlio_fault_matrix()
    freshness = contract["timing"]["provisional_freshness_s"]["fastlio2"]

    assert report["thresholds_s"] == {
        "source": freshness["confidence_source"],
        "canonical_odometry": freshness["canonical_odometry"],
        "lidar_input": freshness["lidar_input"],
        "imu_input": freshness["imu_input"],
        "diagnostic_match": contract["bundle_assembly"]["fastlio2"][
            "diagnostic_match_timeout_s"
        ],
        "future_stamp_tolerance": contract["timing"][
            "future_stamp_tolerance_s"
        ],
        "recovery_dwell": contract["hysteresis"]["recover_dwell_s"],
    }


def test_fault_matrix_covers_required_observable_reasons() -> None:
    report = validate_fastlio_fault_matrix()
    expected = {
        "odometry_and_source_freeze": DegradationReason.ODOMETRY_STALE,
        "lidar_input_freeze": DegradationReason.LIDAR_STALE,
        "imu_input_freeze": DegradationReason.IMU_STALE,
        "effective_diagnostic_missing": DegradationReason.SIGNAL_MISSING,
        "zero_effective_support": DegradationReason.INSUFFICIENT_SUPPORT,
        "native_numeric_invalid": DegradationReason.NUMERIC_INVALID,
        "future_source_stamp": DegradationReason.TIMESTAMP_INVALID,
        "conflicting_duplicate": DegradationReason.TIMESTAMP_INVALID,
        "regressing_source_stamp": DegradationReason.TIMESTAMP_INVALID,
        "malformed_effect_layout": DegradationReason.NUMERIC_INVALID,
        "native_frame_contract_violation": DegradationReason.BACKEND_ERROR,
        "clock_backward": DegradationReason.CLOCK_RESET,
        "missing_scan_then_recovery": DegradationReason.SIGNAL_MISSING,
    }

    for name, reason in expected.items():
        case = _case_by_name(report, name)
        assert case["detected_reasons"] & int(reason)
        assert case["detected_state"] == case["expected_state"]
        assert case["detection_latency_ns"] <= (
            case["detection_deadline_ns"] - case["injected_at_ns"]
        )
        if name == "clock_backward":
            assert case["confidence_at_detection"] == 0.0
        else:
            assert case["confidence_at_detection"] > 0.89


def test_missing_scan_fault_is_one_shot_and_recovers() -> None:
    report = validate_fastlio_fault_matrix()
    case = _case_by_name(report, "missing_scan_then_recovery")

    assert case["recovery_required"] is True
    assert case["recovered_at_ns"] is not None
    assert case["recovered_at_ns"] > case["detected_at_ns"]


def test_fault_report_is_deterministic() -> None:
    assert validate_fastlio_fault_matrix() == validate_fastlio_fault_matrix()
