"""Tests for the ROS-independent locomotion stability diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from anymal_locomotion.stability_diagnostics import (
    DiagnosticThresholds,
    DiagnosticTraceWriter,
    build_diagnostic_report,
    classify_instability,
    evaluate_stability_gate,
    load_diagnostic_trace,
    quaternion_to_rpy_wxyz,
    summarize_samples,
)


FOOT_NAMES = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")


def _sample(
    time_s: float,
    *,
    slip_speed: float = 0.0,
    roll: float = 0.0,
    height: float = 0.6,
    terminated: bool = False,
) -> dict:
    return {
        "time_s": time_s,
        "command": [0.5, 0.0, 0.0],
        "actual_velocity": [0.45, 0.01, 0.02],
        "base_position_w_m": [0.45 * time_s, 0.0, height],
        "base_height_m": height,
        "roll_rad": roll,
        "pitch_rad": 0.0,
        "yaw_rad": 0.0,
        "base_contact_force_n": 0.0,
        "terminated": terminated,
        "truncated": False,
        "feet": {
            name: {
                "normal_force_n": 100.0,
                "tangential_force_n": 10.0,
                "tangential_speed_mps": slip_speed if name == "LF_FOOT" else 0.01,
            }
            for name in FOOT_NAMES
        },
    }


def test_quaternion_to_rpy_wxyz_handles_yaw() -> None:
    half_yaw = math.pi / 4.0
    roll, pitch, yaw = quaternion_to_rpy_wxyz(
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    )
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)
    assert yaw == pytest.approx(math.pi / 2.0)


def test_classifier_reports_foot_slip_before_body_instability() -> None:
    samples = [
        _sample(0.00),
        _sample(0.02, slip_speed=0.4),
        _sample(0.04, slip_speed=0.4),
        _sample(0.06, slip_speed=0.4, roll=0.6),
        _sample(0.08, slip_speed=0.4, roll=0.6, terminated=True),
    ]
    result = classify_instability(
        samples,
        DiagnosticThresholds(
            foot_slip_speed_mps=0.3,
            body_roll_pitch_rad=0.5,
            min_base_height_m=0.35,
            coincidence_window_s=0.01,
        ),
    )
    assert result["classification"] == "foot_slip_first"
    assert result["first_foot_slip_time_s"] == pytest.approx(0.02)
    assert result["first_body_instability_time_s"] == pytest.approx(0.06)


def test_threshold_mapping_rejects_silent_contract_drift() -> None:
    values = {
        "stance_min_normal_force_n": 50.0,
        "foot_slip_speed_mps": 0.6,
        "body_roll_pitch_rad": 0.25,
        "min_base_height_m": 0.45,
        "consecutive_samples": 2,
        "coincidence_window_s": 0.1,
    }
    thresholds = DiagnosticThresholds.from_mapping(values)
    assert thresholds.stance_min_normal_force_n == pytest.approx(50.0)
    with pytest.raises(ValueError, match="unexpected"):
        DiagnosticThresholds.from_mapping({**values, "typo": 1.0})


def test_classifier_does_not_force_ambiguous_failure() -> None:
    samples = [
        _sample(0.00),
        _sample(0.02, terminated=True),
    ]
    result = classify_instability(
        samples,
        DiagnosticThresholds(
            foot_slip_speed_mps=0.3,
            body_roll_pitch_rad=0.5,
            min_base_height_m=0.35,
        ),
    )
    assert result["classification"] == "indeterminate"
    assert result["first_hard_failure_time_s"] == pytest.approx(0.02)


def test_classifier_treats_recovered_slip_as_contained() -> None:
    samples = [
        _sample(0.00, slip_speed=0.7),
        _sample(0.02, slip_speed=0.7),
        _sample(0.04),
    ]
    result = classify_instability(
        samples,
        DiagnosticThresholds(
            stance_min_normal_force_n=50.0,
            foot_slip_speed_mps=0.6,
            body_roll_pitch_rad=0.25,
            min_base_height_m=0.45,
        ),
    )
    assert result["classification"] == "no_instability"
    assert result["contained_foot_slip"]


def test_summary_reports_tracking_and_stance_distributions() -> None:
    samples = [_sample(0.00), _sample(0.02), _sample(0.04)]
    summary = summarize_samples(samples)
    assert summary["sample_count"] == 3
    assert summary["tracking"]["vx_mps"]["mean_actual"] == pytest.approx(0.45)
    assert summary["tracking"]["vx_mps"]["mean_command"] == pytest.approx(0.5)
    assert summary["tracking"]["target_sample_count"] == 3
    assert summary["tracking"]["vx_mps"]["target_mean_absolute_error"] == pytest.approx(0.05)
    assert summary["tracking"]["vx_mps"]["all_absolute_actual"]["p95"] == pytest.approx(0.45)
    assert summary["tracking"]["vx_mps"]["mean_absolute_error"] == pytest.approx(0.05)
    assert (
        summary["feet"]["LF_FOOT"]["stance_tangential_speed_mps"]["sample_count"]
        == 3
    )
    assert summary["event_order"]["classification"] == "not_evaluated"


def test_report_keeps_raw_samples_for_event_review() -> None:
    samples = [_sample(0.00), _sample(0.02)]
    report = build_diagnostic_report(
        samples=samples,
        metadata={"profile": "forward_0_5"},
    )
    assert report["schema_version"] == 1
    assert report["metadata"]["profile"] == "forward_0_5"
    assert report["samples"] == samples


def test_incremental_diagnostic_trace_is_recoverable(tmp_path: Path) -> None:
    report_path = tmp_path / "locomotion_diagnostics.json"
    samples = [{"time_s": 0.02}, {"time_s": 0.04, "terminated": True}]

    writer = DiagnosticTraceWriter(report_path)
    for sample in samples:
        writer.append(sample)
    writer.flush()
    assert writer.path == tmp_path / "locomotion_diagnostics.jsonl"
    assert load_diagnostic_trace(writer.path) == samples
    writer.close()
    assert not report_path.exists()


def test_stability_gate_rejects_tracking_regression() -> None:
    samples = [_sample(index * 0.02) for index in range(30)]
    thresholds = DiagnosticThresholds(
        stance_min_normal_force_n=50.0,
        foot_slip_speed_mps=0.6,
        body_roll_pitch_rad=0.25,
        min_base_height_m=0.45,
        consecutive_samples=2,
        coincidence_window_s=0.1,
    )
    summary = summarize_samples(samples, thresholds=thresholds)
    config = {
        "schema_version": 1,
        "hard_gate": {
            "require_zero_terminations": True,
            "require_zero_truncations": True,
            "require_finite_samples": True,
            "require_no_instability_event": True,
        },
        "tracking_gate": {
            "minimum_target_samples": 25,
            "maximum_mean_absolute_error": {
                "vx_mps": 0.01,
                "vy_mps": 0.2,
                "wz_radps": 0.25,
            },
            "stationary_maximum_mean_absolute_velocity": {
                "vx_mps": 0.15,
                "vy_mps": 0.15,
                "wz_radps": 0.2,
            },
            "stationary_maximum_planar_displacement_m": 0.25,
            "stationary_maximum_yaw_change_rad": 0.2,
        },
    }
    gate = evaluate_stability_gate(
        summary,
        target=(0.5, 0.0, 0.0),
        config=config,
    )
    assert not gate["passed"]
    assert gate["tracking_gate_applied"]
    assert any("vx_mps_mae" in failure for failure in gate["failures"])


def test_stability_gate_can_report_tracking_without_gating_it() -> None:
    samples = [_sample(index * 0.02) for index in range(30)]
    thresholds = DiagnosticThresholds(
        stance_min_normal_force_n=50.0,
        foot_slip_speed_mps=0.6,
        body_roll_pitch_rad=0.25,
        min_base_height_m=0.45,
        consecutive_samples=2,
        coincidence_window_s=0.1,
    )
    summary = summarize_samples(samples, thresholds=thresholds)
    config = {
        "schema_version": 1,
        "hard_gate": {
            "require_zero_terminations": True,
            "require_zero_truncations": True,
            "require_finite_samples": True,
            "require_no_instability_event": True,
        },
        "tracking_gate": {"enabled": False},
    }
    gate = evaluate_stability_gate(
        summary,
        target=(0.5, 0.0, 0.0),
        config=config,
    )
    assert gate["passed"]
    assert not gate["tracking_gate_applied"]
    assert summary["tracking"]["vx_mps"]["target_mean_absolute_error"] > 0.01

    summary["hard_failures"]["terminated_count"] = 1
    hard_failure = evaluate_stability_gate(
        summary,
        target=(0.5, 0.0, 0.0),
        config=config,
    )
    assert not hard_failure["passed"]
    assert "terminated_count=1" in hard_failure["failures"]
