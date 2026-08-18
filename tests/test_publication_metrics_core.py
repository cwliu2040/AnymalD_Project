from __future__ import annotations

import math

import pytest

from anymal_locomotion_ros2.publication_metrics_core import summarize_locomotion, summarize_mechanism


def _sample(index: int) -> dict:
    return {
        "time_s": index * 0.02,
        "command": [1.0, 0.0, 0.0],
        "actual_linear_velocity_body_mps": [0.9, 0.0, 0.0],
        "base_position_w_m": [index * 0.018, 0.0, 0.55],
        "base_height_m": 0.55,
        "roll_rad": 0.01 * index,
        "pitch_rad": 0.0,
        "yaw_rad": 0.0,
        "base_contact_force_n": 0.0,
        "terminated": False,
        "feet": {
            name: {"normal_force_n": 100.0, "tangential_speed_mps": 0.1}
            for name in ("LF", "RF", "LH", "RH")
        },
        "applied_raw_action": [0.01 * index] * 12,
    }


def test_run_metrics_reduce_frames_to_one_experimental_unit() -> None:
    metrics = summarize_locomotion([_sample(index) for index in range(100)])
    assert not metrics["fall"]
    assert not metrics["base_contact"]
    assert not metrics["foot_slip_event"]
    assert metrics["moving_speed_mps"] == pytest.approx(0.9)
    assert metrics["normalized_progress"] == pytest.approx(0.9, rel=0.02)
    assert metrics["roll_pitch_rate_rms_radps"] == pytest.approx(0.5)
    assert metrics["while_stable_roll_pitch_rate_rms_radps"] == pytest.approx(0.5)
    assert metrics["action_rate_rms_per_s"] == pytest.approx(math.sqrt(12) * 0.5)


def test_two_consecutive_stance_slip_ticks_create_event() -> None:
    samples = [_sample(index) for index in range(3)]
    samples[1]["feet"]["LF"]["tangential_speed_mps"] = 0.7
    samples[2]["feet"]["LF"]["tangential_speed_mps"] = 0.8
    assert summarize_locomotion(samples)["foot_slip_event"]


def test_while_stable_rate_excludes_terminal_impact() -> None:
    samples = [_sample(index) for index in range(6)]
    samples[4]["roll_rad"] = 1.0
    samples[4]["base_height_m"] = 0.40
    samples[4]["terminated"] = True
    samples[5]["roll_rad"] = 2.0
    metrics = summarize_locomotion(samples)
    assert metrics["roll_pitch_rate_rms_radps"] > 10.0
    assert metrics["while_stable_roll_pitch_rate_rms_radps"] == pytest.approx(0.5)
    assert metrics["while_stable_sample_count"] == 4
    assert metrics["first_instability_time_s"] == pytest.approx(0.08)


def test_mechanism_reports_degraded_nonzero_fraction() -> None:
    base = {
        "intent_blend": 0.5, "raw_stride_attenuation": 0.2,
        "applied_stride_attenuation": 0.1, "crouch": -0.1,
        "stance_width": 0.1, "action_smoothing": 0.2,
        "structured_delta_l2": 0.3,
    }
    summary = summarize_mechanism([{**base, "safe_scale": 1.0}, {**base, "safe_scale": 0.5}])
    assert summary["degraded_sample_count"] == 1
    assert summary["structured_delta_l2_degraded_nonzero_fraction"] == 1.0
