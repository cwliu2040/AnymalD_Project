from __future__ import annotations

import pytest

from anymal_locomotion_ros2.lio_benchmark_core import PoseSample, get_motion_profile
from anymal_locomotion_ros2.yaw_stress_core import (
    build_blind_review_assignments,
    build_replay_cells,
    counterbalanced_schedule,
    evaluate_yaw_stress_phases,
    output_gap_metrics,
    validate_bag_metadata,
)


@pytest.mark.parametrize("direction", ("left", "right"))
@pytest.mark.parametrize("rate", (0.25, 0.5, 1.0, 1.5, 2.0))
def test_yaw_stress_profiles_share_timeline(direction: str, rate: float) -> None:
    profile = get_motion_profile(
        f"yaw_stress_{direction}_{str(rate).replace('.', '_')}"
    )
    sign = 1.0 if direction == "left" else -1.0
    assert profile.duration_s == pytest.approx(27.0)
    assert profile.command_at(7.0) == pytest.approx((0.0, 0.0, sign * rate))
    assert profile.command_at(15.0) == pytest.approx((0.0, 0.0, sign * rate))
    assert profile.command_at(17.0) == (0.0, 0.0, 0.0)


def test_formal_schedule_has_240_cells_and_is_reproducible() -> None:
    cells = build_replay_cells(
        seeds=(42, 43, 44),
        backends=("liosam", "fastlio2"),
        point_densities=(1.0, 0.75, 0.5, 0.25),
    )
    first = counterbalanced_schedule(cells)
    second = counterbalanced_schedule(cells)
    assert len(first) == 240
    assert first == second
    assert {cell.case_name for cell in first} == {cell.case_name for cell in cells}


def test_blind_review_assignments_hide_case_names_and_are_reproducible() -> None:
    cases = ("yaw_left__fastlio2", "yaw_left__liosam")
    first = build_blind_review_assignments(cases)
    second = build_blind_review_assignments(reversed(cases))
    assert first == second
    assert all("fastlio" not in str(item["blind_id"]) for item in first)
    assert [item["review_index"] for item in first] == [1, 2]


def test_bag_validation_requires_backend_neutral_contract() -> None:
    document = {
        "rosbag2_bagfile_information": {
            "duration": {"nanoseconds": 27_000_000_000},
            "topics_with_message_count": [],
        }
    }
    failures = validate_bag_metadata(document)
    assert any("/lidar/points_raw" in failure for failure in failures)


def test_output_gap_metrics_reports_absolute_and_relative_gaps() -> None:
    metrics = output_gap_metrics((0.0, 0.1, 0.2, 0.7))
    assert metrics["max_gap_s"] == pytest.approx(0.5)
    assert metrics["thresholds"]["0.25"]["count"] == 1
    assert metrics["over_3x_median"]["count"] == 1


def test_output_gap_metrics_ignores_timestamp_roundoff_at_threshold() -> None:
    metrics = output_gap_metrics((0.0, 0.10000353, 0.20000706))
    assert metrics["thresholds"]["0.10"]["count"] == 0


def test_phase_metrics_preserve_unwrapped_multi_turn_yaw() -> None:
    truth = []
    estimate = []
    for index in range(55):
        stamp = 7.0 + index * 0.145
        yaw = index * 0.4
        truth.append(PoseSample(stamp, 0.0, 0.0, 0.0, yaw))
        estimate.append(PoseSample(stamp, 0.0, 0.0, 0.0, yaw * 0.9))
    report = evaluate_yaw_stress_phases(truth, estimate, experiment_start_s=0.0)
    hold = report["phases"]["hold"]["trajectory"]
    assert hold["ground_truth_cumulative_yaw_deg"] > 1000.0
    assert hold["cumulative_yaw_error_deg"] < 0.0
