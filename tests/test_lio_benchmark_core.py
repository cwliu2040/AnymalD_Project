"""Tests for deterministic LIO-SAM benchmark profiles and metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from anymal_locomotion_ros2.lio_benchmark_core import (
    PoseSample,
    evaluate_trajectory,
    get_motion_profile,
    registered_overlap_separation,
)


def test_motion_profile_ramps_and_returns_to_zero() -> None:
    profile = get_motion_profile("forward_3_0")
    assert profile.command_at(0.0) == (0.0, 0.0, 0.0)
    assert profile.command_at(profile.warmup_s) == (0.0, 0.0, 0.0)
    assert profile.command_at(profile.warmup_s + profile.ramp_s) == (
        3.0,
        0.0,
        0.0,
    )
    assert profile.command_at(profile.duration_s) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("name", "expected_displacement"),
    (
        ("loop_out_and_back", 0.0),
        ("loop_back_and_forth", 0.0),
        ("loop_open_forward", 4.5),
        ("loop_open_backward", -4.5),
    ),
)
def test_loop_profiles_integrate_declared_displacement(
    name: str,
    expected_displacement: float,
) -> None:
    profile = get_motion_profile(name)
    integrated_displacement = sum(
        duration_s * 0.5 * (start[0] + end[0])
        for duration_s, start, end in profile.sequence
    )
    assert integrated_displacement == pytest.approx(expected_displacement)


@pytest.mark.parametrize(
    ("name", "target"),
    (
        ("backward_0_5", (-0.5, 0.0, 0.0)),
        ("backward_1_0", (-1.0, 0.0, 0.0)),
        ("backward_2_0", (-2.0, 0.0, 0.0)),
        ("lateral_right_0_75", (0.0, -0.75, 0.0)),
        ("lateral_right_1_5", (0.0, -1.5, 0.0)),
        ("yaw_left_1_0", (0.0, 0.0, 1.0)),
        ("yaw_right_0_5", (0.0, 0.0, -0.5)),
        ("yaw_right_1_0", (0.0, 0.0, -1.0)),
        ("yaw_right_2_0", (0.0, 0.0, -2.0)),
        ("curve_0_5_left_0_5", (0.5, 0.0, 0.5)),
        ("curve_0_5_left_0_25_long", (0.5, 0.0, 0.25)),
        ("curve_0_5_left_0_25_long_shifted", (0.5, 0.0, 0.25)),
        ("curve_0_5_right_0_5", (0.5, 0.0, -0.5)),
        ("curve_1_5_left_1_0", (1.5, 0.0, 1.0)),
        ("curve_1_5_right_1_0", (1.5, 0.0, -1.0)),
        ("curve_3_0_left_0_5", (3.0, 0.0, 0.5)),
        ("curve_3_0_right_0_5", (3.0, 0.0, -0.5)),
        ("combined_long", (1.5, 0.0, 1.0)),
    ),
)
def test_stability_profiles_reach_declared_target(
    name: str,
    target: tuple[float, float, float],
) -> None:
    profile = get_motion_profile(name)
    assert profile.target == target
    assert profile.hold_s > 0.0
    assert profile.command_at(profile.warmup_s + profile.ramp_s) == target
    assert (
        profile.command_at(
            profile.warmup_s + profile.ramp_s + 0.5 * profile.hold_s
        )
        == target
    )
    assert profile.command_at(profile.duration_s) == (0.0, 0.0, 0.0)


def test_trajectory_evaluation_aligns_initial_pose() -> None:
    truth = [
        PoseSample(0.0, 10.0, -3.0, 0.5, 0.2),
        PoseSample(1.0, 11.0, -3.0, 0.5, 0.2),
        PoseSample(2.0, 12.0, -3.0, 0.5, 0.2),
    ]
    estimate = [
        PoseSample(0.0, 0.0, 0.0, 0.0, -0.3),
        PoseSample(1.0, math.cos(0.5), -math.sin(0.5), 0.0, -0.3),
        PoseSample(2.0, 2.0 * math.cos(0.5), -2.0 * math.sin(0.5), 0.0, -0.3),
    ]

    metrics = evaluate_trajectory(truth, estimate)
    assert metrics["translation_ate_rmse_m"] == pytest.approx(0.0, abs=1.0e-9)
    assert metrics["yaw_rmse_deg"] == pytest.approx(0.0, abs=1.0e-9)
    assert metrics["translation_jump_residual_max_m"] == pytest.approx(
        0.0,
        abs=1.0e-9,
    )
    assert metrics["ground_truth_height_drop_max_m"] == pytest.approx(0.0)
    assert metrics["ground_truth_high_speed_scan_count"] == 0
    assert metrics["ground_truth_high_yaw_rate_scan_count"] == 0


def test_trajectory_evaluation_detects_pose_jump() -> None:
    truth = [
        PoseSample(0.0, 0.0, 0.0, 0.0, 0.0),
        PoseSample(1.0, 1.0, 0.0, 0.0, 0.0),
        PoseSample(2.0, 2.0, 0.0, 0.0, 0.0),
    ]
    estimate = [
        PoseSample(0.0, 0.0, 0.0, 0.0, 0.0),
        PoseSample(1.0, 1.0, 0.0, 0.0, 0.0),
        PoseSample(2.0, 2.5, 0.0, 0.0, 0.0),
    ]

    metrics = evaluate_trajectory(truth, estimate)
    assert metrics["translation_jump_residual_max_m"] == pytest.approx(0.5)


def test_trajectory_evaluation_compares_rotating_sensor_frame() -> None:
    truth = [
        PoseSample(0.0, 0.0, 0.0, 0.0, 0.0),
        PoseSample(1.0, 0.0, 0.0, 0.0, math.pi / 2.0),
    ]
    estimate = [
        PoseSample(0.0, 0.2, 0.0, 0.35, 0.0),
        PoseSample(1.0, 0.0, 0.2, 0.35, math.pi / 2.0),
    ]

    metrics = evaluate_trajectory(
        truth,
        estimate,
        ground_truth_sensor_offset_xyz=(0.2, 0.0, 0.35),
    )

    assert metrics["translation_ate_rmse_m"] == pytest.approx(0.0)


def test_registered_wall_separation_uses_overlapping_surfaces() -> None:
    x, z = np.meshgrid(
        np.linspace(-2.0, 2.0, 30),
        np.linspace(0.0, 2.0, 20),
    )
    previous = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    current = previous + np.asarray((0.0, 0.1, 0.0))

    separation = registered_overlap_separation(previous, current)

    assert separation == pytest.approx(0.1)


def test_registered_wall_separation_excludes_horizontal_floor() -> None:
    x, y = np.meshgrid(
        np.linspace(-2.0, 2.0, 30),
        np.linspace(-2.0, 2.0, 30),
    )
    floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))

    separation = registered_overlap_separation(
        floor,
        floor + np.asarray((0.0, 0.0, 0.1)),
    )

    assert separation is None
