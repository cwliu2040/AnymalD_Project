"""Tests for project-owned full SE(3) LiDAR deskew helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from anymal_locomotion_ros2.motion_deskew_core import (
    TimedAngularVelocity,
    TimedPose,
    deskew_points_from_imu_and_odometry,
    deskew_points_from_odometry,
    interpolate_pose,
    relative_translation_in_start_frame,
    select_range_image_points,
    translational_offsets_from_columns,
    translational_offsets_from_odometry,
)


def test_pose_interpolation_and_relative_translation() -> None:
    samples = [
        TimedPose(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TimedPose(2.0, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ]
    start = interpolate_pose(samples, 1.25)
    end = interpolate_pose(samples, 1.75)

    np.testing.assert_allclose(start.position_xyz, (0.5, 0.0, 0.0))
    np.testing.assert_allclose(
        relative_translation_in_start_frame(start, end),
        (1.0, 0.0, 0.0),
    )


def test_relative_translation_is_expressed_in_rotated_start_frame() -> None:
    yaw_90 = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    start = TimedPose(0.0, (0.0, 0.0, 0.0), yaw_90)
    end = TimedPose(1.0, (1.0, 0.0, 0.0), yaw_90)

    np.testing.assert_allclose(
        relative_translation_in_start_frame(start, end),
        (0.0, -1.0, 0.0),
        atol=1.0e-7,
    )


def test_clockwise_ouster_columns_interpolate_translation() -> None:
    offsets = translational_offsets_from_columns(
        np.asarray([512, 256, 0, 768], dtype=np.int32),
        horizon_scan=1024,
        scan_translation_xyz=np.asarray([0.4, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        offsets[:, 0],
        (0.0, 0.1, 0.2, 0.3),
    )


def test_interpolation_rejects_odometry_state_change() -> None:
    samples = [
        TimedPose(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0),
        TimedPose(2.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 1),
    ]
    with pytest.raises(ValueError, match="state changed"):
        interpolate_pose(samples, 1.5)


def test_column_offsets_follow_nonconstant_odometry_motion() -> None:
    samples = [
        TimedPose(1.00, (0.00, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TimedPose(1.05, (0.05, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TimedPose(1.10, (0.20, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ]
    offsets, final_translation = translational_offsets_from_odometry(
        np.asarray([512, 256, 0], dtype=np.int32),
        horizon_scan=1024,
        scan_start_s=1.0,
        scan_period_s=0.1,
        samples=samples,
    )

    # Phases are 0, 0.25 and 0.5. The quadratic-like samples therefore
    # produce 0, 0.025 and 0.05 m, rather than endpoint-linear 0, 0.05, 0.10.
    np.testing.assert_allclose(offsets[:, 0], (0.0, 0.025, 0.05))
    np.testing.assert_allclose(final_translation, (0.05, 0.0, 0.0))


def test_range_image_selection_is_row_major_and_keeps_first_cell_point() -> None:
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    rings = np.asarray([1, 0, 0, 0], dtype=np.uint8)
    times = np.asarray([0.0, 0.0, 0.025, 0.05])

    selected, selected_times, columns = select_range_image_points(
        points,
        rings,
        times,
        n_scan=2,
        horizon_scan=4,
    )

    np.testing.assert_allclose(
        selected,
        (
            (0.0, -1.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
    )
    np.testing.assert_allclose(selected_times, (0.025, 0.0, 0.0))
    assert columns.tolist() == [1, 2, 2]


def test_full_se3_deskew_removes_translation_and_rotation() -> None:
    half_yaw = math.pi / 4.0
    samples = [
        TimedPose(
            0.0,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        TimedPose(
            1.0,
            (1.0, 0.0, 0.0),
            (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
        ),
    ]
    # Both measurements observe the same world point (2, 0, 0).
    points = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    corrected, displacement = deskew_points_from_odometry(
        points,
        np.asarray([0.0, 1.0]),
        samples,
    )

    np.testing.assert_allclose(
        corrected,
        ((2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(displacement, (1.0, 0.0, 0.0))


def test_gyro_rotation_with_odometry_translation_deskews_same_point() -> None:
    poses = [
        TimedPose(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        TimedPose(
            1.0,
            (1.0, 0.0, 0.0),
            (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
        ),
    ]
    angular_velocities = [
        TimedAngularVelocity(0.0, (0.0, 0.0, math.pi / 2.0)),
        TimedAngularVelocity(1.0, (0.0, 0.0, math.pi / 2.0)),
    ]
    points = np.asarray(((2.0, 0.0, 0.0), (0.0, -1.0, 0.0)))

    corrected, displacement = deskew_points_from_imu_and_odometry(
        points,
        np.asarray((0.0, 1.0)),
        angular_velocities,
        poses,
    )

    np.testing.assert_allclose(
        corrected,
        ((2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(displacement, (1.0, 0.0, 0.0))
