"""Tests for FAST-LIO2 pose-to-body-twist conversion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from anymal_locomotion_ros2.slam_odom_adapter_core import (
    body_linear_velocity_from_world,
    body_pose_from_sensor_pose,
    body_velocity_from_pose_delta,
    normalized_quaternion_xyzw,
    rotation_matrix_from_quaternion_xyzw,
)


def test_body_pose_removes_rotated_sensor_offset() -> None:
    half_yaw = math.sin(math.pi / 4.0)
    position, quaternion = body_pose_from_sensor_pose(
        (2.0, 3.0, 1.0),
        (0.0, 0.0, half_yaw, half_yaw),
        (0.20, 0.0, 0.35),
    )

    np.testing.assert_allclose(position, (2.0, 2.8, 0.65), atol=1.0e-12)
    np.testing.assert_allclose(
        quaternion,
        (0.0, 0.0, half_yaw, half_yaw),
        atol=1.0e-12,
    )


def test_normalized_quaternion_rejects_zero_norm() -> None:
    with pytest.raises(ValueError, match="near-zero"):
        normalized_quaternion_xyzw((0.0, 0.0, 0.0, 0.0))


def test_body_velocity_transforms_world_translation_into_previous_body() -> None:
    half_yaw = math.sin(math.pi / 4.0)
    quaternion = (0.0, 0.0, half_yaw, half_yaw)
    linear, angular = body_velocity_from_pose_delta(
        (0.0, 0.0, 0.0),
        quaternion,
        (0.0, 1.0, 0.0),
        quaternion,
        0.5,
    )

    np.testing.assert_allclose(linear, (2.0, 0.0, 0.0), atol=1.0e-9)
    np.testing.assert_allclose(angular, (0.0, 0.0, 0.0), atol=1.0e-9)


def test_body_velocity_reports_relative_yaw_rate() -> None:
    half_yaw = math.sin(math.pi / 8.0)
    linear, angular = body_velocity_from_pose_delta(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, half_yaw, math.cos(math.pi / 8.0)),
        0.5,
    )

    np.testing.assert_allclose(linear, np.zeros(3), atol=1.0e-9)
    np.testing.assert_allclose(angular, (0.0, 0.0, math.pi / 2.0), atol=1.0e-9)


def test_rotation_matrix_is_orthonormal() -> None:
    matrix = rotation_matrix_from_quaternion_xyzw(
        (0.1, -0.2, 0.3, 0.9)
    )
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1.0e-12)


def test_world_linear_velocity_is_rotated_into_current_body() -> None:
    half_yaw = math.sin(math.pi / 4.0)
    velocity = body_linear_velocity_from_world(
        (0.0, 2.0, -0.5),
        (0.0, 0.0, half_yaw, half_yaw),
    )
    np.testing.assert_allclose(velocity, (2.0, 0.0, -0.5), atol=1.0e-9)


def test_world_linear_velocity_rejects_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        body_linear_velocity_from_world(
            (math.nan, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
