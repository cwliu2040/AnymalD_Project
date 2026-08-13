"""ROS-independent pose-to-body-twist helpers for SLAM backend adapters."""

from __future__ import annotations

import math

import numpy as np


def normalized_quaternion_xyzw(
    quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(quaternion_xyzw, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(values))
    if norm < 1.0e-12:
        raise ValueError("quaternion has near-zero norm")
    return values / norm


def rotation_matrix_from_quaternion_xyzw(
    quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    x, y, z, w = normalized_quaternion_xyzw(quaternion_xyzw)
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def body_pose_from_sensor_pose(
    sensor_position_xyz: tuple[float, float, float] | np.ndarray,
    sensor_quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
    sensor_translation_in_body_xyz: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a world sensor pose to the aligned body-frame origin pose.

    ``sensor_translation_in_body_xyz`` is the sensor origin expressed in the
    body frame. The sensor and body axes are required to be aligned; the
    returned orientation is therefore the normalized sensor orientation.
    """

    sensor_position = np.asarray(sensor_position_xyz, dtype=np.float64)
    translation = np.asarray(
        sensor_translation_in_body_xyz,
        dtype=np.float64,
    )
    if (
        sensor_position.shape != (3,)
        or translation.shape != (3,)
        or not np.isfinite(sensor_position).all()
        or not np.isfinite(translation).all()
    ):
        raise ValueError("sensor position and translation must be finite XYZ")
    quaternion = normalized_quaternion_xyzw(sensor_quaternion_xyzw)
    rotation_world_from_body = rotation_matrix_from_quaternion_xyzw(
        quaternion
    )
    body_position = sensor_position - rotation_world_from_body @ translation
    return body_position, quaternion


def _multiply_quaternions_xyzw(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return np.asarray(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dtype=np.float64,
    )


def body_velocity_from_pose_delta(
    previous_position_xyz: tuple[float, float, float] | np.ndarray,
    previous_quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
    current_position_xyz: tuple[float, float, float] | np.ndarray,
    current_quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return linear and angular velocity in the previous body frame."""
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    previous_position = np.asarray(previous_position_xyz, dtype=np.float64)
    current_position = np.asarray(current_position_xyz, dtype=np.float64)
    if (
        previous_position.shape != (3,)
        or current_position.shape != (3,)
        or not np.isfinite(previous_position).all()
        or not np.isfinite(current_position).all()
    ):
        raise ValueError("positions must contain three finite values")

    previous_quaternion = normalized_quaternion_xyzw(previous_quaternion_xyzw)
    current_quaternion = normalized_quaternion_xyzw(current_quaternion_xyzw)
    rotation_world_from_previous = rotation_matrix_from_quaternion_xyzw(
        previous_quaternion
    )
    linear_body = (
        rotation_world_from_previous.T @ (current_position - previous_position)
    ) / dt_s

    inverse_previous = np.asarray(
        (
            -previous_quaternion[0],
            -previous_quaternion[1],
            -previous_quaternion[2],
            previous_quaternion[3],
        ),
        dtype=np.float64,
    )
    relative = normalized_quaternion_xyzw(
        _multiply_quaternions_xyzw(inverse_previous, current_quaternion)
    )
    if relative[3] < 0.0:
        relative = -relative
    vector_norm = float(np.linalg.norm(relative[:3]))
    if vector_norm < 1.0e-12:
        rotation_vector = 2.0 * relative[:3]
    else:
        angle = 2.0 * math.atan2(vector_norm, float(relative[3]))
        rotation_vector = relative[:3] * (angle / vector_norm)
    angular_body = rotation_vector / dt_s
    return linear_body, angular_body


def body_linear_velocity_from_world(
    linear_velocity_world_xyz: tuple[float, float, float] | np.ndarray,
    body_quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """Express a world-frame linear velocity in the current body frame."""
    velocity_world = np.asarray(linear_velocity_world_xyz, dtype=np.float64)
    if velocity_world.shape != (3,) or not np.isfinite(velocity_world).all():
        raise ValueError("world linear velocity must contain three finite values")
    rotation_world_from_body = rotation_matrix_from_quaternion_xyzw(
        body_quaternion_xyzw
    )
    return rotation_world_from_body.T @ velocity_world
