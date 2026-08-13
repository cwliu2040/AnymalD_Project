"""Frame-safe conversion for a vendor body-state odometry authority."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def world_to_body_velocity(
    velocity_world: Sequence[float], quaternion_xyzw: Sequence[float]
) -> np.ndarray:
    velocity = np.asarray(velocity_world, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if velocity.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("velocity and quaternion must have shapes (3,) and (4,)")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("velocity or quaternion contains NaN/Inf")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-9:
        raise ValueError("orientation quaternion has near-zero norm")
    x, y, z, w = quaternion / norm
    rotation_body_to_world = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    return (rotation_body_to_world.T @ velocity).astype(np.float32)


def adapt_linear_velocity(
    velocity: Sequence[float],
    quaternion_xyzw: Sequence[float],
    *,
    input_velocity_frame: str,
    maximum_absolute_velocity_mps: float = 8.0,
) -> np.ndarray:
    frame = input_velocity_frame.strip().lower()
    if frame == "body":
        result = np.asarray(velocity, dtype=np.float32)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError("body velocity must contain three finite values")
    elif frame in ("odom", "world"):
        result = world_to_body_velocity(velocity, quaternion_xyzw)
    else:
        raise ValueError("input_velocity_frame must be 'body' or 'odom'")
    if np.any(np.abs(result) > maximum_absolute_velocity_mps):
        raise ValueError("vendor velocity exceeded the absolute velocity guard")
    return result
