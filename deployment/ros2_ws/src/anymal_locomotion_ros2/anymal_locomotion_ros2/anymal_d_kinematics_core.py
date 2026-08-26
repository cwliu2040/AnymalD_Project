"""Frozen ROS-independent ANYmal-D foot kinematics in model1450 order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FOOT_ORDER = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")
POLICY_INDICES = np.asarray(((0, 4, 5), (2, 8, 9), (1, 6, 7), (3, 10, 11)), dtype=np.int64)
JOINT_ORIGINS_ZERO_BASE_M = np.asarray((
    ((0.304, 0.109, 0.0), (0.373, 0.115, 0.0), (0.373, 0.2955, -0.285)),
    ((0.304, -0.109, 0.0), (0.373, -0.115, 0.0), (0.373, -0.2955, -0.285)),
    ((-0.304, 0.109, 0.0), (-0.373, 0.115, 0.0), (-0.373, 0.2955, -0.285)),
    ((-0.304, -0.109, 0.0), (-0.373, -0.115, 0.0), (-0.373, -0.2955, -0.285)),
), dtype=np.float64)
FOOT_ZERO_BASE_M = np.asarray((
    (0.473, 0.31775, -0.6774598),
    (0.473, -0.31775, -0.6774598),
    (-0.473, 0.31775, -0.6774598),
    (-0.473, -0.31775, -0.6774598),
), dtype=np.float64)
JOINT_AXES_ZERO_BASE = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)))
POLICY_ACTION_SCALE_RAD = 0.5


@dataclass(frozen=True)
class AnymalDFootKinematics:
    foot_position_base_m: np.ndarray
    foot_jacobian_per_rad: np.ndarray
    foot_jacobian_per_policy_action: np.ndarray
    foot_velocity_base_mps: np.ndarray


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    identity = np.eye(3)
    return identity + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def anymal_d_foot_kinematics(
    joint_position_rad: Any,
    joint_velocity_radps: Any | None = None,
) -> AnymalDFootKinematics:
    """Return four feet and 3x12 Jacobians using canonical model1450 joints."""
    position = np.asarray(joint_position_rad, dtype=np.float64)
    velocity = (
        np.zeros(12, dtype=np.float64)
        if joint_velocity_radps is None
        else np.asarray(joint_velocity_radps, dtype=np.float64)
    )
    if position.shape != (12,) or velocity.shape != (12,):
        raise ValueError("ANYmal-D joint position/velocity must have shape (12,)")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("ANYmal-D joint position/velocity must be finite")
    feet = np.zeros((4, 3), dtype=np.float64)
    jacobian = np.zeros((4, 3, 12), dtype=np.float64)
    for foot in range(4):
        indices = POLICY_INDICES[foot]
        angles = position[indices]
        zero_origins = JOINT_ORIGINS_ZERO_BASE_M[foot]
        origin0 = zero_origins[0]
        axis0 = JOINT_AXES_ZERO_BASE[0]
        rotation0 = _rotation(axis0, float(angles[0]))
        origin1 = origin0 + rotation0 @ (zero_origins[1] - zero_origins[0])
        axis1 = rotation0 @ JOINT_AXES_ZERO_BASE[1]
        rotation1 = rotation0 @ _rotation(JOINT_AXES_ZERO_BASE[1], float(angles[1]))
        origin2 = origin1 + rotation1 @ (zero_origins[2] - zero_origins[1])
        axis2 = rotation1 @ JOINT_AXES_ZERO_BASE[2]
        rotation2 = rotation1 @ _rotation(JOINT_AXES_ZERO_BASE[2], float(angles[2]))
        foot_position = origin2 + rotation2 @ (FOOT_ZERO_BASE_M[foot] - zero_origins[2])
        feet[foot] = foot_position
        jacobian[foot, :, indices[0]] = np.cross(axis0, foot_position - origin0)
        jacobian[foot, :, indices[1]] = np.cross(axis1, foot_position - origin1)
        jacobian[foot, :, indices[2]] = np.cross(axis2, foot_position - origin2)
    foot_velocity = np.einsum("fij,j->fi", jacobian, velocity)
    return AnymalDFootKinematics(
        foot_position_base_m=feet,
        foot_jacobian_per_rad=jacobian,
        foot_jacobian_per_policy_action=jacobian * POLICY_ACTION_SCALE_RAD,
        foot_velocity_base_mps=foot_velocity,
    )
