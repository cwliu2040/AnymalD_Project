"""ROS-independent policy contract, observation, and action logic."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import yaml


class PolicyBackend(Protocol):
    """Minimal inference backend interface."""

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        """Return one batch of raw policy actions."""


class GaitMode(IntEnum):
    TRACK = 0
    DECELERATE = 1
    HOLD = 2
    RECOVER = 3


@dataclass(frozen=True)
class GaitModeGovernorConfig:
    """Deployment mirror of the frozen gait-mode v1 contract."""

    degrade_below: float = 0.45
    degrade_dwell_s: float = 0.10
    recover_at_or_above: float = 0.55
    recover_dwell_s: float = 0.50
    hold_minimum_dwell_s: float = 0.50
    command_scale_rate_down_per_s: float = 0.8
    command_scale_rate_up_per_s: float = 2.0

    def validate(self) -> None:
        if not 0.0 <= self.degrade_below < self.recover_at_or_above <= 1.0:
            raise ValueError("gait-mode confidence thresholds are invalid")
        values = (
            self.degrade_dwell_s,
            self.recover_dwell_s,
            self.hold_minimum_dwell_s,
            self.command_scale_rate_down_per_s,
            self.command_scale_rate_up_per_s,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("gait-mode dwell times and rates must be positive")


class GaitModeGovernor:
    """Scalar state machine used by the external 50 Hz policy runtime."""

    def __init__(self, config: GaitModeGovernorConfig | None = None) -> None:
        self.config = config or GaitModeGovernorConfig()
        self.config.validate()
        self.mode = GaitMode.HOLD
        self.command_scale = 0.0
        self.degrade_elapsed_s = 0.0
        self.hold_elapsed_s = 0.0
        self.recover_elapsed_s = 0.0
        self.initialized = False

    def reset(self) -> None:
        self.mode = GaitMode.HOLD
        self.command_scale = 0.0
        self.degrade_elapsed_s = 0.0
        self.hold_elapsed_s = 0.0
        self.recover_elapsed_s = 0.0
        self.initialized = False

    def update(self, confidence: float, valid: bool, dt_s: float) -> float:
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        cfg = self.config
        desired_scale = (
            float(np.clip((confidence - 0.2) / 0.8, 0.0, 1.0))
            if valid
            else 0.0
        )
        if not self.initialized:
            healthy = bool(valid) and confidence >= cfg.recover_at_or_above
            self.mode = GaitMode.TRACK if healthy else GaitMode.HOLD
            self.command_scale = 1.0 if healthy else 0.0
            self.initialized = True
            return self.command_scale

        if self.mode == GaitMode.TRACK:
            unhealthy = desired_scale < 1.0 - 1.0e-6
            self.degrade_elapsed_s = self.degrade_elapsed_s + dt_s if unhealthy else 0.0
            if (not valid) or self.degrade_elapsed_s + 1.0e-6 >= cfg.degrade_dwell_s:
                self.mode = GaitMode.DECELERATE

        if self.mode == GaitMode.DECELERATE:
            self.command_scale = max(
                desired_scale,
                self.command_scale - cfg.command_scale_rate_down_per_s * dt_s,
            )
            if self.command_scale <= 0.0:
                self.mode = GaitMode.HOLD
                self.hold_elapsed_s = 0.0
                self.recover_elapsed_s = 0.0

        if self.mode == GaitMode.HOLD:
            self.command_scale = 0.0
            self.hold_elapsed_s += dt_s
            recover_ready = bool(valid) and confidence >= cfg.recover_at_or_above
            self.recover_elapsed_s = (
                self.recover_elapsed_s + dt_s if recover_ready else 0.0
            )
            if (
                self.hold_elapsed_s + 1.0e-6 >= cfg.hold_minimum_dwell_s
                and self.recover_elapsed_s + 1.0e-6 >= cfg.recover_dwell_s
            ):
                self.mode = GaitMode.RECOVER

        if self.mode == GaitMode.RECOVER:
            if (not valid) or desired_scale + 1.0e-6 < self.command_scale:
                self.mode = GaitMode.DECELERATE
            else:
                self.command_scale = min(
                    desired_scale,
                    self.command_scale + cfg.command_scale_rate_up_per_s * dt_s,
                )
                if self.command_scale >= 1.0:
                    self.mode = GaitMode.TRACK
                    self.command_scale = 1.0
                    self.degrade_elapsed_s = 0.0
        return self.command_scale


@dataclass(frozen=True)
class PolicyContract:
    """Deployment fields loaded from policy_metadata.yaml."""

    joint_order: tuple[str, ...]
    default_joint_positions: tuple[float, ...]
    observation_dimension: int
    action_dimension: int
    action_scale: float
    control_period_s: float
    command_limits: tuple[tuple[float, float], ...]
    observation_terms: tuple[str, ...]

    @classmethod
    def from_metadata(cls, path: str | Path) -> PolicyContract:
        metadata_path = Path(path).expanduser().resolve()
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        action = metadata["action"]
        limits = metadata["command"]["limits"]
        contract = cls(
            joint_order=tuple(metadata["joint_order"]),
            default_joint_positions=tuple(
                float(value) for value in action["default_joint_positions"]
            ),
            observation_dimension=int(metadata["observation"]["dimension"]),
            action_dimension=int(action["dimension"]),
            action_scale=float(action["scale"]),
            control_period_s=float(action["control_period_s"]),
            command_limits=(
                tuple(float(value) for value in limits["vx"]),
                tuple(float(value) for value in limits["vy"]),
                tuple(float(value) for value in limits["wz"]),
            ),
            observation_terms=tuple(
                str(term["name"]) for term in metadata["observation"]["terms"]
            ),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if len(self.joint_order) != 12 or len(set(self.joint_order)) != 12:
            raise ValueError("Policy metadata must contain 12 unique canonical joint names")
        if len(self.default_joint_positions) != 12:
            raise ValueError("Policy metadata must contain 12 default joint positions")
        expected_base_terms = (
            "base_linear_velocity",
            "base_angular_velocity",
            "projected_gravity",
            "velocity_command",
            "relative_joint_position",
            "relative_joint_velocity",
            "previous_action",
        )
        expected_terms = expected_base_terms + (
            ("slam_confidence",) if self.observation_dimension == 51 else ()
        )
        if (
            self.observation_dimension not in (48, 51)
            or self.action_dimension != 12
            or self.observation_terms != expected_terms
        ):
            raise ValueError(
                "Expected the ordered 48-D base contract or 51-D SLAM "
                f"confidence contract, received {self.observation_dimension} -> "
                f"{self.action_dimension} terms={self.observation_terms}"
            )
        if self.action_scale <= 0.0 or self.control_period_s <= 0.0:
            raise ValueError("Action scale and control period must be positive")
        invalid_limits = any(lower > upper for lower, upper in self.command_limits)
        if len(self.command_limits) != 3 or invalid_limits:
            raise ValueError("Command limits must contain ordered vx, vy, and wz ranges")


@dataclass(frozen=True)
class RobotState:
    """One policy tick of canonical robot state."""

    base_linear_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    projected_gravity: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray


@dataclass(frozen=True)
class InferenceResult:
    """Raw inference data and final joint-position targets."""

    observation: np.ndarray
    raw_action: np.ndarray
    joint_targets: np.ndarray


def _vector(values: Sequence[float] | np.ndarray, dimension: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), received {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or Inf")
    return vector


def canonical_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    contract: PolicyContract,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap a ROS JointState by name into deterministic policy order."""
    received_names = tuple(names)
    duplicates = sorted(name for name, count in Counter(received_names).items() if count > 1)
    missing = sorted(set(contract.joint_order) - set(received_names))
    unexpected = sorted(set(received_names) - set(contract.joint_order))
    if duplicates or missing or unexpected:
        raise ValueError(
            "JointState contract mismatch: "
            f"missing={missing}, duplicate={duplicates}, unexpected={unexpected}"
        )
    if len(positions) != len(received_names) or len(velocities) != len(received_names):
        raise ValueError("JointState name, position, and velocity arrays must have equal lengths")
    indices = [received_names.index(name) for name in contract.joint_order]
    canonical_positions = _vector([positions[index] for index in indices], 12, "joint_positions")
    canonical_velocities = _vector(
        [velocities[index] for index in indices],
        12,
        "joint_velocities",
    )
    return canonical_positions, canonical_velocities


def projected_gravity_from_quaternion(
    x: float,
    y: float,
    z: float,
    w: float,
) -> np.ndarray:
    """Project normalized world gravity [0, 0, -1] into the body frame."""
    quaternion = _vector([x, y, z, w], 4, "imu_orientation")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-6:
        raise ValueError("IMU orientation quaternion has near-zero norm")
    x_n, y_n, z_n, w_n = quaternion / norm
    gravity = -np.asarray(
        [
            2.0 * (x_n * z_n - w_n * y_n),
            2.0 * (y_n * z_n + w_n * x_n),
            1.0 - 2.0 * (x_n * x_n + y_n * y_n),
        ],
        dtype=np.float32,
    )
    return gravity


def clamp_command(command: Sequence[float] | np.ndarray, contract: PolicyContract) -> np.ndarray:
    """Clamp body-frame vx, vy, and wz to the training command ranges."""
    command_vector = _vector(command, 3, "velocity_command")
    lower = np.asarray([limits[0] for limits in contract.command_limits], dtype=np.float32)
    upper = np.asarray([limits[1] for limits in contract.command_limits], dtype=np.float32)
    return np.clip(command_vector, lower, upper)


def decelerate_command_toward_zero(
    command: Sequence[float] | np.ndarray,
    elapsed_s: float,
    linear_deceleration: float,
    angular_deceleration: float,
) -> np.ndarray:
    """Slew a body velocity command toward zero without changing its sign."""
    values = _vector(command, 3, "velocity_command")
    if elapsed_s < 0.0:
        raise ValueError("elapsed_s must be non-negative")
    if linear_deceleration <= 0.0 or angular_deceleration <= 0.0:
        raise ValueError("deceleration limits must be positive")
    maximum_change = np.asarray(
        [
            linear_deceleration * elapsed_s,
            linear_deceleration * elapsed_s,
            angular_deceleration * elapsed_s,
        ],
        dtype=np.float32,
    )
    magnitude = np.maximum(np.abs(values) - maximum_change, 0.0)
    return np.copysign(magnitude, values).astype(np.float32, copy=False)


def build_observation(
    state: RobotState,
    command: Sequence[float] | np.ndarray,
    previous_action: Sequence[float] | np.ndarray,
    contract: PolicyContract,
    slam_confidence: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Build the exact 48-D or 51-D observation used during training."""
    default_positions = _vector(contract.default_joint_positions, 12, "default_joint_positions")
    terms: list[np.ndarray] = [
            _vector(state.base_linear_velocity, 3, "base_linear_velocity"),
            _vector(state.base_angular_velocity, 3, "base_angular_velocity"),
            _vector(state.projected_gravity, 3, "projected_gravity"),
            clamp_command(command, contract),
            _vector(state.joint_positions, 12, "joint_positions") - default_positions,
            _vector(state.joint_velocities, 12, "joint_velocities"),
            _vector(previous_action, 12, "previous_action"),
    ]
    if contract.observation_dimension == 51:
        if slam_confidence is None:
            raise ValueError("51-D policy requires slam_confidence observation")
        confidence = _vector(slam_confidence, 3, "slam_confidence")
        if np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise ValueError("slam_confidence values must be in [0, 1]")
        terms.append(confidence)
    elif slam_confidence is not None:
        raise ValueError("48-D policy must not receive slam_confidence observation")
    observation = np.concatenate(terms).astype(np.float32, copy=False)
    if observation.shape != (contract.observation_dimension,):
        raise RuntimeError(f"Observation contract produced unexpected shape {observation.shape}")
    return observation


class PolicyRuntime:
    """Stateful 50 Hz policy runtime independent of ROS message types."""

    def __init__(
        self,
        contract: PolicyContract,
        backend: PolicyBackend,
        *,
        max_abs_policy_action: float = 10.0,
        gait_mode_governor: GaitModeGovernor | None = None,
    ) -> None:
        if max_abs_policy_action <= 0.0:
            raise ValueError("max_abs_policy_action must be positive")
        self.contract = contract
        self.backend = backend
        self.max_abs_policy_action = max_abs_policy_action
        self.gait_mode_governor = gait_mode_governor
        self.previous_action = np.zeros(contract.action_dimension, dtype=np.float32)

    def reset(self) -> None:
        self.previous_action.fill(0.0)
        if self.gait_mode_governor is not None:
            self.gait_mode_governor.reset()

    def seed_previous_action(
        self,
        previous_action: Sequence[float] | np.ndarray,
    ) -> None:
        """Restore the action-history term for a diagnostic state transplant."""
        restored = _vector(
            previous_action,
            self.contract.action_dimension,
            "previous_action",
        )
        self.previous_action = restored.copy()

    def step(
        self,
        state: RobotState,
        command: Sequence[float] | np.ndarray,
        slam_confidence: Sequence[float] | np.ndarray | None = None,
    ) -> InferenceResult:
        effective_command = command
        if self.gait_mode_governor is not None:
            if self.contract.observation_dimension != 51 or slam_confidence is None:
                raise ValueError("gait-mode governor requires the 51-D confidence contract")
            confidence = _vector(slam_confidence, 3, "slam_confidence")
            scale = self.gait_mode_governor.update(
                float(confidence[0]),
                bool(confidence[1] >= 0.5),
                self.contract.control_period_s,
            )
            effective_command = _vector(command, 3, "velocity_command") * scale
        observation = build_observation(
            state,
            effective_command,
            self.previous_action,
            self.contract,
            slam_confidence,
        )
        backend_output = np.asarray(self.backend(observation[None, :]), dtype=np.float32)
        if backend_output.size != self.contract.action_dimension:
            raise ValueError(
                f"Policy backend returned {backend_output.size} actions, "
                f"expected {self.contract.action_dimension}"
            )
        raw_action = backend_output.reshape(self.contract.action_dimension)
        if not np.all(np.isfinite(raw_action)):
            raise ValueError("Policy backend returned NaN or Inf")
        if float(np.max(np.abs(raw_action))) > self.max_abs_policy_action:
            raise ValueError(
                f"Policy action exceeded simulation guard {self.max_abs_policy_action}: "
                f"{float(np.max(np.abs(raw_action)))}"
            )
        defaults = np.asarray(self.contract.default_joint_positions, dtype=np.float32)
        joint_targets = defaults + self.contract.action_scale * raw_action
        self.previous_action = raw_action.copy()
        return InferenceResult(
            observation=observation,
            raw_action=raw_action,
            joint_targets=joint_targets,
        )
