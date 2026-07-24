"""ROS-independent tests for the external policy runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anymal_locomotion_ros2.policy_core import (
    PolicyContract,
    PolicyRuntime,
    RobotState,
    build_observation,
    canonical_joint_state,
    clamp_command,
    projected_gravity_from_quaternion,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def contract() -> PolicyContract:
    return PolicyContract.from_metadata(PROJECT_ROOT / "configs" / "policy_metadata.example.yaml")


def test_joint_state_is_remapped_by_name(contract: PolicyContract) -> None:
    names = tuple(reversed(contract.joint_order))
    positions_by_name = {name: float(index) for index, name in enumerate(contract.joint_order)}
    velocities_by_name = {name: -float(index) for index, name in enumerate(contract.joint_order)}
    positions, velocities = canonical_joint_state(
        names,
        [positions_by_name[name] for name in names],
        [velocities_by_name[name] for name in names],
        contract,
    )
    np.testing.assert_array_equal(positions, np.arange(12, dtype=np.float32))
    np.testing.assert_array_equal(velocities, -np.arange(12, dtype=np.float32))


@pytest.mark.parametrize(
    "names",
    [
        lambda order: order[:-1],
        lambda order: order[:-1] + ("EXTRA",),
        lambda order: order[:-1] + (order[0],),
    ],
)
def test_invalid_joint_state_is_rejected(contract: PolicyContract, names) -> None:
    invalid_names = names(contract.joint_order)
    with pytest.raises(ValueError, match="JointState contract mismatch"):
        canonical_joint_state(
            invalid_names,
            np.zeros(len(invalid_names)),
            np.zeros(len(invalid_names)),
            contract,
        )


def test_projected_gravity_uses_imu_orientation() -> None:
    np.testing.assert_allclose(
        projected_gravity_from_quaternion(0.0, 0.0, 0.0, 1.0),
        [0.0, 0.0, -1.0],
        atol=1.0e-7,
    )
    half_yaw = np.sqrt(0.5)
    np.testing.assert_allclose(
        projected_gravity_from_quaternion(0.0, 0.0, half_yaw, half_yaw),
        [0.0, 0.0, -1.0],
        atol=1.0e-7,
    )
    with pytest.raises(ValueError, match="near-zero"):
        projected_gravity_from_quaternion(0.0, 0.0, 0.0, 0.0)


def test_observation_layout_and_command_clamp(contract: PolicyContract) -> None:
    defaults = np.asarray(contract.default_joint_positions, dtype=np.float32)
    state = RobotState(
        base_linear_velocity=np.asarray([1.0, 2.0, 3.0]),
        base_angular_velocity=np.asarray([4.0, 5.0, 6.0]),
        projected_gravity=np.asarray([0.0, 0.0, -1.0]),
        joint_positions=defaults,
        joint_velocities=np.arange(12, dtype=np.float32),
    )
    command = clamp_command([99.0, -99.0, 99.0], contract)
    np.testing.assert_array_equal(command, [3.0, -1.5, 2.0])
    observation = build_observation(state, command, np.full(12, 0.25), contract)

    assert observation.shape == (48,)
    np.testing.assert_array_equal(observation[0:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(observation[3:6], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(observation[6:9], [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(observation[9:12], [3.0, -1.5, 2.0])
    np.testing.assert_array_equal(observation[12:24], np.zeros(12))
    np.testing.assert_array_equal(observation[24:36], np.arange(12))
    np.testing.assert_array_equal(observation[36:48], np.full(12, 0.25))


def test_runtime_scales_actions_and_remembers_previous_action(contract: PolicyContract) -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.inputs: list[np.ndarray] = []

        def __call__(self, observations: np.ndarray) -> np.ndarray:
            self.inputs.append(observations.copy())
            return np.full((1, 12), 0.25 * len(self.inputs), dtype=np.float32)

    backend = FakeBackend()
    runtime = PolicyRuntime(contract, backend)
    defaults = np.asarray(contract.default_joint_positions, dtype=np.float32)
    state = RobotState(
        base_linear_velocity=np.zeros(3),
        base_angular_velocity=np.zeros(3),
        projected_gravity=np.asarray([0.0, 0.0, -1.0]),
        joint_positions=defaults,
        joint_velocities=np.zeros(12),
    )

    first = runtime.step(state, [0.0, 0.0, 0.0])
    second = runtime.step(state, [0.0, 0.0, 0.0])

    np.testing.assert_allclose(first.joint_targets, defaults + 0.5 * 0.25)
    np.testing.assert_array_equal(backend.inputs[0][0, 36:48], np.zeros(12))
    np.testing.assert_array_equal(backend.inputs[1][0, 36:48], np.full(12, 0.25))
    np.testing.assert_array_equal(second.raw_action, np.full(12, 0.5))


def test_runtime_rejects_non_finite_policy_output(contract: PolicyContract) -> None:
    def invalid_backend(_observations: np.ndarray) -> np.ndarray:
        return np.full((1, 12), np.nan, dtype=np.float32)

    runtime = PolicyRuntime(contract, invalid_backend)
    state = RobotState(
        base_linear_velocity=np.zeros(3),
        base_angular_velocity=np.zeros(3),
        projected_gravity=np.asarray([0.0, 0.0, -1.0]),
        joint_positions=np.asarray(contract.default_joint_positions),
        joint_velocities=np.zeros(12),
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        runtime.step(state, [0.0, 0.0, 0.0])
