"""Tests for deterministic press-and-hold keyboard command semantics."""

from __future__ import annotations

import numpy as np

from anymal_locomotion_ros2.teleop_core import TeleopLimits, TeleopState


def test_motion_keys_use_ros_body_frame_conventions() -> None:
    state = TeleopState()

    state.press("w")
    state.press("q")
    state.press("a")
    np.testing.assert_allclose(state.command(), [0.5, 0.3, 0.5])

    state.release("w")
    state.release("q")
    state.release("a")
    state.press("s")
    state.press("e")
    state.press("d")
    np.testing.assert_allclose(state.command(), [-0.5, -0.3, -0.5])


def test_opposing_keys_cancel_and_space_stops() -> None:
    state = TeleopState()
    for key in ("w", "s", "a", "d", "q", "e"):
        state.press(key)
    np.testing.assert_array_equal(state.command(), np.zeros(3, dtype=np.float32))

    state.release("s")
    state.release("d")
    state.release("e")
    np.testing.assert_allclose(state.command(), [0.5, 0.3, 0.5])
    state.press("space")
    np.testing.assert_array_equal(state.command(), np.zeros(3, dtype=np.float32))


def test_speed_adjustment_never_exceeds_policy_limits() -> None:
    state = TeleopState(
        limits=TeleopLimits(
            forward_speed=1.0,
            lateral_speed=1.0,
            yaw_speed=1.0,
            scale_step=1.0,
            max_scale=10.0,
        )
    )
    for _ in range(20):
        state.increase_speed()
    for key in ("w", "q", "a"):
        state.press(key)
    np.testing.assert_allclose(state.command(), [3.0, 1.5, 2.0])

    state.stop()
    for key in ("s", "e", "d"):
        state.press(key)
    np.testing.assert_allclose(state.command(), [-2.0, -1.5, -2.0])


def test_speed_scale_has_a_positive_floor() -> None:
    state = TeleopState()
    for _ in range(20):
        state.decrease_speed()
    assert state.speed_scale == state.limits.min_scale
