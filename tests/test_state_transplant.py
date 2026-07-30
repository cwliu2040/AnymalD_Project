"""Tests for formal-trace state transplant extraction."""

from __future__ import annotations

import math

import pytest

from anymal_locomotion.state_transplant import (
    StateTransplant,
    build_state_transplant_manifest,
)


def test_build_manifest_aligns_policy_by_joint_state_stamp() -> None:
    observation = [float(index) / 10.0 for index in range(48)]
    observation[9:12] = [0.0, 0.0, 0.817349]
    locomotion = {
        "samples": [
            {
                "time_s": 25.0,
                "command": [0.0, 0.0, 0.817349],
                "base_position_w_m": [14.5, 12.0, 0.56],
                "roll_rad": -0.1,
                "pitch_rad": -0.01,
                "yaw_rad": 2.05,
            }
        ]
    }
    policy = {
        "records": [
            {
                "clock_s": 25.01,
                "state_stamps_s": {"joint_state": 25.0},
                "effective_command": [0.0, 0.0, 0.817349],
                "observation": observation,
                "raw_action": [0.25] * 12,
            },
            {
                "clock_s": 25.03,
                "state_stamps_s": {"joint_state": 25.02},
                "effective_command": [0.0, 0.0, 0.817349],
                "observation": observation[:36] + [0.25] * 12,
                "raw_action": [0.3] * 12,
            },
        ]
    }
    contract = {
        "joints": [
            {"default_position": float(index)}
            for index in range(12)
        ]
    }

    manifest = build_state_transplant_manifest(
        locomotion,
        policy,
        contract,
        source_time_s=25.0,
    )
    state = StateTransplant.from_mapping(manifest)

    assert state.source_time_s == pytest.approx(25.0)
    assert state.policy_clock_s == pytest.approx(25.01)
    assert state.joint_positions[3] == pytest.approx(3.0 + observation[15])
    assert state.previous_action == pytest.approx(observation[36:48])
    assert state.bootstrap_action == pytest.approx([0.25] * 12)
    assert state.expected_observation[36:48] == pytest.approx([0.25] * 12)
    assert math.sqrt(
        sum(value * value for value in state.base_orientation_wxyz)
    ) == pytest.approx(1.0)
    assert manifest["limitations"]["actuator_lstm_state_restored"] is False


def test_build_manifest_rejects_command_misalignment() -> None:
    observation = [0.0] * 48
    locomotion = {
        "samples": [
            {
                "time_s": 25.0,
                "command": [1.0, 0.0, 0.0],
                "base_position_w_m": [0.0, 0.0, 0.5],
                "roll_rad": 0.0,
                "pitch_rad": 0.0,
                "yaw_rad": 0.0,
            }
        ]
    }
    policy = {
        "records": [
            {
                "clock_s": 25.01,
                "state_stamps_s": {"joint_state": 25.0},
                "effective_command": [0.0, 0.0, 0.0],
                "observation": observation,
                "raw_action": [0.0] * 12,
            },
            {
                "clock_s": 25.03,
                "state_stamps_s": {"joint_state": 25.02},
                "effective_command": [0.0, 0.0, 0.0],
                "observation": observation,
                "raw_action": [0.0] * 12,
            },
        ]
    }
    contract = {"joints": [{"default_position": 0.0}] * 12}

    with pytest.raises(ValueError, match="commands differ"):
        build_state_transplant_manifest(
            locomotion,
            policy,
            contract,
            source_time_s=25.0,
        )
