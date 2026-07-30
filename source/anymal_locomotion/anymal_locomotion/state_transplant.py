"""Build and validate deterministic locomotion state-transplant manifests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StateTransplant:
    """Policy and simulator state captured at one 50 Hz joint-state stamp."""

    source_time_s: float
    policy_clock_s: float
    base_position_w_m: tuple[float, float, float]
    base_orientation_wxyz: tuple[float, float, float, float]
    base_linear_velocity_b_mps: tuple[float, float, float]
    base_angular_velocity_b_rps: tuple[float, float, float]
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    previous_action: tuple[float, ...]
    bootstrap_action: tuple[float, ...]
    effective_command: tuple[float, float, float]
    initial_observation: tuple[float, ...]
    expected_observation: tuple[float, ...]
    actuator_lstm_state: dict[str, Any] | None
    replay_spawn: tuple[float, float, float] | None
    action_history: tuple[tuple[float, ...], ...]
    bridge_bootstrap_actions: tuple[tuple[float, ...], ...]
    action_replay_expected_observation: tuple[float, ...] | None
    action_replay_lio_sam_enabled: bool

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "StateTransplant":
        if data.get("schema_version") != 1:
            raise ValueError("state-transplant manifest must use schema_version 1")
        state = data.get("state")
        if not isinstance(state, dict):
            raise ValueError("state-transplant manifest is missing the state mapping")

        def vector(name: str, size: int) -> tuple[float, ...]:
            values = state.get(name)
            if not isinstance(values, list) or len(values) != size:
                raise ValueError(f"state.{name} must contain {size} values")
            result = tuple(float(value) for value in values)
            if not all(math.isfinite(value) for value in result):
                raise ValueError(f"state.{name} contains NaN or Inf")
            return result

        source = data.get("source")
        if not isinstance(source, dict):
            raise ValueError("state-transplant manifest is missing source metadata")
        source_time_s = float(source.get("locomotion_time_s"))
        policy_clock_s = float(source.get("policy_clock_s"))
        if not math.isfinite(source_time_s) or not math.isfinite(policy_clock_s):
            raise ValueError("state-transplant source timestamps must be finite")
        actuator_lstm_state = state.get("actuator_lstm_state")
        if actuator_lstm_state is not None and not isinstance(
            actuator_lstm_state,
            dict,
        ):
            raise ValueError("state.actuator_lstm_state must be a mapping or null")
        replay = data.get("deterministic_action_replay")
        if replay is None:
            replay_spawn = None
            action_history: tuple[tuple[float, ...], ...] = ()
            bridge_bootstrap_actions: tuple[tuple[float, ...], ...] = ()
            action_replay_expected_observation = None
            action_replay_lio_sam_enabled = False
        elif isinstance(replay, dict):
            spawn = replay.get("spawn")
            raw_history = replay.get("actions")
            if not isinstance(spawn, dict) or not isinstance(raw_history, list):
                raise ValueError(
                    "deterministic_action_replay must contain spawn and actions"
                )
            replay_spawn = tuple(
                float(spawn[name]) for name in ("x", "y", "yaw")
            )
            action_history = tuple(
                tuple(float(value) for value in action)
                for action in raw_history
            )
            bridge_bootstrap_actions = tuple(
                tuple(float(value) for value in action)
                for action in replay.get("bridge_bootstrap_actions", [])
            )
            raw_replay_expected = replay.get("expected_observation")
            action_replay_expected_observation = (
                tuple(float(value) for value in raw_replay_expected)
                if isinstance(raw_replay_expected, list)
                else None
            )
            action_replay_lio_sam_enabled = bool(
                replay.get("lio_sam_enabled", False)
            )
            if (
                not all(math.isfinite(value) for value in replay_spawn)
                or not action_history
                or any(
                    len(action) != 12
                    or not all(math.isfinite(value) for value in action)
                    for action in action_history
                )
                or any(
                    len(action) != 12
                    or not all(math.isfinite(value) for value in action)
                    for action in bridge_bootstrap_actions
                )
                or (
                    action_replay_expected_observation is not None
                    and (
                        len(action_replay_expected_observation) != 48
                        or not all(
                            math.isfinite(value)
                            for value in action_replay_expected_observation
                        )
                    )
                )
            ):
                raise ValueError(
                    "deterministic action replay contains invalid values"
                )
        else:
            raise ValueError(
                "deterministic_action_replay must be a mapping or null"
            )
        return cls(
            source_time_s=source_time_s,
            policy_clock_s=policy_clock_s,
            base_position_w_m=vector("base_position_w_m", 3),
            base_orientation_wxyz=vector("base_orientation_wxyz", 4),
            base_linear_velocity_b_mps=vector(
                "base_linear_velocity_b_mps", 3
            ),
            base_angular_velocity_b_rps=vector(
                "base_angular_velocity_b_rps", 3
            ),
            joint_positions=vector("joint_positions", 12),
            joint_velocities=vector("joint_velocities", 12),
            previous_action=vector("previous_action", 12),
            bootstrap_action=vector("bootstrap_action", 12),
            effective_command=vector("effective_command", 3),
            initial_observation=vector("initial_observation", 48),
            expected_observation=vector("expected_observation", 48),
            actuator_lstm_state=actuator_lstm_state,
            replay_spawn=replay_spawn,
            action_history=action_history,
            bridge_bootstrap_actions=bridge_bootstrap_actions,
            action_replay_expected_observation=(
                action_replay_expected_observation
            ),
            action_replay_lio_sam_enabled=(
                action_replay_lio_sam_enabled
            ),
        )


def load_state_transplant(path: str | Path) -> StateTransplant:
    """Load one project-owned JSON transplant manifest."""
    manifest_path = Path(path).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("state-transplant manifest root must be a mapping")
    return StateTransplant.from_mapping(data)


def _quaternion_wxyz_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def build_state_transplant_manifest(
    locomotion_trace: dict[str, Any],
    policy_trace: dict[str, Any],
    policy_contract: dict[str, Any],
    *,
    source_time_s: float,
    timestamp_tolerance_s: float = 0.011,
) -> dict[str, Any]:
    """Align formal simulator and policy traces through the joint-state stamp."""
    samples = locomotion_trace.get("samples")
    records = policy_trace.get("records")
    joints = policy_contract.get("joints")
    if not isinstance(samples, list) or not samples:
        raise ValueError("locomotion trace has no samples")
    if not isinstance(records, list) or not records:
        raise ValueError("policy trace has no records")
    if not isinstance(joints, list) or len(joints) != 12:
        raise ValueError("policy contract must define 12 joints")

    sample = min(samples, key=lambda item: abs(float(item["time_s"]) - source_time_s))
    sample_time_s = float(sample["time_s"])
    if abs(sample_time_s - source_time_s) > timestamp_tolerance_s:
        raise ValueError(
            f"no locomotion sample within {timestamp_tolerance_s}s of {source_time_s}"
        )

    def joint_stamp(record: dict[str, Any]) -> float:
        stamps = record.get("state_stamps_s")
        if not isinstance(stamps, dict) or "joint_state" not in stamps:
            return float("inf")
        return float(stamps["joint_state"])

    record = min(records, key=lambda item: abs(joint_stamp(item) - sample_time_s))
    aligned_joint_stamp_s = joint_stamp(record)
    if abs(aligned_joint_stamp_s - sample_time_s) > timestamp_tolerance_s:
        raise ValueError(
            "policy trace has no joint-state stamp aligned with locomotion "
            f"sample {sample_time_s}"
        )
    observation = tuple(float(value) for value in record["observation"])
    if len(observation) != 48 or not all(math.isfinite(value) for value in observation):
        raise ValueError("aligned policy observation must contain 48 finite values")
    physical_observation = tuple(
        float(value)
        for value in sample.get("native_policy_observation", observation)
    )
    if len(physical_observation) != 48 or not all(
        math.isfinite(value) for value in physical_observation
    ):
        raise ValueError(
            "aligned native policy observation must contain 48 finite values"
        )
    defaults = tuple(float(joint["default_position"]) for joint in joints)
    joint_positions = tuple(
        defaults[index] + physical_observation[12 + index]
        for index in range(12)
    )
    command = tuple(float(value) for value in record["effective_command"])
    simulator_command = tuple(float(value) for value in sample["command"])
    if max(abs(left - right) for left, right in zip(command, simulator_command)) > 1.0e-5:
        raise ValueError(
            "aligned policy and simulator commands differ: "
            f"{command} != {simulator_command}"
        )
    orientation = _quaternion_wxyz_from_rpy(
        float(sample["roll_rad"]),
        float(sample["pitch_rad"]),
        float(sample["yaw_rad"]),
    )
    policy_dt_s = float(
        locomotion_trace.get("metadata", {}).get("policy_dt_s", 0.02)
    )
    expected_stamp_s = sample_time_s + policy_dt_s
    expected_record = min(
        records,
        key=lambda item: abs(joint_stamp(item) - expected_stamp_s),
    )
    if abs(joint_stamp(expected_record) - expected_stamp_s) > timestamp_tolerance_s:
        raise ValueError(
            "policy trace has no post-bootstrap joint-state stamp near "
            f"{expected_stamp_s}"
        )
    expected_observation = tuple(
        float(value) for value in expected_record["observation"]
    )
    bootstrap_action = tuple(float(value) for value in record["raw_action"])
    if (
        len(expected_observation) != 48
        or len(bootstrap_action) != 12
        or not all(
            math.isfinite(value)
            for value in (*expected_observation, *bootstrap_action)
        )
    ):
        raise ValueError(
            "post-bootstrap observation and bootstrap action have invalid dimensions"
        )
    actuator_lstm_state = sample.get("actuator_lstm_state")
    actuator_state_available = isinstance(actuator_lstm_state, dict)
    policy_dt_s = float(
        locomotion_trace.get("metadata", {}).get("policy_dt_s", 0.02)
    )
    replay_samples = [
        item
        for item in samples
        if float(item["time_s"]) <= sample_time_s + 1.0e-9
    ]
    expected_replay_count = int(round(sample_time_s / policy_dt_s))
    replay_actions: list[list[float]] = []
    if len(replay_samples) == expected_replay_count and all(
        abs(float(item["time_s"]) - (index + 1) * policy_dt_s)
        <= timestamp_tolerance_s
        and isinstance(item.get("applied_raw_action"), list)
        and len(item["applied_raw_action"]) == 12
        for index, item in enumerate(replay_samples)
    ):
        replay_actions = [
            [float(value) for value in item["applied_raw_action"]]
            for item in replay_samples
        ]
    metadata = locomotion_trace.get("metadata", {})
    replay_spawn = metadata.get("spawn") if isinstance(metadata, dict) else None
    deterministic_replay = (
        {
            "policy_dt_s": policy_dt_s,
            "lio_sam_enabled": bool(
                metadata.get("lio_sam_enabled", False)
            ),
            "spawn": replay_spawn,
            "actions": replay_actions,
            "bridge_bootstrap_actions": [
                [float(value) for value in item["applied_raw_action"]]
                for item in samples[
                    expected_replay_count : expected_replay_count + 2
                ]
            ],
            "expected_observation": list(
                min(
                    records,
                    key=lambda item: abs(
                        joint_stamp(item)
                        - (sample_time_s + 2.0 * policy_dt_s)
                    ),
                )["observation"]
            ),
        }
        if (
            replay_actions
            and isinstance(replay_spawn, dict)
            and len(samples) >= expected_replay_count + 2
        )
        else None
    )
    return {
        "schema_version": 1,
        "source": {
            "locomotion_time_s": sample_time_s,
            "policy_clock_s": float(record["clock_s"]),
            "policy_joint_state_stamp_s": aligned_joint_stamp_s,
            "post_bootstrap_policy_clock_s": float(
                expected_record["clock_s"]
            ),
            "post_bootstrap_joint_state_stamp_s": joint_stamp(
                expected_record
            ),
        },
        "limitations": {
            "actuator_lstm_state_restored": actuator_state_available,
            "reason": (
                None
                if actuator_state_available
                else (
                    "The formal trace predates actuator checkpoint capture and "
                    "does not contain the 200 Hz recurrent hidden/cell state."
                )
            ),
        },
        "deterministic_action_replay": deterministic_replay,
        "state": {
            "base_position_w_m": [float(value) for value in sample["base_position_w_m"]],
            "base_orientation_wxyz": list(orientation),
            "base_linear_velocity_b_mps": list(physical_observation[0:3]),
            "base_angular_velocity_b_rps": list(physical_observation[3:6]),
            "joint_positions": list(joint_positions),
            "joint_velocities": list(physical_observation[24:36]),
            "previous_action": list(physical_observation[36:48]),
            "bootstrap_action": list(bootstrap_action),
            "effective_command": list(command),
            "initial_observation": list(physical_observation),
            "expected_observation": list(expected_observation),
            "actuator_lstm_state": actuator_lstm_state,
        },
    }


def build_state_transplant_manifest_from_files(
    locomotion_trace_path: str | Path,
    policy_trace_path: str | Path,
    policy_contract_path: str | Path,
    *,
    source_time_s: float,
) -> dict[str, Any]:
    """Read the three source files and return a transplant manifest."""
    locomotion = json.loads(Path(locomotion_trace_path).read_text(encoding="utf-8"))
    policy = json.loads(Path(policy_trace_path).read_text(encoding="utf-8"))
    contract = yaml.safe_load(Path(policy_contract_path).read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (locomotion, policy, contract)):
        raise ValueError("trace and contract roots must be mappings")
    return build_state_transplant_manifest(
        locomotion,
        policy,
        contract,
        source_time_s=source_time_s,
    )
