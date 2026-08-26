#!/usr/bin/env python3
"""Reduce one exact-zero model1450 run to a baseline-envelope record."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOOT_NAMES = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")
CONTACT_THRESHOLD_N = 50.0


def _finite_array(values: Any, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    if shape_tail and (
        value.ndim < len(shape_tail) or value.shape[-len(shape_tail):] != shape_tail
    ):
        raise ValueError(f"{name} must end in shape {shape_tail}, received {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("RMS requires nonempty finite values")
    return float(np.sqrt(np.mean(np.square(array))))


def gait_and_posture_metrics(trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    samples = trace.get("samples", [])
    if len(samples) < 3:
        raise ValueError("baseline trace requires at least three samples")
    times = _finite_array([sample["time_s"] for sample in samples], (), "time")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("baseline timestamps must increase strictly")
    active = np.asarray([
        math.hypot(float(sample["command"][0]), float(sample["command"][1])) >= 0.25
        or abs(float(sample["command"][2])) >= 0.25
        for sample in samples
    ], dtype=bool)
    if int(np.count_nonzero(active)) < 10:
        raise ValueError("baseline trace has insufficient requested-motion samples")

    contact = np.asarray([
        [float(sample["feet"][name]["normal_force_n"]) >= CONTACT_THRESHOLD_N for name in FOOT_NAMES]
        for sample in samples
    ], dtype=bool)
    foot_world = _finite_array(
        [[sample["feet"][name]["position_w_m"] for name in FOOT_NAMES] for sample in samples],
        (4, 3), "foot positions world",
    )
    foot_body = _finite_array(
        [[sample["feet"][name]["position_b_m"] for name in FOOT_NAMES] for sample in samples],
        (4, 3), "foot positions body",
    )
    foot_tangential_speed = _finite_array(
        [[sample["feet"][name]["tangential_speed_mps"] for name in FOOT_NAMES] for sample in samples],
        (4,), "foot tangential speed",
    )
    joint_position = _finite_array(
        [sample["joint_position_rad"] for sample in samples], (12,), "joint position"
    )
    joint_velocity = _finite_array(
        [sample["joint_velocity_radps"] for sample in samples], (12,), "joint velocity"
    )
    torque = _finite_array(
        [sample["applied_joint_torque_nm"] for sample in samples], (12,), "joint torque"
    )
    limits = _finite_array(
        trace.get("metadata", {}).get("joint_position_limits_rad"),
        (12, 2), "joint limits",
    )
    if limits.shape != (12, 2) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("joint limits must contain 12 increasing lower/upper pairs")

    stride_lengths: list[float] = []
    swing_durations: list[float] = []
    foot_clearances: list[float] = []
    touchdown_count = 0
    switch_count = 0
    for foot in range(4):
        transitions = np.flatnonzero(contact[1:, foot] != contact[:-1, foot]) + 1
        switch_count += int(len(transitions))
        touchdown = [int(index) for index in transitions if contact[index, foot]]
        touchdown_count += len(touchdown)
        for before, after in zip(touchdown, touchdown[1:], strict=False):
            displacement = foot_world[after, foot, :2] - foot_world[before, foot, :2]
            stride_lengths.append(float(np.linalg.norm(displacement)))
        lift_off: int | None = None
        for index in transitions:
            index = int(index)
            if not contact[index, foot]:
                lift_off = index
            elif lift_off is not None:
                swing_durations.append(float(times[index] - times[lift_off]))
                swing_slice = slice(lift_off, index + 1)
                reference = np.min(foot_world[swing_slice, :, 2], axis=1)
                clearance = foot_world[swing_slice, foot, 2] - reference
                foot_clearances.append(float(np.max(clearance)))
                lift_off = None
    if not stride_lengths or not swing_durations or not foot_clearances:
        raise ValueError("baseline trace lacks complete stride/swing events")

    active_duration = float(np.sum(np.diff(times, prepend=times[0])[active]))
    if active_duration <= 0.0:
        raise ValueError("active baseline duration must be positive")
    duty_factor = float(np.mean(contact[active]))
    cadence = float(touchdown_count / (4.0 * active_duration))
    contact_switch_rate = float(switch_count / active_duration)
    stance_width_values = []
    for sample_index in np.flatnonzero(active):
        for left, right in ((0, 1), (2, 3)):
            if contact[sample_index, left] and contact[sample_index, right]:
                stance_width_values.append(abs(
                    float(foot_body[sample_index, left, 1] - foot_body[sample_index, right, 1])
                ))
    if not stance_width_values:
        raise ValueError("baseline trace has no paired-stance width samples")

    lower_margin = joint_position - limits[:, 0]
    upper_margin = limits[:, 1] - joint_position
    minimum_joint_margin = float(np.min(np.minimum(lower_margin, upper_margin)[active]))
    dt = np.diff(times, prepend=times[0])
    power = np.sum(np.abs(torque * joint_velocity), axis=1)
    energy_j = float(np.sum(power[active] * dt[active]))
    base_position = _finite_array(
        [sample["base_position_w_m"] for sample in samples], (3,), "base position"
    )
    path_length = float(np.sum(np.linalg.norm(np.diff(base_position[:, :2], axis=0), axis=1)))
    yaw = np.unwrap(np.asarray([float(sample["yaw_rad"]) for sample in samples]))
    yaw_distance = float(np.sum(np.abs(np.diff(yaw))))
    motion_progress = path_length + 0.25 * yaw_distance
    if motion_progress <= 1.0e-6:
        raise ValueError("baseline motion progress must be positive")
    stance_speeds = foot_tangential_speed[active & np.any(contact, axis=1)][
        contact[active & np.any(contact, axis=1)]
    ]
    actual = _finite_array(
        [sample["actual_velocity"] for sample in samples], (3,), "actual velocity"
    )
    stopped = active & (np.linalg.norm(actual[:, :2], axis=1) < 0.10) & (np.abs(actual[:, 2]) < 0.10)

    metrics: dict[str, Any] = {
        "stride_length_m": stride_lengths,
        "cadence_hz": cadence,
        "duty_factor": duty_factor,
        "contact_switch_rate_hz": contact_switch_rate,
        "swing_duration_s": swing_durations,
        "body_height_m": float(np.mean([
            float(sample["base_height_m"]) for sample, enabled in zip(samples, active, strict=True)
            if enabled
        ])),
        "stance_width_m": float(np.mean(stance_width_values)),
        "minimum_joint_margin_rad": minimum_joint_margin,
        "foot_clearance_m": foot_clearances,
        "torque_rms_nm": _rms(torque[active]),
        "energy_per_progress": energy_j / motion_progress,
        "stance_foot_slip_rms_mps": _rms(stance_speeds),
        "stopped_fraction": float(np.mean(stopped[active])),
    }
    sample_units = {
        name: ("complete_stride" if isinstance(value, list) else "complete_run")
        for name, value in metrics.items()
    }
    return metrics, sample_units


def baseline_trace_checks(policy: dict[str, Any]) -> dict[str, bool]:
    records = policy.get("records", [])
    if not records:
        raise ValueError("policy diagnostics must contain records")
    command_exact = True
    zero_model = True
    for record in records:
        observation = np.asarray(record.get("observation"), dtype=np.float32)
        received = np.asarray(record.get("received_command"), dtype=np.float32)
        effective = np.asarray(record.get("effective_command"), dtype=np.float32)
        action = np.asarray(record.get("raw_action"), dtype=np.float32)
        if observation.shape != (48,) or received.shape != (3,) or effective.shape != (3,):
            raise ValueError("baseline policy diagnostics require 48-D observations and 3-D commands")
        if action.shape != (12,) or not np.all(np.isfinite(action)):
            raise ValueError("baseline policy diagnostics require finite 12-D actions")
        command_exact &= bool(
            np.array_equal(received, effective)
            and np.array_equal(observation[9:12], effective)
            and not bool(record.get("watchdog_timed_out"))
        )
        zero_model &= "touchdown_residual" not in record
    return {
        "requested_command_exact_original": bool(command_exact),
        "zero_residual_exact_model1450": bool(zero_model),
        "frame_samples_used_as_independent_replicates": False,
    }


def build_record(
    run_dir: Path, *, profile: str, runtime_profile: str, block: int,
) -> dict[str, Any]:
    trace = json.loads((run_dir / "locomotion_diagnostics.json").read_text(encoding="utf-8"))
    policy = json.loads((run_dir / "policy_diagnostics.json").read_text(encoding="utf-8"))
    driver = json.loads((run_dir / "driver.json").read_text(encoding="utf-8"))
    parity = json.loads((run_dir / "baseline_trace_validation.json").read_text(encoding="utf-8"))
    stability = json.loads((run_dir / "stability_gate.json").read_text(encoding="utf-8"))
    metrics, units = gait_and_posture_metrics(trace)
    checks = baseline_trace_checks(policy)
    checks["zero_residual_exact_model1450"] &= bool(parity.get("passed"))
    checks["frozen_physical_safety_gate_passed"] = bool(
        stability.get("gate", {}).get("passed")
    )
    passed = bool(
        driver.get("passed")
        and checks["requested_command_exact_original"]
        and checks["zero_residual_exact_model1450"]
        and checks["frozen_physical_safety_gate_passed"]
        and checks["frame_samples_used_as_independent_replicates"] is False
    )
    return {
        "schema_version": 1,
        "kind": "slam_low_level_baseline_envelope_run_record",
        "dataset_role": "anti_collapse_calibration_only",
        "identity": {
            "stage": "baseline_envelope", "profile": profile,
            "runtime_profile": runtime_profile, "block_id": block, "arm": "zero",
        },
        "sample_units": units,
        "metrics": metrics,
        "trace": {
            "checks": checks,
            "parity": parity,
            "stability_gate": stability.get("gate"),
        },
        "gate": {
            "passed": passed,
            "role": "baseline_integrity_only_no_slam_effect_claim",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--block", type=int, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("run directory must remain inside the project")
    record = build_record(
        run_dir, profile=args.profile, runtime_profile=args.runtime_profile, block=args.block
    )
    output = run_dir / "baseline_envelope_run_record.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"identity": record["identity"], "gate": record["gate"]}, indent=2))
    return 0 if record["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
