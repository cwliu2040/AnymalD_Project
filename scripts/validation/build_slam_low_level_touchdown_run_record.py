#!/usr/bin/env python3
"""Reduce one block597 touchdown-residual run to a wiring-smoke record."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ENVELOPE_PATH = (ROOT / "outputs/slam_low_level_touchdown_headroom_v1_retry2/"
                 "baseline_envelope/model1450_anti_collapse_envelope.json")
sys.path.insert(0, str(ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))
from anymal_locomotion_ros2.publication_metrics_core import summarize_locomotion


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE = _module(
    "touchdown_baseline_record",
    ROOT / "scripts/validation/build_slam_low_level_baseline_run_record.py",
)


def _load(run_dir: Path, name: str) -> dict:
    return json.loads((run_dir / name).read_text(encoding="utf-8"))


def _touchdown_metrics(samples: list[dict]) -> dict[str, float]:
    times = np.asarray([float(row["time_s"]) for row in samples])
    dt = np.diff(times)
    feet = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")
    force = np.asarray([[row["feet"][foot]["normal_force_n"] for foot in feet] for row in samples])
    height = np.asarray([[row["feet"][foot]["position_w_m"][2] for foot in feet] for row in samples])
    roll = np.unwrap(np.asarray([float(row["roll_rad"]) for row in samples]))
    pitch = np.unwrap(np.asarray([float(row["pitch_rad"]) for row in samples]))
    rp_rate = np.hypot(np.diff(roll) / dt, np.diff(pitch) / dt)
    touchdown = np.argwhere((force[1:] >= 50.0) & (force[:-1] < 50.0))
    vertical, impulse, body_impulse = [], [], []
    for index, foot in touchdown:
        i = int(index) + 1
        vertical.append(abs(float((height[i, foot] - height[i - 1, foot]) / dt[i - 1])))
        end = min(i + 3, len(samples) - 1)
        impulse.append(float(np.sum(force[i:end, foot] * dt[i - 1:end - 1])))
        body_impulse.append(float(np.sum(rp_rate[i - 1:end - 1] * dt[i - 1:end - 1])))
    angular = np.asarray([row["actual_angular_velocity_body_radps"] for row in samples])
    scan_rotation = np.linalg.norm(angular, axis=1) * 0.1
    if not vertical:
        raise ValueError("run has no touchdown events")
    return {
        "touchdown_vertical_speed_abs_mps": float(np.mean(vertical)),
        "touchdown_contact_impulse_ns": float(np.mean(impulse)),
        "touchdown_roll_pitch_rate_impulse_rad": float(np.mean(body_impulse)),
        "lidar_scan_time_rotation_rad": float(np.mean(scan_rotation)),
    }


def _trace_checks(policy: dict, arm: str) -> tuple[dict[str, bool], float]:
    maximum = 0.0
    checks = {
        "requested_command_exact_across_arms": True,
        "model1450_observation_command_exact_original": True,
        "invalid_or_stale_exact_hard_stop": True,
        "residual_linf_bound": True,
        "residual_only_in_eligible_late_swing": True,
        "horizontal_foot_correction_first_order": True,
        "zero_arm_bit_exact_model1450": True,
    }
    for row in policy["records"]:
        observation = np.asarray(row["observation"], dtype=np.float32)
        received = np.asarray(row["received_command"], dtype=np.float32)
        effective = np.asarray(row["effective_command"], dtype=np.float32)
        raw = np.asarray(row["raw_action"], dtype=np.float32)
        td = row.get("touchdown_residual")
        if td is None:
            raise ValueError("touchdown experiment diagnostic is missing")
        residual = np.asarray(td["action_residual"], dtype=np.float32)
        adapted = np.asarray(td["adapted_action"], dtype=np.float32)
        maximum = max(maximum, float(np.max(np.abs(residual))))
        checks["requested_command_exact_across_arms"] &= bool(
            np.array_equal(received, effective) and not row["watchdog_timed_out"]
        )
        checks["model1450_observation_command_exact_original"] &= bool(
            np.array_equal(observation[9:12], effective)
        )
        eligible = td["mode"] == "valid_touchdown_residual"
        checks["residual_only_in_eligible_late_swing"] &= bool(
            eligible or np.array_equal(residual, np.zeros(12, dtype=np.float32))
        )
        if not td["wiring_valid"]:
            checks["invalid_or_stale_exact_hard_stop"] &= bool(
                np.array_equal(residual, np.zeros(12, dtype=np.float32))
            )
        horizontal = np.asarray(td["horizontal_first_order_correction_m"])
        checks["horizontal_foot_correction_first_order"] &= bool(
            np.max(np.abs(horizontal)) <= 1.0e-5
        )
        checks["zero_arm_bit_exact_model1450"] &= bool(
            arm != "zero" or (np.array_equal(residual, np.zeros(12, dtype=np.float32))
            and np.array_equal(adapted, raw))
        )
    checks["residual_linf_bound"] = maximum <= 0.0500001
    return checks, maximum


def _within_envelope(value, bound: dict) -> bool:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not array.size or not np.all(np.isfinite(array)):
        return False
    lower_value = float(np.quantile(array, 0.01)) if array.size > 1 else float(array[0])
    upper_value = float(np.quantile(array, 0.99)) if array.size > 1 else float(array[0])
    return lower_value >= float(bound["lower"]) and upper_value <= float(bound["upper"])


def _anti_collapse(gait: dict, profile: str, stability_passed: bool,
                   command_exact: bool) -> dict[str, bool]:
    envelope = json.loads(ENVELOPE_PATH.read_text(encoding="utf-8"))
    if envelope.get("frozen") is not True or envelope.get("passed") is not True:
        raise ValueError("anti-collapse envelope is not frozen PASS")
    bounds = envelope["profiles"][profile]
    passed = {name: _within_envelope(gait[name], bounds[name]) for name in bounds}
    shuffle = ("stride_length_m", "cadence_hz", "duty_factor",
               "contact_switch_rate_hz", "swing_duration_s")
    posture = ("body_height_m", "stance_width_m", "minimum_joint_margin_rad",
               "foot_clearance_m", "torque_rms_nm", "energy_per_progress",
               "stance_foot_slip_rms_mps")
    return {
        "stop_envelope_pass": passed["stopped_fraction"],
        "shuffle_envelope_pass": all(passed[name] for name in shuffle),
        "posture_envelope_pass": stability_passed and all(passed[name] for name in posture),
        "limiter_equivalence_pass": command_exact,
    }


def build(run_dir: Path, backend: str, profile: str, block: int, arm: str, stage: str) -> dict:
    trace = _load(run_dir, "locomotion_diagnostics.json")
    policy = _load(run_dir, "policy_diagnostics.json")
    offline = _load(run_dir, "offline_usability.json")
    stability = _load(run_dir, "stability_gate.json")
    driver = _load(run_dir, "driver.json")
    locomotion = summarize_locomotion(trace["samples"])
    gait, _ = BASELINE.gait_and_posture_metrics(trace)
    checks, maximum = _trace_checks(policy, arm)
    records = offline["false_stop"]["records"]
    eligible = [row for row in records if row.get("requested_motion") and
                row.get("usable_next_horizon") is not None and float(row.get("safe_scale", 0.0)) > 0.0]
    hazard = sum(not bool(row["usable_next_horizon"]) for row in eligible) / len(eligible) if eligible else 0.0
    angular = np.asarray([row["actual_angular_velocity_body_radps"] for row in trace["samples"]])
    command = np.asarray([row["command"] for row in trace["samples"]])
    active = (np.linalg.norm(command[:, :2], axis=1) >= 0.25) | (np.abs(command[:, 2]) >= 0.25)
    actual_linear = np.asarray([
        row["actual_linear_velocity_body_mps"] for row in trace["samples"]
    ])
    moving_linear_speed = float(np.mean(np.linalg.norm(actual_linear[active, :2], axis=1)))
    stopped_fraction = float(np.mean(
        (np.linalg.norm(actual_linear[active, :2], axis=1) < 0.10)
        & (np.abs(angular[active, 2]) < 0.10)
    ))
    normalized_progress = locomotion["normalized_progress"]
    if normalized_progress is None:
        times = np.asarray([float(row["time_s"]) for row in trace["samples"]])
        dt = np.diff(times)
        requested_yaw_distance = float(np.sum(np.abs(command[:-1, 2]) * dt))
        yaw = np.unwrap(np.asarray([float(row["yaw_rad"]) for row in trace["samples"]]))
        realized_yaw_distance = float(np.sum(np.abs(np.diff(yaw))))
        if requested_yaw_distance <= 1.0e-9:
            raise ValueError("run has neither translational nor yaw progress denominator")
        normalized_progress = realized_yaw_distance / requested_yaw_distance
    metrics = {
        **locomotion, **_touchdown_metrics(trace["samples"]),
        "moving_linear_speed_mps": moving_linear_speed,
        "realized_yaw_rate_radps": float(np.mean(np.abs(angular[active, 2]))),
        "normalized_progress": float(normalized_progress),
        "stopped_fraction": stopped_fraction,
        "valid_requested_usable_next_horizon_failure_fraction": float(hazard),
        "tracking_restricted_mean_survival_time_s": float(offline["tracking"]["survival_time_s"]),
    }
    anti = _anti_collapse(
        gait, profile, bool(stability["gate"]["passed"]),
        checks["requested_command_exact_across_arms"],
    )
    passed = bool(driver["passed"] and stability["gate"]["passed"] and all(checks.values()))
    return {
        "schema_version": 1, "kind": "slam_low_level_touchdown_headroom_run_record",
        "dataset_role": ("excluded_wiring" if stage == "wiring_smoke" else "causal_mechanism_development"),
        "identity": {"stage": stage, "backend": backend,
        "profile": profile, "block_id": block, "arm": arm}, "metrics": metrics,
        "trace": {"checks": checks, "maximum_residual_linf": maximum},
        "anti_collapse": anti, "gate": {"passed": passed, "role": "wiring_integrity_only"},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("fastlio2", "liosam"), required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--arm", choices=("zero", "touchdown_soft_low", "touchdown_soft"), required=True)
    parser.add_argument("--stage", choices=("wiring_smoke", "pilot"), required=True)
    parser.add_argument("--output-name", default="touchdown_headroom_run_record.json")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    record = build(run_dir, args.backend, args.profile, args.block, args.arm, args.stage)
    if Path(args.output_name).name != args.output_name:
        raise ValueError("output name must be a basename")
    (run_dir / args.output_name).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"identity": record["identity"], "gate": record["gate"]}, indent=2))
    return 0 if record["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
