#!/usr/bin/env python3
"""Verify untreated prefix, short scale pulse, recovery, and Arm-B actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))
from anymal_locomotion_ros2.lio_benchmark_core import get_motion_profile


def validate_trace(
    diagnostics: dict, driver: dict, protocol: dict, release: dict, arm: str,
) -> dict:
    records = diagnostics.get("records", [])
    if not records:
        raise ValueError("pulse diagnostics require records")
    treatment = protocol["treatment"]
    contract = protocol["trace_contract"]
    if arm not in treatment.get("arms", {}):
        raise ValueError(f"arm is not defined by the protocol: {arm}")
    assigned_value = treatment["arms"][arm]
    assigned = np.asarray(
        assigned_value if isinstance(assigned_value, list) else [assigned_value] * 3,
        dtype=np.float32,
    )
    start = float(treatment["pulse_start_relative_to_profile_s"])
    duration = float(treatment["pulse_duration_s"])
    end = start + duration
    grace = float(contract["boundary_grace_s"])
    command_atol = float(contract["command_atol"])
    action_atol = float(contract["action_atol"])
    driver_pulse = driver.get("command_scale_pulse", {})
    driver_contract = bool(
        driver_pulse.get("enabled") is True
        and float(driver_pulse.get("start_s", -1)) == start
        and float(driver_pulse.get("duration_s", -1)) == duration
        and np.allclose(
            np.asarray(driver_pulse.get("scales_xyz", [driver_pulse.get("scale", -1)] * 3), dtype=np.float32),
            assigned,
            rtol=0.0,
            atol=0.0,
        )
        and int(driver_pulse.get("publish_count", 0)) > 0
    )
    origin = driver.get("profile_start_clock_s")
    if origin is None or not np.isfinite(float(origin)):
        raise ValueError("driver is missing finite profile start clock")
    profile = get_motion_profile(driver["profile"])
    observations = np.asarray([row["observation"] for row in records], dtype=np.float32)
    actions = np.asarray([row["raw_action"] for row in records], dtype=np.float32)
    clocks = np.asarray([row["clock_s"] for row in records], dtype=np.float64)
    elapsed = clocks - float(origin)
    received = np.asarray([row["received_command"] for row in records], dtype=np.float32)
    effective = np.asarray([row["effective_command"] for row in records], dtype=np.float32)
    expected_action = ReferenceEvaluator(
        onnx.load(str(PROJECT_ROOT / release["policy"]["path"]))
    ).run(None, {"observation": observations})[0]
    action_error = float(np.max(np.abs(actions - expected_action)))
    observation_command_error = float(np.max(np.abs(observations[:, 9:12] - effective)))
    phase_masks = {
        "untreated_prefix": (
            elapsed >= float(profile.warmup_s + profile.ramp_s) + grace
        ) & (elapsed < start - grace),
        "pulse": (elapsed >= start + grace) & (elapsed < end - grace),
        "recovered": (elapsed >= end + grace) & (elapsed < end + 0.5),
    }
    phase_reports = {}
    for phase, mask in phase_masks.items():
        indices = np.flatnonzero(mask)
        errors = []
        for index in indices:
            base = np.asarray(profile.command_at(float(elapsed[index])), dtype=np.float32)
            scales = assigned if phase == "pulse" else np.ones(3, dtype=np.float32)
            errors.append(float(np.max(np.abs(received[index] - scales * base))))
        maximum = max(errors) if errors else None
        phase_reports[phase] = {
            "record_count": int(len(indices)),
            "maximum_command_error": maximum,
            "passed": bool(indices.size and maximum is not None and maximum <= command_atol),
        }
    maximum_pre_age = float(contract["pre_pulse_sample_max_age_s"])
    uncontaminated_edge = start - grace
    pre_candidates = np.flatnonzero(
        (elapsed >= uncontaminated_edge - maximum_pre_age)
        & (elapsed < uncontaminated_edge)
    )
    if not len(pre_candidates):
        raise ValueError("no pre-pulse observation within frozen matching window")
    pre_index = int(pre_candidates[-1])
    checks = {
        "driver_pulse_contract": driver_contract,
        "untreated_prefix_exact_common_command": phase_reports["untreated_prefix"]["passed"],
        "pulse_exact_assigned_scale": phase_reports["pulse"]["passed"],
        "recovered_exact_common_command": phase_reports["recovered"]["passed"],
        "observation_uses_effective_command": observation_command_error <= command_atol,
        "arm_b_action_formula_parity": action_error <= action_atol,
        "finite_trace": bool(np.all(np.isfinite(observations)) and np.all(np.isfinite(actions))),
    }
    return {
        "schema_version": 1, "kind": "slam_speed_pulse_trace_validation",
        "arm": arm, "assigned_scales": assigned.astype(float).tolist(), "record_count": len(records),
        "passed": bool(all(checks.values())), "checks": checks,
        "phases": phase_reports, "maximum_action_error": action_error,
        "maximum_observation_command_error": observation_command_error,
        "pre_pulse": {
            "clock_s": float(clocks[pre_index]), "elapsed_s": float(elapsed[pre_index]),
            "age_to_uncontaminated_edge_s": float(uncontaminated_edge - elapsed[pre_index]),
            "observation": observations[pre_index].astype(float).tolist(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for p in (args.diagnostics, args.driver, args.protocol, args.release)]
    if any(not p.is_relative_to(PROJECT_ROOT) or not p.is_file() for p in paths):
        raise ValueError("inputs must be existing project-local files")
    report = validate_trace(
        json.loads(paths[0].read_text()), json.loads(paths[1].read_text()),
        yaml.safe_load(paths[2].read_text()), yaml.safe_load(paths[3].read_text()), args.arm,
    )
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("output must remain inside project")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
