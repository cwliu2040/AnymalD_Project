#!/usr/bin/env python3
"""Reduce one matched-prefix command-pulse run to one experimental unit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))
from anymal_locomotion_ros2.publication_metrics_core import summarize_locomotion


def _load(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def build_record(
    run_dir: Path, *, backend: str, profile: str, block: int, arm: str,
    stage: str, protocol: dict,
) -> dict:
    locomotion_trace = _load(run_dir, "locomotion_diagnostics.json")
    policy = _load(run_dir, "policy_diagnostics.json")
    driver = _load(run_dir, "driver.json")
    offline = _load(run_dir, "offline_usability.json")
    map_result = _load(run_dir, "map_consistency.json")
    stability = _load(run_dir, "stability_gate.json")
    pulse_trace = _load(run_dir, "speed_pulse_trace_validation.json")
    start_clock = float(driver["profile_start_clock_s"]) + float(
        protocol["treatment"]["pulse_start_relative_to_profile_s"]
    )
    end_clock = start_clock + float(protocol["treatment"]["pulse_duration_s"])
    valid_by_clock = {
        round(float(row["clock_s"]) * 1e9): bool(
            row["observation"][49] >= 0.5 and row["observation"][50] <= 1.0
        )
        for row in policy["records"]
    }
    eligible = [
        row for row in offline["false_stop"]["records"]
        if start_clock <= int(row["policy_clock_ns"]) * 1e-9 < end_clock
        and row.get("requested_motion")
        and row.get("usable_next_horizon") is not None
        and valid_by_clock.get(int(row["policy_clock_ns"])) is True
    ]
    hazard = (
        sum(not bool(row["usable_next_horizon"]) for row in eligible) / len(eligible)
        if eligible else None
    )
    pre_observation = pulse_trace["pre_pulse"]["observation"]
    pre_tracking_valid = bool(pre_observation[49] >= 0.5 and pre_observation[50] <= 1.0)
    intention_to_treat_hazard = float(hazard) if hazard is not None else 1.0
    window_samples = [
        sample for sample in locomotion_trace["samples"]
        if start_clock <= float(sample["time_s"]) < end_clock
    ]
    if len(window_samples) < 2:
        raise ValueError("pulse window requires at least two locomotion samples")
    window = summarize_locomotion(window_samples)
    full = summarize_locomotion(locomotion_trace["samples"])
    return {
        "schema_version": 1, "kind": "slam_speed_pulse_causal_run_record",
        "dataset_role": protocol.get("dataset_role", "excluded_causal_development"),
        "experimental_unit": "scheduled_live_run",
        "identity": {"stage": stage, "backend": backend, "profile": profile, "block_id": block, "arm": arm},
        "pulse_window": {"start_clock_s": start_clock, "end_clock_s": end_clock, "sample_count": len(window_samples)},
        "metrics": {
            "pulse_window_intention_to_treat_failure_fraction": intention_to_treat_hazard,
            "pulse_window_valid_requested_usable_next_horizon_failure_fraction": hazard,
            "pulse_window_valid_requested_usable_next_horizon_count": len(eligible),
            "pre_pulse_tracking_valid": pre_tracking_valid,
            "pulse_window_moving_speed_mps": window["moving_speed_mps"],
            "pulse_window_normalized_progress": window["normalized_progress"],
            "pulse_window_stance_weighted_foot_slip_rms_mps": window["stance_weighted_foot_slip_rms_mps"],
            "pulse_window_roll_pitch_rate_rms_radps": window["roll_pitch_rate_rms_radps"],
            "fall": full["fall"], "base_contact": full["base_contact"],
            "tracking_restricted_mean_survival_time_s": offline["tracking"]["survival_time_s"],
            "map_registration_valid": bool(map_result["outcome"]["map_registration_valid"]),
        },
        "pulse_trace": pulse_trace,
        "gate": {
            "passed": bool(
                pulse_trace["passed"] and offline["gate"]["passed"]
                and map_result["gate"]["passed"]
            ),
            "role": "data_integrity_only_policy_or_slam_failure_is_outcome",
            "stability_outcome_passed": bool(stability["gate"]["passed"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--backend", choices=("fastlio2", "liosam"), required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--stage", choices=("wiring_smoke", "causal_pilot"), required=True)
    args = parser.parse_args()
    run_dir, protocol_path = args.run_dir.resolve(), args.protocol.resolve()
    protocol = __import__("yaml").safe_load(protocol_path.read_text())
    if args.arm not in protocol.get("treatment", {}).get("arms", {}):
        raise ValueError(f"arm is not defined by the protocol: {args.arm}")
    record = build_record(run_dir, backend=args.backend, profile=args.profile, block=args.block, arm=args.arm, stage=args.stage, protocol=protocol)
    (run_dir / "speed_pulse_run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"identity": record["identity"], "gate": record["gate"]}, indent=2))
    return 0 if record["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
