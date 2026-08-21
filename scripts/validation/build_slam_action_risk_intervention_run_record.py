#!/usr/bin/env python3
"""Reduce one intervention live run to one causal-pilot experimental unit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))
from anymal_locomotion_ros2.publication_metrics_core import summarize_locomotion


def _load(run_dir: Path, name: str) -> dict:
    path = run_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def valid_requested_hazard(
    policy: dict, offline: dict,
) -> tuple[float | None, int, int]:
    """Return future-usability failure only while current tracking is valid."""
    valid_by_clock = {
        round(float(row["clock_s"]) * 1.0e9): bool(row["observation"][49] >= 0.5)
        for row in policy["records"]
    }
    offline_records = offline["false_stop"]["records"]
    unmatched = sum(
        int(row["policy_clock_ns"]) not in valid_by_clock for row in offline_records
    )
    eligible = [
        row for row in offline_records
        if row.get("requested_motion")
        and row.get("usable_next_horizon") is not None
        and valid_by_clock.get(int(row["policy_clock_ns"])) is True
    ]
    fraction = (
        sum(not bool(row["usable_next_horizon"]) for row in eligible) / len(eligible)
        if eligible else None
    )
    return fraction, len(eligible), unmatched


def build_record(
    run_dir: Path, *, backend: str, profile: str, block: int, arm: str,
    stage: str,
) -> dict:
    trace = _load(run_dir, "locomotion_diagnostics.json")
    policy = _load(run_dir, "policy_diagnostics.json")
    offline = _load(run_dir, "offline_usability.json")
    map_result = _load(run_dir, "map_consistency.json")
    stability = _load(run_dir, "stability_gate.json")
    intervention = _load(run_dir, "intervention_trace_validation.json")
    locomotion = summarize_locomotion(trace["samples"])
    hazard_fraction, hazard_count, unmatched_policy_clocks = valid_requested_hazard(
        policy, offline
    )
    return {
        "schema_version": 1,
        "kind": "slam_action_risk_intervention_run_record",
        "dataset_role": "excluded_causal_development",
        "experimental_unit": "scheduled_live_run",
        "identity": {
            "stage": stage, "backend": backend, "profile": profile,
            "block_id": block, "arm": arm,
        },
        "metrics": {
            **locomotion,
            "valid_requested_usable_next_horizon_failure_fraction": hazard_fraction,
            "valid_requested_usable_next_horizon_count": hazard_count,
            "offline_policy_clock_unmatched_count": unmatched_policy_clocks,
            "tracking_restricted_mean_survival_time_s": offline["tracking"]["survival_time_s"],
            "tracking_event_observed": offline["tracking"]["event_observed"],
            "tracking_valid_fraction": offline["tracking"]["tracking_valid_fraction"],
            "translation_ate_rmse_m": offline["trajectory"]["translation_ate_rmse_m"],
            "map_registration_valid": bool(map_result["outcome"]["map_registration_valid"]),
        },
        "intervention": intervention,
        "gate": {
            "passed": bool(
                intervention["passed"] and offline["gate"]["passed"]
                and map_result["gate"]["passed"]
                and unmatched_policy_clocks == 0
            ),
            "role": "data_integrity_only_policy_or_slam_failure_is_outcome",
            "frame_samples_are_independent_replicates": False,
            "stability_outcome_passed": bool(stability["gate"]["passed"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("fastlio2", "liosam"), required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--arm", choices=("smooth", "zero", "antismooth"), required=True)
    parser.add_argument("--stage", choices=("wiring_smoke", "pilot", "expanded_only_after_pilot_inconclusive"), required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("run directory must remain inside the project")
    record = build_record(
        run_dir, backend=args.backend, profile=args.profile, block=args.block,
        arm=args.arm, stage=args.stage,
    )
    (run_dir / "intervention_run_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"identity": record["identity"], "gate": record["gate"]}, indent=2))
    return 0 if record["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
