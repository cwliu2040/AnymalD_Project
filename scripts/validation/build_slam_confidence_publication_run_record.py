#!/usr/bin/env python3
"""Normalize one accepted publication cell into one run-level record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))

from anymal_locomotion_ros2.publication_metrics_core import summarize_locomotion, summarize_mechanism


def _load(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("fastlio2", "liosam"), required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--arm", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument(
        "--dataset-role",
        choices=("formal", "excluded_smoke", "excluded_pilot", "excluded_calibration"),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("run directory must remain inside the project")
    trace = _load(run_dir / "locomotion_diagnostics.json")
    offline = _load(run_dir / "offline_usability.json")
    map_result = _load(run_dir / "map_consistency.json")
    stability = _load(run_dir / "stability_gate.json")
    mechanism = _load(run_dir / "mechanism_sidecar.json", required=False)
    locomotion = summarize_locomotion(trace["samples"])
    false_stop = offline["false_stop"]
    record = {
        "schema_version": 1,
        "kind": "slam_confidence_publication_run_record",
        "dataset_role": args.dataset_role,
        "experimental_unit": "scheduled_live_run",
        "identity": {
            "backend": args.backend, "profile": args.profile, "condition": args.condition,
            "paired_block_id": args.block, "arm": args.arm,
        },
        "metrics": {
            **locomotion,
            "confidence_stop_known_label_count": false_stop["confidence_stop_known_label_count"],
            "false_stop_count": false_stop["false_stop_count"],
            "false_stop_fraction": false_stop["false_stop_fraction"],
            "tracking_restricted_mean_survival_time_s": offline["tracking"]["survival_time_s"],
            "tracking_event_observed": offline["tracking"]["event_observed"],
            "tracking_valid_fraction": offline["tracking"]["tracking_valid_fraction"],
            "translation_ate_rmse_m": offline["trajectory"]["translation_ate_rmse_m"],
            "map_registration_valid": bool(
                map_result.get("outcome", {}).get("map_registration_valid", True)
            ),
            "map_reference_distance_p50_m": map_result["metrics"]["reference_distance_p50_m"],
            "map_reference_distance_p95_m": map_result["metrics"]["reference_distance_p95_m"],
            "map_duplicate_surface_fraction": map_result["metrics"]["duplicate_surface_fraction"],
        },
        "mechanism": summarize_mechanism(mechanism.get("records", [])),
        "outcomes": {
            "stability_gate_passed": bool(stability["gate"]["passed"]),
            "stability_failures": list(stability["gate"].get("failures", [])),
            "map_registration_valid": bool(
                map_result.get("outcome", {}).get("map_registration_valid", True)
            ),
            "map_registration_failure_reason": map_result.get("outcome", {}).get(
                "map_registration_failure_reason"
            ),
        },
        "gate": {
            "passed": bool(offline["gate"]["passed"] and map_result["gate"]["passed"]),
            "role": "data_integrity_only",
            "policy_or_slam_failure_retained_as_outcome": True,
            "frame_samples_are_independent_replicates": False,
        },
    }
    output = run_dir / "publication_run_record.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"identity": record["identity"], "gate": record["gate"]}, indent=2))
    return 0 if record["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
