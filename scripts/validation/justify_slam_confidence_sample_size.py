#!/usr/bin/env python3
"""Choose formal paired-block count from the frozen disjoint pilot matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))

from anymal_locomotion_ros2.publication_statistics_core import paired_values
from anymal_locomotion_ros2.sample_size_core import first_precision_qualified_block_count


def validate_pilot_records(records: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    matrix = protocol["sample_size_pilot_matrix"]
    expected = {
        (backend, profile, condition, int(block), arm)
        for backend in matrix["backends"] for profile in matrix["profiles"]
        for condition in matrix["perception_conditions"]
        for block in matrix["paired_block_ids"] for arm in matrix["policy_arms"]
    }
    observed_list = [
        (
            record["identity"]["backend"], record["identity"]["profile"],
            record["identity"]["condition"], int(record["identity"]["paired_block_id"]),
            record["identity"]["arm"],
        )
        for record in records
    ]
    roles = {record.get("dataset_role") for record in records}
    pilot_blocks = {value[3] for value in observed_list}
    formal_blocks = set(protocol["live_matrix"]["paired_block_ids"])
    prior_pilot_blocks = set(matrix.get("prior_pilot_block_ids_must_be_disjoint", []))
    checks = {
        "excluded_pilot_role_only": roles == {"excluded_pilot"},
        "expected_record_count": len(records) == int(matrix["expected_run_count"]),
        "unique_identities": len(observed_list) == len(set(observed_list)),
        "exact_frozen_pilot_schedule": set(observed_list) == expected,
        "all_run_gates_passed": all(record.get("gate", {}).get("passed") for record in records),
        "pilot_blocks_disjoint_from_formal": not (pilot_blocks & formal_blocks),
        "pilot_blocks_disjoint_from_prior_pilot": not (
            pilot_blocks & prior_pilot_blocks
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _cluster_effects(records: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    pairs = paired_values(records, treatment="C", control="D", metric=metric)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for pair in pairs:
        identity = pair["pair"]
        effect = pair["difference"]
        grouped[(identity["profile"], int(identity["paired_block_id"]))].append(effect)
    by_profile: dict[str, list[float]] = defaultdict(list)
    for (profile, _block), effects in sorted(grouped.items()):
        by_profile[profile].append(float(sum(effects) / len(effects)))
    return dict(by_profile)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/slam_confidence_publication_protocol.yaml")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root, output = args.input_root.expanduser().resolve(), args.output.expanduser().resolve()
    if not root.is_relative_to(PROJECT_ROOT) or not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("sample-size paths must remain inside the project")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    records = [json.loads(path.read_text()) for path in sorted(root.glob("**/publication_run_record.json"))]
    completeness = validate_pilot_records(records, protocol)
    if not completeness["passed"]:
        report = {
            "schema_version": 1, "kind": "slam_confidence_sample_size_justification",
            "pilot_completeness": completeness, "passed": False,
            "formal_collection_may_start": False,
            "reason": "disjoint excluded pilot matrix is incomplete or invalid",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1
    planning = protocol["statistics"]["sample_size_planning"]
    candidates = planning["candidate_paired_block_counts"]
    resamples, seed = int(planning["simulation_resamples"]), int(planning["seed"])
    targets = planning["precision_halfwidth_targets"]
    metrics = {}
    for index, metric in enumerate(
        ("stance_weighted_foot_slip_rms_mps", "roll_pitch_rate_rms_radps", "tracking_restricted_mean_survival_time_s")
    ):
        metrics[metric] = first_precision_qualified_block_count(
            _cluster_effects(records, metric), candidates=candidates,
            halfwidth_target=float(targets[metric]), resamples=resamples, seed=seed + index * 100,
        )
    metrics["normalized_progress_difference"] = first_precision_qualified_block_count(
        _cluster_effects(records, "normalized_progress"), candidates=candidates,
        halfwidth_target=float(targets["normalized_progress_difference"]),
        resamples=resamples, seed=seed + 400,
        lower_bound=float(
            planning["efficiency_noninferiority_normalized_progress_difference"]
        ),
    )
    for index, metric in enumerate(
        ("fall", "base_contact", "foot_slip_event", "completion")
    ):
        metrics[f"{metric}_risk_difference"] = first_precision_qualified_block_count(
            _cluster_effects(records, metric), candidates=candidates,
            halfwidth_target=float(targets["binary_risk_difference"]),
            resamples=resamples, seed=seed + 500 + index * 100,
        )
    selected_values = [value["selected_paired_block_count"] for value in metrics.values()]
    passed = all(value is not None for value in selected_values)
    selected = max(selected_values) if passed else None
    challenge_frozen = bool(
        protocol.get("challenge_calibration_matrix", {}).get(
            "formal_condition_frozen", False
        )
    )
    report = {
        "schema_version": 1, "kind": "slam_confidence_sample_size_justification",
        "pilot_completeness": completeness, "planning": planning,
        "endpoint_precision": metrics, "selected_formal_paired_block_count": selected,
        "protocol_current_formal_paired_block_count": len(protocol["live_matrix"]["paired_block_ids"]),
        "protocol_revision_required": bool(selected and selected != len(protocol["live_matrix"]["paired_block_ids"])),
        "challenge_condition_frozen": challenge_frozen,
        "formal_collection_may_start": passed and challenge_frozen,
        "passed": passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("selected_formal_paired_block_count", "protocol_revision_required", "formal_collection_may_start", "passed")}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
