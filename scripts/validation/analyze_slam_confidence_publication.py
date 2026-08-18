#!/usr/bin/env python3
"""Analyze complete run-level A/B/C/D records under the frozen protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))

from anymal_locomotion_ros2.publication_statistics_core import (
    backend_interaction,
    paired_contrast,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/slam_confidence_publication_protocol.yaml")
    parser.add_argument("--allow-excluded", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int)
    return parser.parse_args()


def _records(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("**/publication_run_record.json"))
    ]


def validate_formal_completeness(
    records: list[dict[str, Any]], protocol: dict[str, Any], roles: set[str],
) -> dict[str, Any]:
    matrix = protocol["live_matrix"]
    expected = {
        (backend, profile, condition, int(block), arm)
        for backend in matrix["backends"]
        for profile in matrix["profiles"]
        for condition in matrix["perception_conditions"]
        for block in matrix["paired_block_ids"]
        for arm in matrix["policy_arms"]
    }
    observed_list = [
        (
            record["identity"]["backend"], record["identity"]["profile"],
            record["identity"]["condition"], int(record["identity"]["paired_block_id"]),
            record["identity"]["arm"],
        )
        for record in records
    ]
    observed = set(observed_list)
    checks = {
        "formal_role_only": roles == {"formal"},
        "expected_record_count": len(records) == int(matrix["expected_run_count"]),
        "unique_cell_identity": len(observed) == len(observed_list),
        "exact_frozen_schedule": observed == expected,
        "all_run_gates_passed": all(record.get("gate", {}).get("passed") for record in records),
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> int:
    args = _parse_args()
    root = args.input_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_relative_to(PROJECT_ROOT) or not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("analysis paths must remain inside the project")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    records = _records(root)
    if not records:
        raise ValueError("no publication run records found")
    roles = {record["dataset_role"] for record in records}
    if roles != {"formal"} and not args.allow_excluded:
        raise ValueError("non-formal records require --allow-excluded and remain descriptive only")
    resamples = args.bootstrap_resamples or int(protocol["statistics"]["bootstrap_resamples"])
    gradual = [record for record in records if record["identity"]["condition"] == "gradual_support_loss"]
    if not gradual:
        raise ValueError("confirmatory analysis requires gradual_support_loss records")
    contrasts = {}
    for treatment, control, contrast_id in (("C", "D", "learned_gait"), ("C", "B", "learned_total"), ("B", "A", "supervisor")):
        contrasts[contrast_id] = {
            metric: paired_contrast(
                gradual, treatment=treatment, control=control, metric=metric,
                resamples=resamples,
            )
            for metric in (
                "stance_weighted_foot_slip_rms_mps",
                "while_stable_roll_pitch_rate_rms_radps",
                "tracking_restricted_mean_survival_time_s",
                "normalized_progress",
                "normalized_progress_per_elapsed_second",
            )
        }
    binary_contrasts = {
        contrast_id: {
            metric: paired_contrast(
                gradual, treatment=treatment, control=control, metric=metric,
                resamples=resamples,
            )
            for metric in ("fall", "base_contact", "foot_slip_event", "completion")
        }
        for treatment, control, contrast_id in (
            ("C", "D", "learned_gait"), ("C", "B", "learned_total"), ("B", "A", "supervisor")
        )
    }
    interactions = {
        metric: backend_interaction(
            gradual, treatment="C", control="D", metric=metric, resamples=resamples,
        )
        for metric in (
            "stance_weighted_foot_slip_rms_mps",
            "while_stable_roll_pitch_rate_rms_radps",
            "tracking_restricted_mean_survival_time_s",
            "normalized_progress",
            "normalized_progress_per_elapsed_second",
        )
    }
    learned = contrasts["learned_gait"]
    efficiency = learned["normalized_progress"]
    efficiency_lower = efficiency[
        "mean_difference_95pct_cluster_bootstrap"
    ]["lower"]
    efficiency_margin = float(
        protocol["statistics"]["sample_size_planning"][
            "efficiency_noninferiority_normalized_progress_difference"
        ]
    )
    conclusion = {
        "mechanism_supported": all(
            record["mechanism"].get("applied_stride_attenuation_degraded_nonzero_fraction", 0.0) > 0.0
            for record in gradual if record["identity"]["arm"] == "C"
        ),
        "slip_direction_supported": learned["stance_weighted_foot_slip_rms_mps"]["mean_difference"] < 0.0,
        "roll_pitch_rate_direction_supported": learned[
            "while_stable_roll_pitch_rate_rms_radps"
        ]["mean_difference"] < 0.0,
        "survival_direction_supported": learned["tracking_restricted_mean_survival_time_s"]["mean_difference"] > 0.0,
        "efficiency_noninferiority_95pct_lower_ge_margin": (
            efficiency_lower >= efficiency_margin
        ),
    }
    completeness = validate_formal_completeness(records, protocol, roles)
    cluster_counts = [
        value["mean_difference_95pct_cluster_bootstrap"]["cluster_count"]
        for contrast in contrasts.values() for value in contrast.values()
    ]
    inference_valid = completeness["passed"] and min(cluster_counts) >= 5
    report = {
        "schema_version": 1,
        "kind": "slam_confidence_publication_analysis",
        "dataset_roles": sorted(roles),
        "formal_claim_allowed": completeness["passed"],
        "formal_completeness": completeness,
        "cluster_inference_valid": inference_valid,
        "run_record_count": len(records),
        "experimental_unit": "scheduled_live_run",
        "frame_samples_are_independent_replicates": False,
        "contrasts": contrasts,
        "binary_risk_difference_contrasts": binary_contrasts,
        "backend_interactions": interactions,
        "learned_gait_efficiency_difference": {
            **efficiency,
            "noninferiority_margin": efficiency_margin,
        },
        "evidence_rule": conclusion,
        "complete_support": inference_valid and all(conclusion.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("dataset_roles", "formal_claim_allowed", "run_record_count", "evidence_rule", "complete_support")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
