#!/usr/bin/env python3
"""Evaluate one frozen J1-J0 or J2-J1 matched-motion block offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from anymal_locomotion.joint_training_contract import (
    evaluate_body_lidar_mechanism,
    evaluate_directional_metrics,
    evaluate_matched_motion,
    evaluate_paired_noninferiority,
    event_aligned_body_lidar_summary,
)


def _array(record: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(record[name], dtype=np.float64)


def evaluate_block(
    candidate: dict[str, Any],
    comparator: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Reduce paired run records into the exact go/no-go block fields."""
    comparison = f"{candidate['arm']}-{comparator['arm']}"
    if comparison not in {"J1-J0", "J2-J1"}:
        raise ValueError(f"unsupported joint-training comparison: {comparison}")
    identity_fields = ("backend", "profile", "block")
    identity_passed = all(candidate[name] == comparator[name] for name in identity_fields)
    candidate_times = _array(candidate, "times_s")
    comparator_times = _array(comparator, "times_s")
    timestamps_passed = np.array_equal(candidate_times, comparator_times)
    command_integrity = np.array_equal(
        _array(candidate, "requested_commands"),
        _array(comparator, "requested_commands"),
    )
    integrity_passed = bool(identity_passed and timestamps_passed and command_integrity)

    matched_cfg = protocol["future_evaluation"]["matched_motion"]
    matched = evaluate_matched_motion(
        candidate_times,
        _array(candidate, "requested_commands"),
        _array(comparator, "requested_commands"),
        _array(candidate, "body_linear_velocity_mps"),
        _array(comparator, "body_linear_velocity_mps"),
        _array(candidate, "body_angular_velocity_radps")[:, 2],
        _array(comparator, "body_angular_velocity_radps")[:, 2],
        maximum_abs_moving_linear_speed_difference_mps=matched_cfg[
            "maximum_abs_moving_linear_speed_difference_mps"
        ],
        maximum_abs_realized_yaw_rate_difference_radps=matched_cfg[
            "maximum_abs_realized_yaw_rate_difference_radps"
        ],
        minimum_linear_progress_ratio=matched_cfg["minimum_linear_progress_ratio"],
        minimum_yaw_progress_ratio=matched_cfg["minimum_yaw_progress_ratio"],
        maximum_stopped_fraction_excess=matched_cfg["maximum_stopped_fraction_excess"],
        minimum_comparator_requested_progress_fraction=matched_cfg[
            "minimum_comparator_requested_progress_fraction"
        ],
        maximum_candidate_stopped_fraction=matched_cfg[
            "maximum_candidate_stopped_fraction"
        ],
    )

    # Use comparator transitions for both arms so the physical window is paired;
    # avoiding a confidence drop is then captured by the separate SLAM gate.
    event_reference = _array(comparator, "localization_state")
    summary_args = {
        "times_s": candidate_times,
        "evaluation_localization_state": event_reference,
        "requested_commands": _array(candidate, "requested_commands"),
        "post_event_window_s": protocol["future_evaluation"]["body_lidar_event_window_s"],
        "lidar_scan_time_s": protocol["future_evaluation"]["lidar_scan_time_s"],
    }
    candidate_body = event_aligned_body_lidar_summary(
        body_linear_velocity_mps=_array(candidate, "body_linear_velocity_mps"),
        body_angular_velocity_radps=_array(candidate, "body_angular_velocity_radps"),
        **summary_args,
    )
    comparator_body = event_aligned_body_lidar_summary(
        body_linear_velocity_mps=_array(comparator, "body_linear_velocity_mps"),
        body_angular_velocity_radps=_array(comparator, "body_angular_velocity_radps"),
        **summary_args,
    )
    direction_cfg = protocol["future_evaluation"]["directional_gate"]
    body_lidar = evaluate_body_lidar_mechanism(
        candidate_body,
        comparator_body,
        minimum_improvement_fraction=direction_cfg["minimum_improvement_fraction"],
        maximum_regression_fraction=direction_cfg["maximum_regression_fraction"],
    )
    anti_collapse = evaluate_paired_noninferiority(
        candidate["gait_safety_metrics"],
        comparator["gait_safety_metrics"],
        protocol["future_evaluation"]["paired_noninferiority_metrics"],
    )
    slam = evaluate_directional_metrics(
        candidate["slam_metrics"],
        comparator["slam_metrics"],
        protocol["future_evaluation"]["slam_metric_directions"],
        minimum_improvement_fraction=direction_cfg["minimum_improvement_fraction"],
        maximum_regression_fraction=direction_cfg["maximum_regression_fraction"],
        required_improvements=direction_cfg["slam_required_improvements"],
    )
    candidate_safety = int(candidate.get("safety_event_count", 0))
    comparator_safety = int(comparator.get("safety_event_count", 0))
    safety_event_excess = candidate_safety > comparator_safety
    return {
        "schema_version": 1,
        "comparison": comparison,
        "backend": candidate["backend"],
        "profile": candidate["profile"],
        "block": candidate["block"],
        "integrity_passed": integrity_passed,
        "matched_motion_passed": bool(matched["passed"]),
        "body_lidar_passed": bool(body_lidar["passed"]),
        "slam_direction_passed": bool(slam["passed"]),
        "anti_collapse_passed": bool(anti_collapse["passed"]),
        "safety_event_excess": safety_event_excess,
        "details": {
            "matched_motion": matched,
            "candidate_body_lidar": candidate_body,
            "comparator_body_lidar": comparator_body,
            "body_lidar_gate": body_lidar,
            "anti_collapse": anti_collapse,
            "slam_direction": slam,
            "candidate_safety_event_count": candidate_safety,
            "comparator_safety_event_count": comparator_safety,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/slam_confidence_joint_training_v1.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    comparator = json.loads(args.comparator.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    report = evaluate_block(candidate, comparator, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
