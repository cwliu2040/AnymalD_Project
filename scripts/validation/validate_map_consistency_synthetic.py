#!/usr/bin/env python3
"""Validate the publication map metric against synthetic no-split/split maps."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))

from anymal_locomotion_ros2.map_consistency_core import evaluate_map_consistency


def synthetic_maps() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal = np.column_stack(
        (np.linspace(0.0, 6.0, 601), np.zeros(601), np.zeros(601))
    )
    vertical = np.column_stack(
        (np.zeros(401), np.linspace(0.0, 4.0, 401), np.zeros(401))
    )
    diagonal = np.column_stack(
        (np.linspace(1.0, 4.0, 301), np.linspace(1.0, 2.5, 301), np.zeros(301))
    )
    reference = np.vstack((horizontal, vertical, diagonal))
    yaw = 0.08
    rotation = np.array(((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw))))
    no_split = reference.copy()
    no_split[:, :2] = no_split[:, :2] @ rotation.T + np.array((0.18, -0.12))
    split_copy = no_split[::2].copy()
    split_copy[:, 1] += 0.25
    split = np.vstack((no_split, split_copy))
    return reference, no_split, split


def validate_synthetic() -> dict:
    reference, no_split, split = synthetic_maps()
    healthy = evaluate_map_consistency(no_split, reference)
    broken = evaluate_map_consistency(split, reference)
    checks = {
        "no_split_reference_p95_le_0_03m": healthy.reference_distance_p95_m <= 0.03,
        "no_split_duplicate_fraction_le_0_01": healthy.duplicate_surface_fraction <= 0.01,
        "split_duplicate_fraction_ge_0_10": broken.duplicate_surface_fraction >= 0.10,
        "split_p95_ge_0_15m": broken.reference_distance_p95_m >= 0.15,
    }
    return {
        "schema_version": 1,
        "kind": "map_consistency_synthetic_gate",
        "no_split": asdict(healthy),
        "split": asdict(broken),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "docs/validation/slam_confidence_map_consistency_synthetic.json",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("output must remain inside project")
    result = validate_synthetic()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
