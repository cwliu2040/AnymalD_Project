#!/usr/bin/env python3
"""Validate curated five-profile model48+estimator15 recovery regression evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = PROJECT_ROOT / "configs/slam_confidence_sim_release_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_regression(release: dict[str, Any]) -> dict[str, Any]:
    evidence = release["recovery_regression_evidence"]
    checkpoint = PROJECT_ROOT / release["arms"]["C"]["checkpoint_path"]
    estimator = PROJECT_ROOT / release["common_runtime"]["velocity_estimator_metadata_path"]
    expected_checkpoint_sha = release["arms"]["C"]["checkpoint_sha256"]
    expected_estimator_sha = release["common_runtime"]["velocity_estimator_metadata_sha256"]
    profile_results = {}
    all_seeds = []
    total_environments = total_hard_terminated = 0
    for name, configured in evidence["profiles"].items():
        path = PROJECT_ROOT / configured["path"]
        document = json.loads(path.read_text(encoding="utf-8"))
        source_checkpoint = Path(document["checkpoint"])
        estimator_path = PROJECT_ROOT / document["velocity_estimator"]["metadata"]
        checks = {
            "report_sha256": _sha256(path) == configured["sha256"],
            "behavior_gate_passed": document["behavior_gate"]["passed"] is True,
            "all_behavior_checks_passed": all(document["behavior_gate"]["checks"].values()),
            "num_envs": int(document["num_envs"]) == int(evidence["num_envs_per_profile"]),
            "steps": int(document["steps"]) == int(evidence["steps_per_profile"]),
            "profile_id": document["behavior_gate_profile"]["profile_id"] == configured["profile_id"],
            "seed": int(document["behavior_gate_profile"]["seed"]) == int(configured["seed"]),
            "checkpoint_filename": source_checkpoint.name == checkpoint.name,
            "checkpoint_sha256": _sha256(checkpoint) == expected_checkpoint_sha,
            "estimator_metadata_path": estimator_path.resolve() == estimator.resolve(),
            "estimator_metadata_sha256": _sha256(estimator) == expected_estimator_sha,
            "ground_truth_offline_only": document["velocity_estimator"]["ground_truth_role"] == "offline_evaluator_only",
            "hard_termination_fraction": float(document["termination_summary"]["hard_terminated_env_fraction"])
            <= float(evidence["maximum_distinct_hard_terminated_env_fraction"]),
        }
        hard_count = int(document["termination_summary"]["hard_terminated_env_count"])
        total_environments += int(document["num_envs"])
        total_hard_terminated += hard_count
        all_seeds.append(int(configured["seed"]))
        profile_results[name] = {
            "checks": checks,
            "hard_terminated_env_count": hard_count,
            "hard_terminated_env_fraction": float(document["termination_summary"]["hard_terminated_env_fraction"]),
            "passed": all(checks.values()),
        }
    aggregate_checks = {
        "five_profiles": len(profile_results) == 5,
        "five_distinct_seeds": len(set(all_seeds)) == 5,
        "all_profiles_passed": all(value["passed"] for value in profile_results.values()),
        "tracked_checkpoint_path": checkpoint.is_file() and "checkpoints" in checkpoint.parts,
    }
    return {
        "schema_version": 1,
        "kind": "model48_estimator15_recovery_regression_gate",
        "profiles": profile_results,
        "aggregate": {
            "checks": aggregate_checks,
            "total_environment_count": total_environments,
            "total_distinct_hard_terminated_env_count": total_hard_terminated,
            "pooled_distinct_hard_terminated_env_fraction": total_hard_terminated / total_environments,
        },
        "passed": all(aggregate_checks.values()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "docs/validation/slam_confidence_model48_estimator15_regression_gate.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    release_path, output = args.release.resolve(), args.output.resolve()
    if not release_path.is_relative_to(PROJECT_ROOT) or not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("regression paths must remain inside the project")
    release = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    report = validate_regression(release)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"aggregate": report["aggregate"], "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
