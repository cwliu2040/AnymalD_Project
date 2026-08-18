#!/usr/bin/env python3
"""Backfill offline/map/run records for already accepted publication cells."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/validation"))

from run_slam_confidence_publication_matrix import (
    _load_yaml,
    stability_artifact_data_valid,
    validate_artifacts,
    validate_raw_bag,
)


DEFAULT_RELEASE = PROJECT_ROOT / "configs/slam_confidence_sim_release_v1.yaml"


def infer_identity(run_dir: Path) -> dict[str, Any]:
    parts = run_dir.parts
    backend_indices = [index for index, value in enumerate(parts) if value in {"fastlio2", "liosam"}]
    if not backend_indices:
        raise ValueError(f"cannot infer backend from {run_dir}")
    index = backend_indices[-1]
    try:
        block = parts[index + 3]
        arm = parts[index + 4]
        if not block.startswith("block_") or not arm.startswith("arm_"):
            raise ValueError
        return {
            "backend": parts[index], "profile": parts[index + 1], "condition": parts[index + 2],
            "block": int(block.removeprefix("block_")), "arm": arm.removeprefix("arm_"),
        }
    except (IndexError, ValueError) as exc:
        raise ValueError(f"run path does not follow publication layout: {run_dir}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--dataset-role",
        choices=("formal", "excluded_smoke", "excluded_pilot", "excluded_calibration"),
        required=True,
    )
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    release_path = args.release.expanduser().resolve()
    if not release_path.is_relative_to(PROJECT_ROOT) or not release_path.is_file():
        raise ValueError("release manifest must be a project-local file")
    release = _load_yaml(release_path)
    artifacts = validate_artifacts(release)
    environment = os.environ.copy()
    project_source = str(PROJECT_ROOT / "source/anymal_locomotion")
    existing_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        f"{project_source}:{existing_python_path}"
        if existing_python_path else project_source
    )
    estimator_environment = environment.copy()
    estimator_environment["PYTHONPATH"] = (
        f"{PROJECT_ROOT / 'deployment/python_vendor'}:{environment['PYTHONPATH']}"
    )
    run_dirs: list[Path] = []
    for value in args.input_root:
        root = value.expanduser().resolve()
        if not root.is_relative_to(PROJECT_ROOT):
            raise ValueError("input roots must remain inside the project")
        run_dirs.extend(sorted(path.parent.parent for path in root.glob("**/raw_bag/metadata.yaml")))
    identities: set[tuple[Any, ...]] = set()
    results = []
    for run_dir in run_dirs:
        identity = infer_identity(run_dir)
        key = tuple(identity[field] for field in ("backend", "profile", "condition", "block", "arm"))
        if key in identities:
            raise ValueError(f"duplicate cell identity across input roots: {key}")
        identities.add(key)
        commands = (
            ((
                sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_velocity_estimator_replay.py"),
                "--bag", str(run_dir / "raw_bag"),
                "--estimator-metadata", artifacts["common"]["velocity_estimator_metadata_path"],
                "--policy-metadata", artifacts[identity["arm"]]["metadata_path"],
                "--sync-tolerance-s", artifacts["common"]["velocity_estimator_sync_tolerance_s"],
                "--output", str(run_dir / "velocity_estimator_replay.json"),
            ), estimator_environment),
            ((
                sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_slam_confidence_publication_run.py"),
                "--bag", str(run_dir / "raw_bag"), "--policy-diagnostics", str(run_dir / "policy_diagnostics.json"),
                "--arm", identity["arm"], "--output", str(run_dir / "offline_usability.json"),
            ), environment),
            ((
                sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_slam_map_consistency_run.py"),
                "--bag", str(run_dir / "raw_bag"), "--output", str(run_dir / "map_consistency.json"),
            ), environment),
            ((
                sys.executable, str(PROJECT_ROOT / "scripts/validation/build_slam_confidence_publication_run_record.py"),
                "--run-dir", str(run_dir), "--backend", identity["backend"],
                "--profile", identity["profile"], "--condition", identity["condition"],
                "--block", str(identity["block"]), "--arm", identity["arm"],
                "--dataset-role", args.dataset_role,
            ), environment),
        )
        returncodes = [
            subprocess.run(command, cwd=PROJECT_ROOT, env=command_environment, check=False).returncode
            for command, command_environment in commands
        ]
        original = json.loads((run_dir / "cell.json").read_text(encoding="utf-8"))
        stability = json.loads((run_dir / "stability_gate.json").read_text(encoding="utf-8"))
        record = json.loads((run_dir / "publication_run_record.json").read_text(encoding="utf-8"))
        collection_failures = []
        if not original.get("driver_passed") or not original.get("diagnostics_passed"):
            collection_failures.append("live_driver_or_diagnostics")
        if not stability_artifact_data_valid(stability):
            collection_failures.append("stability_artifact_data_integrity")
        if not validate_raw_bag(run_dir / "raw_bag").get("passed"):
            collection_failures.append("raw_bag_data_integrity")
        if any(code != 0 for code in returncodes):
            collection_failures.append("derived_artifact_generation")
        if not record.get("gate", {}).get("passed"):
            collection_failures.append("publication_record_data_integrity")
        backfill = {
            "schema_version": 1,
            "kind": "slam_confidence_publication_backfill_disposition",
            "identity": identity,
            "source_cell_preserved": True,
            "returncodes": returncodes,
            "collection_valid": not collection_failures,
            "collection_failures": collection_failures,
            "policy_or_slam_failure_retained_as_outcome": bool(
                not stability.get("gate", {}).get("passed")
                or not record.get("outcomes", {}).get("map_registration_valid", False)
            ),
        }
        (run_dir / "publication_backfill.json").write_text(
            json.dumps(backfill, indent=2) + "\n", encoding="utf-8"
        )
        results.append({**backfill, "passed": backfill["collection_valid"]})
    report = {
        "schema_version": 1,
        "kind": "slam_confidence_publication_backfill_summary",
        "cell_count": len(results),
        "passed_cell_count": sum(value["passed"] for value in results),
        "failed_cell_count": sum(not value["passed"] for value in results),
        "source_attempt_manifests_preserved": True,
        "cells": results,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if not output.is_relative_to(PROJECT_ROOT):
            raise ValueError("output must remain inside the project")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if results and all(value["passed"] for value in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
