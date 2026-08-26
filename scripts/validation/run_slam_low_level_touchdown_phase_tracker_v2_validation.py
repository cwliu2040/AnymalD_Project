#!/usr/bin/env python3
"""Plan or execute the fail-closed fresh phase-tracker v2 validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/slam_low_level_touchdown_phase_tracker_v2.yaml"
RELEASE_PATH = PROJECT_ROOT / "configs/slam_low_level_touchdown_phase_tracker_v2_release.yaml"
BASELINE_RUNNER_PATH = PROJECT_ROOT / "scripts/validation/run_slam_low_level_touchdown_baseline.py"
VALIDATOR_PATH = PROJECT_ROOT / "scripts/validation/validate_slam_low_level_touchdown_phase_tracker_v2.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE = _module("touchdown_baseline_runner", BASELINE_RUNNER_PATH)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _artifact(relative: str, expected: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"release artifact mismatch: {relative}")
    return path


def validate_release(release: dict[str, Any], config_path: Path) -> None:
    if release["config"]["sha256"] != _sha256(config_path):
        raise ValueError("phase v2 config hash mismatch")
    for section, pairs in (
        ("candidate", (("path", "sha256"),)),
        ("development_report", (("path", "sha256"),)),
        ("base_policy", (("path", "sha256"), ("metadata_path", "metadata_sha256"))),
        ("velocity_estimator", (("metadata_path", "metadata_sha256"), ("onnx_path", "onnx_sha256"))),
        ("stability_gate", (("path", "sha256"),)),
    ):
        value = release[section]
        for path_key, hash_key in pairs:
            _artifact(str(value[path_key]), str(value[hash_key]))
    candidate = json.loads((PROJECT_ROOT / release["candidate"]["path"]).read_text())
    if candidate.get("development_candidate") is not True or candidate.get("frozen") is not False:
        raise ValueError("phase v2 candidate must remain nondeployable before fresh validation")


def schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    fresh = config["data_roles"]["fresh_validation"]
    rows = [{
        "stage": "phase_tracker_v2_fresh_validation",
        "dataset_role": "phase_estimator_fresh_validation_only",
        "profile": str(profile), "block_id": int(block),
        "simulation_seed": int(block), "arm": "zero",
    } for profile in fresh["profiles"] for block in fresh["blocks"]]
    if len(rows) != int(fresh["expected_run_count"]):
        raise ValueError("fresh phase schedule count mismatch")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs/slam_low_level_touchdown_phase_tracker_v2",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--release", type=Path, default=RELEASE_PATH)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    release_path = args.release.expanduser().resolve()
    if any(not path.is_relative_to(PROJECT_ROOT) for path in (output_root, config_path, release_path)):
        raise ValueError("all paths must remain inside project")
    if not config_path.is_file() or not release_path.is_file():
        raise ValueError("phase v2 config/release is missing")
    config, release = _load(config_path), _load(release_path)
    validate_release(release, config_path)
    rows = schedule(config)
    plan = {
        "stage": "phase_tracker_v2_fresh_validation",
        "selected_count": len(rows),
        "blocks": config["data_roles"]["fresh_validation"]["blocks"],
        "profiles": config["data_roles"]["fresh_validation"]["profiles"],
        "execution_authorized": release["boundaries"]["live_execution_authorized"],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    boundaries = release["boundaries"]
    if boundaries.get("live_execution_authorized") is not True or boundaries.get("authorized_stages") != [
        "phase_tracker_v2_fresh_validation"
    ]:
        raise ValueError("fresh phase validation execution is not authorized")
    if boundaries.get("exact_zero_residual_only") is not True:
        raise ValueError("fresh phase validation must use exact-zero residual only")
    stage_root = output_root / "fresh_validation"
    if stage_root.exists():
        raise ValueError("fresh validation output already exists")
    snapshot = BASELINE.capture_execution_baseline()
    manifest = {
        "schema_version": 1, "kind": "touchdown_phase_tracker_v2_fresh_matrix",
        "config_sha256": _sha256(config_path), "release_sha256": _sha256(release_path),
        "selected_count": len(rows), "schedule": rows, "worktree_snapshot": snapshot,
    }
    BASELINE._write(stage_root / "run_manifest.json", manifest)
    results = []
    for row in rows:
        result = BASELINE.execute_cell(row, release, stage_root, args.domain_id, release_path)
        results.append(result)
        if result["stop_required"]:
            break
    collection_passed = len(results) == len(rows) and all(row["passed"] for row in results)
    validation_returncode = None
    if collection_passed:
        command = (
            sys.executable, str(VALIDATOR_PATH), "--records-root", str(stage_root),
            "--config", str(config_path),
            "--candidate", str(PROJECT_ROOT / release["candidate"]["path"]),
            "--development-report", str(PROJECT_ROOT / release["development_report"]["path"]),
            "--report", str(stage_root / "phase_tracker_v2_fresh_validation.json"),
            "--frozen-artifact", str(PROJECT_ROOT / "exported/slam_low_level_touchdown_phase_tracker_v2/frozen.json"),
        )
        validation_returncode = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
        collection_passed = validation_returncode == 0
    summary = {
        **manifest, "executed_count": len(results), "results": results,
        "collection_passed": collection_passed,
        "stopped_early": len(results) < len(rows),
        "validation_returncode": validation_returncode,
        "block_597_executed": False,
    }
    BASELINE._write(stage_root / "matrix_summary.json", summary)
    print(json.dumps({
        "selected_count": len(rows), "executed_count": len(results),
        "collection_passed": collection_passed,
        "stopped_early": summary["stopped_early"],
    }, indent=2))
    return 0 if collection_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
