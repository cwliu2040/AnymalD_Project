#!/usr/bin/env python3
"""Validate the complete confidence behavior profile matrix and safety overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OVERLAY = (
    PROJECT_ROOT / "configs/slam_confidence_behavior_holdout_safety_overlay.yaml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"Path escapes project root: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def validate_reports(
    overlay: dict[str, Any],
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen profile checks and distinct-env safety overlay."""
    if overlay.get("schema_version") != 1:
        raise ValueError("behavior safety overlay schema_version must be 1")
    if overlay.get("application") != "each_profile":
        raise ValueError("behavior safety overlay must apply to each profile")
    required = list(overlay["required_profiles"])
    if len(required) != len(set(required)):
        raise ValueError("behavior safety overlay contains duplicate profiles")
    threshold = float(overlay["hard_termination_env_fraction_max"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("hard-termination environment threshold must be in [0, 1]")

    profiles: dict[str, dict[str, Any]] = {}
    checkpoints: set[str] = set()
    for report in reports:
        profile_id = report["behavior_gate_profile"]["profile_id"]
        if profile_id in profiles:
            raise ValueError(f"duplicate behavior report for {profile_id}")
        gate_passed = bool(report["behavior_gate"]["passed"])
        derived = report["behavior_gate"]["derived"]
        env_count = int(derived["hard_terminated_env_count"])
        num_envs = int(report["num_envs"])
        recorded_fraction = float(derived["hard_terminated_env_fraction"])
        computed_fraction = env_count / num_envs
        if abs(recorded_fraction - computed_fraction) > 1.0e-12:
            raise ValueError(f"inconsistent hard-termination env fraction for {profile_id}")
        overlay_passed = computed_fraction <= threshold
        profiles[profile_id] = {
            "behavior_gate_passed": gate_passed,
            "hard_terminated_env_count": env_count,
            "hard_terminated_env_fraction": computed_fraction,
            "overlay_passed": overlay_passed,
            "passed": gate_passed and overlay_passed,
        }
        checkpoints.add(str(Path(report["checkpoint"]).resolve()))

    missing = sorted(set(required) - set(profiles))
    unexpected = sorted(set(profiles) - set(required))
    if missing or unexpected:
        raise ValueError(
            f"behavior report profile mismatch: missing={missing}, unexpected={unexpected}"
        )
    if len(checkpoints) != 1:
        raise ValueError("behavior reports do not reference one common checkpoint")
    ordered_profiles = {profile_id: profiles[profile_id] for profile_id in required}
    return {
        "checkpoint": checkpoints.pop(),
        "profiles": ordered_profiles,
        "passed": all(profile["passed"] for profile in ordered_profiles.values()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    overlay_path = _project_file(args.overlay)
    report_paths = [_project_file(path) for path in args.report]
    parity_path = _project_file(args.parity_report)
    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    behavior = validate_reports(overlay, reports)
    parity_checkpoint = str(Path(parity["artifacts"]["checkpoint"]["path"]).resolve())
    same_checkpoint = parity_checkpoint == behavior["checkpoint"]
    parity_passed = bool(parity["passed"]) and same_checkpoint
    summary = {
        "schema_version": 1,
        "kind": "slam_confidence_behavior_qualification",
        "overlay": {
            "path": str(overlay_path),
            "sha256": _sha256(overlay_path),
            "overlay_id": overlay["overlay_id"],
            "hard_termination_env_fraction_max": float(
                overlay["hard_termination_env_fraction_max"]
            ),
        },
        "behavior": behavior,
        "parity": {
            "path": str(parity_path),
            "sha256": _sha256(parity_path),
            "passed": parity_passed,
            "same_checkpoint": same_checkpoint,
        },
        "reports": [
            {"path": str(path), "sha256": _sha256(path)} for path in report_paths
        ],
        "passed": behavior["passed"] and parity_passed,
    }
    output_path = args.output.expanduser().resolve()
    if output_path != PROJECT_ROOT and PROJECT_ROOT not in output_path.parents:
        raise ValueError(f"Output path escapes project root: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
