#!/usr/bin/env python3
"""Reconstruct and verify a fixed-scale policy trace against frozen Arm B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_trace(diagnostics: dict, release: dict, arm: str, *, atol: float = 1e-5) -> dict:
    config = release["arms"][arm]
    scale = float(config["assigned_scale"])
    records = diagnostics.get("records", [])
    observations = np.asarray([row["observation"] for row in records], dtype=np.float32)
    recorded = np.asarray([row["raw_action"] for row in records], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 51:
        raise ValueError("speed-scale diagnostics require nonempty 51-D observations")
    if recorded.shape != (len(observations), 12):
        raise ValueError("speed-scale diagnostics require 12-D actions")
    base_path = (PROJECT_ROOT / release["base_policy"]["path"]).resolve()
    if _sha256(base_path) != release["base_policy"]["sha256"]:
        raise ValueError("base policy SHA-256 mismatch")
    transformed = observations.copy()
    tracking = (observations[:, 49:50] >= 0.5) & (observations[:, 50:51] <= 1.0)
    effective_scale = np.where(tracking, scale, 0.0).astype(np.float32)
    transformed[:, 9:12] = observations[:, 9:12] * effective_scale
    expected = ReferenceEvaluator(onnx.load(str(base_path))).run(
        None, {"observation": transformed}
    )[0]
    error = np.abs(recorded - expected)
    maximum_error = float(np.max(error))
    valid = tracking[:, 0]
    invalid = ~valid
    valid_command_error = np.abs(
        transformed[valid, 9:12] - observations[valid, 9:12] * scale
    )
    invalid_command = transformed[invalid, 9:12]
    checks = {
        "formula_parity": bool(maximum_error <= atol),
        "finite_observation_and_action": bool(
            np.all(np.isfinite(observations)) and np.all(np.isfinite(recorded))
        ),
        "valid_command_exact_assigned_scale": bool(
            not len(valid_command_error) or float(np.max(valid_command_error)) == 0.0
        ),
        "invalid_stale_command_exact_zero": bool(
            not len(invalid_command) or np.array_equal(invalid_command, np.zeros_like(invalid_command))
        ),
        "assigned_scale_matches_arm": bool(0.0 < scale <= 1.0),
    }
    return {
        "schema_version": 1,
        "kind": "slam_speed_scale_trace_validation",
        "arm": arm,
        "assigned_command_scale": scale,
        "realized_effective_command_scale": scale,
        "realized_scale_basis": "framewise_reconstruction_against_hash_locked_arm_B",
        "record_count": len(records),
        "tracking_valid_record_count": int(np.sum(valid)),
        "invalid_or_stale_record_count": int(np.sum(invalid)),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "maximum_formula_error": maximum_error,
        "atol": atol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--arm", choices=("scale_100", "scale_075", "scale_050", "scale_025"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    paths = [value.expanduser().resolve() for value in (args.diagnostics, args.release)]
    if any(not path.is_relative_to(PROJECT_ROOT) or not path.is_file() for path in paths):
        raise ValueError("inputs must be existing project-local files")
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("output must remain inside the project")
    report = validate_trace(
        json.loads(paths[0].read_text(encoding="utf-8")),
        yaml.safe_load(paths[1].read_text(encoding="utf-8")),
        args.arm, atol=args.atol,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
