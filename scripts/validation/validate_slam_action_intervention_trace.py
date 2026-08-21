#!/usr/bin/env python3
"""Verify recorded intervention actions against the frozen Arm-B formula."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_trace(
    diagnostics: dict, release: dict, arm: str, *, atol: float = 1.0e-5,
) -> dict:
    config = release["arms"][arm]
    records = diagnostics.get("records", [])
    observations = np.asarray([row["observation"] for row in records], dtype=np.float32)
    recorded = np.asarray([row["raw_action"] for row in records], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 51:
        raise ValueError("intervention diagnostics require nonempty 51-D observations")
    if recorded.shape != (len(observations), 12):
        raise ValueError("intervention diagnostics require 12-D actions")
    base_path = PROJECT_ROOT / release["base_policy"]["path"]
    evaluator = ReferenceEvaluator(onnx.load(str(base_path)))
    baseline = evaluator.run(None, {"observation": observations})[0]
    alpha = float(config["alpha"])
    limit = float(release["raw_action_linf_limit"])
    residual = np.clip(alpha * (observations[:, 36:48] - baseline), -limit, limit)
    residual *= np.clip(observations[:, 49:50], 0.0, 1.0)
    expected = baseline + residual
    error = np.abs(recorded - expected)
    invalid = observations[:, 49] < 0.5
    invalid_error = np.abs(recorded[invalid] - baseline[invalid])
    realized = recorded - baseline
    maximum_error = float(np.max(error))
    maximum_residual = float(np.max(np.abs(realized)))
    valid_realized = realized[~invalid]
    nonzero_fraction = float(np.mean(np.any(np.abs(valid_realized) > atol, axis=1))) if len(valid_realized) else 0.0
    checks = {
        "formula_parity": bool(maximum_error <= atol),
        "residual_linf_bound": bool(
            maximum_residual <= limit + np.finfo(np.float32).eps
        ),
        "invalid_exact_arm_B": bool(
            not len(invalid_error) or float(np.max(invalid_error)) <= atol
        ),
        "zero_exact_arm_B": bool(
            arm != "zero" or float(np.max(np.abs(realized))) <= atol
        ),
        "nonzero_arm_realized": bool(arm == "zero" or nonzero_fraction > 0.0),
    }
    return {
        "schema_version": 1,
        "kind": "slam_action_intervention_trace_validation",
        "arm": arm,
        "alpha": alpha,
        "record_count": len(records),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "maximum_formula_error": maximum_error,
        "maximum_realized_residual": maximum_residual,
        "mean_realized_residual_l2": float(np.mean(np.linalg.norm(realized, axis=1))),
        "valid_nonzero_residual_fraction": nonzero_fraction,
        "atol": atol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--arm", choices=("smooth", "zero", "antismooth"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    args = parser.parse_args()
    paths = [value.expanduser().resolve() for value in (args.diagnostics, args.release)]
    if any(not path.is_relative_to(PROJECT_ROOT) or not path.is_file() for path in paths):
        raise ValueError("inputs must be existing project-local files")
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("output must remain inside the project")
    diagnostics = json.loads(paths[0].read_text(encoding="utf-8"))
    release = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    report = validate_trace(diagnostics, release, args.arm, atol=args.atol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
