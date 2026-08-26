#!/usr/bin/env python3
"""Reconstruct exact model1450 actions for a zero-residual baseline trace."""

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
    diagnostics: dict, release: dict, *, atol: float = 1.0e-5,
) -> dict:
    records = diagnostics.get("records", [])
    observations = np.asarray([row["observation"] for row in records], dtype=np.float32)
    recorded = np.asarray([row["raw_action"] for row in records], dtype=np.float32)
    received = np.asarray([row["received_command"] for row in records], dtype=np.float32)
    effective = np.asarray([row["effective_command"] for row in records], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 48:
        raise ValueError("baseline diagnostics require nonempty 48-D observations")
    if recorded.shape != (len(observations), 12):
        raise ValueError("baseline diagnostics require 12-D actions")
    policy_path = (PROJECT_ROOT / release["base_policy"]["path"]).resolve()
    model = onnx.load(str(policy_path))
    if len(model.graph.input) != 1:
        raise ValueError("model1450 must expose exactly one observation input")
    input_name = model.graph.input[0].name
    evaluator = ReferenceEvaluator(model)
    reconstructed = np.asarray(
        evaluator.run(None, {input_name: observations})[0], dtype=np.float32
    )
    action_error = np.abs(recorded - reconstructed)
    command_error = np.maximum(
        np.abs(received - effective), np.abs(observations[:, 9:12] - effective)
    )
    watchdog_clear = all(not bool(row.get("watchdog_timed_out")) for row in records)
    checks = {
        "model1450_action_parity": bool(float(np.max(action_error)) <= atol),
        "requested_effective_observation_command_exact": bool(
            float(np.max(command_error)) == 0.0
        ),
        "command_watchdog_never_timed_out": watchdog_clear,
        "observation_and_action_finite": bool(
            np.all(np.isfinite(observations)) and np.all(np.isfinite(recorded))
        ),
    }
    return {
        "schema_version": 1,
        "kind": "slam_low_level_model1450_zero_residual_trace_validation",
        "record_count": len(records),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "maximum_action_reconstruction_error": float(np.max(action_error)),
        "maximum_command_error": float(np.max(command_error)),
        "atol": atol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    args = parser.parse_args()
    diagnostics_path = args.diagnostics.expanduser().resolve()
    release_path = args.release.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if any(
        not path.is_relative_to(PROJECT_ROOT) or not path.is_file()
        for path in (diagnostics_path, release_path)
    ):
        raise ValueError("inputs must be existing project-local files")
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("output must remain inside the project")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    release = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    report = validate_trace(diagnostics, release, atol=args.atol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
