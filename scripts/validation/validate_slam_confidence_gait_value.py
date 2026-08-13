#!/usr/bin/env python3
"""Compare 51-D gait behavior with the model1450 confidence-limiter baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE = PROJECT_ROOT / "configs/slam_confidence_gait_value_gate.yaml"


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


def _ratio(candidate: float, baseline: float, label: str) -> float:
    if baseline <= 0.0:
        raise ValueError(f"baseline metric must be positive: {label}={baseline}")
    return candidate / baseline


def compare_reports(
    gate: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Apply the frozen gait-value criteria to one matched A/B pair."""
    if gate.get("schema_version") != 1:
        raise ValueError("gait-value gate schema_version must be 1")
    for field in ("steps", "num_envs"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"A/B {field} mismatch")
    if baseline["behavior_gate_profile"] != candidate["behavior_gate_profile"]:
        raise ValueError("A/B behavior gate profile mismatch")
    baseline_estimator = baseline.get("velocity_estimator", {}).get("metadata")
    candidate_estimator = candidate.get("velocity_estimator", {}).get("metadata")
    if not baseline_estimator or baseline_estimator != candidate_estimator:
        raise ValueError("A/B must use the same proprioceptive velocity estimator")

    required_windows = list(gate["required_windows"])
    if any(name not in baseline["windows"] for name in required_windows) or any(
        name not in candidate["windows"] for name in required_windows
    ):
        raise ValueError("A/B report is missing a required behavior window")
    metrics = list(gate["gait_metrics"])
    ratios: dict[str, dict[str, float]] = {}
    for window in required_windows:
        ratios[window] = {
            metric: _ratio(
                float(candidate["windows"][window][metric]),
                float(baseline["windows"][window][metric]),
                f"{window}.{metric}",
            )
            for metric in metrics
        }

    thresholds = gate["thresholds"]
    transition_ratios = [
        ratios[window][metric]
        for window in ("degraded_late", "invalid_settled")
        for metric in metrics
    ]
    transition_composite = sum(transition_ratios) / len(transition_ratios)
    candidate_hard_fraction = float(
        candidate["termination_summary"]["hard_terminated_env_fraction"]
    )
    checks = {
        "baseline_behavior_gate": bool(baseline["behavior_gate"]["passed"]),
        "candidate_behavior_gate": bool(candidate["behavior_gate"]["passed"]),
        "candidate_hard_termination_environments": candidate_hard_fraction
        <= float(thresholds["hard_terminated_env_fraction_max"]),
        "healthy_tracking_not_regressed": float(
            candidate["windows"]["healthy"]["linear_rmse"]
        )
        - float(baseline["windows"]["healthy"]["linear_rmse"])
        <= float(thresholds["healthy_linear_rmse_regression_max_mps"]),
        "recovery_tracking_not_regressed": float(
            candidate["windows"]["recovery_settled"]["linear_rmse"]
        )
        - float(baseline["windows"]["recovery_settled"]["linear_rmse"])
        <= float(thresholds["recovery_linear_rmse_regression_max_mps"]),
        "healthy_gait_not_regressed": max(ratios["healthy"].values())
        <= float(thresholds["healthy_each_gait_ratio_max"]),
        "recovery_gait_not_regressed": max(ratios["recovery_settled"].values())
        <= float(thresholds["recovery_each_gait_ratio_max"]),
        "degraded_invalid_no_gait_metric_regressed": max(transition_ratios)
        <= float(thresholds["degraded_invalid_each_gait_ratio_max"]),
        "degraded_invalid_gait_value_added": transition_composite
        <= float(thresholds["degraded_invalid_gait_composite_ratio_max"]),
    }
    return {
        "gate_id": gate["gate_id"],
        "ratios_candidate_over_baseline": ratios,
        "derived": {
            "degraded_invalid_gait_composite_ratio": transition_composite,
            "candidate_hard_terminated_env_fraction": candidate_hard_fraction,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    gate_path = _project_file(args.gate)
    baseline_path = _project_file(args.baseline)
    candidate_path = _project_file(args.candidate)
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    result = compare_reports(gate, baseline, candidate)
    result.update(
        {
            "schema_version": 1,
            "kind": "slam_confidence_gait_value_comparison",
            "gate": {"path": str(gate_path), "sha256": _sha256(gate_path)},
            "baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
            "candidate": {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
        }
    )
    output_path = args.output.expanduser().resolve()
    if output_path != PROJECT_ROOT and PROJECT_ROOT not in output_path.parents:
        raise ValueError(f"Output path escapes project root: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
