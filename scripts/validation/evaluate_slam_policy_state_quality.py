#!/usr/bin/env python3
"""Evaluate policy-consumed SLAM body velocity against offline simulator truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "liosam_policy_state_quality.yaml"


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"document must contain a mapping: {path}")
    return value


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return vector


def _percentile(values: np.ndarray, percentile: float) -> np.ndarray:
    if not len(values):
        return np.full(3, math.nan, dtype=np.float64)
    return np.percentile(values, percentile, axis=0)


def evaluate_policy_state_quality(
    policy_diagnostics: dict[str, Any],
    locomotion_diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return a machine-readable pass/fail result for one closed-loop run."""
    failures: list[str] = []
    if config.get("schema_version") != 1:
        raise ValueError("policy-state quality config schema_version must be 1")
    runtime = config.get("runtime", {})
    if runtime.get("runtime_ground_truth_input") is not False:
        raise ValueError("runtime_ground_truth_input must be false")
    if runtime.get("offline_ground_truth_evaluator_only") is not True:
        raise ValueError("offline ground truth must be evaluator-only")

    records = policy_diagnostics.get("records")
    samples = locomotion_diagnostics.get("samples")
    if not isinstance(records, list) or not isinstance(samples, list):
        raise ValueError("diagnostics must contain records and samples lists")
    evaluation = config["evaluation"]
    gate = config["gate"]
    start_s = float(evaluation["start_time_s"])
    end_margin_s = float(evaluation["end_margin_s"])
    truth_tolerance_s = float(evaluation["nearest_truth_tolerance_s"])
    minimum_samples = int(evaluation["minimum_matched_samples"])

    truth: list[tuple[float, np.ndarray]] = []
    missing_truth_field_count = 0
    for sample in samples:
        if "actual_linear_velocity_body_mps" not in sample:
            missing_truth_field_count += 1
            continue
        try:
            truth.append(
                (
                    float(sample["time_s"]),
                    _finite_vector(
                        sample["actual_linear_velocity_body_mps"],
                        3,
                        "actual body linear velocity",
                    ),
                )
            )
        except (TypeError, ValueError):
            continue
    truth.sort(key=lambda item: item[0])
    if not truth:
        return {
            "schema_version": 1,
            "kind": "slam_policy_state_quality",
            "contract_id": config.get("contract_id"),
            "passed": False,
            "failures": [
                "locomotion diagnostics do not contain finite "
                "actual_linear_velocity_body_mps offline truth"
            ],
            "metrics": {
                "matched_sample_count": 0,
                "missing_truth_field_count": missing_truth_field_count,
            },
            "boundaries": {
                "runtime_ground_truth_used": False,
                "native_deskew_required": bool(
                    runtime.get("native_deskew_required")
                ),
            },
        }

    end_s = truth[-1][0] - end_margin_s
    truth_times = np.asarray([item[0] for item in truth], dtype=np.float64)
    observed: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    clocks: list[float] = []
    source_stamps: list[float] = []
    unmatched_count = 0
    malformed_record_count = 0
    for record in records:
        try:
            clock_s = float(record["clock_s"])
            if clock_s < start_s or clock_s > end_s:
                continue
            observation = _finite_vector(record["observation"], len(record["observation"]), "observation")
            if len(observation) < 3:
                raise ValueError("observation is shorter than three values")
            source_stamp_s = float(record["state_stamps_s"]["odometry"])
            if not math.isfinite(clock_s) or not math.isfinite(source_stamp_s):
                raise ValueError("policy state timestamps must be finite")
        except (KeyError, TypeError, ValueError):
            malformed_record_count += 1
            continue
        index = int(np.searchsorted(truth_times, clock_s))
        candidates = [
            candidate
            for candidate in (index - 1, index)
            if 0 <= candidate < len(truth)
        ]
        if not candidates:
            unmatched_count += 1
            continue
        nearest = min(candidates, key=lambda candidate: abs(truth_times[candidate] - clock_s))
        if abs(truth_times[nearest] - clock_s) > truth_tolerance_s:
            unmatched_count += 1
            continue
        clocks.append(clock_s)
        source_stamps.append(source_stamp_s)
        observed.append(observation[:3])
        actual.append(truth[nearest][1])

    if len(observed) < minimum_samples:
        failures.append(
            f"matched_sample_count={len(observed)}<{minimum_samples}"
        )
    if malformed_record_count:
        failures.append(
            f"malformed_policy_record_count={malformed_record_count}>0"
        )

    observed_array = np.asarray(observed, dtype=np.float64).reshape((-1, 3))
    actual_array = np.asarray(actual, dtype=np.float64).reshape((-1, 3))
    clocks_array = np.asarray(clocks, dtype=np.float64)
    stamps_array = np.asarray(source_stamps, dtype=np.float64)
    absolute_error = np.abs(observed_array - actual_array)
    mae = absolute_error.mean(axis=0) if len(absolute_error) else np.full(3, math.nan)
    error_p95 = _percentile(absolute_error, 95.0)
    error_max = absolute_error.max(axis=0) if len(absolute_error) else np.full(3, math.nan)
    ages = clocks_array - stamps_array
    negative_age_count = int(np.count_nonzero(ages < -1.0e-9))
    age_p95 = float(np.percentile(ages, 95.0)) if len(ages) else math.nan
    age_max = float(np.max(ages)) if len(ages) else math.nan

    stamp_deltas = np.diff(stamps_array)
    regression_count = int(np.count_nonzero(stamp_deltas < 0.0))
    unique_stamps = np.unique(stamps_array)
    if len(unique_stamps) >= 2 and unique_stamps[-1] > unique_stamps[0]:
        source_rate_hz = float(
            (len(unique_stamps) - 1) / (unique_stamps[-1] - unique_stamps[0])
        )
    else:
        source_rate_hz = 0.0

    advanced = np.concatenate(([True], stamp_deltas > 0.0)) if len(stamps_array) else np.zeros(0, dtype=bool)
    advanced_velocity = observed_array[advanced]
    advanced_stamps = stamps_array[advanced]
    velocity_deltas = np.diff(advanced_velocity, axis=0)
    time_deltas = np.diff(advanced_stamps)
    valid_deltas = time_deltas > 0.0
    acceleration = (
        np.abs(velocity_deltas[valid_deltas] / time_deltas[valid_deltas, None])
        if np.any(valid_deltas)
        else np.empty((0, 3), dtype=np.float64)
    )
    acceleration_p95 = _percentile(acceleration, 95.0)
    acceleration_max = acceleration.max(axis=0) if len(acceleration) else np.full(3, math.nan)

    def check_scalar(name: str, value: float, maximum: float) -> None:
        if not math.isfinite(value) or value > maximum:
            failures.append(f"{name}={value:.6g}>{maximum:.6g}")

    def check_vector(name: str, values: np.ndarray, maxima: Any) -> None:
        limits = _finite_vector(maxima, 3, f"{name} limits")
        for axis, value, limit in zip("xyz", values, limits, strict=True):
            check_scalar(f"{name}.{axis}", float(value), float(limit))

    age_gate = gate["state_age_s"]
    check_scalar("state_age_s.p95", age_p95, float(age_gate["p95_max"]))
    check_scalar("state_age_s.max", age_max, float(age_gate["absolute_max"]))
    if negative_age_count:
        failures.append(f"negative_state_age_count={negative_age_count}>0")
    if regression_count > int(gate["timestamp_regression_count_max"]):
        failures.append(
            f"timestamp_regression_count={regression_count}>"
            f"{int(gate['timestamp_regression_count_max'])}"
        )
    minimum_rate = float(gate["source_update_rate_hz_min"])
    if not math.isfinite(source_rate_hz) or source_rate_hz < minimum_rate:
        failures.append(
            f"source_update_rate_hz={source_rate_hz:.6g}<{minimum_rate:.6g}"
        )
    error_gate = gate["body_linear_velocity_error_mps"]
    check_vector("velocity_error_mae_mps", mae, error_gate["mean_absolute_max"])
    check_vector("velocity_error_p95_mps", error_p95, error_gate["p95_absolute_max"])
    check_vector("velocity_error_max_mps", error_max, error_gate["absolute_max"])
    acceleration_gate = gate["body_linear_acceleration_mps2"]
    check_vector(
        "observed_acceleration_p95_mps2",
        acceleration_p95,
        acceleration_gate["p95_absolute_max"],
    )
    check_vector(
        "observed_acceleration_max_mps2",
        acceleration_max,
        acceleration_gate["absolute_max"],
    )

    return {
        "schema_version": 1,
        "kind": "slam_policy_state_quality",
        "contract_id": config.get("contract_id"),
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "matched_sample_count": len(observed),
            "unmatched_policy_record_count": unmatched_count,
            "malformed_policy_record_count": malformed_record_count,
            "missing_truth_field_count": missing_truth_field_count,
            "state_age_s": {"p95": age_p95, "max": age_max},
            "negative_state_age_count": negative_age_count,
            "source_update_rate_hz": source_rate_hz,
            "timestamp_regression_count": regression_count,
            "body_linear_velocity_error_mps": {
                "mean_absolute": mae.tolist(),
                "p95_absolute": error_p95.tolist(),
                "absolute_max": error_max.tolist(),
            },
            "body_linear_acceleration_mps2": {
                "p95_absolute": acceleration_p95.tolist(),
                "absolute_max": acceleration_max.tolist(),
            },
        },
        "boundaries": {
            "runtime_ground_truth_used": False,
            "offline_ground_truth_evaluator_only": True,
            "native_deskew_required": bool(runtime["native_deskew_required"]),
            "mapping_confidence_authority_topic": runtime[
                "mapping_confidence_authority_topic"
            ],
            "policy_odometry_topic": runtime["policy_odometry_topic"],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-diagnostics", type=Path, required=True)
    parser.add_argument("--locomotion-diagnostics", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = evaluate_policy_state_quality(
        _load_mapping(args.policy_diagnostics),
        _load_mapping(args.locomotion_diagnostics),
        _load_mapping(args.config),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
