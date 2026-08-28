#!/usr/bin/env python3
"""Calibrate the v2 causal scan-motion proxy from actual-backend paired runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs/slam_confidence_causal_calibration_v1.yaml"
RECORD_NAME = "touchdown_headroom_run_record_v3.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _project_path(value: str, *, file_required: bool = True) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"path escapes repository: {value}")
    if file_required and not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_source(protocol: dict[str, Any]) -> None:
    source = protocol["source"]
    for path_key, hash_key in (
        ("run_manifest", "run_manifest_sha256"),
        ("matrix_summary", "matrix_summary_sha256"),
        ("source_protocol", "source_protocol_sha256"),
        ("source_release", "source_release_sha256"),
    ):
        path = _project_path(source[path_key])
        if _sha256(path) != source[hash_key]:
            raise ValueError(f"source hash mismatch: {source[path_key]}")
    if protocol["execution"]["retrospective_analysis_authorized"] is not True:
        raise ValueError("retrospective calibration analysis is not authorized")
    for forbidden in (
        "fresh_simulation_collection_authorized",
        "ppo_training_authorized",
        "teacher_student_authorized",
        "live_deployment_authorized",
        "physical_robot_authorized",
    ):
        if protocol["execution"][forbidden] is not False:
            raise ValueError(f"forbidden execution gate is open: {forbidden}")


def scan_translation_rms(samples: list[dict[str, Any]], scan_time_s: float) -> float:
    times = np.asarray([row["time_s"] for row in samples], dtype=np.float64)
    velocity = np.asarray(
        [row["actual_linear_velocity_body_mps"] for row in samples], dtype=np.float64
    )
    command = np.asarray([row["command"] for row in samples], dtype=np.float64)
    if times.ndim != 1 or velocity.shape != (times.size, 3) or command.shape != (times.size, 3):
        raise ValueError("invalid locomotion trace layout")
    dt = np.diff(times)
    valid_dt = np.isfinite(dt) & (dt > 0.0)
    acceleration = np.zeros_like(velocity[1:])
    acceleration[valid_dt] = np.diff(velocity, axis=0)[valid_dt] / dt[valid_dt, None]
    translation = np.linalg.norm(0.5 * acceleration * scan_time_s**2, axis=1)
    active = (np.linalg.norm(command[1:, :2], axis=1) >= 0.25) | (
        np.abs(command[1:, 2]) >= 0.25
    )
    eligible = valid_dt & active & np.isfinite(translation)
    if not np.any(eligible):
        raise ValueError("trace has no active finite scan-motion samples")
    return float(np.sqrt(np.mean(np.square(translation[eligible]))))


def load_runs(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    source_root = _project_path(protocol["source"]["root"], file_required=False)
    paths = sorted(source_root.glob(f"**/{RECORD_NAME}"))
    if len(paths) != int(protocol["source"]["expected_run_count"]):
        raise ValueError(f"expected 72 records, found {len(paths)}")
    scan_time = float(protocol["motion_proxy"]["lidar_scan_time_s"])
    digest = hashlib.sha256()
    runs: list[dict[str, Any]] = []
    for record_path in paths:
        trace_path = record_path.parent / "locomotion_diagnostics.json"
        for path in (record_path, trace_path):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(bytes.fromhex(_sha256(path)))
        record = json.loads(record_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        identity, metrics = record["identity"], record["metrics"]
        checks = record["trace"]["checks"]
        integrity = bool(
            record["gate"]["passed"]
            and checks["requested_command_exact_across_arms"]
            and checks["model1450_observation_command_exact_original"]
            and not metrics["fall"]
            and not metrics["base_contact"]
        )
        runs.append(
            {
                **identity,
                "integrity": integrity,
                "translation_m": scan_translation_rms(trace["samples"], scan_time),
                "rotation_rad": float(metrics["lidar_scan_time_rotation_rad"]),
                "moving_speed_mps": float(metrics["moving_linear_speed_mps"]),
                "yaw_rate_radps": float(metrics["realized_yaw_rate_radps"]),
                "hazard": float(
                    metrics["valid_requested_usable_next_horizon_failure_fraction"]
                ),
                "survival_s": float(metrics["tracking_restricted_mean_survival_time_s"]),
            }
        )
    return runs, digest.hexdigest()


def paired_contrasts(runs: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    split = protocol["split"]
    comparator = split["comparator_arm"]
    groups: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in runs:
        key = (row["backend"], row["profile"], int(row["block_id"]))
        groups.setdefault(key, {})[row["arm"]] = row
    contrasts: list[dict[str, Any]] = []
    max_speed = float(protocol["eligibility"]["maximum_absolute_moving_linear_speed_difference_mps"])
    max_yaw = float(protocol["eligibility"]["maximum_absolute_realized_yaw_rate_difference_radps"])
    expected_arms = set(split["arms"])
    for (backend, profile, block), arms in sorted(groups.items()):
        if set(arms) != expected_arms:
            raise ValueError(f"incomplete paired group: {(backend, profile, block)}")
        zero = arms[comparator]
        for arm, row in sorted(arms.items()):
            if arm == comparator:
                continue
            speed_delta = row["moving_speed_mps"] - zero["moving_speed_mps"]
            yaw_delta = row["yaw_rate_radps"] - zero["yaw_rate_radps"]
            eligible = bool(
                row["integrity"]
                and zero["integrity"]
                and abs(speed_delta) <= max_speed
                and abs(yaw_delta) <= max_yaw
            )
            contrasts.append(
                {
                    "backend": backend,
                    "profile": profile,
                    "block_id": block,
                    "arm": arm,
                    "eligible": eligible,
                    "speed_delta_mps": speed_delta,
                    "yaw_rate_delta_radps": yaw_delta,
                    "delta_translation_squared": row["translation_m"] ** 2 - zero["translation_m"] ** 2,
                    "delta_rotation_squared": row["rotation_rad"] ** 2 - zero["rotation_rad"] ** 2,
                    "delta_hazard": row["hazard"] - zero["hazard"],
                    "delta_survival_s": row["survival_s"] - zero["survival_s"],
                }
            )
    return contrasts


def _nonnegative_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.std(x, axis=0)
    scale[scale < 1.0e-12] = 1.0
    normalized = x / scale
    candidates = [np.zeros(x.shape[1], dtype=np.float64)]
    for index in range(x.shape[1]):
        denominator = float(np.dot(normalized[:, index], normalized[:, index]))
        coefficient = max(0.0, float(np.dot(normalized[:, index], y)) / denominator) if denominator else 0.0
        value = np.zeros(x.shape[1], dtype=np.float64)
        value[index] = coefficient
        candidates.append(value)
    coefficient = np.linalg.lstsq(normalized, y, rcond=None)[0]
    if np.all(coefficient >= 0.0):
        candidates.append(coefficient)
    best = min(candidates, key=lambda value: float(np.mean(np.square(normalized @ value - y))))
    return best, scale


def analyze(protocol: dict[str, Any], contrasts: list[dict[str, Any]]) -> dict[str, Any]:
    calibration_routes = set(protocol["split"]["calibration_routes"])
    held_routes = set(protocol["split"]["held_out_routes"])
    gate = protocol["success_gate"]
    results: dict[str, Any] = {}
    for backend in protocol["source"]["actual_backends_required"]:
        train = [r for r in contrasts if r["backend"] == backend and r["profile"] in calibration_routes and r["eligible"]]
        held = [r for r in contrasts if r["backend"] == backend and r["profile"] in held_routes and r["eligible"]]
        if len(train) < int(protocol["eligibility"]["minimum_calibration_contrasts_per_backend"]):
            raise ValueError(f"insufficient calibration contrasts: {backend}")
        if len(held) < int(protocol["eligibility"]["minimum_held_out_contrasts_per_backend"]):
            raise ValueError(f"insufficient held-out contrasts: {backend}")
        keys = ("delta_translation_squared", "delta_rotation_squared")
        x_train = np.asarray([[row[key] for key in keys] for row in train], dtype=np.float64)
        y_train = np.asarray([row["delta_hazard"] for row in train], dtype=np.float64)
        coefficient, scale = _nonnegative_fit(x_train, y_train)
        x_held = np.asarray([[row[key] for key in keys] for row in held], dtype=np.float64)
        y_held = np.asarray([row["delta_hazard"] for row in held], dtype=np.float64)
        prediction = (x_held / scale) @ coefficient
        null_mse = float(np.mean(np.square(y_held)))
        model_mse = float(np.mean(np.square(prediction - y_held)))
        improvement = (null_mse - model_mse) / null_mse if null_mse > 0.0 else 0.0
        nonzero = np.abs(y_held) > 1.0e-9
        sign_accuracy = float(np.mean(np.sign(prediction[nonzero]) == np.sign(y_held[nonzero]))) if np.any(nonzero) else 0.0
        checks = {
            "held_out_mse_improvement": improvement >= float(gate["minimum_relative_held_out_mse_improvement_over_zero_effect"]),
            "nonzero_target_sign_accuracy": sign_accuracy >= float(gate["minimum_nonzero_target_sign_accuracy"]),
            "positive_motion_coefficient": int(np.sum(coefficient > 0.0)) >= int(gate["minimum_positive_motion_coefficients"]),
        }
        results[backend] = {
            "calibration_contrast_count": len(train),
            "held_out_contrast_count": len(held),
            "standardized_nonnegative_coefficients": dict(zip(keys, coefficient.astype(float), strict=True)),
            "feature_scales": dict(zip(keys, scale.astype(float), strict=True)),
            "held_out_null_mse": null_mse,
            "held_out_model_mse": model_mse,
            "held_out_relative_mse_improvement": improvement,
            "held_out_nonzero_target_sign_accuracy": sign_accuracy,
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed = all(value["passed"] for value in results.values())
    return {
        "backend_results": results,
        "passed": passed,
        "decision": "CALIBRATION_PASS" if passed else "CALIBRATION_FAIL",
        "proxy_parameters_frozen": passed,
        "ppo_may_start": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = _load_yaml(protocol_path)
    validate_source(protocol)
    runs, evidence_hash = load_runs(protocol)
    contrasts = paired_contrasts(runs, protocol)
    analysis = analyze(protocol, contrasts)
    output = (args.output or _project_path(protocol["execution"]["output"], file_required=False)).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output escapes repository")
    protocol_snapshot = output.parent / "protocol_executed.yaml"
    protocol_text = protocol_path.read_text(encoding="utf-8")
    if protocol_snapshot.exists() and protocol_snapshot.read_text(encoding="utf-8") != protocol_text:
        raise ValueError("existing executed protocol snapshot differs")
    protocol_snapshot.parent.mkdir(parents=True, exist_ok=True)
    protocol_snapshot.write_text(protocol_text, encoding="utf-8")
    report = {
        "schema_version": 1,
        "kind": "slam_confidence_causal_calibration_v1",
        "protocol_sha256": _sha256(protocol_path),
        "protocol_snapshot": str(protocol_snapshot.relative_to(ROOT)),
        "source_evidence_sha256": evidence_hash,
        "run_count": len(runs),
        "contrast_count": len(contrasts),
        "eligible_contrast_count": sum(row["eligible"] for row in contrasts),
        "split": protocol["split"],
        **analysis,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if analysis["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
