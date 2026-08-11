#!/usr/bin/env python3
"""Fit and gate one backend's native-deskew SLAM confidence pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS_PACKAGE = PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"
sys.path.insert(0, str(ROS_PACKAGE))

from anymal_locomotion_ros2.slam_confidence_calibration_core import (  # noqa: E402
    CalibrationGates,
    CausalFeatureTransform,
    FEATURE_SCHEMAS,
    FEATURE_TRANSFORM_CONFIG,
    SUPPORT_FORECAST_GUARDS,
    apply_support_forecast_guard,
    calibration_fingerprint,
    event_metrics,
    fit_isotonic,
    fit_logistic,
    frame_metrics,
    isotonic_predict,
    logistic_predict,
    stable_capture_split,
)
from anymal_locomotion_ros2.slam_confidence_estimator import (  # noqa: E402
    ARTIFACT_PROVENANCE_SCHEMA_VERSION,
    RUNTIME_PROVENANCE_EXPECTATIONS,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(value: str, *, must_exist: bool) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path escapes project root: {path}")
    if must_exist and not path.is_file():
        raise ValueError(f"file does not exist: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=sorted(FEATURE_SCHEMAS))
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifact")
    parser.add_argument("--split-manifest")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="fit and report through threshold-validation without reading holdout rows",
    )
    args = parser.parse_args()
    inputs = [_path(value, must_exist=True) for value in args.input]
    report_path = _path(args.report, must_exist=False)
    captures = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    failures = []
    groups = []
    rows = []
    for path, capture in zip(inputs, captures, strict=True):
        if capture.get("backend_id") != args.backend:
            failures.append(f"backend mismatch: {path}")
        if capture.get("deskew_mode") != "native":
            failures.append(f"non-native deskew capture rejected: {path}")
        if capture.get("odometry_outage_semantics") != "source_advance_gap":
            failures.append(f"outdated outage label semantics: {path}")
        group = str(capture.get("capture_group", ""))
        if not group:
            failures.append(f"missing capture_group: {path}")
            continue
        groups.append(group)
        transform = CausalFeatureTransform(args.backend)
        capture_rows = sorted(
            capture.get("rows", []), key=lambda row: int(row["evaluation_stamp_ns"])
        )
        for row in capture_rows:
            if row.get("usable_next_0_5s") is None:
                continue
            vector = transform.transform(row)
            enriched = {
                **row,
                "capture_group": group,
                "capture_instance": str(path.relative_to(PROJECT_ROOT)),
                "feature_vector": vector,
            }
            rows.append(enriched)
    split_manifest_path = None
    if args.split_manifest:
        split_manifest_path = _path(args.split_manifest, must_exist=True)
        split_manifest = yaml.safe_load(
            split_manifest_path.read_text(encoding="utf-8")
        )
        registered = {
            str(capture["group"]): str(capture["role"])
            for capture in split_manifest["captures"]
        }
        allowed_roles = {
            "train", "calibration", "threshold_validation", "final_holdout"
        }
        if any(registered.get(group) not in allowed_roles for group in groups):
            raise ValueError("split manifest does not assign every capture group")
        split = {group: registered[group] for group in sorted(set(groups))}
    else:
        split = stable_capture_split(groups)
    valid_rows = [row for row in rows if row["feature_vector"] is not None]
    abrupt_row_count = sum(bool(row.get("abrupt_hard_fault")) for row in valid_rows)
    model_rows = [row for row in valid_rows if not row.get("abrupt_hard_fault")]
    source_rows = [row for row in rows if row.get("source_stamp_ns") is not None]
    complete_source_rows = [
        row for row in source_rows if row["feature_vector"] is not None
    ]
    coverage = (
        len(complete_source_rows) / len(source_rows) if source_rows else 0.0
    )
    gates = CalibrationGates()
    if len(set(groups)) < gates.minimum_capture_groups:
        failures.append("minimum_capture_groups")
    if coverage < gates.source_signal_coverage_min:
        failures.append("source_signal_coverage")

    model = isotonic = None
    role_metrics = {}
    split_inventory = {}
    if model_rows:
        for row in model_rows:
            row["split"] = split[row["capture_group"]]
        inventory_roles = [
            "train",
            "calibration",
            "threshold_validation",
        ]
        if not args.development_only:
            inventory_roles.append("final_holdout")
        for role in inventory_roles:
            selected = [row for row in model_rows if row["split"] == role]
            labels = np.asarray(
                [int(row["usable_next_0_5s"]) for row in selected],
                dtype=int,
            )
            inventory_events = event_metrics(
                selected,
                np.ones(len(selected), dtype=float),
                gates.low_threshold,
            )
            split_inventory[role] = {
                "sample_count": len(selected),
                "capture_groups": sorted(
                    {row["capture_group"] for row in selected}
                ),
                "failure_prevalence": (
                    float(np.mean(labels == 0)) if len(labels) else None
                ),
                "gradual_event_count": inventory_events[
                    "gradual_event_count"
                ],
            }
        train = [row for row in model_rows if row["split"] == "train"]
        calibrate = [row for row in model_rows if row["split"] == "calibration"]
        threshold_validation = [
            row
            for row in model_rows
            if row["split"] == "threshold_validation"
        ]
        holdout = [row for row in model_rows if row["split"] == "final_holdout"]
        try:
            train_x = np.asarray([row["feature_vector"] for row in train], dtype=float)
            train_y = np.asarray([int(row["usable_next_0_5s"]) for row in train], dtype=float)
            model = fit_logistic(train_x, train_y)
            calibration_x = np.asarray([row["feature_vector"] for row in calibrate], dtype=float)
            calibration_y = np.asarray(
                [int(row["usable_next_0_5s"]) for row in calibrate],
                dtype=float,
            )
            isotonic = fit_isotonic(logistic_predict(model, calibration_x), calibration_y)
            evaluation_roles = [
                ("train", train),
                ("calibration", calibrate),
                ("threshold_validation", threshold_validation),
            ]
            if not args.development_only:
                evaluation_roles.append(("final_holdout", holdout))
            for role, selected in evaluation_roles:
                x = np.asarray([row["feature_vector"] for row in selected], dtype=float)
                y = np.asarray([int(row["usable_next_0_5s"]) for row in selected], dtype=int)
                confidence = apply_support_forecast_guard(
                    args.backend,
                    x,
                    isotonic_predict(isotonic, logistic_predict(model, x)),
                )
                role_metrics[role] = {
                    **frame_metrics(
                        y,
                        confidence,
                        clusters=[
                            str(row["capture_instance"])
                            for row in selected
                        ],
                    ),
                    **event_metrics(selected, confidence, gates.low_threshold),
                    "capture_groups": sorted({row["capture_group"] for row in selected}),
                    "failure_prevalence": float(np.mean(y == 0)) if len(y) else None,
                }
        except ValueError as error:
            failures.append(f"fit_or_split_class_coverage: {error}")
    else:
        failures.append("no_valid_rows")

    event_inventory_rows = [
        row
        for row in model_rows
        if not args.development_only or split[row["capture_group"]] != "final_holdout"
    ]
    total_event_inventory = event_metrics(
        event_inventory_rows,
        np.ones(len(event_inventory_rows), dtype=float),
        gates.low_threshold,
    )
    total_gradual_count = total_event_inventory[
        "independent_gradual_capture_count"
    ]
    if (
        not args.development_only
        and total_gradual_count < gates.minimum_independent_gradual_events
    ):
        failures.append("minimum_independent_gradual_events")
    evaluation_role = (
        "threshold_validation" if args.development_only else "final_holdout"
    )
    holdout_metrics = role_metrics.get(evaluation_role, {})
    for key, threshold, comparison in (
        ("auroc", gates.frame_auroc_min, "min"),
        ("low_confidence_event_recall", gates.gradual_event_degrade_recall_min, "min"),
        ("median_lead_time_s", gates.gradual_event_median_lead_time_s_min, "min"),
        ("healthy_false_low_time_fraction", gates.healthy_false_low_time_fraction_max, "max"),
    ):
        value = holdout_metrics.get(key)
        numerical_tolerance = 1.0e-6 if key == "median_lead_time_s" else 0.0
        if (
            value is None
            or (
                comparison == "min"
                and value + numerical_tolerance < threshold
            )
            or (comparison == "max" and value > threshold)
        ):
            failures.append(f"{evaluation_role}_{key}")
    if holdout_metrics.get("support_hard_gate_event_count", 0):
        for key, threshold in (
            ("support_hard_gate_advance_recall", 0.80),
            ("support_hard_gate_median_lead_time_s", 0.20),
        ):
            value = holdout_metrics.get(key)
            if value is None or value + 1.0e-6 < threshold:
                failures.append(f"{evaluation_role}_{key}")

    failures = sorted(set(failures))
    report = {
        "schema_version": 1,
        "backend_id": args.backend,
        "status": (
            "development_passed"
            if args.development_only and not failures
            else "passed"
            if not failures
            else "pilot_blocked"
        ),
        "runtime_activation_authorized": not failures and not args.development_only,
        "deskew_mode": "native",
        "input_files": [str(path.relative_to(PROJECT_ROOT)) for path in inputs],
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in inputs
        },
        "implementation_sha256": {
            "calibration_cli": _sha256(Path(__file__).resolve()),
            "calibration_core": _sha256(
                ROS_PACKAGE
                / "anymal_locomotion_ros2"
                / "slam_confidence_calibration_core.py"
            ),
            "contract_config": _sha256(
                PROJECT_ROOT / "configs/slam_confidence_contract.yaml"
            ),
            "calibration_config": _sha256(
                PROJECT_ROOT / "configs/slam_confidence_calibration.yaml"
            ),
        },
        "split_manifest": (
            str(split_manifest_path.relative_to(PROJECT_ROOT))
            if split_manifest_path
            else None
        ),
        "split_manifest_sha256": (
            _sha256(split_manifest_path) if split_manifest_path else None
        ),
        "capture_group_split": split,
        "row_count": len(rows),
        "feature_complete_row_count": len(valid_rows),
        "abrupt_hard_fault_row_count_excluded": abrupt_row_count,
        "gradual_model_row_count": len(model_rows),
        "source_signal_coverage": coverage,
        "feature_schema": list(FEATURE_SCHEMAS[args.backend]),
        "feature_transform": FEATURE_TRANSFORM_CONFIG,
        "support_forecast_guard": SUPPORT_FORECAST_GUARDS[args.backend],
        "metrics": role_metrics,
        "split_inventory": split_inventory,
        "total_independent_gradual_events": total_gradual_count,
        "gate_failures": failures,
    }
    report["report_fingerprint_sha256"] = calibration_fingerprint(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not failures and not args.development_only and args.artifact:
        artifact_path = _path(args.artifact, must_exist=False)
        artifact = {
            "schema_version": 1,
            "artifact_provenance_schema_version": (
                ARTIFACT_PROVENANCE_SCHEMA_VERSION
            ),
            "backend_id": args.backend,
            "calibration_id": f"native-v1-{report['report_fingerprint_sha256'][:12]}",
            "deskew_mode": "native",
            "prediction_horizon_s": 0.5,
            "feature_schema": list(FEATURE_SCHEMAS[args.backend]),
            "feature_transform": FEATURE_TRANSFORM_CONFIG,
            "support_forecast_guard": SUPPORT_FORECAST_GUARDS[args.backend],
            "estimator_family": (
                "standardized_logistic_plus_isotonic_with_causal_support_guard"
            ),
            "logistic": model,
            "isotonic": isotonic,
            "thresholds": {
                "degrade_below": 0.45,
                "invalidate_below": 0.25,
                "recover_at_or_above": 0.55,
            },
            "provenance": {
                "report": str(report_path.relative_to(PROJECT_ROOT)),
                "report_fingerprint_sha256": report[
                    "report_fingerprint_sha256"
                ],
                "implementation_sha256": report["implementation_sha256"],
                "input_sha256": report["input_sha256"],
                "runtime_contract": RUNTIME_PROVENANCE_EXPECTATIONS[
                    args.backend
                ],
            },
        }
        artifact["artifact_fingerprint_sha256"] = calibration_fingerprint(artifact)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
