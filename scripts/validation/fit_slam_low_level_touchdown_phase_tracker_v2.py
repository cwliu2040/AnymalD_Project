#!/usr/bin/env python3
"""Fit the temporal phase-tracker v2 development candidate on retired data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/slam_low_level_touchdown_phase_tracker_v2.yaml"
V1_FITTER_PATH = PROJECT_ROOT / "scripts/validation/fit_slam_low_level_touchdown_phase_estimator.py"
ROS_SOURCE = PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = _module("touchdown_phase_v1_fit", V1_FITTER_PATH)

import sys
sys.path.insert(0, str(ROS_SOURCE))
from anymal_locomotion_ros2.touchdown_phase_tracker_core import (  # noqa: E402
    initial_touchdown_phase_tracker_state,
    update_tracker_from_probabilities,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("phase tracker config must be a mapping")
    return value


def _v1_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": {"expected_run_count": 36},
        "inputs": {
            "history_offsets_samples_newest_first": config["inputs"][
                "history_offsets_samples_newest_first"
            ]
        },
        "labels_offline_only": config["labels_offline_only"],
    }


def _ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    augmented = np.column_stack((features, np.ones(len(features))))
    penalty = np.eye(augmented.shape[1]) * alpha
    penalty[-1, -1] = 0.0
    coefficient = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ target)
    return coefficient[:-1], float(coefficient[-1])


def fit_candidate(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    features = np.concatenate([row["features"] for row in rows])
    late = np.concatenate([row["late"] for row in rows])
    swing = np.concatenate([row["swing"] for row in rows])
    progress = np.concatenate([row["progress"] for row in rows])
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (features - mean) / scale
    weights = []
    intercepts = []
    progress_weights = []
    progress_intercepts = []
    model = config["model"]
    for foot in range(4):
        target = np.zeros(len(features), dtype=np.int64)
        target[swing[:, foot]] = 1
        target[late[:, foot]] = 2
        classifier = LogisticRegression(
            C=float(model["logistic_c"]), solver=str(model["logistic_solver"]),
            max_iter=int(model["logistic_max_iterations"]), class_weight=None,
            random_state=0,
        ).fit(standardized, target)
        if not np.array_equal(classifier.classes_, [0, 1, 2]):
            raise ValueError("all three phase classes must exist for every foot")
        weights.append(classifier.coef_)
        intercepts.append(classifier.intercept_)
        weight, intercept = _ridge(
            standardized[swing[:, foot]], progress[swing[:, foot], foot],
            float(model["ridge_alpha"]),
        )
        progress_weights.append(weight)
        progress_intercepts.append(intercept)
    tracker = dict(config["tracker"])
    tracker["late_swing_progress_minimum"] = float(
        config["labels_offline_only"]["late_swing_progress_minimum"]
    )
    return {
        "schema_version": 1,
        "kind": "slam_low_level_touchdown_temporal_phase_tracker",
        "estimator_id": config["estimator_id"],
        "development_candidate": True,
        "frozen": False,
        "passed": False,
        "history_offsets_samples": config["inputs"]["history_offsets_samples_newest_first"],
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "classifier_weight": np.asarray(weights).tolist(),
        "classifier_intercept": np.asarray(intercepts).tolist(),
        "progress_weight": np.asarray(progress_weights).tolist(),
        "progress_intercept": progress_intercepts,
        "tracker": tracker,
        "runtime_inputs": ["joint_position_history", "joint_velocity_history", "previous_action_history"],
        "contact_truth_runtime_input": False,
    }


def _predict(row: dict[str, Any], artifact: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    features = row["features"]
    standardized = (
        features - np.asarray(artifact["feature_mean"])
    ) / np.asarray(artifact["feature_scale"])
    weights = np.asarray(artifact["classifier_weight"])
    intercept = np.asarray(artifact["classifier_intercept"])
    logits = np.einsum("tn,fcn->tfc", standardized, weights) + intercept[None, :, :]
    logits -= np.max(logits, axis=2, keepdims=True)
    probability = np.exp(np.clip(logits, -60.0, 0.0))
    probability /= np.sum(probability, axis=2, keepdims=True)
    progress = np.clip(
        standardized @ np.asarray(artifact["progress_weight"]).T
        + np.asarray(artifact["progress_intercept"]), 0.0, 1.0,
    )
    predicted = np.zeros_like(row["late"], dtype=bool)
    state = initial_touchdown_phase_tracker_state()
    for index in range(len(features)):
        result = update_tracker_from_probabilities(
            probability[index], progress[index], state, artifact["tracker"]
        )
        if not result.tracking_valid:
            raise ValueError("development tracker became invalid")
        predicted[index] = result.eligible_late_swing
        state = result.state
    return predicted, progress


def evaluate(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    predicted_parts = []
    actual_parts = []
    progress_parts = []
    truth_progress_parts = []
    swing_parts = []
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        predicted, progress = _predict(row, artifact)
        part = {
            "predicted": predicted, "actual": row["late"], "progress": progress,
            "truth_progress": row["progress"], "swing": row["swing"],
        }
        by_profile.setdefault(row["profile"], []).append(part)
        predicted_parts.append(predicted)
        actual_parts.append(row["late"])
        progress_parts.append(progress)
        truth_progress_parts.append(row["progress"])
        swing_parts.append(row["swing"])

    def metrics(parts: list[dict[str, Any]]) -> dict[str, Any]:
        predicted = np.concatenate([part["predicted"] for part in parts])
        actual = np.concatenate([part["actual"] for part in parts])
        progress = np.concatenate([part["progress"] for part in parts])
        truth = np.concatenate([part["truth_progress"] for part in parts])
        swing = np.concatenate([part["swing"] for part in parts])
        true_positive = int(np.count_nonzero(predicted & actual))
        positive = int(np.count_nonzero(predicted))
        actual_positive = int(np.count_nonzero(actual))
        false_positive = int(np.count_nonzero(predicted & ~actual))
        negative = int(np.count_nonzero(~actual))
        return {
            "late_swing_precision": true_positive / positive if positive else 0.0,
            "late_swing_recall": true_positive / actual_positive if actual_positive else 0.0,
            "false_trigger_fraction": false_positive / negative if negative else 1.0,
            "swing_progress_mae": float(np.mean(np.abs(progress[swing] - truth[swing]))),
            "positive_predictions_by_foot": [int(value) for value in np.sum(predicted, axis=0)],
        }

    all_parts = [{
        "predicted": value[0], "actual": value[1], "progress": value[2],
        "truth_progress": value[3], "swing": value[4],
    } for value in zip(
        predicted_parts, actual_parts, progress_parts, truth_progress_parts, swing_parts, strict=True
    )]
    return {
        "aggregate": metrics(all_parts),
        "by_profile": {name: metrics(parts) for name, parts in by_profile.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--artifact", type=Path,
        default=PROJECT_ROOT / "exported/slam_low_level_touchdown_phase_tracker_v2/candidate.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=PROJECT_ROOT / "docs/validation/slam_low_level_touchdown_phase_tracker_v2_development.json",
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    root = (PROJECT_ROOT / config["data_roles"]["development_only"]["baseline_root"]).resolve()
    rows = V1.load_rows(root, _v1_compatible_config(config))
    expected_blocks = set(config["data_roles"]["development_only"]["blocks"])
    if {int(row["block"]) for row in rows} != expected_blocks:
        raise ValueError("development block inventory mismatch")
    artifact = fit_candidate(rows, config)
    artifact.update({
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": _sha256(config_path),
        "development_blocks": sorted(expected_blocks),
        "fresh_validation_blocks": config["data_roles"]["fresh_validation"]["blocks"],
    })
    development = evaluate(rows, artifact)
    report = {
        "schema_version": 1,
        "kind": "slam_low_level_touchdown_phase_tracker_v2_development_only",
        "passed": None, "frozen": False,
        "claim_allowed": False,
        "old_v1_holdout_reused_as_validation": False,
        "metrics": development,
        "fresh_validation_required": config["data_roles"]["fresh_validation"],
        "fresh_validation_gates": config["fresh_validation_gates"],
    }
    artifact_path = args.artifact.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if any(not path.is_relative_to(PROJECT_ROOT) for path in (artifact_path, report_path)):
        raise ValueError("outputs must remain inside project")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    report.update({
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": _sha256(config_path),
        "candidate_path": str(artifact_path.relative_to(PROJECT_ROOT)),
        "candidate_sha256": _sha256(artifact_path),
    })
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "development_only": True,
        "aggregate": development["aggregate"],
        "fresh_validation_blocks": config["data_roles"]["fresh_validation"]["blocks"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
