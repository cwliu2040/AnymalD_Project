#!/usr/bin/env python3
"""Fit and one-shot gate the frozen history-only touchdown phase estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/slam_low_level_touchdown_phase_estimator_v1.yaml"
FOOT_NAMES = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _confirmed_contact(raw: np.ndarray, count: int) -> np.ndarray:
    if raw.ndim != 2 or raw.shape[1] != 4 or count < 1:
        raise ValueError("contact labels require [time,4] and a positive confirmation count")
    result = np.empty_like(raw, dtype=bool)
    result[0] = raw[0]
    state = raw[0].copy()
    pending = np.zeros(4, dtype=np.int64)
    for index in range(1, len(raw)):
        changed = raw[index] != state
        pending[changed] += 1
        pending[~changed] = 0
        accept = changed & (pending >= count)
        state[accept] = raw[index, accept]
        pending[accept] = 0
        result[index] = state
    return result


def _swing_labels(contact: np.ndarray, late_minimum: float) -> tuple[np.ndarray, np.ndarray]:
    progress = np.zeros(contact.shape, dtype=np.float64)
    valid_swing = np.zeros(contact.shape, dtype=bool)
    for foot in range(4):
        transitions = np.flatnonzero(contact[1:, foot] != contact[:-1, foot]) + 1
        liftoff: int | None = None
        for transition in transitions:
            index = int(transition)
            if not contact[index, foot]:
                liftoff = index
            elif liftoff is not None and index > liftoff:
                frames = np.arange(liftoff, index)
                progress[frames, foot] = (
                    np.arange(len(frames), dtype=np.float64) / float(len(frames))
                )
                valid_swing[frames, foot] = True
                liftoff = None
    late = valid_swing & (progress >= late_minimum)
    return progress, late


def _identity(path: Path, root: Path) -> tuple[str, int]:
    relative = path.relative_to(root)
    if len(relative.parts) < 4 or not relative.parts[1].startswith("block_"):
        raise ValueError(f"unexpected baseline trace path: {path}")
    return relative.parts[0], int(relative.parts[1].removeprefix("block_"))


def load_rows(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = sorted(root.glob("*/block_*/zero/locomotion_diagnostics.json"))
    if len(paths) != int(config["source"]["expected_run_count"]):
        raise ValueError("phase estimator source run count mismatch")
    offsets = tuple(int(value) for value in config["inputs"]["history_offsets_samples_newest_first"])
    start = max(offsets) + 1
    threshold = float(config["labels_offline_only"]["contact_force_threshold_n"])
    confirmation = int(config["labels_offline_only"]["transition_confirmation_samples"])
    late_minimum = float(config["labels_offline_only"]["late_swing_progress_minimum"])
    rows = []
    for path in paths:
        profile, block = _identity(path, root)
        trace = json.loads(path.read_text(encoding="utf-8"))
        samples = trace["samples"]
        position = np.asarray([row["joint_position_rad"] for row in samples], dtype=np.float64)
        velocity = np.asarray([row["joint_velocity_radps"] for row in samples], dtype=np.float64)
        action = np.asarray([row["applied_raw_action"] for row in samples], dtype=np.float64)
        raw_contact = np.asarray([
            [float(row["feet"][foot]["normal_force_n"]) >= threshold for foot in FOOT_NAMES]
            for row in samples
        ], dtype=bool)
        if any(value.shape != (len(samples), 12) for value in (position, velocity, action)):
            raise ValueError(f"invalid joint/action trace shape: {path}")
        if not all(np.all(np.isfinite(value)) for value in (position, velocity, action)):
            raise ValueError(f"nonfinite joint/action trace: {path}")
        contact = _confirmed_contact(raw_contact, confirmation)
        progress, late = _swing_labels(contact, late_minimum)
        features = []
        for index in range(start, len(samples)):
            features.append(np.concatenate([
                np.concatenate((
                    position[index-offset], velocity[index-offset], action[index-1-offset],
                ))
                for offset in offsets
            ]))
        rows.append({
            "profile": profile, "block": block,
            "features": np.asarray(features, dtype=np.float64),
            "progress": progress[start:], "late": late[start:],
            "swing": (~contact)[start:],
        })
    return rows


def _stack(rows: list[dict[str, Any]], blocks: set[int], key: str) -> np.ndarray:
    selected = [row[key] for row in rows if int(row["block"]) in blocks]
    if not selected:
        raise ValueError(f"empty split: {sorted(blocks)}")
    return np.concatenate(selected, axis=0)


def _ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    augmented = np.column_stack((features, np.ones(len(features))))
    penalty = np.eye(augmented.shape[1]) * alpha
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ target)
    return coefficients[:-1], float(coefficients[-1])


def _scores(features: np.ndarray, artifact: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    standardized = (
        features - np.asarray(artifact["feature_mean"])
    ) / np.asarray(artifact["feature_scale"])
    logits = standardized @ np.asarray(artifact["classifier_weight"]).T + np.asarray(
        artifact["classifier_intercept"]
    )
    confidence = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    progress = np.clip(
        standardized @ np.asarray(artifact["progress_weight"]).T
        + np.asarray(artifact["progress_intercept"]), 0.0, 1.0,
    )
    return confidence, progress


def evaluate(rows: list[dict[str, Any]], blocks: set[int], artifact: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    threshold = float(config["model"]["runtime_confidence_threshold"])
    late_minimum = float(config["labels_offline_only"]["late_swing_progress_minimum"])

    def metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
        features = np.concatenate([row["features"] for row in selected])
        actual = np.concatenate([row["late"] for row in selected])
        progress_true = np.concatenate([row["progress"] for row in selected])
        swing = np.concatenate([row["swing"] for row in selected])
        confidence, progress = _scores(features, artifact)
        predicted = (confidence >= threshold) & (progress >= late_minimum)
        true_positive = int(np.count_nonzero(predicted & actual))
        predicted_positive = int(np.count_nonzero(predicted))
        actual_positive = int(np.count_nonzero(actual))
        false_positive = int(np.count_nonzero(predicted & ~actual))
        negative = int(np.count_nonzero(~actual))
        return {
            "late_swing_precision": true_positive / predicted_positive if predicted_positive else 0.0,
            "late_swing_recall": true_positive / actual_positive if actual_positive else 0.0,
            "false_trigger_fraction": false_positive / negative if negative else 1.0,
            "swing_progress_mae": float(np.mean(np.abs(progress[swing] - progress_true[swing]))),
            "positive_predictions_by_foot": [int(value) for value in np.sum(predicted, axis=0)],
            "sample_count": len(features),
        }

    selected = [row for row in rows if int(row["block"]) in blocks]
    aggregate = metrics(selected)
    by_profile = {
        profile: metrics([row for row in selected if row["profile"] == profile])
        for profile in config["split_by_complete_block"]["profiles_required"]
    }
    gates = config["gates_unchanged_after_fit"]
    failures = []
    if aggregate["late_swing_precision"] < float(gates["aggregate_minimum_late_swing_precision"]):
        failures.append("aggregate_precision")
    if aggregate["false_trigger_fraction"] > float(gates["aggregate_maximum_false_trigger_fraction"]):
        failures.append("aggregate_false_trigger")
    if aggregate["late_swing_recall"] < float(gates["aggregate_minimum_late_swing_recall"]):
        failures.append("aggregate_recall")
    if aggregate["swing_progress_mae"] > float(gates["aggregate_maximum_swing_progress_mae"]):
        failures.append("aggregate_progress_mae")
    for profile, value in by_profile.items():
        if value["late_swing_precision"] < float(gates["each_profile_minimum_late_swing_precision"]):
            failures.append(f"profile_precision:{profile}")
        if value["late_swing_recall"] < float(gates["each_profile_minimum_late_swing_recall"]):
            failures.append(f"profile_recall:{profile}")
    if gates["all_four_feet_must_have_positive_predictions"] and any(
        value == 0 for value in aggregate["positive_predictions_by_foot"]
    ):
        failures.append("missing_foot_predictions")
    return {"passed": not failures, "failures": failures, "aggregate": aggregate, "by_profile": by_profile}


def fit(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    train_blocks = set(int(value) for value in config["split_by_complete_block"]["train"])
    features = _stack(rows, train_blocks, "features")
    late = _stack(rows, train_blocks, "late")
    progress = _stack(rows, train_blocks, "progress")
    swing = _stack(rows, train_blocks, "swing")
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (features - mean) / scale
    classifier_weight = []
    classifier_intercept = []
    progress_weight = []
    progress_intercept = []
    model_config = config["model"]
    for foot in range(4):
        classifier = LogisticRegression(
            C=float(model_config["logistic_c"]),
            solver=str(model_config["logistic_solver"]),
            max_iter=int(model_config["logistic_max_iterations"]),
            class_weight=None,
            random_state=0,
        ).fit(standardized, late[:, foot].astype(np.int64))
        classifier_weight.append(classifier.coef_[0])
        classifier_intercept.append(float(classifier.intercept_[0]))
        weight, intercept = _ridge(
            standardized[swing[:, foot]], progress[swing[:, foot], foot],
            float(model_config["ridge_alpha"]),
        )
        progress_weight.append(weight)
        progress_intercept.append(intercept)
    return {
        "schema_version": 1,
        "kind": "slam_low_level_touchdown_history_phase_estimator",
        "estimator_id": config["estimator_id"],
        "joint_order": None,
        "history_offsets_samples": config["inputs"]["history_offsets_samples_newest_first"],
        "feature_terms_per_offset": config["inputs"]["terms_per_offset"],
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "classifier_weight": np.asarray(classifier_weight).tolist(),
        "classifier_intercept": classifier_intercept,
        "progress_weight": np.asarray(progress_weight).tolist(),
        "progress_intercept": progress_intercept,
        "confidence_threshold": float(model_config["runtime_confidence_threshold"]),
        "late_swing_progress_minimum": float(config["labels_offline_only"]["late_swing_progress_minimum"]),
        "runtime_inputs": ["joint_position_history", "joint_velocity_history", "previous_action_history"],
        "contact_truth_runtime_input": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--artifact", type=Path,
        default=PROJECT_ROOT / "exported/slam_low_level_touchdown_phase_estimator_v1/phase_estimator.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=PROJECT_ROOT / "docs/validation/slam_low_level_touchdown_phase_estimator_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    root = (PROJECT_ROOT / config["source"]["baseline_root"]).resolve()
    envelope_path = (PROJECT_ROOT / config["source"]["envelope_path"]).resolve()
    if not root.is_dir() or not envelope_path.is_file():
        raise ValueError("frozen baseline source is missing")
    if _sha256(envelope_path) != config["source"]["envelope_sha256"]:
        raise ValueError("frozen baseline envelope SHA-256 mismatch")
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    if envelope.get("frozen") is not True or envelope.get("passed") is not True:
        raise ValueError("baseline envelope is not frozen PASS")
    rows = load_rows(root, config)
    artifact = fit(rows, config)
    first_trace = json.loads(next(root.glob("*/block_*/zero/locomotion_diagnostics.json")).read_text())
    artifact["joint_order"] = first_trace["metadata"]["joint_order"]
    calibration_blocks = set(int(value) for value in config["split_by_complete_block"]["calibration_report_only"])
    holdout_blocks = set(int(value) for value in config["split_by_complete_block"]["final_holdout_one_shot"])
    calibration = evaluate(rows, calibration_blocks, artifact, config)
    holdout = evaluate(rows, holdout_blocks, artifact, config)
    passed = calibration["passed"] and holdout["passed"]
    artifact.update({
        "frozen": passed, "passed": passed,
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": _sha256(config_path),
        "baseline_envelope_path": str(envelope_path.relative_to(PROJECT_ROOT)),
        "baseline_envelope_sha256": _sha256(envelope_path),
        "split": config["split_by_complete_block"],
    })
    report = {
        "schema_version": 1,
        "kind": "slam_low_level_touchdown_phase_estimator_gate",
        "passed": passed, "frozen": passed,
        "calibration": calibration, "final_holdout": holdout,
        "gates": config["gates_unchanged_after_fit"],
        "artifact_path": str(args.artifact.expanduser().resolve().relative_to(PROJECT_ROOT)),
    }
    for output, value in ((args.artifact, artifact), (args.report, report)):
        resolved = output.expanduser().resolve()
        if not resolved.is_relative_to(PROJECT_ROOT):
            raise ValueError("outputs must remain inside the project")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": passed,
        "calibration_failures": calibration["failures"],
        "holdout_failures": holdout["failures"],
        "calibration": calibration["aggregate"],
        "holdout": holdout["aggregate"],
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
