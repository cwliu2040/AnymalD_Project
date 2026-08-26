#!/usr/bin/env python3
"""Fit and blockwise-evaluate an offline component-conditioned SLAM risk model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/slam_component_risk_model_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "exported/slam_component_risk_model_v1/model_report.json"
ARMS = ("control", "uniform_075", "preserve_yaw", "preserve_translation")
STATE_FEATURE_NAMES = (
    "slam_confidence_probability",
    "slam_tracking_valid",
    "slam_normalized_age",
    "requested_command_linear_x",
    "requested_command_linear_y",
    "requested_command_yaw",
    "base_linear_velocity_x",
    "base_linear_velocity_y",
    "base_angular_velocity_z",
    "joint_velocity_rms",
    "previous_action_rms",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_file(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
        raise ValueError(f"required project file is missing: {relative}")
    return path


def validate_source(config: dict[str, Any]) -> Path:
    source = config["source"]
    for key, relative in (
        ("protocol_sha256", source["protocol"]),
        ("manifest_sha256", f"{source['root']}/run_manifest.json"),
        ("decision_sha256", f"{source['root']}/decision.json"),
    ):
        path = _project_file(relative)
        if _sha256(path) != source[key]:
            raise ValueError(f"source hash mismatch: {relative}")
    decision = json.loads(_project_file(f"{source['root']}/decision.json").read_text())
    if decision.get("decision", {}).get("status") != source["required_decision"]:
        raise ValueError("source decision does not satisfy the required gate")
    return (PROJECT_ROOT / source["root"]).resolve()


def load_records(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("**/speed_pulse_run_record.json"))
    ]
    source = config["source"]
    expected_blocks = set(map(int, source["blocks"]))
    arms = tuple(config["validation"].get("candidate_arms", ARMS))
    identities = []
    for record in records:
        identity = record.get("identity", {})
        key = (
            identity.get("backend"), identity.get("profile"),
            int(identity.get("block_id", -1)), identity.get("arm"),
        )
        identities.append(key)
        target = record.get("metrics", {}).get(config["target"]["name"])
        if (
            record.get("dataset_role") != source["dataset_role"]
            or identity.get("stage") != "causal_pilot"
            or key[2] not in expected_blocks
            or key[3] not in arms
            or not record.get("gate", {}).get("passed")
            or not record.get("pulse_trace", {}).get("passed")
            or target is None
            or not math.isfinite(float(target))
        ):
            raise ValueError(f"invalid source record: {key}")
    if len(records) != int(source["expected_run_count"]):
        raise ValueError("source record count mismatch")
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate source run identity")
    expected = {
        (backend, profile, block, arm)
        for backend in ("fastlio2", "liosam")
        for profile in ("curve_1_5_right_1_0", "curve_1_5_left_1_0")
        for block in expected_blocks
        for arm in arms
    }
    if set(identities) != expected:
        raise ValueError("source schedule is incomplete or unexpected")
    return records


def state_features(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != (51,) or not np.all(np.isfinite(observation)):
        raise ValueError("pre-treatment observation must be finite 51-D")
    return np.asarray((
        observation[48], observation[49], observation[50],
        observation[9], observation[10], observation[11],
        observation[0], observation[1], observation[5],
        float(np.sqrt(np.mean(np.square(observation[24:36])))),
        float(np.sqrt(np.mean(np.square(observation[36:48])))),
    ), dtype=np.float64)


def action_features(
    family: str, state: np.ndarray, scales_xyz: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    state = np.asarray(state, dtype=np.float64)
    scales = np.asarray(scales_xyz, dtype=np.float64)
    if state.shape != (len(STATE_FEATURE_NAMES),) or scales.shape != (3,):
        raise ValueError("state/action feature shape mismatch")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0) or np.any(scales > 1.0):
        raise ValueError("candidate scales must be finite in (0, 1]")
    if family == "state_only":
        return state.copy(), STATE_FEATURE_NAMES
    confidence, command_yaw = state[0], state[5]
    if family == "scalar_action":
        mean_scale = float(np.mean(scales))
        reduction = 1.0 - mean_scale
        extra = np.asarray((
            mean_scale, reduction,
            reduction * command_yaw,
            reduction * confidence,
        ))
        names = (
            "candidate_mean_xyz_scale", "candidate_mean_reduction",
            "mean_reduction_x_requested_yaw",
            "mean_reduction_x_slam_confidence",
        )
    elif family == "component_action":
        translation_scale = float(np.mean(scales[:2]))
        yaw_scale = float(scales[2])
        translation_reduction = 1.0 - translation_scale
        yaw_reduction = 1.0 - yaw_scale
        extra = np.asarray((
            translation_scale, yaw_scale,
            translation_reduction, yaw_reduction,
            translation_reduction * command_yaw,
            yaw_reduction * command_yaw,
            translation_reduction * confidence,
            yaw_reduction * confidence,
            translation_reduction * yaw_reduction,
        ))
        names = (
            "candidate_translation_scale", "candidate_yaw_scale",
            "candidate_translation_reduction", "candidate_yaw_reduction",
            "translation_reduction_x_requested_yaw",
            "yaw_reduction_x_requested_yaw",
            "translation_reduction_x_slam_confidence",
            "yaw_reduction_x_slam_confidence",
            "translation_reduction_x_yaw_reduction",
        )
    elif family == "component_action_v2_additive_directional":
        translation_reduction = 1.0 - float(np.mean(scales[:2]))
        yaw_reduction = 1.0 - float(scales[2])
        extra = np.asarray((
            translation_reduction,
            yaw_reduction,
            translation_reduction * command_yaw,
            yaw_reduction * command_yaw,
        ))
        names = (
            "candidate_translation_reduction",
            "candidate_yaw_reduction",
            "translation_reduction_x_requested_yaw",
            "yaw_reduction_x_requested_yaw",
        )
    else:
        raise ValueError(f"unknown model family: {family}")
    return np.concatenate((state, extra)), STATE_FEATURE_NAMES + names


def _row(record: dict[str, Any], family: str, target_name: str) -> dict[str, Any]:
    trace = record["pulse_trace"]
    state = state_features(np.asarray(trace["pre_pulse"]["observation"]))
    scales = np.asarray(trace["assigned_scales"], dtype=np.float64)
    features, names = action_features(family, state, scales)
    identity = record["identity"]
    return {
        "identity": identity,
        "state": state,
        "scales": scales,
        "features": features,
        "feature_names": names,
        "target": float(record["metrics"][target_name]),
        "speed": float(record["metrics"]["pulse_window_moving_speed_mps"]),
        "safety": bool(record["metrics"].get("fall") or record["metrics"].get("base_contact")),
    }


def fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("invalid ridge training arrays")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x)), standardized))
    regularizer = np.eye(design.shape[1]) * float(penalty)
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    standardized = (x - model["mean"]) / model["scale"]
    values = model["coefficients"][0] + standardized @ model["coefficients"][1:]
    return np.clip(values, 0.0, 1.0)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target, prediction = map(lambda value: np.asarray(value, dtype=np.float64), (target, prediction))
    return {
        "run_count": int(len(target)),
        "brier": float(np.mean(np.square(prediction - target))),
        "mae": float(np.mean(np.abs(prediction - target))),
        "target_mean": float(np.mean(target)),
        "prediction_mean": float(np.mean(prediction)),
    }


def evaluate_family(
    records: list[dict[str, Any]], config: dict[str, Any], family: str,
) -> dict[str, Any]:
    target_name = config["target"]["name"]
    rows = [_row(record, family, target_name) for record in records]
    arms = tuple(config["validation"].get("candidate_arms", ARMS))
    blocks = tuple(map(int, config["validation"]["held_out_blocks"]))
    penalty = float(config["models"]["l2_penalty"])
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    folds = []
    selection_groups = []
    for block in blocks:
        train_indices = [i for i, row in enumerate(rows) if int(row["identity"]["block_id"]) != block]
        test_indices = [i for i, row in enumerate(rows) if int(row["identity"]["block_id"]) == block]
        train_x = np.asarray([rows[i]["features"] for i in train_indices])
        train_y = np.asarray([rows[i]["target"] for i in train_indices])
        model = fit_ridge(train_x, train_y, penalty)
        predictions[test_indices] = predict_ridge(
            model, np.asarray([rows[i]["features"] for i in test_indices])
        )
        folds.append({
            "held_out_block": block,
            "train_blocks": sorted({int(rows[i]["identity"]["block_id"]) for i in train_indices}),
            "train_run_count": len(train_indices), "test_run_count": len(test_indices),
        })
        for backend in ("fastlio2", "liosam"):
            for profile in ("curve_1_5_right_1_0", "curve_1_5_left_1_0"):
                group = [row for row in rows if row["identity"]["backend"] == backend and row["identity"]["profile"] == profile and int(row["identity"]["block_id"]) == block]
                if {row["identity"]["arm"] for row in group} != set(arms):
                    raise ValueError("incomplete action-selection group")
                state = np.mean([row["state"] for row in group], axis=0)
                candidates = []
                for order, arm in enumerate(arms):
                    actual = next(row for row in group if row["identity"]["arm"] == arm)
                    candidate_x, _ = action_features(family, state, actual["scales"])
                    predicted = float(predict_ridge(model, candidate_x[None, :])[0])
                    candidates.append((predicted, -float(np.mean(actual["scales"])), order, actual))
                _, _, _, selected = min(candidates)
                uniform = next(row for row in group if row["identity"]["arm"] == "uniform_075")
                control = next(row for row in group if row["identity"]["arm"] == "control")
                selection_groups.append({
                    "backend": backend, "profile": profile, "block_id": block,
                    "selected_arm": selected["identity"]["arm"],
                    "selected_hazard": selected["target"],
                    "uniform_hazard": uniform["target"], "control_hazard": control["target"],
                    "selected_speed_mps": selected["speed"], "uniform_speed_mps": uniform["speed"],
                    "selected_safety_event": selected["safety"],
                    "candidate_predictions": {
                        item[3]["identity"]["arm"]: item[0] for item in candidates
                    },
                })
    if not np.all(np.isfinite(predictions)):
        raise ValueError("cross-validation did not predict every run")
    full_model = fit_ridge(
        np.asarray([row["features"] for row in rows]),
        np.asarray([row["target"] for row in rows]), penalty,
    )
    return {
        "family": family,
        "feature_names": list(rows[0]["feature_names"]),
        "cross_validation": {"folds": folds, "metrics": _metrics(np.asarray([row["target"] for row in rows]), predictions)},
        "selection_groups": selection_groups,
        "full_fit": {key: value.astype(float).tolist() for key, value in full_model.items()},
    }


def decide(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    gate = config["decision_gate"]["all_required"]
    primary_family = config["models"].get("primary_family", "component_action")
    component = results[primary_family]
    groups = component["selection_groups"]
    liosam = [group for group in groups if group["backend"] == "liosam"]
    fast = [group for group in groups if group["backend"] == "fastlio2"]
    selected_arms = [group["selected_arm"] for group in groups]
    arms = tuple(config["validation"].get("candidate_arms", ARMS))
    component_arms = tuple(arm for arm in arms if arm not in ("control", "uniform_075"))
    component_fraction = sum(arm in component_arms for arm in selected_arms) / len(selected_arms)
    lio_hazard_delta = float(np.mean([g["selected_hazard"] - g["uniform_hazard"] for g in liosam]))
    lio_speed_delta = float(np.mean([g["selected_speed_mps"] - g["uniform_speed_mps"] for g in liosam]))
    fast_hazard_delta = float(np.mean([g["selected_hazard"] - g["control_hazard"] for g in fast]))
    conditions = {
        "complete_source_integrity": True,
        "component_brier_not_worse_than_state_only": component["cross_validation"]["metrics"]["brier"] <= results["state_only"]["cross_validation"]["metrics"]["brier"],
        "component_brier_not_worse_than_scalar_action": component["cross_validation"]["metrics"]["brier"] <= results["scalar_action"]["cross_validation"]["metrics"]["brier"],
        "liosam_selected_minus_uniform_mean_hazard": lio_hazard_delta <= float(gate["liosam_selected_minus_uniform_mean_hazard_maximum"]),
        "liosam_selected_nonworse_groups": sum(g["selected_hazard"] <= g["uniform_hazard"] for g in liosam) >= int(gate["liosam_selected_nonworse_groups_minimum"]),
        "liosam_selected_minus_uniform_mean_speed_mps": lio_speed_delta >= float(gate["liosam_selected_minus_uniform_mean_speed_mps_minimum"]),
        "fastlio2_selected_minus_control_mean_hazard": fast_hazard_delta <= float(gate["fastlio2_selected_minus_control_mean_hazard_maximum"]),
        "selected_component_arm_fraction": component_fraction >= float(gate["selected_component_arm_fraction_minimum"]),
        "selected_distinct_arm_count": len(set(selected_arms)) >= int(gate["selected_distinct_arm_count_minimum"]),
        "no_selected_arm_safety_event": not any(group["selected_safety_event"] for group in groups),
    }
    passed = all(conditions.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "claim_allowed": False,
        "conditions": conditions,
        "summary": {
            "component_brier": component["cross_validation"]["metrics"]["brier"],
            "state_only_brier": results["state_only"]["cross_validation"]["metrics"]["brier"],
            "scalar_action_brier": results["scalar_action"]["cross_validation"]["metrics"]["brier"],
            "liosam_selected_minus_uniform_mean_hazard": lio_hazard_delta,
            "liosam_selected_nonworse_groups": sum(g["selected_hazard"] <= g["uniform_hazard"] for g in liosam),
            "liosam_selected_minus_uniform_mean_speed_mps": lio_speed_delta,
            "fastlio2_selected_minus_control_mean_hazard": fast_hazard_delta,
            "selected_component_arm_fraction": component_fraction,
            "selected_arm_counts": {arm: selected_arms.count(arm) for arm in arms},
        },
        "next_step": config["decision_gate"]["pass_next_step" if passed else "fail_next_step"],
    }


def build_report(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    root = validate_source(config)
    records = load_records(root, config)
    families = tuple(config["models"].get(
        "family_order", ("state_only", "scalar_action", "component_action")
    ))
    results = {family: evaluate_family(records, config, family) for family in families}
    return {
        "schema_version": 1,
        "kind": "slam_component_conditioned_risk_model_report",
        "config_sha256": _sha256(config_path),
        "source_record_count": len(records),
        "runtime_forbidden_inputs": ["backend_id", "profile_id", "ground_truth", "future_label", "intervention_id"],
        "models": results,
        "decision": decide(results, config),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not config_path.is_relative_to(PROJECT_ROOT) or not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("config and output must remain inside the project")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("boundaries", {}).get("offline_model_fit_authorized") is not True:
        raise ValueError("offline model fit is not authorized")
    for key in ("ros_policy_wiring_authorized", "live_execution_authorized", "ppo_training_authorized", "default_switch_authorized", "physical_robot_authorized"):
        if config["boundaries"].get(key) is not False:
            raise ValueError(f"unsafe boundary must remain false: {key}")
    report = build_report(config, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": report["decision"]}, indent=2))
    return 0 if report["decision"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
