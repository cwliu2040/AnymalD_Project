#!/usr/bin/env python3
"""Measure formal-policy action sensitivity to offline GT velocity replacement."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "liosam_policy_state_quality.yaml"


def _load(path: Path) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"document must contain a mapping: {path}")
    return value


def action_difference_metrics(values: np.ndarray) -> dict[str, float]:
    differences = np.asarray(values, dtype=np.float64)
    if differences.ndim != 2 or differences.shape[1] != 12:
        raise ValueError("action differences must have shape (N, 12)")
    if not len(differences) or not np.isfinite(differences).all():
        return {"sample_count": 0, "mean_absolute": math.nan, "absolute_max": math.nan}
    absolute = np.abs(differences)
    return {
        "sample_count": int(len(absolute)),
        "mean_absolute": float(absolute.mean()),
        "absolute_max": float(absolute.max()),
    }


def evaluate(
    policy_diagnostics: dict[str, Any],
    locomotion_diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    vendor = PROJECT_ROOT / "deployment" / "python_vendor"
    sys.path.insert(0, str(vendor))
    from onnx.reference import ReferenceEvaluator

    qualification = config["qualification"]
    model = ReferenceEvaluator(
        str((PROJECT_ROOT / qualification["policy_path"]).resolve())
    )
    input_name = model.input_names[0]
    records = policy_diagnostics["records"]
    samples = locomotion_diagnostics["samples"]
    truth_times = np.asarray([sample["time_s"] for sample in samples])
    observations: list[np.ndarray] = []
    counterfactuals: list[np.ndarray] = []
    recorded_actions: list[np.ndarray] = []
    active: list[bool] = []
    stopped: list[bool] = []
    has_been_active = False
    for record in records:
        clock_s = float(record["clock_s"])
        index = int(np.argmin(np.abs(truth_times - clock_s)))
        if abs(float(truth_times[index]) - clock_s) > 0.015:
            continue
        observation = np.asarray(record["observation"], dtype=np.float32)
        truth_velocity = np.asarray(
            samples[index]["actual_linear_velocity_body_mps"],
            dtype=np.float32,
        )
        command = np.asarray(record["effective_command"], dtype=np.float32)
        is_active = bool(np.max(np.abs(command)) >= 0.05)
        has_been_active = has_been_active or is_active
        observations.append(observation)
        replacement = observation.copy()
        replacement[:3] = truth_velocity
        counterfactuals.append(replacement)
        recorded_actions.append(
            np.asarray(record["raw_action"], dtype=np.float32)
        )
        active.append(is_active)
        stopped.append(has_been_active and not is_active)

    observed_array = np.stack(observations)
    counterfactual_array = np.stack(counterfactuals)
    recorded_array = np.stack(recorded_actions)
    observed_actions = np.asarray(
        model.run(None, {input_name: observed_array})[0], dtype=np.float32
    )
    counterfactual_actions = np.asarray(
        model.run(None, {input_name: counterfactual_array})[0],
        dtype=np.float32,
    )
    replay_error = float(np.max(np.abs(observed_actions - recorded_array)))
    differences = counterfactual_actions - observed_actions
    active_metrics = action_difference_metrics(differences[np.asarray(active)])
    stopped_metrics = action_difference_metrics(differences[np.asarray(stopped)])
    limits = config["gate"]["policy_action_counterfactual"]
    failures: list[str] = []

    def check(name: str, value: float, limit_name: str) -> None:
        limit = float(limits[limit_name])
        if not math.isfinite(value) or value > limit:
            failures.append(f"{name}={value:.6g}>{limit:.6g}")

    check(
        "recorded_action_replay_absolute_max",
        replay_error,
        "recorded_action_replay_absolute_max",
    )
    for phase, metrics in (
        ("active", active_metrics),
        ("stopped", stopped_metrics),
    ):
        check(
            f"{phase}_mean_absolute",
            metrics["mean_absolute"],
            f"{phase}_mean_absolute_max",
        )
        check(
            f"{phase}_absolute_max",
            metrics["absolute_max"],
            f"{phase}_absolute_max",
        )
    return {
        "schema_version": 1,
        "kind": "policy_state_action_sensitivity",
        "passed": not failures,
        "failures": failures,
        "formal_policy_sha256": qualification["policy_sha256"],
        "metrics": {
            "recorded_action_replay_absolute_max": replay_error,
            "active": active_metrics,
            "stopped": stopped_metrics,
        },
        "boundaries": {
            "runtime_ground_truth_used": False,
            "offline_ground_truth_replacement_only": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-diagnostics", type=Path, required=True)
    parser.add_argument("--locomotion-diagnostics", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        _load(args.policy_diagnostics),
        _load(args.locomotion_diagnostics),
        _load(args.config),
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
