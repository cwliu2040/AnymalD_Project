#!/usr/bin/env python3
"""Evaluate one estimator artifact on its environment-disjoint holdout gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "anymal_locomotion"))

from anymal_locomotion.velocity_estimator.dataset import VelocityDataset, build_history_windows, concatenate_datasets, velocity_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.metadata.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not metadata_path.is_relative_to(PROJECT_ROOT) or not output_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("metadata and output must be inside the repository")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = concatenate_datasets(
        [VelocityDataset.load(PROJECT_ROOT / item["path"]) for item in metadata["datasets"]]
    )
    holdout_ids = np.asarray(metadata["split"]["holdout_environment_ids"], dtype=np.int64)
    features, labels = build_history_windows(dataset, holdout_ids)
    from onnx.reference import ReferenceEvaluator

    model_path = metadata_path.parent / metadata["artifacts"]["onnx"]["path"]
    session = ReferenceEvaluator(str(model_path))
    predictions = session.run(None, {session.input_names[0]: features})[0]
    metrics = velocity_metrics(predictions, labels)
    torchscript_path = metadata_path.parent / metadata["artifacts"]["torchscript"]["path"]
    torchscript = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
    parity_features = features[: min(256, features.shape[0])]
    with torch.inference_mode():
        torchscript_predictions = torchscript(
            torch.from_numpy(parity_features)
        ).cpu().numpy()
    onnx_parity_predictions = session.run(
        None, {session.input_names[0]: parity_features}
    )[0]
    parity_max_abs_error = float(
        np.max(np.abs(torchscript_predictions - onnx_parity_predictions))
    )
    gate_path = PROJECT_ROOT / "configs" / "proprioceptive_velocity_estimator_gate.yaml"
    thresholds = yaml.safe_load(gate_path.read_text(encoding="utf-8"))["thresholds"]
    checks = {
        "axis_mae": all(a <= b for a, b in zip(metrics["axis_mae_mps"], thresholds["axis_mae_max_mps"])),
        "axis_rmse": all(a <= b for a, b in zip(metrics["axis_rmse_mps"], thresholds["axis_rmse_max_mps"])),
        "vector_p95": metrics["vector_p95_error_mps"] <= thresholds["vector_p95_error_max_mps"],
        "maximum_error": metrics["maximum_vector_error_mps"] <= thresholds["maximum_single_sample_vector_error_mps"],
        "non_finite": metrics["non_finite_predictions"] <= thresholds["non_finite_predictions_max"],
        "stopped_samples_present": metrics["stopped_samples"] > 0,
        "stopped_planar_bias": metrics["stopped_planar_bias_mps"] is not None and metrics["stopped_planar_bias_mps"] <= thresholds["stopped_planar_bias_max_mps"],
        "stopped_vertical_bias": metrics["stopped_vertical_bias_mps"] is not None and metrics["stopped_vertical_bias_mps"] <= thresholds["stopped_vertical_bias_max_mps"],
        "torchscript_onnx_parity": parity_max_abs_error
        <= thresholds["torchscript_onnx_max_abs_error"],
    }
    report = {
        "schema_version": 1,
        "gate_id": "anymal-d-proprioceptive-velocity-v1",
        "metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
        "metrics": metrics,
        "export_parity": {
            "samples": int(parity_features.shape[0]),
            "torchscript_onnx_max_abs_error": parity_max_abs_error,
        },
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
