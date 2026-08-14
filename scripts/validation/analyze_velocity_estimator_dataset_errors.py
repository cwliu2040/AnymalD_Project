#!/usr/bin/env python3
"""Break estimator errors down by capture without changing the frozen gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "anymal_locomotion"))

from anymal_locomotion.velocity_estimator.dataset import (
    VelocityDataset,
    build_history_windows,
    velocity_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.metadata.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not metadata_path.is_relative_to(PROJECT_ROOT) or not output_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("metadata and output must remain inside the repository")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    holdout = set(int(value) for value in metadata["split"]["holdout_environment_ids"])
    model_path = metadata_path.parent / metadata["artifacts"]["torchscript"]["path"]
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    reports = []
    environment_offset = 0
    for item in metadata["datasets"]:
        dataset = VelocityDataset.load(PROJECT_ROOT / item["path"])
        unique = np.unique(dataset.environment_ids)
        selected_global = sorted(
            value for value in holdout
            if environment_offset <= value < environment_offset + unique.size
        )
        selected_local_indices = np.asarray(
            [value - environment_offset for value in selected_global], dtype=np.int64
        )
        selected_local_ids = unique[selected_local_indices]
        features, labels = build_history_windows(dataset, selected_local_ids)
        batches = []
        with torch.inference_mode():
            for start in range(0, features.shape[0], 4096):
                batches.append(model(torch.from_numpy(features[start:start + 4096])).numpy())
        predictions = np.concatenate(batches)
        error = np.linalg.norm(predictions - labels, axis=1)
        worst = int(np.argmax(error))
        reports.append(
            {
                "dataset": item["path"],
                "holdout_environment_count": len(selected_global),
                "metrics": velocity_metrics(predictions, labels),
                "worst": {
                    "vector_error_mps": float(error[worst]),
                    "prediction_mps": predictions[worst].astype(float).tolist(),
                    "label_mps": labels[worst].astype(float).tolist(),
                },
            }
        )
        environment_offset += unique.size
    report = {"schema_version": 1, "metadata": str(metadata_path.relative_to(PROJECT_ROOT)), "captures": reports}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
