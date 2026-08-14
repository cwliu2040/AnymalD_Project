#!/usr/bin/env python3
"""Train and export the project-owned proprioceptive velocity estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source" / "anymal_locomotion"
sys.path.insert(0, str(SOURCE_ROOT))

from anymal_locomotion.velocity_estimator.contract import HISTORY_LENGTH, STEP_DIMENSION
from anymal_locomotion.velocity_estimator.dataset import (
    VelocityDataset,
    build_history_windows,
    concatenate_datasets,
    holdout_environment_ids,
    velocity_metrics,
)
from anymal_locomotion.velocity_estimator.model import ProprioceptiveVelocityEstimator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NormalizedEstimator(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.model((history - self.mean) / self.std)


def _batched_predict(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run a large environment-disjoint holdout without GPU-sized allocation."""
    predictions = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(
                device, non_blocking=True
            )
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--vertical-transition-weight",
        type=float,
        default=0.0,
        help="Additional MSE weight for rare high-|vz| fall/contact transitions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Project-local qualified estimator checkpoint used for fine-tuning.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset_paths = [path.expanduser().resolve() for path in args.dataset]
    output_dir = args.output_dir.expanduser().resolve()
    if any(not path.is_relative_to(PROJECT_ROOT) for path in dataset_paths) or not output_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("dataset and output directory must remain inside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gate_path = PROJECT_ROOT / "configs" / "proprioceptive_velocity_estimator_gate.yaml"
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    dataset = concatenate_datasets([VelocityDataset.load(path) for path in dataset_paths])
    train_envs, holdout_envs = holdout_environment_ids(
        dataset.environment_ids,
        fraction=float(gate["holdout"]["fraction"]),
        seed=int(gate["holdout"]["seed"]),
    )
    train_x, train_y = build_history_windows(dataset, train_envs)
    holdout_x, holdout_y = build_history_windows(dataset, holdout_envs)
    initial_checkpoint_path = (
        args.initial_checkpoint.expanduser().resolve()
        if args.initial_checkpoint is not None
        else None
    )
    if initial_checkpoint_path is not None:
        if not initial_checkpoint_path.is_relative_to(PROJECT_ROOT):
            raise ValueError("initial checkpoint must remain inside the repository")
        initial_checkpoint = torch.load(
            initial_checkpoint_path, map_location="cpu", weights_only=False
        )
        step_mean = np.asarray(initial_checkpoint["step_mean"], dtype=np.float32)
        step_std = np.asarray(initial_checkpoint["step_std"], dtype=np.float32)
    else:
        step_mean = train_x.reshape(-1, STEP_DIMENSION).mean(axis=0).astype(np.float32)
        step_std = train_x.reshape(-1, STEP_DIMENSION).std(axis=0).astype(np.float32)
        step_std = np.maximum(step_std, 1.0e-4)
    mean = np.tile(step_mean, HISTORY_LENGTH)
    std = np.tile(step_std, HISTORY_LENGTH)
    normalized_train_x = (train_x - mean) / std
    training_speed = np.linalg.norm(train_y, axis=1)
    speed_weight = 1.0 + 3.0 * np.clip(training_speed - 1.0, 0.0, 2.0)
    airborne_weight = 1.0 + 2.0 * (
        np.sum(train_x[:, -4:] > 0.5, axis=1) == 0
    ).astype(np.float32)
    vertical_transition_weight = 1.0 + args.vertical_transition_weight * np.clip(
        np.abs(train_y[:, 2]) - 0.2, 0.0, 2.0
    )
    sample_weight = (
        speed_weight * airborne_weight * vertical_transition_weight
    ).astype(np.float32)

    device = torch.device(args.device)
    model = ProprioceptiveVelocityEstimator(HISTORY_LENGTH * STEP_DIMENSION).to(device)
    if initial_checkpoint_path is not None:
        model.load_state_dict(initial_checkpoint["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # The deployment gate includes a hard maximum-error bound.  Squared loss
    # intentionally keeps rare push/airborne transitions influential instead
    # of reducing them to the linear tail of a robust loss.
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(normalized_train_x),
            torch.from_numpy(train_y),
            torch.from_numpy(sample_weight),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for features, labels, weights in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            squared_error = torch.mean(torch.square(prediction - labels), dim=1)
            loss = torch.mean(squared_error * weights)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * labels.shape[0]
            count += labels.shape[0]
        history.append({"epoch": epoch + 1, "training_loss": total / count})

    wrapped = NormalizedEstimator(model.eval(), mean, std).to(device).eval()
    holdout_prediction = _batched_predict(
        wrapped,
        holdout_x,
        device=device,
        batch_size=args.batch_size,
    )
    metrics = velocity_metrics(holdout_prediction, holdout_y)

    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step_mean": step_mean,
            "step_std": step_std,
            "history_length": HISTORY_LENGTH,
            "step_dimension": STEP_DIMENSION,
        },
        checkpoint_path,
    )
    # Export from CPU so recurrent initial states remain portable and are not
    # traced with a hard-coded cuda:0 device.
    wrapped = wrapped.to("cpu").eval()
    example = torch.zeros((1, HISTORY_LENGTH * STEP_DIMENSION), device="cpu")
    torchscript_path = output_dir / "velocity_estimator.pt"
    torch.jit.script(wrapped).save(str(torchscript_path))
    onnx_path = output_dir / "velocity_estimator.onnx"
    torch.onnx.export(
        wrapped,
        example,
        onnx_path,
        input_names=["proprioceptive_history"],
        output_names=["base_linear_velocity"],
        dynamic_axes={"proprioceptive_history": {0: "batch"}, "base_linear_velocity": {0: "batch"}},
        opset_version=17,
    )
    artifact = {
        "schema_version": 1,
        "contract_id": "anymal-d-proprioceptive-velocity-v1",
        "history_length": HISTORY_LENGTH,
        "step_dimension": STEP_DIMENSION,
        "input_dimension": HISTORY_LENGTH * STEP_DIMENSION,
        "output_frame": "base_link",
        "datasets": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha256(path)}
            for path in dataset_paths
        ],
        "split": {"unit": "simulator_environment", "train_environment_ids": train_envs.tolist(), "holdout_environment_ids": holdout_envs.tolist()},
        "training": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "loss": "transition_weighted_mean_squared_error", "vertical_transition_weight": args.vertical_transition_weight, "initial_checkpoint": ({"path": str(initial_checkpoint_path.relative_to(PROJECT_ROOT)), "sha256": _sha256(initial_checkpoint_path)} if initial_checkpoint_path is not None else None), "history": history},
        "architecture": {"type": "windowed_gru", "hidden_dimension": 128, "layers": 2, "head_hidden_dimension": 128, "activation": "elu"},
        "normalization": {"step_mean": step_mean.astype(float).tolist(), "step_std": step_std.astype(float).tolist()},
        "holdout_metrics": metrics,
        "artifacts": {
            "torchscript": {"path": torchscript_path.name, "sha256": _sha256(torchscript_path)},
            "onnx": {"path": onnx_path.name, "sha256": _sha256(onnx_path)},
            "checkpoint": {"path": checkpoint_path.name, "sha256": _sha256(checkpoint_path)},
        },
        "contract_sha256": _sha256(PROJECT_ROOT / "configs" / "proprioceptive_velocity_estimator_contract.yaml"),
        "gate_sha256": _sha256(gate_path),
    }
    artifact_path = output_dir / "velocity_estimator_metadata.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(artifact_path), "holdout_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
