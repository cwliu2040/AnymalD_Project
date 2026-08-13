"""Dataset validation and leakage-safe temporal window construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contract import HISTORY_LENGTH, STEP_DIMENSION


@dataclass(frozen=True)
class VelocityDataset:
    steps: np.ndarray
    labels: np.ndarray
    environment_ids: np.ndarray
    episode_ids: np.ndarray
    timestamps_s: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "VelocityDataset":
        with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as data:
            required = {
                "steps",
                "labels",
                "environment_ids",
                "episode_ids",
                "timestamps_s",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"velocity dataset missing arrays: {missing}")
            if "schema_version" in data.files and int(data["schema_version"]) != 1:
                raise ValueError("velocity dataset schema_version must be 1")
            if "contract_id" in data.files and str(data["contract_id"]) != (
                "anymal-d-proprioceptive-velocity-v1"
            ):
                raise ValueError("velocity dataset contract_id mismatch")
            if "ground_truth_role" in data.files and str(data["ground_truth_role"]) != (
                "supervised_label_and_offline_evaluator_only"
            ):
                raise ValueError("ground-truth labels have an invalid role declaration")
            dataset = cls(
                steps=np.asarray(data["steps"], dtype=np.float32),
                labels=np.asarray(data["labels"], dtype=np.float32),
                environment_ids=np.asarray(data["environment_ids"], dtype=np.int64),
                episode_ids=np.asarray(data["episode_ids"], dtype=np.int64),
                timestamps_s=np.asarray(data["timestamps_s"], dtype=np.float64),
            )
        dataset.validate()
        return dataset

    def validate(self) -> None:
        count = self.steps.shape[0]
        if self.steps.shape != (count, STEP_DIMENSION):
            raise ValueError(f"steps must have shape (N, {STEP_DIMENSION})")
        if self.labels.shape != (count, 3):
            raise ValueError("labels must have shape (N, 3)")
        for name, values in (
            ("environment_ids", self.environment_ids),
            ("episode_ids", self.episode_ids),
            ("timestamps_s", self.timestamps_s),
        ):
            if values.shape != (count,):
                raise ValueError(f"{name} must have shape (N,)")
        if count == 0 or not np.all(np.isfinite(self.steps)) or not np.all(
            np.isfinite(self.labels)
        ) or not np.all(np.isfinite(self.timestamps_s)):
            raise ValueError("velocity dataset is empty or contains NaN/Inf")


def holdout_environment_ids(
    environment_ids: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by simulator environment, never by adjacent temporal rows."""
    unique = np.unique(environment_ids)
    if unique.size < 2:
        raise ValueError("at least two simulator environments are required")
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout fraction must be in (0, 1)")
    shuffled = np.random.default_rng(seed).permutation(unique)
    holdout_count = min(unique.size - 1, max(1, round(unique.size * fraction)))
    return np.sort(shuffled[holdout_count:]), np.sort(shuffled[:holdout_count])


def build_history_windows(
    dataset: VelocityDataset,
    selected_environment_ids: np.ndarray,
    *,
    history_length: int = HISTORY_LENGTH,
    expected_period_s: float = 0.02,
    maximum_gap_s: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Build oldest-to-newest windows without crossing env/episode boundaries."""
    selected = set(int(value) for value in selected_environment_ids)
    histories: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    order = np.lexsort(
        (dataset.timestamps_s, dataset.episode_ids, dataset.environment_ids)
    )
    steps = dataset.steps[order]
    targets = dataset.labels[order]
    envs = dataset.environment_ids[order]
    episodes = dataset.episode_ids[order]
    stamps = dataset.timestamps_s[order]
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and envs[end] == envs[start]
            and episodes[end] == episodes[start]
        ):
            end += 1
        if int(envs[start]) in selected and end - start >= history_length:
            group_stamps = stamps[start:end]
            gaps = np.diff(group_stamps)
            for local_end in range(history_length - 1, end - start):
                window_start = local_end - history_length + 1
                window_gaps = gaps[window_start:local_end]
                if np.any(window_gaps <= 0.0) or np.any(window_gaps > maximum_gap_s):
                    continue
                duration = group_stamps[local_end] - group_stamps[window_start]
                expected = expected_period_s * (history_length - 1)
                if abs(duration - expected) > maximum_gap_s:
                    continue
                histories.append(
                    steps[start + window_start : start + local_end + 1].reshape(-1)
                )
                labels.append(targets[start + local_end])
        start = end
    if not histories:
        raise ValueError("no valid temporal windows could be built")
    return np.stack(histories).astype(np.float32), np.stack(labels).astype(np.float32)


def velocity_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if predictions.shape != labels.shape or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("predictions and labels must both have shape (N, 3)")
    finite = np.all(np.isfinite(predictions), axis=1)
    safe_predictions = np.where(np.isfinite(predictions), predictions, 0.0)
    error = safe_predictions - labels
    vector_error = np.linalg.norm(error, axis=1)
    stopped = np.linalg.norm(labels[:, :2], axis=1) <= 0.10
    stopped_planar = np.linalg.norm(safe_predictions[stopped, :2], axis=1)
    stopped_vertical = np.abs(safe_predictions[stopped, 2])
    return {
        "samples": int(labels.shape[0]),
        "non_finite_predictions": int((~finite).sum()),
        "axis_mae_mps": np.mean(np.abs(error), axis=0).astype(float).tolist(),
        "axis_rmse_mps": np.sqrt(np.mean(np.square(error), axis=0)).astype(float).tolist(),
        "vector_p95_error_mps": float(np.percentile(vector_error, 95)),
        "maximum_vector_error_mps": float(np.max(vector_error)),
        "stopped_samples": int(stopped.sum()),
        "stopped_planar_bias_mps": (
            float(np.mean(stopped_planar)) if stopped_planar.size else None
        ),
        "stopped_vertical_bias_mps": (
            float(np.mean(stopped_vertical)) if stopped_vertical.size else None
        ),
    }


def concatenate_datasets(datasets: list[VelocityDataset]) -> VelocityDataset:
    """Concatenate captures while keeping every capture's episode IDs distinct."""
    if not datasets:
        raise ValueError("at least one velocity dataset is required")
    steps = []
    labels = []
    environment_ids = []
    episode_ids = []
    timestamps = []
    environment_offset = 0
    episode_offset = 0
    for dataset in datasets:
        dataset.validate()
        unique_envs, remapped_envs = np.unique(
            dataset.environment_ids, return_inverse=True
        )
        pairs = np.stack((dataset.environment_ids, dataset.episode_ids), axis=1)
        _, remapped_episodes = np.unique(pairs, axis=0, return_inverse=True)
        steps.append(dataset.steps)
        labels.append(dataset.labels)
        environment_ids.append(remapped_envs.astype(np.int64) + environment_offset)
        episode_ids.append(remapped_episodes.astype(np.int64) + episode_offset)
        timestamps.append(dataset.timestamps_s)
        environment_offset += unique_envs.size
        episode_offset += int(remapped_episodes.max()) + 1
    result = VelocityDataset(
        steps=np.concatenate(steps),
        labels=np.concatenate(labels),
        environment_ids=np.concatenate(environment_ids),
        episode_ids=np.concatenate(episode_ids),
        timestamps_s=np.concatenate(timestamps),
    )
    result.validate()
    return result
