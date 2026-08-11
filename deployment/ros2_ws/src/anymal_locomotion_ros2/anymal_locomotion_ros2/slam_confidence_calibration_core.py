"""Deterministic, explainable offline calibration for SLAM confidence.

This module is ROS-independent.  Ground truth is accepted only in the labelled
capture files consumed here; runtime extractors never import this module's
dataset tooling or subscribe to ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


FEATURE_SCHEMAS = {
    "fastlio2": (
        "log1p_effective_features",
        "effective_support_ratio",
        "effective_log_trend_per_s",
        "effective_projected_ratio_0_2s",
        "effective_support_memory",
        "linear_speed_mps",
        "yaw_rate_deg_s",
        "support_motion_load",
        "support_motion_exposure",
        "source_age_s",
    ),
    "liosam": (
        "log1p_corner_features",
        "log1p_surface_features",
        "corner_support_ratio",
        "surface_support_ratio",
        "corner_log_trend_per_s",
        "surface_log_trend_per_s",
        "corner_projected_ratio_0_2s",
        "surface_projected_ratio_0_2s",
        "corner_support_memory",
        "surface_support_memory",
        "linear_speed_mps",
        "yaw_rate_deg_s",
        "support_motion_load",
        "support_motion_exposure",
        "source_age_s",
        "degenerate",
        "odom_unavailable",
        "insufficient_support",
    ),
}

FEATURE_TRANSFORM_CONFIG = {
    "schema": "causal_support_motion_v3",
    "peak_half_life_s": 30.0,
    "support_memory_recovery_s": 20.0,
    "motion_load_yaw_scale": 0.02,
    "support_projection_horizon_s": {
        "fastlio2": 0.20,
        "liosam": 0.20,
    },
}

SUPPORT_FORECAST_GUARDS = {
    "fastlio2": {
        "projected_ratio_indices": [3],
        "zero_at_or_below": 0.24,
        "one_at_or_above": 0.3244,
    },
    "liosam": {
        "projected_ratio_indices": [6, 7],
        "support_trend_indices": [4, 5],
        "falling_trend_below_per_s": -0.25,
        "zero_at_or_below": 0.25,
        "one_at_or_above": 0.472,
        "motion_exposure_index": 13,
        "motion_load_index": 12,
        "low_exposure_floor_until": 0.105,
        "high_motion_load_release": 0.40,
        "low_exposure_confidence_floor": 0.60,
        "motion_risk_ratio_indices": [2, 3],
        "motion_risk_load_index": 12,
        "motion_risk_load_at_or_above": 0.03,
        "motion_risk_ratio_from": 0.65,
        "motion_risk_ratio_through": 0.72,
        "motion_risk_linear_speed_index": 10,
        "motion_risk_linear_speed_from": 0.10,
        "motion_risk_linear_speed_through": 0.20,
        "motion_risk_confidence_cap": 0.40,
        "degenerate_index": 15,
        "degenerate_confidence_cap": 0.40,
        "low_exposure_release_projected_at_or_below": 0.20,
        "odom_unavailable_index": 16,
        "insufficient_support_index": 17,
    },
}


class CausalFeatureTransform:
    """Build deployment-identical temporal features without future samples.

    The slowly decaying peak is learned after process start from source-stamped
    observations only.  Repeated 20 Hz evaluation rows for one 10 Hz scan do
    not update the state twice.
    """

    def __init__(self, backend_id: str) -> None:
        if backend_id not in FEATURE_SCHEMAS:
            raise ValueError(f"unsupported backend: {backend_id}")
        self.backend_id = backend_id
        self._last_stamp_ns: int | None = None
        self._last_values: tuple[float, ...] | None = None
        self._peaks: tuple[float, ...] | None = None
        self._last_vector: list[float] | None = None
        self._memories: tuple[float, ...] | None = None
        self._motion_exposure = 0.0

    def transform(self, row: dict[str, Any]) -> list[float] | None:
        features = row.get("features", {})
        age = row.get("source_age_s")
        stamp = row.get("source_stamp_ns")
        try:
            if self.backend_id == "fastlio2":
                values = (max(0.0, float(features["effective_features"])),)
                state_tail = (float(age),)
            else:
                values = (
                    max(0.0, float(features["corner_features"])),
                    max(0.0, float(features["surface_features"])),
                )
                state_tail = (
                    float(age),
                    float(bool(features.get("degenerate", False))),
                    float(not bool(features.get("odom_available", True))),
                    float(values[0] <= 10.0 or values[1] <= 100.0),
                )
            stamp_ns = int(stamp)
        except (KeyError, TypeError, ValueError):
            return None
        try:
            linear_speed = max(0.0, float(features.get("linear_speed_mps", 0.0)))
            yaw_rate = max(0.0, float(features.get("yaw_rate_deg_s", 0.0)))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (*values, *state_tail, linear_speed, yaw_rate)):
            return None
        if self._last_stamp_ns is not None and stamp_ns < self._last_stamp_ns:
            self.__init__(self.backend_id)
        if stamp_ns == self._last_stamp_ns:
            return list(self._last_vector) if self._last_vector is not None else None

        if self._last_stamp_ns is None:
            dt_s = 0.0
            peaks = tuple(max(value, 1.0) for value in values)
            trends = tuple(0.0 for _ in values)
        else:
            dt_s = max((stamp_ns - self._last_stamp_ns) * 1.0e-9, 1.0e-6)
            decay = math.exp(
                -math.log(2.0)
                * dt_s
                / float(FEATURE_TRANSFORM_CONFIG["peak_half_life_s"])
            )
            assert self._peaks is not None and self._last_values is not None
            peaks = tuple(
                max(value, peak * decay, 1.0)
                for value, peak in zip(values, self._peaks, strict=True)
            )
            trends = tuple(
                (math.log1p(value) - math.log1p(previous)) / dt_s
                for value, previous in zip(values, self._last_values, strict=True)
            )
        ratios = tuple(
            min(1.5, value / peak)
            for value, peak in zip(values, peaks, strict=True)
        )
        if self._memories is None:
            memories = ratios
        else:
            memories = tuple(
                min(
                    ratio,
                    memory
                    + dt_s
                    / float(FEATURE_TRANSFORM_CONFIG["support_memory_recovery_s"]),
                )
                for ratio, memory in zip(ratios, self._memories, strict=True)
            )
        logs = tuple(math.log1p(value) for value in values)
        projected_ratios = tuple(
            min(
                1.5,
                ratio
                * math.exp(
                    min(0.0, trend)
                    * float(
                        FEATURE_TRANSFORM_CONFIG[
                            "support_projection_horizon_s"
                        ][self.backend_id]
                    )
                ),
            )
            for ratio, trend in zip(ratios, trends, strict=True)
        )
        support_deficit = max(0.0, 1.0 - min(ratios))
        motion_load = (
            linear_speed
            + float(FEATURE_TRANSFORM_CONFIG["motion_load_yaw_scale"])
            * yaw_rate
        ) * support_deficit
        if support_deficit < 0.10:
            self._motion_exposure = max(
                0.0, self._motion_exposure - 0.5 * dt_s
            )
        else:
            self._motion_exposure += motion_load * dt_s
        vector = [
            *logs,
            *ratios,
            *trends,
            *projected_ratios,
            *memories,
            linear_speed,
            yaw_rate,
            motion_load,
            self._motion_exposure,
            *state_tail,
        ]
        self._last_stamp_ns = stamp_ns
        self._last_values = values
        self._peaks = peaks
        self._memories = memories
        self._last_vector = vector
        return list(vector)


@dataclass(frozen=True)
class CalibrationGates:
    source_signal_coverage_min: float = 0.99
    frame_auroc_min: float = 0.80
    healthy_false_low_time_fraction_max: float = 0.05
    gradual_event_degrade_recall_min: float = 0.80
    gradual_event_median_lead_time_s_min: float = 0.20
    minimum_independent_gradual_events: int = 20
    minimum_capture_groups: int = 5
    low_threshold: float = 0.45


def stable_capture_split(groups: list[str]) -> dict[str, str]:
    """Split by complete capture group, never by frames."""

    unique = sorted(set(groups), key=lambda item: hashlib.sha256(item.encode()).hexdigest())
    count = len(unique)
    if count < 4:
        return {group: "train" for group in unique}
    train_count = max(1, round(0.40 * count))
    calibration_count = max(1, round(0.20 * count))
    validation_count = max(1, round(0.20 * count))
    if train_count + calibration_count + validation_count >= count:
        train_count = count - 3
        calibration_count = 1
        validation_count = 1
    roles = (
        ["train"] * train_count
        + ["calibration"] * calibration_count
        + ["threshold_validation"] * validation_count
        + ["final_holdout"]
        * (count - train_count - calibration_count - validation_count)
    )
    return dict(zip(unique, roles, strict=True))


def feature_vector(backend_id: str, row: dict[str, Any]) -> list[float] | None:
    """Stateless compatibility helper; prefer one transform per capture."""

    return CausalFeatureTransform(backend_id).transform(row)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int = 4000,
    learning_rate: float = 0.03,
    l2: float = 1.0e-3,
) -> dict[str, Any]:
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("invalid logistic training shapes")
    if features.shape[0] < 2 or len(np.unique(labels)) != 2:
        raise ValueError("logistic training requires both label classes")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1.0e-9] = 1.0
    normalized = (features - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(iterations):
        probabilities = _sigmoid(design @ weights)
        gradient = design.T @ (probabilities - labels) / len(labels)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
    }


def logistic_predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    return _sigmoid(design @ weights)


def fit_isotonic(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, list[float]]:
    """Pool-adjacent-violators mapping for probability calibration."""

    if len(probabilities) < 2 or len(np.unique(labels)) != 2:
        raise ValueError("isotonic calibration requires both label classes")
    order = np.argsort(probabilities, kind="stable")
    x = probabilities[order]
    y = labels[order]
    blocks: list[list[float]] = []
    for index, target in enumerate(y):
        blocks.append([float(index), float(index), 1.0, float(target)])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[3] / left[2] <= right[3] / right[2]:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    thresholds: list[float] = []
    values: list[float] = []
    for start, end, weight, total in blocks:
        thresholds.append(float(x[int(end)]))
        values.append(float(total / weight))
    return {"upper_probability": thresholds, "calibrated_value": values}


def isotonic_predict(model: dict[str, list[float]], values: np.ndarray) -> np.ndarray:
    thresholds = np.asarray(model["upper_probability"], dtype=np.float64)
    calibrated = np.asarray(model["calibrated_value"], dtype=np.float64)
    indices = np.searchsorted(thresholds, values, side="left")
    return calibrated[np.minimum(indices, len(calibrated) - 1)]


def apply_support_forecast_guard(
    backend_id: str, features: np.ndarray, confidence: np.ndarray
) -> np.ndarray:
    """Apply a conservative causal support envelope."""

    config = SUPPORT_FORECAST_GUARDS[backend_id]
    projected = np.min(
        features[:, config["projected_ratio_indices"]], axis=1
    )
    lower = float(config["zero_at_or_below"])
    upper = float(config["one_at_or_above"])
    guard = np.clip((projected - lower) / (upper - lower), 0.0, 1.0)
    if "support_trend_indices" in config:
        trends = np.min(
            features[:, config["support_trend_indices"]], axis=1
        )
        insufficient = (
            features[:, int(config["insufficient_support_index"])] > 0.5
        )
        support_ratio = np.min(
            features[:, config["motion_risk_ratio_indices"]], axis=1
        )
        motion_risk = (
            features[:, int(config["motion_risk_load_index"])]
            >= float(config["motion_risk_load_at_or_above"])
        ) & (
            support_ratio >= float(config["motion_risk_ratio_from"])
        ) & (
            support_ratio <= float(config["motion_risk_ratio_through"])
        ) & (
            features[:, int(config["motion_risk_linear_speed_index"])]
            >= float(config["motion_risk_linear_speed_from"])
        ) & (
            features[:, int(config["motion_risk_linear_speed_index"])]
            <= float(config["motion_risk_linear_speed_through"])
        )
        falling = trends < float(config["falling_trend_below_per_s"])
        guard = np.where(falling | insufficient, guard, 1.0)
    guarded = np.minimum(confidence, guard)
    if "motion_exposure_index" in config:
        operational = (
            features[:, int(config["odom_unavailable_index"])] > 0.5
        )
        guarded[motion_risk] = np.minimum(
            guarded[motion_risk],
            float(config["motion_risk_confidence_cap"]),
        )
        # LIO-SAM raises its degeneracy flag during the mapping-odometry
        # bootstrap gap as well as during an active weak-geometry update.  The
        # former is already handled by the hard odometry-availability state and
        # must not accumulate false-low confidence time; only cap an update
        # that actually produced usable mapping odometry.
        degenerate = (
            features[:, int(config["degenerate_index"])] > 0.5
        ) & ~operational
        guarded[degenerate] = np.minimum(
            guarded[degenerate],
            float(config["degenerate_confidence_cap"]),
        )
        exposure = features[:, int(config["motion_exposure_index"])]
        insufficient = (
            features[:, int(config["insufficient_support_index"])] > 0.5
        )
        low_exposure = (
            exposure < float(config["low_exposure_floor_until"])
        ) & (
            features[:, int(config["motion_load_index"])]
            < float(config["high_motion_load_release"])
        ) & (
            projected
            > float(
                config["low_exposure_release_projected_at_or_below"]
            )
        ) & ~operational & ~insufficient & ~motion_risk
        guarded[low_exposure] = np.maximum(
            guarded[low_exposure],
            float(config["low_exposure_confidence_floor"]),
        )
    return guarded


def frame_metrics(
    labels: np.ndarray,
    confidence: np.ndarray,
    *,
    clusters: list[str] | None = None,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    if len(labels) == 0:
        return {
            "sample_count": 0,
            "auroc": None,
            "brier": None,
            "ece": None,
            "reliability_bins": [],
        }
    brier = float(np.mean((confidence - labels) ** 2))
    bins = []
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if not np.any(mask):
            continue
        observed = float(np.mean(labels[mask]))
        predicted = float(np.mean(confidence[mask]))
        fraction = float(np.mean(mask))
        ece += fraction * abs(observed - predicted)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_confidence": predicted,
                "usable_fraction": observed,
            }
        )
    auroc = None
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count and negative_count:
        order = np.argsort(confidence, kind="stable")
        sorted_confidence = confidence[order]
        ranks = np.empty(len(confidence), dtype=np.float64)
        start = 0
        while start < len(confidence):
            end = start + 1
            while (
                end < len(confidence)
                and sorted_confidence[end] == sorted_confidence[start]
            ):
                end += 1
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
            start = end
        positive_rank_sum = float(np.sum(ranks[labels == 1]))
        wins = positive_rank_sum - positive_count * (positive_count + 1) / 2
        auroc = float(wins / (positive_count * negative_count))
    result = {
        "sample_count": len(labels),
        "auroc": auroc,
        "brier": brier,
        "ece": float(ece),
        "reliability_bins": bins,
    }
    if clusters is not None:
        result["cluster_bootstrap_95_ci"] = _cluster_bootstrap_95_ci(
            labels,
            confidence,
            clusters,
            iterations=bootstrap_iterations,
        )
    return result


def _cluster_bootstrap_95_ci(
    labels: np.ndarray,
    confidence: np.ndarray,
    clusters: list[str],
    *,
    iterations: int,
) -> dict[str, Any]:
    """Deterministic capture-cluster bootstrap for probability metrics."""

    if len(clusters) != len(labels):
        raise ValueError("bootstrap clusters must match metric rows")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    unique = sorted(set(clusters))
    if not unique:
        raise ValueError("bootstrap requires at least one cluster")
    indices = {
        cluster: np.asarray(
            [index for index, value in enumerate(clusters) if value == cluster],
            dtype=np.int64,
        )
        for cluster in unique
    }
    rng = np.random.default_rng(0)
    samples: dict[str, list[float]] = {"auroc": [], "brier": [], "ece": []}
    for _ in range(iterations):
        selected = rng.choice(unique, size=len(unique), replace=True)
        draw = np.concatenate([indices[str(cluster)] for cluster in selected])
        metrics = frame_metrics(labels[draw], confidence[draw])
        for name in samples:
            value = metrics[name]
            if value is not None:
                samples[name].append(float(value))
    return {
        "unit": "capture_instance",
        "confidence_level": 0.95,
        "iterations": iterations,
        "cluster_count": len(unique),
        **{
            name: (
                {
                    "lower": float(np.quantile(values, 0.025)),
                    "upper": float(np.quantile(values, 0.975)),
                    "valid_draws": len(values),
                }
                if values
                else None
            )
            for name, values in samples.items()
        },
    }


def event_metrics(
    rows: list[dict[str, Any]],
    confidence: np.ndarray,
    low_threshold: float,
) -> dict[str, Any]:
    events: list[tuple[str, int, int]] = []
    by_group: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        instance = str(row.get("capture_instance", row["capture_group"]))
        by_group.setdefault(instance, []).append(index)
    for group, indices in by_group.items():
        active_start = None
        last = None
        for index in indices:
            gradual = row_is_gradual_failure(rows[index])
            if gradual and active_start is None:
                active_start = index
            if not gradual and active_start is not None:
                events.append((group, active_start, int(last)))
                active_start = None
            last = index
        if active_start is not None and last is not None:
            events.append((group, active_start, last))
    detected = 0
    leads = []
    for group, start, end in events:
        onset_ns = int(rows[start]["evaluation_stamp_ns"])
        candidates = [
            index for index, row in enumerate(rows)
            if str(row.get("capture_instance", row["capture_group"])) == group
            and onset_ns - 500_000_000
            <= int(row["evaluation_stamp_ns"])
            <= onset_ns
            and confidence[index] < low_threshold
        ]
        if candidates:
            detected += 1
            earliest_ns = int(rows[candidates[0]]["evaluation_stamp_ns"])
            leads.append((onset_ns - earliest_ns) * 1.0e-9)
    healthy = np.asarray([row.get("usable_next_0_5s") is True for row in rows])
    false_low = float(np.mean(confidence[healthy] < low_threshold)) if np.any(healthy) else None
    support_onsets = []
    for group, indices in by_group.items():
        previously_failed = False
        for index in indices:
            failed = bool(int(rows[index].get("operational_reasons", 0)) & 32)
            if failed and not previously_failed:
                support_onsets.append((group, index))
            previously_failed = failed
    support_detected = 0
    support_leads = []
    for group, onset in support_onsets:
        onset_ns = int(rows[onset]["evaluation_stamp_ns"])
        candidates = [
            index
            for index, row in enumerate(rows)
            if str(row.get("capture_instance", row["capture_group"])) == group
            and onset_ns - 500_000_000 <= int(row["evaluation_stamp_ns"]) <= onset_ns
            and confidence[index] < low_threshold
        ]
        if candidates:
            support_detected += 1
            support_leads.append(
                (onset_ns - int(rows[candidates[0]]["evaluation_stamp_ns"]))
                * 1.0e-9
            )
    return {
        "gradual_event_count": len(events),
        "independent_gradual_capture_count": len({event[0] for event in events}),
        "low_confidence_event_recall": detected / len(events) if events else None,
        "median_lead_time_s": float(np.median(leads)) if leads else None,
        "healthy_false_low_time_fraction": false_low,
        "support_hard_gate_event_count": len(support_onsets),
        "support_hard_gate_advance_recall": (
            support_detected / len(support_onsets) if support_onsets else None
        ),
        "support_hard_gate_median_lead_time_s": (
            float(np.median(support_leads)) if support_leads else None
        ),
    }


def row_is_gradual_failure(row: dict[str, Any]) -> bool:
    return row.get("usable_next_0_5s") is False and not bool(row.get("abrupt_hard_fault"))


def calibration_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
