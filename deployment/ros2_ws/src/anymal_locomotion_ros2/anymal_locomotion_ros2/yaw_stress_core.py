"""ROS-independent contracts for the native-deskew yaw-stress experiment."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .lio_benchmark_core import (
    PoseSample,
    evaluate_trajectory,
    interpolate_pose_sample,
    wrap_angle,
)


REQUIRED_BAG_TOPICS = frozenset(
    {
        "/clock",
        "/odom",
        "/imu/data",
        "/lidar/points_raw",
        "/cmd_vel",
        "/joint_states",
        "/tf",
        "/tf_static",
    }
)


@dataclass(frozen=True)
class PhaseWindow:
    name: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class ReplayCell:
    profile: str
    seed: int
    backend: str
    point_density: float

    @property
    def case_name(self) -> str:
        density = f"{self.point_density * 100:.0f}"
        return (
            f"{self.profile}__seed_{self.seed}__{self.backend}"
            f"__uniform_density_{density}"
        )


def yaw_stress_phase_windows() -> tuple[PhaseWindow, ...]:
    return (
        PhaseWindow("warmup", 0.0, 5.0),
        PhaseWindow("ramp_up", 5.0, 7.0),
        PhaseWindow("hold", 7.0, 15.0),
        PhaseWindow("ramp_down", 15.0, 17.0),
        PhaseWindow("recovery", 17.0, 27.0),
    )


def yaw_stress_profile_name(direction: str, yaw_rate_rps: float) -> str:
    if direction not in {"left", "right"}:
        raise ValueError("direction must be left or right")
    if yaw_rate_rps not in {0.25, 0.5, 1.0, 1.5, 2.0}:
        raise ValueError("yaw rate is not in the v1 experiment grid")
    return f"yaw_stress_{direction}_{str(yaw_rate_rps).replace('.', '_')}"


def build_replay_cells(
    *,
    seeds: Iterable[int],
    backends: Iterable[str],
    point_densities: Iterable[float],
) -> list[ReplayCell]:
    profiles = [
        yaw_stress_profile_name(direction, rate)
        for rate in (0.25, 0.5, 1.0, 1.5, 2.0)
        for direction in ("left", "right")
    ]
    cells: list[ReplayCell] = []
    for seed in seeds:
        for profile in profiles:
            for density in point_densities:
                for backend in backends:
                    if backend not in {"liosam", "fastlio2"}:
                        raise ValueError(f"unsupported backend: {backend}")
                    if not math.isfinite(density) or not 0.0 < density <= 1.0:
                        raise ValueError("point density must be in (0, 1]")
                    cells.append(ReplayCell(profile, int(seed), backend, density))
    return cells


def counterbalanced_schedule(
    cells: Iterable[ReplayCell], *, schedule_seed: int = 20260804
) -> list[ReplayCell]:
    """Return a deterministic schedule while alternating paired backends."""
    grouped: dict[tuple[str, int, float], list[ReplayCell]] = {}
    for cell in cells:
        grouped.setdefault(
            (cell.profile, cell.seed, cell.point_density), []
        ).append(cell)
    keys = sorted(grouped)
    random.Random(schedule_seed).shuffle(keys)
    scheduled: list[ReplayCell] = []
    for index, key in enumerate(keys):
        pair = sorted(grouped[key], key=lambda cell: cell.backend)
        if len(pair) == 2 and index % 2:
            pair.reverse()
        scheduled.extend(pair)
    return scheduled


def build_blind_review_assignments(
    case_names: Iterable[str], *, review_seed: int = 20260804
) -> list[dict[str, str | int]]:
    ordered = sorted(set(case_names))
    random.Random(review_seed).shuffle(ordered)
    return [
        {
            "review_index": index,
            "blind_id": f"yaw_review_{index:04d}",
            "case_name": case_name,
        }
        for index, case_name in enumerate(ordered, start=1)
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bag_metadata(document: dict[str, Any]) -> list[str]:
    info = document.get("rosbag2_bagfile_information", document)
    topics = info.get("topics_with_message_count", [])
    names = {
        entry.get("topic_metadata", {}).get("name")
        for entry in topics
        if isinstance(entry, dict)
    }
    failures = [
        f"missing required topic {name}"
        for name in sorted(REQUIRED_BAG_TOPICS - names)
    ]
    duration = info.get("duration", {}).get("nanoseconds")
    if not isinstance(duration, int) or duration < 26_000_000_000:
        failures.append("bag duration must cover the 27 s experiment timeline")
    for entry in topics:
        metadata = entry.get("topic_metadata", {}) if isinstance(entry, dict) else {}
        if metadata.get("name") in REQUIRED_BAG_TOPICS and entry.get("message_count", 0) < 1:
            failures.append(f"required topic {metadata.get('name')} has no messages")
    return failures


def output_gap_metrics(
    stamps_s: Iterable[float],
    thresholds_s: tuple[float, ...] = (0.10, 0.25, 0.50),
) -> dict[str, Any]:
    stamps = np.asarray(tuple(stamps_s), dtype=np.float64)
    if stamps.size < 2:
        return {"sample_count": int(stamps.size), "max_gap_s": None}
    gaps = np.diff(stamps)
    positive = gaps[gaps > 0.0]
    median = float(np.median(positive)) if positive.size else None
    result: dict[str, Any] = {
        "sample_count": int(stamps.size),
        "median_period_s": median,
        "max_gap_s": float(gaps.max()),
        "thresholds": {},
    }
    comparison_tolerance_s = 1.0e-4
    for threshold in thresholds_s:
        selected = gaps[gaps > threshold + comparison_tolerance_s]
        result["thresholds"][f"{threshold:.2f}"] = {
            "count": int(selected.size),
            "total_excess_s": float(np.maximum(selected - threshold, 0.0).sum()),
        }
    if median is not None:
        selected = gaps[gaps > 3.0 * median]
        result["over_3x_median"] = {
            "count": int(selected.size),
            "threshold_s": 3.0 * median,
        }
    return result


def evaluate_yaw_stress_phases(
    ground_truth: list[PoseSample],
    estimate: list[PoseSample],
    *,
    experiment_start_s: float,
    ground_truth_sensor_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    ordered_truth = sorted(ground_truth, key=lambda sample: sample.stamp_s)
    ordered_estimate = sorted(estimate, key=lambda sample: sample.stamp_s)
    if len(ordered_truth) < 2 or len(ordered_estimate) < 2:
        return {"status": "insufficient_samples"}
    ordered_estimate = [
        sample
        for sample in ordered_estimate
        if ordered_truth[0].stamp_s <= sample.stamp_s <= ordered_truth[-1].stamp_s
    ]
    if len(ordered_estimate) < 2:
        return {"status": "timestamps_do_not_overlap"}
    first_estimate = ordered_estimate[0]
    first_truth = interpolate_pose_sample(ordered_truth, first_estimate.stamp_s)
    global_yaw_alignment = wrap_angle(first_truth.yaw - first_estimate.yaw)
    all_truth = [
        interpolate_pose_sample(ordered_truth, sample.stamp_s)
        for sample in ordered_estimate
    ]
    all_abs_yaw_errors = np.abs(
        np.asarray(
            [
                wrap_angle(sample.yaw + global_yaw_alignment - truth.yaw)
                for sample, truth in zip(ordered_estimate, all_truth, strict=True)
            ]
        )
    )
    for window in yaw_stress_phase_windows():
        start = experiment_start_s + window.start_s
        end = experiment_start_s + window.end_s
        truth = [sample for sample in ground_truth if start <= sample.stamp_s <= end]
        estimated = [sample for sample in estimate if start <= sample.stamp_s <= end]
        if len(truth) < 2 or len(estimated) < 2:
            phases[window.name] = {"status": "insufficient_samples"}
            continue
        metrics = evaluate_trajectory(
            truth,
            estimated,
            ground_truth_sensor_offset_xyz=ground_truth_sensor_offset_xyz,
        )
        truth_unwrapped = np.unwrap([sample.yaw for sample in truth])
        estimate_unwrapped = np.unwrap([sample.yaw for sample in estimated])
        metrics["ground_truth_cumulative_yaw_deg"] = math.degrees(
            float(truth_unwrapped[-1] - truth_unwrapped[0])
        )
        metrics["estimate_cumulative_yaw_deg"] = math.degrees(
            float(estimate_unwrapped[-1] - estimate_unwrapped[0])
        )
        metrics["cumulative_yaw_error_deg"] = (
            metrics["estimate_cumulative_yaw_deg"]
            - metrics["ground_truth_cumulative_yaw_deg"]
        )
        matched_truth = [
            interpolate_pose_sample(ordered_truth, sample.stamp_s)
            for sample in estimated
        ]
        globally_aligned_yaw_errors = np.asarray(
            [
                wrap_angle(
                    sample.yaw + global_yaw_alignment - truth_sample.yaw
                )
                for sample, truth_sample in zip(
                    estimated, matched_truth, strict=True
                )
            ],
            dtype=np.float64,
        )
        metrics["global_aligned_yaw_rmse_deg"] = math.degrees(
            float(np.sqrt(np.mean(np.square(globally_aligned_yaw_errors))))
        )
        metrics["global_aligned_yaw_error_max_deg"] = math.degrees(
            float(np.abs(globally_aligned_yaw_errors).max())
        )
        actual_yaw_rates = np.abs(
            np.asarray([sample.yaw_rate_rps for sample in truth], dtype=np.float64)
        )
        metrics["ground_truth_abs_yaw_rate_rps"] = {
            "p10": float(np.percentile(actual_yaw_rates, 10.0)),
            "median": float(np.median(actual_yaw_rates)),
            "p90": float(np.percentile(actual_yaw_rates, 90.0)),
        }
        phases[window.name] = {"status": "measured", "trajectory": metrics}
    warmup_errors = [
        error
        for sample, error in zip(ordered_estimate, all_abs_yaw_errors, strict=True)
        if experiment_start_s <= sample.stamp_s <= experiment_start_s + 5.0
    ]
    baseline_p95 = (
        float(np.percentile(warmup_errors, 95.0)) if warmup_errors else None
    )
    recovered_at_s = None
    if baseline_p95 is not None:
        recovery = [
            (sample.stamp_s, float(error))
            for sample, error in zip(
                ordered_estimate, all_abs_yaw_errors, strict=True
            )
            if experiment_start_s + 17.0 <= sample.stamp_s <= experiment_start_s + 27.0
        ]
        for index in range(max(0, len(recovery) - 2)):
            if all(error <= baseline_p95 for _, error in recovery[index:index + 3]):
                recovered_at_s = recovery[index][0] - (experiment_start_s + 17.0)
                break
    return {
        "phases": phases,
        "yaw_recovery": {
            "warmup_abs_yaw_error_p95_deg": (
                math.degrees(baseline_p95)
                if baseline_p95 is not None else None
            ),
            "three_sample_recovery_time_s": recovered_at_s,
            "not_recovered_within_window": recovered_at_s is None,
        },
    }
