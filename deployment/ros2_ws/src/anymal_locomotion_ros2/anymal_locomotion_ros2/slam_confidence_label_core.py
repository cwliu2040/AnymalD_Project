"""Offline-only evaluation-grid pose errors and short-horizon usability labels."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class OfflinePoseSample:
    stamp_ns: int
    x: float
    y: float
    z: float
    yaw: float
    roll: float = 0.0
    pitch: float = 0.0

    def __post_init__(self) -> None:
        if type(self.stamp_ns) is not int:
            raise TypeError("pose stamp_ns must be an integer")
        if not all(
            math.isfinite(value)
            for value in (
                self.x,
                self.y,
                self.z,
                self.yaw,
                self.roll,
                self.pitch,
            )
        ):
            raise ValueError("offline pose values must be finite")


@dataclass(frozen=True)
class EvaluationRecord:
    """Runtime-available source provenance at one logical evaluation tick."""

    evaluation_stamp_ns: int
    source_stamp_ns: int | None
    hard_failure: bool = False

    def __post_init__(self) -> None:
        if type(self.evaluation_stamp_ns) is not int:
            raise TypeError("evaluation_stamp_ns must be an integer")
        if self.source_stamp_ns is not None and type(self.source_stamp_ns) is not int:
            raise TypeError("source_stamp_ns must be an integer or None")
        if type(self.hard_failure) is not bool:
            raise TypeError("hard_failure must be a boolean")


@dataclass(frozen=True)
class OfflineLabelConfig:
    prediction_horizon_ns: int
    odometry_outage_ns: int
    pose_error_sustain_ns: int
    translation_error_m: float
    yaw_error_deg: float
    translation_jump_residual_m: float
    yaw_jump_residual_deg: float

    def __post_init__(self) -> None:
        for field_name in (
            "prediction_horizon_ns",
            "odometry_outage_ns",
            "pose_error_sustain_ns",
        ):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an integer")
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "translation_error_m",
            "yaw_error_deg",
            "translation_jump_residual_m",
            "yaw_jump_residual_deg",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")


@dataclass(frozen=True)
class OfflineLabelSample:
    evaluation_stamp_ns: int
    source_stamp_ns: int | None
    source_age_ns: int | None
    translation_error_m: float | None
    yaw_error_deg: float | None
    translation_jump_residual_m: float | None
    yaw_jump_residual_deg: float | None
    usable_next_horizon: bool | None
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class _SourceMetric:
    translation_error_m: float
    yaw_error_deg: float
    translation_jump_residual_m: float | None
    yaw_jump_residual_deg: float | None


def build_evaluation_grid(
    *,
    start_ns: int,
    end_ns: int,
    period_ns: int = 50_000_000,
) -> tuple[int, ...]:
    """Return epoch-aligned logical ticks covering ``[start_ns, end_ns]``."""

    for name, value in (
        ("start_ns", start_ns),
        ("end_ns", end_ns),
        ("period_ns", period_ns),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
    if period_ns <= 0:
        raise ValueError("period_ns must be positive")
    if end_ns < start_ns:
        raise ValueError("end_ns must not precede start_ns")
    first_ns = ((start_ns + period_ns - 1) // period_ns) * period_ns
    return tuple(range(first_ns, end_ns + 1, period_ns))


def generate_offline_labels(
    *,
    evaluations: list[EvaluationRecord],
    estimates: list[OfflinePoseSample],
    ground_truth: list[OfflinePoseSample],
    config: OfflineLabelConfig,
    ground_truth_sensor_offset_xyz: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
) -> list[OfflineLabelSample]:
    """Generate evaluation-anchored labels without runtime GT dependency."""

    _require_strictly_increasing(
        [sample.evaluation_stamp_ns for sample in evaluations],
        "evaluation",
    )
    _require_strictly_increasing(
        [sample.stamp_ns for sample in estimates],
        "estimate",
    )
    _require_strictly_increasing(
        [sample.stamp_ns for sample in ground_truth],
        "ground truth",
    )
    if not evaluations:
        return []
    if len(estimates) < 1 or len(ground_truth) < 2:
        raise ValueError("offline labels require estimate and ground-truth poses")
    offset = tuple(float(value) for value in ground_truth_sensor_offset_xyz)
    if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
        raise ValueError("ground-truth sensor offset must be finite XYZ")

    estimates_by_stamp = {sample.stamp_ns: sample for sample in estimates}
    source_stamps = []
    for evaluation in evaluations:
        stamp_ns = evaluation.source_stamp_ns
        if stamp_ns is not None and stamp_ns not in source_stamps:
            source_stamps.append(stamp_ns)
    matched = [
        (
            stamp_ns,
            estimates_by_stamp[stamp_ns],
            _interpolate_pose(ground_truth, stamp_ns),
        )
        for stamp_ns in source_stamps
        if stamp_ns in estimates_by_stamp
        and ground_truth[0].stamp_ns <= stamp_ns <= ground_truth[-1].stamp_ns
    ]
    if not matched:
        raise ValueError("no source pose overlaps estimate and ground truth")

    _, first_estimate, first_truth = matched[0]
    first_truth_position = _sensor_position(first_truth, offset)
    yaw_alignment = _wrap_angle(first_truth.yaw - first_estimate.yaw)
    cosine = math.cos(yaw_alignment)
    sine = math.sin(yaw_alignment)
    metrics: dict[int, _SourceMetric] = {}
    previous_aligned: tuple[float, float, float, float] | None = None
    previous_truth: tuple[float, float, float, float] | None = None

    for stamp_ns, estimate, truth in matched:
        dx = estimate.x - first_estimate.x
        dy = estimate.y - first_estimate.y
        aligned = (
            first_truth_position[0] + cosine * dx - sine * dy,
            first_truth_position[1] + sine * dx + cosine * dy,
            first_truth_position[2] + estimate.z - first_estimate.z,
            _wrap_angle(estimate.yaw + yaw_alignment),
        )
        truth_position = _sensor_position(truth, offset)
        truth_pose = (*truth_position, truth.yaw)
        translation_error = math.dist(aligned[:3], truth_pose[:3])
        yaw_error = abs(math.degrees(_wrap_angle(aligned[3] - truth.yaw)))
        translation_residual = None
        yaw_residual = None
        if previous_aligned is not None and previous_truth is not None:
            estimate_step = tuple(
                aligned[index] - previous_aligned[index]
                for index in range(3)
            )
            truth_step = tuple(
                truth_pose[index] - previous_truth[index]
                for index in range(3)
            )
            translation_residual = math.sqrt(
                sum(
                    (estimate_step[index] - truth_step[index]) ** 2
                    for index in range(3)
                )
            )
            estimate_yaw_step = _wrap_angle(
                aligned[3] - previous_aligned[3]
            )
            truth_yaw_step = _wrap_angle(
                truth_pose[3] - previous_truth[3]
            )
            yaw_residual = abs(
                math.degrees(
                    _wrap_angle(estimate_yaw_step - truth_yaw_step)
                )
            )
        metrics[stamp_ns] = _SourceMetric(
            translation_error,
            yaw_error,
            translation_residual,
            yaw_residual,
        )
        previous_aligned = aligned
        previous_truth = truth_pose

    base_reasons: list[set[str]] = []
    translation_high: list[bool] = []
    yaw_high: list[bool] = []
    last_source_stamp_ns: int | None = None
    last_source_advance_evaluation_ns: int | None = None
    for evaluation in evaluations:
        reasons: set[str] = set()
        metric = (
            metrics.get(evaluation.source_stamp_ns)
            if evaluation.source_stamp_ns is not None
            else None
        )
        age_ns = (
            evaluation.evaluation_stamp_ns - evaluation.source_stamp_ns
            if evaluation.source_stamp_ns is not None
            else None
        )
        if evaluation.hard_failure:
            reasons.add("backend_hard_failure")
        if metric is None or age_ns is None or age_ns < 0:
            reasons.add("pose_unavailable")
        else:
            if (
                last_source_stamp_ns is None
                or evaluation.source_stamp_ns != last_source_stamp_ns
            ):
                last_source_stamp_ns = evaluation.source_stamp_ns
                last_source_advance_evaluation_ns = (
                    evaluation.evaluation_stamp_ns
                )
            assert last_source_advance_evaluation_ns is not None
            if (
                evaluation.evaluation_stamp_ns
                - last_source_advance_evaluation_ns
                > config.odometry_outage_ns
            ):
                reasons.add("odometry_outage")
        if metric is not None:
            if (
                metric.translation_jump_residual_m is not None
                and metric.translation_jump_residual_m
                > config.translation_jump_residual_m
            ):
                reasons.add("translation_jump")
            if (
                metric.yaw_jump_residual_deg is not None
                and metric.yaw_jump_residual_deg
                > config.yaw_jump_residual_deg
            ):
                reasons.add("yaw_jump")
        base_reasons.append(reasons)
        translation_high.append(
            metric is not None
            and metric.translation_error_m > config.translation_error_m
        )
        yaw_high.append(
            metric is not None
            and metric.yaw_error_deg > config.yaw_error_deg
        )

    translation_sustained = _sustained_mask(
        evaluations,
        translation_high,
        config.pose_error_sustain_ns,
    )
    yaw_sustained = _sustained_mask(
        evaluations,
        yaw_high,
        config.pose_error_sustain_ns,
    )
    for index, sustained in enumerate(translation_sustained):
        if sustained:
            base_reasons[index].add("translation_error_sustained")
    for index, sustained in enumerate(yaw_sustained):
        if sustained:
            base_reasons[index].add("yaw_error_sustained")

    last_evaluation_ns = evaluations[-1].evaluation_stamp_ns
    results: list[OfflineLabelSample] = []
    for index, evaluation in enumerate(evaluations):
        horizon_end_ns = (
            evaluation.evaluation_stamp_ns + config.prediction_horizon_ns
        )
        usable: bool | None
        horizon_reasons: set[str] = set()
        if horizon_end_ns > last_evaluation_ns:
            usable = None
        else:
            for future_index in range(index, len(evaluations)):
                if (
                    evaluations[future_index].evaluation_stamp_ns
                    > horizon_end_ns
                ):
                    break
                horizon_reasons.update(base_reasons[future_index])
            usable = not horizon_reasons
        metric = (
            metrics.get(evaluation.source_stamp_ns)
            if evaluation.source_stamp_ns is not None
            else None
        )
        source_age_ns = (
            evaluation.evaluation_stamp_ns - evaluation.source_stamp_ns
            if evaluation.source_stamp_ns is not None
            else None
        )
        results.append(
            OfflineLabelSample(
                evaluation_stamp_ns=evaluation.evaluation_stamp_ns,
                source_stamp_ns=evaluation.source_stamp_ns,
                source_age_ns=source_age_ns,
                translation_error_m=(
                    metric.translation_error_m if metric is not None else None
                ),
                yaw_error_deg=(
                    metric.yaw_error_deg if metric is not None else None
                ),
                translation_jump_residual_m=(
                    metric.translation_jump_residual_m
                    if metric is not None
                    else None
                ),
                yaw_jump_residual_deg=(
                    metric.yaw_jump_residual_deg
                    if metric is not None
                    else None
                ),
                usable_next_horizon=usable,
                failure_reasons=tuple(sorted(horizon_reasons)),
            )
        )
    return results


def _require_strictly_increasing(values: list[int], name: str) -> None:
    if any(after <= before for before, after in zip(values, values[1:])):
        raise ValueError(f"{name} stamps must be strictly increasing")


def _interpolate_pose(
    samples: list[OfflinePoseSample],
    stamp_ns: int,
) -> OfflinePoseSample:
    if stamp_ns < samples[0].stamp_ns or stamp_ns > samples[-1].stamp_ns:
        raise ValueError("interpolation stamp is outside the pose range")
    upper = 0
    while upper < len(samples) and samples[upper].stamp_ns < stamp_ns:
        upper += 1
    if upper < len(samples) and samples[upper].stamp_ns == stamp_ns:
        return samples[upper]
    before = samples[upper - 1]
    after = samples[upper]
    ratio = (stamp_ns - before.stamp_ns) / (
        after.stamp_ns - before.stamp_ns
    )
    return OfflinePoseSample(
        stamp_ns=stamp_ns,
        x=before.x + ratio * (after.x - before.x),
        y=before.y + ratio * (after.y - before.y),
        z=before.z + ratio * (after.z - before.z),
        yaw=_wrap_angle(
            before.yaw + ratio * _wrap_angle(after.yaw - before.yaw)
        ),
        roll=before.roll + ratio * (after.roll - before.roll),
        pitch=before.pitch + ratio * (after.pitch - before.pitch),
    )


def _sensor_position(
    pose: OfflinePoseSample,
    offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    sr, cr = math.sin(pose.roll), math.cos(pose.roll)
    sp, cp = math.sin(pose.pitch), math.cos(pose.pitch)
    sy, cy = math.sin(pose.yaw), math.cos(pose.yaw)
    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    rotated = tuple(
        sum(rotation[row][column] * offset[column] for column in range(3))
        for row in range(3)
    )
    return (
        pose.x + rotated[0],
        pose.y + rotated[1],
        pose.z + rotated[2],
    )


def _sustained_mask(
    evaluations: list[EvaluationRecord],
    high: list[bool],
    sustain_ns: int,
) -> list[bool]:
    mask = [False] * len(evaluations)
    run_start: int | None = None
    for index in range(len(evaluations) + 1):
        is_high = index < len(evaluations) and high[index]
        if is_high and run_start is None:
            run_start = index
        if not is_high and run_start is not None:
            run_end = index - 1
            duration_ns = (
                evaluations[run_end].evaluation_stamp_ns
                - evaluations[run_start].evaluation_stamp_ns
            )
            if duration_ns >= sustain_ns:
                for run_index in range(run_start, run_end + 1):
                    mask[run_index] = True
            run_start = None
    return mask
