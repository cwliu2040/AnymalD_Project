"""Tests for offline-only SLAM confidence target generation."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from anymal_locomotion_ros2.slam_confidence_label_core import (
    EvaluationRecord,
    OfflineLabelConfig,
    OfflinePoseSample,
    build_evaluation_grid,
    generate_offline_labels,
)


def _config() -> OfflineLabelConfig:
    return OfflineLabelConfig(
        prediction_horizon_ns=500_000_000,
        odometry_outage_ns=300_000_000,
        pose_error_sustain_ns=200_000_000,
        translation_error_m=0.10,
        yaw_error_deg=1.0,
        translation_jump_residual_m=0.20,
        yaw_jump_residual_deg=2.0,
    )


def test_label_config_matches_machine_readable_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    target = yaml.safe_load(
        (project_root / "configs/slam_confidence_contract.yaml").read_text(
            encoding="utf-8"
        )
    )["offline_target"]
    config = _config()

    assert config.prediction_horizon_ns == round(
        target["prediction_horizon_s"] * 1.0e9
    )
    assert config.odometry_outage_ns == round(
        target["odometry_outage_s"] * 1.0e9
    )
    assert config.pose_error_sustain_ns == round(
        target["pose_error_sustain_s"] * 1.0e9
    )
    assert config.translation_error_m == target["translation_error_m"]
    assert config.yaw_error_deg == target["yaw_error_deg"]
    assert config.translation_jump_residual_m == target[
        "translation_jump_residual_m"
    ]
    assert config.yaw_jump_residual_deg == target[
        "yaw_jump_residual_deg"
    ]


def _straight_poses(
    *,
    estimate_bias_after_ns: int | None = None,
    bias_m: float = 0.0,
) -> tuple[list[OfflinePoseSample], list[OfflinePoseSample]]:
    truth = []
    estimate = []
    for stamp_ns in range(0, 2_000_000_001, 50_000_000):
        x = stamp_ns * 1.0e-9
        truth.append(OfflinePoseSample(stamp_ns, x, 0.0, 0.0, 0.0))
        bias = (
            bias_m
            if estimate_bias_after_ns is not None
            and stamp_ns >= estimate_bias_after_ns
            else 0.0
        )
        estimate.append(
            OfflinePoseSample(stamp_ns, x + bias, 0.0, 0.0, 0.0)
        )
    return truth, estimate


def _evaluations(
    *,
    start_ns: int = 0,
    end_ns: int = 2_000_000_000,
) -> list[EvaluationRecord]:
    return [
        EvaluationRecord(stamp_ns, stamp_ns)
        for stamp_ns in build_evaluation_grid(
            start_ns=start_ns,
            end_ns=end_ns,
        )
    ]


def test_grid_is_epoch_aligned_and_inclusive() -> None:
    assert build_evaluation_grid(
        start_ns=1,
        end_ns=150_000_000,
    ) == (50_000_000, 100_000_000, 150_000_000)


def test_healthy_trajectory_is_usable_until_incomplete_horizon_tail() -> None:
    truth, estimate = _straight_poses()
    labels = generate_offline_labels(
        evaluations=_evaluations(),
        estimates=estimate,
        ground_truth=truth,
        config=_config(),
    )

    assert labels[0].translation_error_m == 0.0
    assert labels[0].usable_next_horizon is True
    assert labels[-1].usable_next_horizon is None


def test_initial_se2_alignment_does_not_fit_later_drift() -> None:
    truth, estimate = _straight_poses(
        estimate_bias_after_ns=500_000_000,
        bias_m=0.15,
    )
    labels = generate_offline_labels(
        evaluations=_evaluations(),
        estimates=estimate,
        ground_truth=truth,
        config=_config(),
    )

    drifted = next(
        sample for sample in labels if sample.evaluation_stamp_ns == 600_000_000
    )
    assert math.isclose(drifted.translation_error_m or 0.0, 0.15)


def test_sustained_translation_error_marks_forecast_unusable() -> None:
    truth, estimate = _straight_poses(
        estimate_bias_after_ns=1_000_000_000,
        bias_m=0.15,
    )
    labels = generate_offline_labels(
        evaluations=_evaluations(),
        estimates=estimate,
        ground_truth=truth,
        config=_config(),
    )

    before = next(
        sample for sample in labels if sample.evaluation_stamp_ns == 600_000_000
    )
    assert before.usable_next_horizon is False
    assert "translation_error_sustained" in before.failure_reasons


def test_short_error_transient_does_not_trigger_sustained_label() -> None:
    truth, estimate = _straight_poses()
    estimate = [
        OfflinePoseSample(
            sample.stamp_ns,
            sample.x
            + (0.15 if 1_000_000_000 <= sample.stamp_ns < 1_150_000_000 else 0.0),
            sample.y,
            sample.z,
            sample.yaw,
        )
        for sample in estimate
    ]
    labels = generate_offline_labels(
        evaluations=_evaluations(),
        estimates=estimate,
        ground_truth=truth,
        config=_config(),
    )

    relevant = next(
        sample for sample in labels if sample.evaluation_stamp_ns == 900_000_000
    )
    assert "translation_error_sustained" not in relevant.failure_reasons


def test_jump_and_source_outage_are_horizon_failures() -> None:
    truth, estimate = _straight_poses()
    estimate = [
        OfflinePoseSample(
            sample.stamp_ns,
            sample.x + (0.25 if sample.stamp_ns == 1_000_000_000 else 0.0),
            sample.y,
            sample.z,
            sample.yaw,
        )
        for sample in estimate
    ]
    evaluations = _evaluations()
    for index in range(23, 31):
        evaluations[index] = EvaluationRecord(
            evaluations[index].evaluation_stamp_ns,
            1_100_000_000,
        )
    labels = generate_offline_labels(
        evaluations=evaluations,
        estimates=estimate,
        ground_truth=truth,
        config=_config(),
    )

    jump_forecast = next(
        sample for sample in labels if sample.evaluation_stamp_ns == 600_000_000
    )
    outage_forecast = next(
        sample for sample in labels if sample.evaluation_stamp_ns == 1_200_000_000
    )
    assert "translation_jump" in jump_forecast.failure_reasons
    assert "odometry_outage" in outage_forecast.failure_reasons


def test_fixed_pipeline_latency_is_not_an_odometry_outage() -> None:
    truth, estimate = _straight_poses()
    evaluations = [
        EvaluationRecord(stamp_ns, max(0, stamp_ns - 350_000_000))
        for stamp_ns in build_evaluation_grid(
            start_ns=350_000_000,
            end_ns=1_500_000_000,
        )
    ]
    labels = generate_offline_labels(
        evaluations=evaluations,
        estimates=estimate,
        ground_truth=truth,
        config=_config(),
    )
    assert all(
        "odometry_outage" not in sample.failure_reasons for sample in labels
    )


def test_yaw_interpolation_uses_wrapped_short_path() -> None:
    truth = [
        OfflinePoseSample(0, 0.0, 0.0, 0.0, math.radians(179.0)),
        OfflinePoseSample(
            1_000_000_000,
            1.0,
            0.0,
            0.0,
            math.radians(-179.0),
        ),
    ]
    estimate = [
        OfflinePoseSample(500_000_000, 0.5, 0.0, 0.0, math.pi),
    ]
    evaluations = [EvaluationRecord(500_000_000, 500_000_000)]
    labels = generate_offline_labels(
        evaluations=evaluations,
        estimates=estimate,
        ground_truth=truth,
        config=_config(),
    )

    assert (labels[0].yaw_error_deg or 0.0) < 1.0e-9
