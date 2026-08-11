from __future__ import annotations

import numpy as np

from anymal_locomotion_ros2.slam_confidence_calibration_core import (
    CausalFeatureTransform,
    FEATURE_SCHEMAS,
    event_metrics,
    feature_vector,
    fit_isotonic,
    fit_logistic,
    frame_metrics,
    isotonic_predict,
    logistic_predict,
    stable_capture_split,
)


def test_split_is_deterministic_and_keeps_groups_whole() -> None:
    groups = [f"capture-{index}" for index in range(10)]
    assert stable_capture_split(groups) == stable_capture_split(list(reversed(groups)))
    assert set(stable_capture_split(groups).values()) == {
        "train", "calibration", "threshold_validation", "final_holdout"
    }


def test_backend_features_are_not_directly_interchangeable() -> None:
    fast = feature_vector(
        "fastlio2",
        {
            "source_stamp_ns": 1,
            "source_age_s": 0.1,
            "features": {"effective_features": 100},
        },
    )
    lio = feature_vector(
        "liosam",
        {
            "source_age_s": 0.1,
            "source_stamp_ns": 1,
            "features": {
                "corner_features": 20,
                "surface_features": 200,
                "degenerate": True,
                "odom_available": False,
            },
        },
    )
    assert len(fast) == len(FEATURE_SCHEMAS["fastlio2"])
    assert len(lio) == len(FEATURE_SCHEMAS["liosam"])
    assert len(fast) != len(lio)


def test_causal_features_update_once_per_source_and_remember_degradation() -> None:
    transform = CausalFeatureTransform("fastlio2")
    first = transform.transform(
        {
            "source_stamp_ns": 1_000_000_000,
            "source_age_s": 0.01,
            "features": {"effective_features": 1000},
        }
    )
    repeated = transform.transform(
        {
            "source_stamp_ns": 1_000_000_000,
            "source_age_s": 0.06,
            "features": {"effective_features": 1000},
        }
    )
    degraded = transform.transform(
        {
            "source_stamp_ns": 1_100_000_000,
            "source_age_s": 0.01,
            "features": {"effective_features": 100},
        }
    )
    recovered = transform.transform(
        {
            "source_stamp_ns": 1_200_000_000,
            "source_age_s": 0.01,
            "features": {"effective_features": 1000},
        }
    )
    assert repeated == first
    assert degraded[1] < 0.2
    assert recovered[4] < 0.2


def test_logistic_and_isotonic_are_deterministic_and_bounded() -> None:
    features = np.asarray([[0.0], [0.2], [0.8], [1.0]])
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    model = fit_logistic(features, labels, iterations=500)
    raw = logistic_predict(model, features)
    calibrated = isotonic_predict(fit_isotonic(raw, labels), raw)
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))
    assert list(calibrated) == sorted(calibrated)


def test_metrics_separate_abrupt_faults_from_gradual_events() -> None:
    rows = [
        {
            "capture_group": "bag-a",
            "evaluation_stamp_ns": index * 50_000_000,
            "usable_next_0_5s": not (4 <= index <= 7 or index == 10),
            "abrupt_hard_fault": index == 10,
        }
        for index in range(12)
    ]
    confidence = np.asarray([0.9, 0.9, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.9, 0.9, 0.0, 0.9])
    metrics = event_metrics(rows, confidence, 0.45)
    assert metrics["gradual_event_count"] == 1
    assert metrics["independent_gradual_capture_count"] == 1
    assert metrics["low_confidence_event_recall"] == 1.0
    assert metrics["median_lead_time_s"] == 0.1


def test_frame_metrics_reports_probability_quality() -> None:
    metrics = frame_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]))
    assert metrics["auroc"] == 1.0
    assert metrics["brier"] < 0.1
    assert metrics["ece"] >= 0.0
    assert metrics["reliability_bins"]


def test_frame_metrics_reports_deterministic_cluster_bootstrap() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 1])
    confidence = np.asarray([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    clusters = ["bag-a", "bag-a", "bag-b", "bag-b", "bag-c", "bag-c"]
    first = frame_metrics(
        labels,
        confidence,
        clusters=clusters,
        bootstrap_iterations=100,
    )
    second = frame_metrics(
        labels,
        confidence,
        clusters=clusters,
        bootstrap_iterations=100,
    )
    assert first["cluster_bootstrap_95_ci"] == second["cluster_bootstrap_95_ci"]
    assert first["cluster_bootstrap_95_ci"]["cluster_count"] == 3
    assert first["cluster_bootstrap_95_ci"]["brier"]["valid_draws"] == 100
