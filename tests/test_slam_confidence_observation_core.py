from __future__ import annotations

from pathlib import Path

from anymal_locomotion_ros2.slam_confidence_estimator import (
    CalibratedConfidenceEstimator,
)
from anymal_locomotion_ros2.slam_confidence_observation_core import (
    confidence_identity_valid,
    ppo_confidence_observation,
)


def test_ppo_observation_is_normalized_but_not_connected_to_policy() -> None:
    observation = ppo_confidence_observation(
        confidence=0.8,
        tracking_valid=True,
        source_stamp_valid=True,
        confidence_age_s=0.25,
        receipt_age_s=0.01,
    )
    assert observation.slam_confidence == 0.8
    assert observation.slam_tracking_valid == 1.0
    assert observation.confidence_age_normalized == 0.5
    assert observation.receipt_watchdog_valid


def test_receipt_watchdog_fail_closes_stale_high_confidence() -> None:
    observation = ppo_confidence_observation(
        confidence=0.99,
        tracking_valid=True,
        source_stamp_valid=True,
        confidence_age_s=0.01,
        receipt_age_s=0.16,
    )
    assert observation.slam_confidence == 0.0
    assert observation.slam_tracking_valid == 0.0
    assert observation.confidence_age_normalized == 1.0
    assert not observation.receipt_watchdog_valid


def test_invalid_source_stamp_fail_closes_the_complete_vector() -> None:
    observation = ppo_confidence_observation(
        confidence=0.99,
        tracking_valid=True,
        source_stamp_valid=False,
        confidence_age_s=0.01,
        receipt_age_s=0.01,
    )
    assert observation.slam_confidence == 0.0
    assert observation.slam_tracking_valid == 0.0
    assert observation.confidence_age_normalized == 1.0
    assert observation.receipt_watchdog_valid


def test_confidence_identity_requires_exact_selected_calibration() -> None:
    values = {
        "schema_version": 1,
        "expected_schema_version": 1,
        "backend_id": "fastlio2",
        "calibration_id": "native-v1-1e6cf8347be1",
        "expected_backend_id": "fastlio2",
        "expected_calibration_id": "native-v1-1e6cf8347be1",
    }
    assert confidence_identity_valid(**values)
    assert not confidence_identity_valid(
        **{**values, "calibration_id": "native-v1-wrong"}
    )
    assert not confidence_identity_valid(
        **{**values, "backend_id": "liosam"}
    )
    assert not confidence_identity_valid(
        **{**values, "schema_version": 2}
    )


def test_both_backend_artifacts_feed_the_same_ppo_ready_adapter() -> None:
    config_dir = (
        Path(__file__).resolve().parents[1]
        / "deployment/ros2_ws/src/anymal_locomotion_ros2/config"
    )
    cases = (
        (
            "fastlio2",
            "slam_confidence_fastlio2_native_v2.json",
            {"effective_features": 5000},
        ),
        (
            "liosam",
            "slam_confidence_liosam_native_v3.json",
            {
                "corner_features": 500,
                "surface_features": 8000,
                "linear_speed_mps": 0.1,
                "yaw_rate_deg_s": 1.0,
                "degenerate": False,
                "odom_available": True,
            },
        ),
    )
    for backend_id, filename, features in cases:
        estimator = CalibratedConfidenceEstimator(
            str(config_dir / filename), backend_id=backend_id
        )
        confidence = estimator.observe(
            {
                "source_stamp_ns": 1_000_000_000,
                "source_age_s": 0.02,
                "features": features,
            }
        )
        observation = ppo_confidence_observation(
            confidence=confidence,
            tracking_valid=True,
            source_stamp_valid=True,
            confidence_age_s=0.02,
            receipt_age_s=0.01,
        )
        assert 0.0 <= observation.slam_confidence <= 1.0
        assert observation.slam_tracking_valid == 1.0
        assert observation.confidence_age_normalized == 0.04
        assert observation.receipt_watchdog_valid
