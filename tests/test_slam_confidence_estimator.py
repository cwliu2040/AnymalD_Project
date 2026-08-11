from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from anymal_locomotion_ros2.slam_confidence_calibration_core import (
    FEATURE_SCHEMAS,
    FEATURE_TRANSFORM_CONFIG,
    SUPPORT_FORECAST_GUARDS,
    calibration_fingerprint,
)
from anymal_locomotion_ros2.slam_confidence_estimator import (
    ARTIFACT_PROVENANCE_SCHEMA_VERSION,
    CalibratedConfidenceEstimator,
    CausalPoseMotion,
    RUNTIME_PROVENANCE_EXPECTATIONS,
)


def _artifact() -> dict:
    feature_count = len(FEATURE_SCHEMAS["fastlio2"])
    payload = {
        "schema_version": 1,
        "artifact_provenance_schema_version": ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "backend_id": "fastlio2",
        "calibration_id": "native-v1-abcdef123456",
        "deskew_mode": "native",
        "feature_schema": list(FEATURE_SCHEMAS["fastlio2"]),
        "feature_transform": FEATURE_TRANSFORM_CONFIG,
        "support_forecast_guard": SUPPORT_FORECAST_GUARDS["fastlio2"],
        "logistic": {
            "mean": [0.0] * feature_count,
            "scale": [1.0] * feature_count,
            "weights": [0.0] * (feature_count + 1),
        },
        "isotonic": {
            "upper_probability": [0.5],
            "calibrated_value": [0.75],
        },
        "thresholds": {
            "degrade_below": 0.45,
            "invalidate_below": 0.25,
            "recover_at_or_above": 0.55,
        },
        "provenance": {
            "report_fingerprint_sha256": "abcdef123456" + "0" * 52,
            "runtime_contract": RUNTIME_PROVENANCE_EXPECTATIONS["fastlio2"],
        },
    }
    payload["artifact_fingerprint_sha256"] = calibration_fingerprint(payload)
    return payload


def test_estimator_validates_artifact_and_returns_calibrated_score(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    estimator = CalibratedConfidenceEstimator(str(path), backend_id="fastlio2")
    assert estimator.observe(
        {
            "source_stamp_ns": 1,
            "source_age_s": 0.01,
            "features": {"effective_features": 100},
        }
    ) == 0.75
    assert estimator.calibration_id == "native-v1-abcdef123456"


def test_estimator_can_advance_temporal_state_while_holding_wire_score(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    estimator = CalibratedConfidenceEstimator(str(path), backend_id="fastlio2")
    first = estimator.observe(
        {
            "source_stamp_ns": 1,
            "source_age_s": 0.01,
            "features": {"effective_features": 100},
        }
    )
    held = estimator.observe(
        {
            "source_stamp_ns": 2,
            "source_age_s": 0.01,
            "features": {"effective_features": 0},
        },
        update_confidence=False,
    )
    assert held == first
    assert estimator.confidence == first


def test_estimator_rejects_runtime_provenance_mismatch(tmp_path) -> None:
    payload = _artifact()
    payload["provenance"]["runtime_contract"] = {
        **payload["provenance"]["runtime_contract"],
        "timestamp_contract": "wrong",
    }
    unsigned = dict(payload)
    unsigned.pop("artifact_fingerprint_sha256")
    payload["artifact_fingerprint_sha256"] = calibration_fingerprint(unsigned)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime provenance"):
        CalibratedConfidenceEstimator(str(path), backend_id="fastlio2")


def test_pose_motion_is_causal_and_handles_wrapped_yaw() -> None:
    motion = CausalPoseMotion()
    first = (0.0, 0.0, 0.0, 0.0, 0.0, math.sin(1.55), math.cos(1.55))
    second = (1.0, 0.0, 0.0, 0.0, 0.0, math.sin(-1.55), math.cos(-1.55))
    assert motion.observe(1_000_000_000, first)["linear_speed_mps"] == 0.0
    observed = motion.observe(2_000_000_000, second)
    assert observed["linear_speed_mps"] == 1.0
    assert observed["yaw_rate_deg_s"] < 10.0


def test_installed_native_backend_artifacts_match_runtime_schema() -> None:
    config_dir = (
        Path(__file__).resolve().parents[1]
        / "deployment"
        / "ros2_ws"
        / "src"
        / "anymal_locomotion_ros2"
        / "config"
    )
    expected = {
        "fastlio2": (
            "slam_confidence_fastlio2_native_v2.json",
            "native-v1-1e6cf8347be1",
        ),
        "liosam": (
            "slam_confidence_liosam_native_v3.json",
            "native-v1-edc098b0bd98",
        ),
    }
    for backend_id, (filename, calibration_id) in expected.items():
        artifact_path = config_dir / filename
        estimator = CalibratedConfidenceEstimator(
            str(artifact_path), backend_id=backend_id
        )
        assert estimator.calibration_id == calibration_id
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        unsigned = dict(payload)
        fingerprint = unsigned.pop("artifact_fingerprint_sha256")
        assert calibration_fingerprint(unsigned) == fingerprint

    summary_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "validation"
        / "slam_confidence_final_holdout_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary["backends"]) == set(expected)
    for backend_id, (filename, calibration_id) in expected.items():
        evidence = summary["backends"][backend_id]
        assert evidence["calibration_id"] == calibration_id
        artifact = json.loads((config_dir / filename).read_text(encoding="utf-8"))
        assert (
            evidence["artifact_fingerprint_sha256"]
            == artifact["artifact_fingerprint_sha256"]
        )
