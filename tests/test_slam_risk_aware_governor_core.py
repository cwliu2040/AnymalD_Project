from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = (
    ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2/anymal_locomotion_ros2"
    / "risk_aware_governor_core.py"
)
spec = importlib.util.spec_from_file_location("risk_aware_governor_core", CORE_PATH)
assert spec and spec.loader
CORE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = CORE
spec.loader.exec_module(CORE)


def _decision(**overrides):
    values = {
        "confidence": 0.45,
        "tracking_valid": True,
        "confidence_age_s": 0.1,
        "receipt_age_s": 0.05,
        "prediction_available": True,
        "selected_scale": 0.75,
        "predicted_unusable_probability": 0.08,
    }
    values.update(overrides)
    return CORE.arbitrate_command_scale(**values)


def test_valid_risk_path_replaces_B_soft_throttle() -> None:
    decision = _decision()
    assert decision.mode == CORE.GovernorMode.RISK_AWARE
    assert decision.legacy_b_scale == 0.3125
    assert decision.command_scale == 0.75
    assert decision.command_scale > decision.legacy_b_scale


def test_invalid_and_stale_high_slam_are_exact_hard_stops() -> None:
    for overrides in (
        {"tracking_valid": False, "confidence": 0.99},
        {"confidence_age_s": 0.500001, "confidence": 0.99},
        {"receipt_age_s": 0.150001, "confidence": 0.99},
        {"confidence": float("nan")},
    ):
        decision = _decision(**overrides)
        assert decision.mode == CORE.GovernorMode.HARD_STOP
        assert decision.command_scale == 0.0


def test_missing_or_invalid_risk_prediction_falls_back_to_exact_B() -> None:
    expected = CORE.legacy_b_scale(0.45, True)
    missing = _decision(prediction_available=False)
    invalid = _decision(selected_scale=float("nan"))
    out_of_range = _decision(predicted_unusable_probability=1.1)
    for decision in (missing, invalid, out_of_range):
        assert decision.mode == CORE.GovernorMode.LEGACY_B_FALLBACK
        assert decision.command_scale == expected


def test_unsafe_prediction_stops_instead_of_silently_using_B() -> None:
    decision = _decision(predicted_unusable_probability=0.100001)
    assert decision.mode == CORE.GovernorMode.RISK_STOP
    assert decision.command_scale == 0.0
    assert decision.legacy_b_scale == 0.3125


def test_phase1_config_matches_core_and_forbids_live_use() -> None:
    document = yaml.safe_load(
        (ROOT / "configs/slam_risk_aware_governor_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    cfg = CORE.RiskAwareGovernorConfig()
    assert document["arm_B_parity"]["confidence_floor"] == cfg.confidence_floor
    assert document["arm_B_parity"]["confidence_span"] == cfg.confidence_span
    assert document["freshness"]["maximum_confidence_age_s"] == cfg.maximum_confidence_age_s
    assert document["freshness"]["maximum_receipt_age_s"] == cfg.maximum_receipt_age_s
    assert document["boundaries"]["live_execution_authorized"] is False
    assert document["boundaries"]["candidate_risk_model_present"] is False
    assert "rclpy" not in CORE_PATH.read_text(encoding="utf-8")
