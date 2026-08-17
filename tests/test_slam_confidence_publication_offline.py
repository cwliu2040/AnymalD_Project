from __future__ import annotations

import importlib.util
from pathlib import Path

from anymal_locomotion_ros2.slam_confidence_label_core import OfflineLabelSample


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/evaluate_slam_confidence_publication_run.py"


def _module():
    spec = importlib.util.spec_from_file_location("publication_offline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _label(stamp_ns: int, usable: bool | None) -> OfflineLabelSample:
    return OfflineLabelSample(
        stamp_ns, stamp_ns, 0, 0.0, 0.0, 0.0, 0.0, usable, (),
    )


def _record(clock_s: float, confidence: float, valid: float) -> dict:
    return {
        "clock_s": clock_s,
        "received_command": [1.0, 0.0, 0.0],
        "observation": [0.0] * 48 + [confidence, valid, 0.0],
    }


def test_false_stop_uses_future_offline_usability_not_current_tracking() -> None:
    module = _module()
    result = module.evaluate_false_stops(
        labels=[_label(0, True), _label(50_000_000, False), _label(100_000_000, None)],
        policy_records=[
            _record(0.01, 0.0, 0.0),
            _record(0.06, 0.0, 0.0),
            _record(0.11, 0.0, 0.0),
        ],
        arm="D",
    )
    assert result["confidence_stop_known_label_count"] == 2
    assert result["false_stop_count"] == 1
    assert result["false_stop_fraction"] == 0.5


def test_arm_a_never_attributes_a_confidence_stop() -> None:
    module = _module()
    record = _record(0.01, 0.0, 0.0)
    record["observation"] = record["observation"][:48]
    result = module.evaluate_false_stops(
        labels=[_label(0, True)], policy_records=[record], arm="A",
    )
    assert result["confidence_stop_known_label_count"] == 0
    assert result["false_stop_fraction"] is None


def test_label_config_reuses_frozen_confidence_contract() -> None:
    module = _module()
    import yaml
    contract = yaml.safe_load((ROOT / "configs/slam_confidence_contract.yaml").read_text())
    config = module.label_config(contract)
    assert config.prediction_horizon_ns == 500_000_000
    assert config.translation_error_m == 0.10


def test_tracking_survival_detects_sustained_loss() -> None:
    module = _module()
    confidence = [
        {"evaluation_stamp_ns": stamp, "tracking_valid": valid}
        for stamp, valid in (
            (0, True), (50_000_000, True), (100_000_000, False),
            (200_000_000, False), (350_000_000, False), (400_000_000, True),
        )
    ]
    result = module.evaluate_tracking_survival(confidence)
    assert result["event_observed"]
    assert result["survival_time_s"] == 0.1


def test_tracking_survival_is_right_censored_without_sustained_loss() -> None:
    module = _module()
    confidence = [
        {"evaluation_stamp_ns": stamp, "tracking_valid": valid}
        for stamp, valid in (
            (0, True), (100_000_000, False), (200_000_000, True), (300_000_000, True),
        )
    ]
    result = module.evaluate_tracking_survival(confidence)
    assert not result["event_observed"]
    assert result["survival_time_s"] == 0.3
