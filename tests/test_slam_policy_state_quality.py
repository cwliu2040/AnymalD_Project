"""Tests for the offline LIO-SAM policy-state quality gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "evaluate_slam_policy_state_quality.py"
SPEC = importlib.util.spec_from_file_location("policy_state_quality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "liosam_policy_state_quality.yaml").read_text()
    )


def _diagnostics(*, error: float = 0.0, age: float = 0.01) -> tuple[dict, dict]:
    records = []
    samples = []
    for index in range(400):
        time_s = index * 0.02
        truth = [1.0, -0.2, 0.05]
        observation = [truth[0] + error, truth[1], truth[2]] + [0.0] * 45
        records.append(
            {
                "clock_s": time_s,
                "observation": observation,
                "state_stamps_s": {"odometry": time_s - age},
            }
        )
        samples.append(
            {
                "time_s": time_s,
                "actual_linear_velocity_body_mps": truth,
            }
        )
    return {"records": records}, {"samples": samples}


def test_quality_gate_passes_fresh_accurate_body_velocity() -> None:
    policy, locomotion = _diagnostics()
    result = MODULE.evaluate_policy_state_quality(policy, locomotion, _config())
    assert result["passed"] is True
    assert result["boundaries"]["runtime_ground_truth_used"] is False
    assert result["metrics"]["source_update_rate_hz"] > 30.0


def test_quality_gate_rejects_large_velocity_error() -> None:
    policy, locomotion = _diagnostics(error=0.8)
    result = MODULE.evaluate_policy_state_quality(policy, locomotion, _config())
    assert result["passed"] is False
    assert any("velocity_error_mae_mps.x" in item for item in result["failures"])


def test_quality_gate_rejects_stale_policy_state() -> None:
    policy, locomotion = _diagnostics(age=0.3)
    result = MODULE.evaluate_policy_state_quality(policy, locomotion, _config())
    assert result["passed"] is False
    assert any("state_age_s.p95" in item for item in result["failures"])


def test_quality_gate_fails_closed_without_offline_truth_field() -> None:
    policy, locomotion = _diagnostics()
    for sample in locomotion["samples"]:
        sample.pop("actual_linear_velocity_body_mps")
    result = MODULE.evaluate_policy_state_quality(policy, locomotion, _config())
    assert result["passed"] is False
    assert result["metrics"]["matched_sample_count"] == 0


def test_contract_separates_mapping_authority_from_policy_state() -> None:
    config = _config()
    runtime = config["runtime"]
    assert runtime["mapping_confidence_authority_topic"] == "/slam/odom"
    assert runtime["policy_odometry_topic"] == "/slam/policy_odom"
    assert runtime["runtime_ground_truth_input"] is False
    assert runtime["native_deskew_required"] is True
