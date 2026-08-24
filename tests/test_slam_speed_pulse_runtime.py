from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))
from anymal_locomotion_ros2.lio_benchmark_core import get_motion_profile


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load("pulse_runner", "scripts/validation/run_slam_speed_pulse_causal_pilot.py")
ANALYZER = _load("pulse_analyzer", "scripts/validation/analyze_slam_speed_pulse_causal_pilot.py")


def _yaml(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text())


def test_release_is_hash_locked_and_live_closed_after_attempt1() -> None:
    release = _yaml("configs/slam_speed_pulse_causal_release_v2.yaml")
    protocol_path = ROOT / release["protocol"]["path"]
    RUNNER.validate_release(release, protocol_path)
    assert release["boundaries"]["live_execution_authorized"] is False
    assert release["boundaries"]["authorized_stages"] == []
    with pytest.raises(ValueError, match="live execution is not authorized"):
        RUNNER.require_execution_authorization(release, "wiring_smoke")


@pytest.mark.parametrize("arm", ("scale_100", "scale_075", "scale_050", "scale_025"))
def test_synthetic_trace_verifies_prefix_pulse_and_recovery(arm: str) -> None:
    onnx = pytest.importorskip("onnx")
    ReferenceEvaluator = pytest.importorskip("onnx.reference").ReferenceEvaluator
    trace_module = _load("pulse_trace", "scripts/validation/validate_slam_speed_pulse_trace.py")
    protocol = _yaml("configs/slam_speed_pulse_causal_pilot_v2.yaml")
    release = _yaml("configs/slam_speed_pulse_causal_release_v2.yaml")
    scale = protocol["treatment"]["arms"][arm]
    origin = 1.0
    elapsed_values = [5.5, 7.16, 7.32, 7.5, 7.9, 8.1, 8.3]
    profile = get_motion_profile("curve_1_5_right_1_0")
    rng = np.random.default_rng(20260822)
    observations = rng.normal(size=(len(elapsed_values), 51)).astype(np.float32)
    observations[:, 48:51] = np.asarray([0.7, 1.0, 0.1], dtype=np.float32)
    received = []
    for index, elapsed in enumerate(elapsed_values):
        base = np.asarray(profile.command_at(elapsed), dtype=np.float32)
        assigned = scale if 7.25 <= elapsed < 8.0 else 1.0
        command = assigned * base
        received.append(command)
        observations[index, 9:12] = command
    evaluator = ReferenceEvaluator(onnx.load(str(ROOT / release["policy"]["path"])))
    actions = evaluator.run(None, {"observation": observations})[0]
    diagnostics = {"records": [
        {"clock_s": origin + elapsed, "received_command": command.tolist(), "effective_command": command.tolist(), "observation": observation.tolist(), "raw_action": action.tolist()}
        for elapsed, command, observation, action in zip(elapsed_values, received, observations, actions)
    ]}
    driver = {"profile": profile.name, "profile_start_clock_s": origin, "command_scale_pulse": {"enabled": True, "start_s": 7.25, "duration_s": 0.75, "scale": scale, "publish_count": 30}}
    report = trace_module.validate_trace(diagnostics, driver, protocol, release, arm)
    assert report["passed"], report
    assert all(value["passed"] for value in report["phases"].values())


def test_pre_pulse_match_detects_divergent_state() -> None:
    protocol = _yaml("configs/slam_speed_pulse_causal_pilot_v2.yaml")
    base = np.zeros(51).tolist()
    arms = {
        arm: {"pulse_trace": {"pre_pulse": {"elapsed_s": 7.16, "observation": list(base)}}}
        for arm in ANALYZER.ARMS
    }
    passed, _ = ANALYZER._pre_match(arms, protocol)
    assert passed
    arms["scale_025"]["pulse_trace"]["pre_pulse"]["observation"][0] = 1.0
    passed, detail = ANALYZER._pre_match(arms, protocol)
    assert not passed
    assert not detail["base_linear_velocity"]["passed"]


def test_pre_pulse_match_detects_excess_sample_time_skew() -> None:
    protocol = _yaml("configs/slam_speed_pulse_causal_pilot_v2.yaml")
    arms = {
        arm: {
            "pulse_trace": {
                "pre_pulse": {"elapsed_s": 7.16, "observation": np.zeros(51).tolist()}
            }
        }
        for arm in ANALYZER.ARMS
    }
    arms["scale_025"]["pulse_trace"]["pre_pulse"]["elapsed_s"] = 7.10
    passed, detail = ANALYZER._pre_match(arms, protocol)
    assert not passed
    assert not detail["sample_time_alignment"]["passed"]


def test_composite_joint_velocity_matching_allows_phase_peak_but_limits_rms() -> None:
    protocol = _yaml("configs/slam_speed_pulse_causal_pilot_v2.yaml")
    arms = {
        arm: {
            "pulse_trace": {
                "pre_pulse": {"elapsed_s": 7.16, "observation": np.zeros(51).tolist()}
            }
        }
        for arm in ANALYZER.ARMS
    }
    arms["scale_025"]["pulse_trace"]["pre_pulse"]["observation"][24] = 3.0
    passed, detail = ANALYZER._pre_match(arms, protocol)
    assert passed, detail
    assert detail["joint_velocity"]["maximum_pairwise_linf"] == 3.0
    assert detail["joint_velocity"]["maximum_pairwise_rms"] < 1.25
    for index in range(24, 36):
        arms["scale_025"]["pulse_trace"]["pre_pulse"]["observation"][index] = 2.0
    passed, detail = ANALYZER._pre_match(arms, protocol)
    assert not passed
    assert detail["joint_velocity"]["maximum_pairwise_linf"] < 3.25
    assert detail["joint_velocity"]["maximum_pairwise_rms"] > 1.25


def test_randomized_itt_risk_retains_pre_pulse_tracking_loss_as_failure() -> None:
    assert ANALYZER._risk({"metrics": {
        "pulse_window_valid_requested_usable_next_horizon_failure_fraction": None,
        "pulse_window_valid_requested_usable_next_horizon_count": 0,
    }}) == 1.0
    assert ANALYZER._risk({"metrics": {
        "pulse_window_valid_requested_usable_next_horizon_failure_fraction": 0.25,
        "pulse_window_valid_requested_usable_next_horizon_count": 10,
    }}) == 0.25


def test_causal_analyzer_passes_matched_monotonic_fixture() -> None:
    protocol = _yaml("configs/slam_speed_pulse_causal_pilot_v2.yaml")
    validator = _load(
        "pulse_protocol_fixture",
        "scripts/validation/validate_slam_speed_pulse_causal_protocol.py",
    )
    records = []
    for row in validator.build_schedule(protocol, "causal_pilot"):
        pre_observation = np.zeros(51)
        if row["backend"] == "fastlio2" and row["profile"] == "curve_1_5_right_1_0" and row["block_id"] == 567 and row["arm"] == "scale_025":
            pre_observation[24] = 10.0
        records.append({
            "dataset_role": "excluded_causal_development",
            "identity": {
                "stage": row["stage"], "backend": row["backend"],
                "profile": row["profile"], "block_id": row["block_id"],
                "arm": row["arm"],
            },
            "gate": {"passed": True},
            "pulse_trace": {
                "passed": True,
                "pre_pulse": {"elapsed_s": 7.16, "observation": pre_observation.tolist()},
            },
            "metrics": {
                "pulse_window_valid_requested_usable_next_horizon_failure_fraction": 0.2 * row["assigned_scale"],
                "pulse_window_moving_speed_mps": row["assigned_scale"],
                "fall": False, "base_contact": False,
            },
        })
    report = ANALYZER.analyze_records(records, protocol, "causal_pilot")
    assert report["decision"]["status"] == "PASS", report
    assert not report["pre_pulse_matching"]["fastlio2/curve_1_5_right_1_0/567"]["passed"]


def test_fastlio2_ceiling_does_not_block_prespecified_liosam_headroom() -> None:
    protocol = _yaml("configs/slam_speed_pulse_causal_pilot_v2.yaml")
    validator = _load(
        "pulse_protocol_ceiling_fixture",
        "scripts/validation/validate_slam_speed_pulse_causal_protocol.py",
    )
    records = []
    for row in validator.build_schedule(protocol, "causal_pilot"):
        if row["backend"] == "fastlio2":
            hazard = 0.0 if row["arm"] == "scale_100" else 0.01
        else:
            hazard = 0.2 * row["assigned_scale"]
        records.append({
            "dataset_role": "excluded_causal_development",
            "identity": {
                "stage": row["stage"], "backend": row["backend"],
                "profile": row["profile"], "block_id": row["block_id"],
                "arm": row["arm"],
            },
            "gate": {"passed": True},
            "pulse_trace": {
                "passed": True,
                "pre_pulse": {"elapsed_s": 7.16, "observation": np.zeros(51).tolist()},
            },
            "metrics": {
                "pulse_window_valid_requested_usable_next_horizon_failure_fraction": hazard,
                "pulse_window_moving_speed_mps": row["assigned_scale"],
                "fall": False, "base_contact": False,
            },
        })
    report = ANALYZER.analyze_records(records, protocol, "causal_pilot")
    assert report["decision"]["status"] == "PASS", report
    assert all(report["gate"]["fastlio2_safeguards"])
    assert all(
        report["strata"][f"fastlio2/{profile}"]["full_scale_ceiling"]
        for profile in protocol["stages"]["causal_pilot"]["profiles"]
    )
