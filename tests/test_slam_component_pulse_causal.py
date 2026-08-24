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


def _yaml(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


VALIDATOR = _load(
    "component_protocol", "scripts/validation/validate_slam_speed_pulse_causal_protocol.py"
)
ANALYZER = _load(
    "component_analyzer", "scripts/validation/analyze_slam_component_pulse_causal_pilot.py"
)
RUNNER = _load(
    "component_runner", "scripts/validation/run_slam_speed_pulse_causal_pilot.py"
)


def test_component_protocol_is_fresh_balanced_and_live_closed() -> None:
    protocol = _yaml("configs/slam_component_pulse_causal_pilot_v1.yaml")
    report = VALIDATOR.validate_protocol(protocol)
    assert report["passed"], report["failures"]
    assert report["stage_counts"] == {"wiring_smoke": 8, "causal_pilot": 64}
    assert protocol["frozen_fresh_blocks"] == {
        "wiring_smoke": [571], "causal_pilot": [572, 573, 574, 575]
    }
    assert {
        tuple(row["assigned_scales"])
        for row in report["schedules"]["causal_pilot"]
    } == {
        (1.0, 1.0, 1.0),
        (0.75, 0.75, 0.75),
        (0.75, 0.75, 1.0),
        (1.0, 1.0, 0.75),
    }
    assert protocol["boundaries"]["live_execution_authorized"] is False


def test_component_release_is_hash_locked_and_live_closed() -> None:
    release = _yaml("configs/slam_component_pulse_causal_release_v1.yaml")
    protocol_path = ROOT / release["protocol"]["path"]
    RUNNER.validate_release(release, protocol_path)
    with pytest.raises(ValueError, match="live execution is not authorized"):
        RUNNER.require_execution_authorization(release, "wiring_smoke")


@pytest.mark.parametrize(
    "arm", ("control", "uniform_075", "preserve_yaw", "preserve_translation")
)
def test_component_trace_verifies_xyz_pulse(arm: str) -> None:
    onnx = pytest.importorskip("onnx")
    ReferenceEvaluator = pytest.importorskip("onnx.reference").ReferenceEvaluator
    trace_module = _load(
        f"component_trace_{arm}", "scripts/validation/validate_slam_speed_pulse_trace.py"
    )
    protocol = _yaml("configs/slam_component_pulse_causal_pilot_v1.yaml")
    release = _yaml("configs/slam_component_pulse_causal_release_v1.yaml")
    scales = np.asarray(protocol["treatment"]["arms"][arm], dtype=np.float32)
    origin = 1.0
    elapsed_values = [5.5, 7.16, 7.32, 7.5, 7.9, 8.1, 8.3]
    profile = get_motion_profile("curve_1_5_right_1_0")
    rng = np.random.default_rng(20260824)
    observations = rng.normal(size=(len(elapsed_values), 51)).astype(np.float32)
    observations[:, 48:51] = np.asarray([0.7, 1.0, 0.1], dtype=np.float32)
    received = []
    for index, elapsed in enumerate(elapsed_values):
        base = np.asarray(profile.command_at(elapsed), dtype=np.float32)
        command = (scales if 7.25 <= elapsed < 8.0 else 1.0) * base
        received.append(command)
        observations[index, 9:12] = command
    evaluator = ReferenceEvaluator(onnx.load(str(ROOT / release["policy"]["path"])))
    actions = evaluator.run(None, {"observation": observations})[0]
    diagnostics = {"records": [
        {
            "clock_s": origin + elapsed,
            "received_command": command.tolist(),
            "effective_command": command.tolist(),
            "observation": observation.tolist(),
            "raw_action": action.tolist(),
        }
        for elapsed, command, observation, action
        in zip(elapsed_values, received, observations, actions)
    ]}
    driver = {
        "profile": profile.name,
        "profile_start_clock_s": origin,
        "command_scale_pulse": {
            "enabled": True,
            "start_s": 7.25,
            "duration_s": 0.75,
            "scales_xyz": scales.tolist(),
            "publish_count": 30,
        },
    }
    report = trace_module.validate_trace(diagnostics, driver, protocol, release, arm)
    assert report["passed"], report


def _fixture_record(row: dict) -> dict:
    if row["backend"] == "liosam":
        hazard = {
            "control": 0.45,
            "uniform_075": 0.40,
            "preserve_yaw": 0.20,
            "preserve_translation": 0.50,
        }[row["arm"]]
    else:
        hazard = {
            "control": 0.10,
            "uniform_075": 0.09,
            "preserve_yaw": 0.10,
            "preserve_translation": 0.15,
        }[row["arm"]]
    speed = {
        "control": 1.0,
        "uniform_075": 0.75,
        "preserve_yaw": 0.80,
        "preserve_translation": 0.90,
    }[row["arm"]]
    return {
        "dataset_role": "excluded_causal_development",
        "identity": {
            "stage": row["stage"], "backend": row["backend"],
            "profile": row["profile"], "block_id": row["block_id"], "arm": row["arm"],
        },
        "gate": {"passed": True},
        "pulse_trace": {
            "passed": True,
            "pre_pulse": {"elapsed_s": 7.16, "observation": np.zeros(51).tolist()},
        },
        "metrics": {
            "pulse_window_intention_to_treat_failure_fraction": hazard,
            "pulse_window_moving_speed_mps": speed,
            "fall": False,
            "base_contact": False,
        },
    }


def test_component_analyzer_requires_better_than_uniform_not_only_control() -> None:
    protocol = _yaml("configs/slam_component_pulse_causal_pilot_v1.yaml")
    records = [
        _fixture_record(row) for row in VALIDATOR.build_schedule(protocol, "causal_pilot")
    ]
    report = ANALYZER.analyze_records(records, protocol, "causal_pilot")
    assert report["decision"]["status"] == "PASS", report
    assert report["gate"]["qualified_candidates"] == ["preserve_yaw"]

    for record in records:
        if record["identity"]["backend"] == "liosam" and record["identity"]["arm"] == "preserve_yaw":
            record["metrics"]["pulse_window_intention_to_treat_failure_fraction"] = 0.40
    report = ANALYZER.analyze_records(records, protocol, "causal_pilot")
    assert report["decision"]["status"] == "FAIL"
    assert not report["gate"]["conditions"]["at_least_one_component_candidate_beats_uniform_on_liosam"]
