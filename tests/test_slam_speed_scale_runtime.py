from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load("speed_runner", "scripts/validation/run_slam_speed_scale_causal_pilot.py")
ANALYZER = _load("speed_analyzer", "scripts/validation/analyze_slam_speed_scale_causal_pilot.py")


def _release() -> dict:
    return yaml.safe_load((ROOT / "configs/slam_speed_scale_causal_release_v1.yaml").read_text())


def _protocol() -> dict:
    return yaml.safe_load((ROOT / "configs/slam_speed_scale_causal_pilot_v1.yaml").read_text())


def test_release_is_hash_locked_and_live_execution_is_closed_after_pilot() -> None:
    release = _release()
    RUNNER.validate_release(release, ROOT / release["protocol"]["path"])
    assert release["boundaries"]["live_execution_authorized"] is False
    assert release["boundaries"]["authorized_stages"] == []
    with pytest.raises(ValueError, match="live execution is not authorized"):
        RUNNER.require_execution_authorization(release, "causal_pilot")


@pytest.mark.parametrize("arm", ("scale_100", "scale_075", "scale_050", "scale_025"))
def test_trace_reconstructs_fixed_scale_and_invalid_zero(arm: str) -> None:
    onnx = pytest.importorskip("onnx")
    ReferenceEvaluator = pytest.importorskip("onnx.reference").ReferenceEvaluator
    trace = _load("speed_trace", "scripts/validation/validate_slam_speed_scale_trace.py")
    release = _release()
    config = release["arms"][arm]
    rng = np.random.default_rng(20260822)
    observations = rng.normal(size=(16, 51)).astype(np.float32)
    observations[:, 48] = rng.random(16, dtype=np.float32)
    observations[:, 49] = 1.0
    observations[:, 50] = 0.25
    observations[::3, 49] = 0.0
    observations[1::4, 50] = 1.25
    evaluator = ReferenceEvaluator(onnx.load(str(ROOT / config["policy_path"])))
    actions = evaluator.run(None, {"observation": observations})[0]
    diagnostics = {
        "records": [
            {"observation": observation.tolist(), "raw_action": action.tolist()}
            for observation, action in zip(observations, actions)
        ]
    }
    report = trace.validate_trace(diagnostics, release, arm)
    assert report["passed"], report
    assert report["assigned_command_scale"] == config["assigned_scale"]
    assert report["checks"]["invalid_stale_command_exact_zero"]


def _record(row: dict, *, speed: float, hazard: float) -> dict:
    return {
        "dataset_role": "excluded_causal_development",
        "identity": {
            "stage": row["stage"], "backend": row["backend"], "profile": row["profile"],
            "block_id": row["block_id"], "arm": row["arm"],
        },
        "gate": {"passed": True},
        "treatment_trace": {
            "passed": True, "assigned_command_scale": row["assigned_scale"],
            "checks": {"invalid_stale_command_exact_zero": True},
        },
        "metrics": {
            "moving_speed_mps": speed,
            "normalized_progress": speed,
            "normalized_progress_per_elapsed_second": speed,
            "valid_requested_usable_next_horizon_failure_fraction": hazard,
            "tracking_restricted_mean_survival_time_s": 10.0,
            "stance_weighted_foot_slip_rms_mps": 0.1,
            "while_stable_roll_pitch_rate_rms_radps": 0.2,
            "fall": False, "base_contact": False,
        },
    }


def test_causal_analyzer_passes_frozen_monotonic_fixture() -> None:
    protocol = _protocol()
    schedule = _load(
        "speed_protocol_for_fixture",
        "scripts/validation/validate_slam_speed_scale_causal_protocol.py",
    ).build_schedule(protocol, "causal_pilot")
    records = [
        _record(row, speed=row["assigned_scale"], hazard=0.2 * row["assigned_scale"])
        for row in schedule
    ]
    report = ANALYZER.analyze_records(records, protocol, "causal_pilot")
    assert report["decision"]["status"] == "PASS", json.dumps(report, indent=2)
