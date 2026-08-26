from __future__ import annotations

import importlib.util
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


def _yaml(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


VALIDATOR = _load(
    "identification_validator", "scripts/validation/validate_slam_speed_pulse_causal_protocol.py"
)
ANALYZER = _load(
    "identification_analyzer", "scripts/validation/analyze_slam_component_identification_pilot.py"
)
RUNNER = _load(
    "identification_runner", "scripts/validation/run_slam_speed_pulse_causal_pilot.py"
)


def test_identification_protocol_freezes_two_levels_and_validation_reserve() -> None:
    protocol = _yaml("configs/slam_component_identification_pilot_v1.yaml")
    report = VALIDATOR.validate_protocol(protocol)
    assert report["passed"], report["failures"]
    assert report["stage_counts"] == {"wiring_smoke": 12, "causal_pilot": 96}
    assert protocol["frozen_fresh_blocks"] == {
        "wiring_smoke": [576], "causal_pilot": [577, 578, 579, 580]
    }
    assert protocol["disjointness"]["reserved_not_collected_in_this_protocol"] == [581, 582, 583, 584]
    assert {
        tuple(row["assigned_scales"])
        for row in report["schedules"]["causal_pilot"]
    } == set(VALIDATOR.IDENTIFICATION_SCALES)


def test_identification_release_is_hash_locked_and_live_closed() -> None:
    release = _yaml("configs/slam_component_identification_release_v1.yaml")
    RUNNER.validate_release(release, ROOT / release["protocol"]["path"])
    with pytest.raises(ValueError, match="live execution is not authorized"):
        RUNNER.require_execution_authorization(release, "wiring_smoke")


def _record(row: dict, *, safety: bool = False) -> dict:
    scales = np.asarray(row["assigned_scales"], dtype=np.float64)
    hazard = float(np.clip(0.7 - 0.3 * scales[0] + 0.2 * scales[2], 0.0, 1.0))
    return {
        "dataset_role": "excluded_component_identification_development",
        "identity": {
            "stage": row["stage"], "backend": row["backend"],
            "profile": row["profile"], "block_id": row["block_id"], "arm": row["arm"],
        },
        "gate": {"passed": True},
        "pulse_trace": {
            "passed": True,
            "assigned_scales": scales.tolist(),
            "pre_pulse": {"elapsed_s": 7.16, "observation": np.zeros(51).tolist()},
        },
        "metrics": {
            "pulse_window_intention_to_treat_failure_fraction": hazard,
            "pulse_window_moving_speed_mps": float(np.mean(scales)),
            "fall": safety, "base_contact": safety,
        },
    }


def test_identification_analyzer_accepts_complete_varying_fixture() -> None:
    protocol = _yaml("configs/slam_component_identification_pilot_v1.yaml")
    records = [_record(row) for row in VALIDATOR.build_schedule(protocol, "causal_pilot")]
    report = ANALYZER.analyze_records(records, protocol, "causal_pilot")
    assert report["decision"]["status"] == "PASS", report
    assert all(report["gate"]["conditions"].values())


def test_identification_analyzer_rejects_repeated_arm_specific_safety() -> None:
    protocol = _yaml("configs/slam_component_identification_pilot_v1.yaml")
    records = []
    for row in VALIDATOR.build_schedule(protocol, "causal_pilot"):
        repeated = (
            row["backend"] == "liosam"
            and row["profile"] == "curve_1_5_right_1_0"
            and row["arm"] == "reduce_yaw_075"
            and row["block_id"] in (577, 578)
        )
        records.append(_record(row, safety=repeated))
    report = ANALYZER.analyze_records(records, protocol, "causal_pilot")
    assert report["decision"]["status"] == "FAIL"
    assert not report["gate"]["conditions"]["no_repeated_arm_specific_safety_harm"]
