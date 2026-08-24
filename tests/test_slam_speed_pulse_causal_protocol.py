from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/validation/validate_slam_speed_pulse_causal_protocol.py"
SPEC = importlib.util.spec_from_file_location("pulse_protocol", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _protocol() -> dict:
    return yaml.safe_load((ROOT / "configs/slam_speed_pulse_causal_pilot_v2.yaml").read_text())


def test_frozen_pulse_protocol_is_fresh_and_unauthorized() -> None:
    protocol = _protocol()
    report = MODULE.validate_protocol(protocol)
    assert report["passed"], report["failures"]
    assert report["stage_counts"] == {"wiring_smoke": 8, "causal_pilot": 64}
    assert protocol["stages"]["wiring_smoke"]["block_ids"] == [566]
    assert protocol["stages"]["causal_pilot"]["block_ids"] == [567, 568, 569, 570]
    assert protocol["common"]["point_density_timeline_s"]["healthy"] == 7.25
    assert protocol["treatment"]["pulse_start_relative_to_profile_s"] == 7.25
    assert protocol["boundaries"]["live_execution_authorized"] is False


def test_every_arm_has_identical_pulse_timing_and_distinct_scale() -> None:
    protocol = _protocol()
    schedule = MODULE.build_schedule(protocol, "causal_pilot")
    assert len(schedule) == 64
    assert {row["assigned_scale"] for row in schedule} == {1.0, 0.75, 0.5, 0.25}
    assert protocol["treatment"]["pulse_start_relative_to_profile_s"] == 7.25
    assert protocol["treatment"]["pulse_duration_s"] == 0.75
