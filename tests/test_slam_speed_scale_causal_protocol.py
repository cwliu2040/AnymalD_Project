from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/validation/validate_slam_speed_scale_causal_protocol.py"
spec = importlib.util.spec_from_file_location("speed_scale_protocol", PATH)
assert spec and spec.loader
MODULE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)


def _protocol() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/slam_speed_scale_causal_pilot_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_hash_locked_phase3b_protocol_and_counts_validate() -> None:
    protocol = _protocol()
    report = MODULE.validate_protocol(protocol)
    assert report["passed"], report["failures"]
    assert report["stage_counts"] == {"wiring_smoke": 8, "causal_pilot": 32}
    assert protocol["boundaries"]["live_execution_authorized"] is False
    assert protocol["implementation_gates"]["fixed_scale_runtime_wiring_complete"] is True
    assert protocol["implementation_gates"]["artifact_hash_lock_complete"] is True
    assert protocol["implementation_gates"]["wiring_smoke_authorized"] is False
    assert protocol["implementation_gates"]["causal_pilot_authorized"] is False
    assert protocol["implementation_gates"]["action_conditioned_model_training_authorized"] is False


def test_schedule_is_complete_balanced_and_disjoint() -> None:
    protocol = _protocol()
    schedule = MODULE.build_schedule(protocol, "causal_pilot")
    identities = {
        (row["backend"], row["profile"], row["block_id"], row["arm"])
        for row in schedule
    }
    assert len(identities) == 32
    assert {row["block_id"] for row in schedule} == {561, 562}
    assert {row["assigned_scale"] for row in schedule} == {1.0, 0.75, 0.5, 0.25}
    for backend in ("fastlio2", "liosam"):
        for profile in ("curve_1_5_right_1_0", "lateral_right_1_5"):
            for block in (561, 562):
                rows = [
                    row for row in schedule
                    if row["backend"] == backend
                    and row["profile"] == profile
                    and row["block_id"] == block
                ]
                assert {row["arm"] for row in rows} == set(MODULE.ARMS)
                assert sorted(row["arm_order"] for row in rows) == [0, 1, 2, 3]


def test_validator_rejects_live_or_reused_blocks() -> None:
    protocol = _protocol()
    protocol["boundaries"]["live_execution_authorized"] = True
    protocol["stages"]["causal_pilot"]["block_ids"] = [559, 562]
    report = MODULE.validate_protocol(protocol)
    assert not report["passed"]
    assert "boundary must be false: live_execution_authorized" in report["failures"]
    assert "causal_pilot reuses a forbidden block" in report["failures"]
