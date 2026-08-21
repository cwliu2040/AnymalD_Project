from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
ROS_SOURCE = ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = _module(
    "action_intervention_core",
    ROS_SOURCE / "anymal_locomotion_ros2/action_intervention_core.py",
)
VALIDATOR = _module(
    "validate_action_intervention_protocol",
    ROOT / "scripts/validation/validate_slam_action_risk_intervention_protocol.py",
)
RUNNER = _module(
    "run_action_intervention_pilot",
    ROOT / "scripts/validation/run_slam_action_risk_intervention_pilot.py",
)
ANALYZER = _module(
    "analyze_action_intervention_pilot",
    ROOT / "scripts/validation/analyze_slam_action_risk_intervention_pilot.py",
)
RECORD_BUILDER = _module(
    "build_action_intervention_record",
    ROOT / "scripts/validation/build_slam_action_risk_intervention_run_record.py",
)


def test_zero_and_invalid_are_exact_baseline() -> None:
    baseline = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(2, 12)
    previous = -baseline
    zero, zero_delta = CORE.apply_previous_action_intervention(
        baseline, previous, np.asarray([True, True]), alpha=0.0,
        residual_limit=0.05,
    )
    invalid, invalid_delta = CORE.apply_previous_action_intervention(
        baseline, previous, np.asarray([False, False]), alpha=0.1,
        residual_limit=0.05,
    )
    assert np.array_equal(zero, baseline)
    assert np.count_nonzero(zero_delta) == 0
    assert np.array_equal(invalid, baseline)
    assert np.count_nonzero(invalid_delta) == 0


def test_signed_interventions_are_symmetric_before_clipping() -> None:
    baseline = np.full((1, 12), 0.2, dtype=np.float32)
    previous = np.full((1, 12), 0.1, dtype=np.float32)
    _, smooth = CORE.apply_previous_action_intervention(
        baseline, previous, np.asarray([True]), alpha=0.1,
        residual_limit=0.05,
    )
    _, antismooth = CORE.apply_previous_action_intervention(
        baseline, previous, np.asarray([True]), alpha=-0.1,
        residual_limit=0.05,
    )
    np.testing.assert_allclose(smooth, -antismooth)


def test_intervention_respects_linf_limit() -> None:
    baseline = np.ones((3, 12), dtype=np.float32)
    previous = -baseline
    _, residual = CORE.apply_previous_action_intervention(
        baseline, previous, np.asarray([True, False, True]), alpha=1.0,
        residual_limit=0.05,
    )
    assert float(np.max(np.abs(residual))) <= 0.05 + np.finfo(np.float32).eps
    assert np.count_nonzero(residual[1]) == 0


def test_frozen_protocol_and_schedule_validate() -> None:
    path = ROOT / "configs/slam_action_risk_intervention_pilot.yaml"
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    result = VALIDATOR.validate_protocol(protocol)
    assert result["passed"], result["failures"]
    assert result["stage_counts"] == {
        "wiring_smoke": 6,
        "pilot": 24,
        "expanded_only_after_pilot_inconclusive": 72,
    }
    zero_rows = [
        row for row in result["schedules"]["pilot"] if row["arm"] == "zero"
    ]
    assert len(zero_rows) == 8


def test_release_hashes_match_frozen_protocol_and_artifacts() -> None:
    protocol_path = ROOT / "configs/slam_action_risk_intervention_pilot.yaml"
    release = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_release_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    RUNNER.validate_release(release, protocol_path)


def test_execution_requires_explicit_stage_authorization() -> None:
    release = {"boundaries": {
        "live_execution_authorized": False,
        "authorized_stages": [],
    }}
    try:
        RUNNER.require_execution_authorization(release, "wiring_smoke")
    except ValueError as exc:
        assert "not authorized" in str(exc)
    else:
        raise AssertionError("unauthorized live execution was accepted")
    release["boundaries"] = {
        "live_execution_authorized": True,
        "authorized_stages": ["wiring_smoke"],
    }
    RUNNER.require_execution_authorization(release, "wiring_smoke")
    try:
        RUNNER.require_execution_authorization(release, "pilot")
    except ValueError as exc:
        assert "stage is not explicitly authorized" in str(exc)
    else:
        raise AssertionError("unauthorized stage was accepted")


def test_pilot_requires_recomputed_wiring_pass(tmp_path: Path) -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    root = tmp_path / "wiring_smoke"
    root.mkdir()
    (root / "run_manifest.json").write_text(json.dumps({
        "stage": "wiring_smoke",
        "protocol_sha256": RUNNER._sha256(
            ROOT / "configs/slam_action_risk_intervention_pilot.yaml"
        ),
    }))
    try:
        RUNNER.require_stage_prerequisite(tmp_path, protocol, "pilot")
    except ValueError as exc:
        assert "requires wiring_smoke=WIRING_PASS" in str(exc)
    else:
        raise AssertionError("pilot bypassed an incomplete wiring smoke")


def test_expansion_requires_pilot_inconclusive(tmp_path: Path) -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "run_manifest.json").write_text(json.dumps({
        "stage": "pilot",
        "protocol_sha256": RUNNER._sha256(
            ROOT / "configs/slam_action_risk_intervention_pilot.yaml"
        ),
    }))
    for index, record in enumerate(_pilot_records()):
        cell = root / f"cell_{index}"
        cell.mkdir()
        (cell / "intervention_run_record.json").write_text(json.dumps(record))
    try:
        RUNNER.require_stage_prerequisite(
            tmp_path, protocol, "expanded_only_after_pilot_inconclusive",
        )
    except ValueError as exc:
        assert "requires pilot=INCONCLUSIVE; got PASS" in str(exc)
    else:
        raise AssertionError("expansion bypassed a PASS pilot decision")


def test_execution_refuses_existing_stage_artifacts(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.json").write_text("{}")
    try:
        RUNNER.require_fresh_stage_output(tmp_path)
    except ValueError as exc:
        assert "already contains execution artifacts" in str(exc)
    else:
        raise AssertionError("existing execution output was accepted")


def test_schedule_stops_immediately_after_a_required_stop() -> None:
    visited = []

    def execute(row: dict) -> dict:
        visited.append(row["id"])
        return {"stop_required": row["id"] == 2}

    results = RUNNER.execute_schedule(
        [{"id": 1}, {"id": 2}, {"id": 3}], execute,
    )
    assert visited == [1, 2]
    assert len(results) == 2


def _pilot_records() -> list[dict]:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    records = []
    action = {"smooth": 1.0, "zero": 2.0, "antismooth": 3.0}
    body = {"smooth": 0.1, "zero": 0.2, "antismooth": 0.3}
    survival = {"smooth": 11.0, "zero": 10.0, "antismooth": 9.0}
    hazard = {"smooth": 0.1, "zero": 0.2, "antismooth": 0.3}
    for row in VALIDATOR.build_stage_schedule(protocol, "pilot"):
        arm = row["arm"]
        records.append({
            "dataset_role": "excluded_causal_development",
            "identity": {
                "stage": "pilot", "backend": row["backend"],
                "profile": row["profile"], "block_id": row["block_id"],
                "arm": arm,
            },
            "metrics": {
                "action_rate_rms_per_s": action[arm],
                "while_stable_roll_pitch_rate_rms_radps": body[arm],
                "tracking_restricted_mean_survival_time_s": survival[arm],
                "valid_requested_usable_next_horizon_failure_fraction": hazard[arm],
                "moving_speed_mps": 1.0,
                "fall": False, "base_contact": False,
            },
            "intervention": {
                "passed": True,
                "maximum_realized_residual": 0.0 if arm == "zero" else 0.01,
                "checks": {
                    "zero_exact_arm_B": True,
                    "invalid_exact_arm_B": True,
                    "nonzero_arm_realized": True,
                },
            },
            "gate": {"passed": True},
        })
    return records


def test_frozen_pilot_decision_passes_only_complete_directional_chain() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    report = ANALYZER.analyze_records(_pilot_records(), protocol, "pilot")
    assert report["decision"]["status"] == "PASS"
    assert report["gate"]["ordered_action_rate_strata"] == 4
    assert report["gate"]["ordered_body_rate_strata"] == 4


def test_frozen_pilot_decision_fails_joint_body_and_survival_harm() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    records = _pilot_records()
    for record in records:
        if record["identity"]["backend"] == "fastlio2" and record["identity"]["arm"] == "smooth":
            record["metrics"]["while_stable_roll_pitch_rate_rms_radps"] = 0.4
            record["metrics"]["tracking_restricted_mean_survival_time_s"] = 9.0
    report = ANALYZER.analyze_records(records, protocol, "pilot")
    assert report["decision"]["status"] == "FAIL"
    assert report["gate"]["body_rate_and_survival_harm"]


def test_frozen_pilot_decision_is_inconclusive_for_isolated_hazard_direction() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    records = _pilot_records()
    for record in records:
        if record["identity"]["backend"] == "liosam" and record["identity"]["arm"] == "smooth":
            record["metrics"]["valid_requested_usable_next_horizon_failure_fraction"] = 0.25
    report = ANALYZER.analyze_records(records, protocol, "pilot")
    assert report["decision"]["status"] == "INCONCLUSIVE"


def test_hazard_endpoint_excludes_currently_invalid_policy_ticks() -> None:
    policy = {
        "records": [
            {"clock_s": 1.0, "observation": [0.0] * 49 + [1.0, 0.0]},
            {"clock_s": 2.0, "observation": [0.0] * 49 + [0.0, 1.0]},
            {"clock_s": 3.0, "observation": [0.0] * 49 + [1.0, 0.0]},
        ]
    }
    offline = {
        "false_stop": {"records": [
            {"policy_clock_ns": 1_000_000_000, "requested_motion": True, "usable_next_horizon": False},
            {"policy_clock_ns": 2_000_000_000, "requested_motion": True, "usable_next_horizon": False},
            {"policy_clock_ns": 3_000_000_000, "requested_motion": True, "usable_next_horizon": True},
        ]}
    }
    fraction, count, unmatched = RECORD_BUILDER.valid_requested_hazard(policy, offline)
    assert fraction == 0.5
    assert count == 2
    assert unmatched == 0
