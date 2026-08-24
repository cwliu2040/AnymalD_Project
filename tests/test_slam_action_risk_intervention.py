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


def test_v2_fresh_block_protocol_and_release_validate() -> None:
    protocol_path = ROOT / "configs/slam_action_risk_intervention_pilot_v2.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    result = VALIDATOR.validate_protocol(protocol)
    assert result["passed"], result["failures"]
    assert {
        row["block_id"] for rows in result["schedules"].values() for row in rows
    } == set(range(551, 560))
    release = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_release_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert release["boundaries"]["live_execution_authorized"] is False
    assert release["boundaries"]["authorized_stages"] == []
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


def test_execution_baseline_hashes_dirty_files_without_requiring_commit(monkeypatch) -> None:
    dirty = "configs/slam_action_risk_intervention_release_v1.yaml"

    def fake_git(*args: str) -> str:
        values = {
            ("branch", "--show-current"): "exp/slam-fastlio2",
            ("diff", "--name-only", "--"): dirty,
            ("diff", "--cached", "--name-only", "--"): "",
            ("ls-files", "--others", "--exclude-standard"): "outputs/ignored.json",
            ("rev-parse", "HEAD"): "abc123",
        }
        return values[args]

    monkeypatch.setattr(RUNNER, "_git", fake_git)
    baseline = RUNNER.capture_execution_baseline()
    assert baseline["git_commit"] == "abc123"
    assert baseline["project_owned_dirty"]
    assert [item["path"] for item in baseline["dirty_files"]] == [dirty]
    assert len(baseline["dirty_files"][0]["sha256"]) == 64


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


def test_v2_reuses_hash_locked_wiring_and_recomputes_raw_gate(
    tmp_path: Path, monkeypatch,
) -> None:
    protocol_path = tmp_path / "configs" / "v1.yaml"
    protocol_path.parent.mkdir()
    protocol_path.write_text("schema_version: 1\n")
    root = tmp_path / "evidence" / "wiring_smoke"
    root.mkdir(parents=True)
    manifest_path = root / "run_manifest.json"
    protocol_sha = RUNNER._sha256(protocol_path)
    manifest_path.write_text(json.dumps({"protocol_sha256": protocol_sha}))

    class FakeAnalyzer:
        @staticmethod
        def load_records(path: Path) -> list[dict]:
            assert path == root
            return [{"raw": True}]

        @staticmethod
        def analyze_records(records: list[dict], protocol: dict, stage: str) -> dict:
            assert records == [{"raw": True}]
            assert protocol == {"schema_version": 1}
            assert stage == "wiring_smoke"
            return {"decision": {"status": "WIRING_PASS"}}

    monkeypatch.setattr(RUNNER, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(RUNNER, "_analyzer_module", lambda: FakeAnalyzer)
    release = {"prerequisites": {"wiring_smoke": {
        "output_path": "evidence/wiring_smoke",
        "protocol_path": "configs/v1.yaml",
        "protocol_sha256": protocol_sha,
        "manifest_sha256": RUNNER._sha256(manifest_path),
        "required_decision": "WIRING_PASS",
    }}}
    RUNNER.require_stage_prerequisite(
        tmp_path / "unused", {}, "pilot", tmp_path / "configs/v2.yaml", release,
    )


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


def test_cell_bound_uses_recorded_formula_parity_tolerance() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    record = _pilot_records()[0]
    record["intervention"]["maximum_realized_residual"] = 0.05000042915344238
    record["intervention"]["atol"] = 1.0e-5
    assert RUNNER.cell_stop_reasons(
        record, {"arm": record["identity"]["arm"]}, protocol,
    ) == []
    record["intervention"]["maximum_realized_residual"] = 0.05002
    assert "intervention_linf_violation" in RUNNER.cell_stop_reasons(
        record, {"arm": record["identity"]["arm"]}, protocol,
    )


def test_analyzer_bound_uses_recorded_formula_parity_tolerance() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot.yaml").read_text(
            encoding="utf-8"
        )
    )
    records = _pilot_records()
    records[0]["intervention"]["maximum_realized_residual"] = 0.05000042915344238
    records[0]["intervention"]["atol"] = 1.0e-5
    report = ANALYZER.analyze_records(records, protocol, "pilot")
    assert report["integrity"]["passed"]
    assert report["decision"]["status"] == "PASS"


def test_runner_uses_repository_onnx_vendor_only_for_estimator_replay() -> None:
    source = (
        ROOT / "scripts/validation/run_slam_action_risk_intervention_pilot.py"
    ).read_text(encoding="utf-8")
    assert 'deployment_vendor = str(PROJECT_ROOT / "deployment/python_vendor")' in source
    assert 'environment=estimator_env' in source


def test_runner_keeps_verbose_process_logs_out_of_controller_stdout() -> None:
    source = (
        ROOT / "scripts/validation/run_slam_action_risk_intervention_pilot.py"
    ).read_text(encoding="utf-8")
    assert 'run_dir / "launch.log"' in source
    assert 'run_dir / f"{name}.log"' in source
    assert "stderr=subprocess.STDOUT" in source


def _pilot_records(
    protocol_name: str = "slam_action_risk_intervention_pilot.yaml",
) -> list[dict]:
    protocol = yaml.safe_load(
        (ROOT / "configs" / protocol_name).read_text(
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


def test_v2_single_simulation_fall_is_retained_and_inconclusive() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    records = _pilot_records("slam_action_risk_intervention_pilot_v2.yaml")
    smooth = next(record for record in records if (
        record["identity"]["backend"] == "fastlio2"
        and record["identity"]["profile"] == "curve_1_5_right_1_0"
        and record["identity"]["block_id"] == 552
        and record["identity"]["arm"] == "smooth"
    ))
    smooth["metrics"]["fall"] = True
    smooth["metrics"]["base_contact"] = True
    assert RUNNER.cell_stop_reasons(
        smooth, {"arm": "smooth"}, protocol,
    ) == []
    report = ANALYZER.analyze_records(records, protocol, "pilot")
    assert report["decision"]["status"] == "INCONCLUSIVE"
    assert report["gate"]["simulation_safety"]["paired_excess_by_stratum"] == {
        "fastlio2/curve_1_5_right_1_0": [552]
    }
    assert not report["gate"]["simulation_safety"]["route_fail"]


def test_v2_repeated_paired_smooth_specific_harm_fails_route() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    records = _pilot_records("slam_action_risk_intervention_pilot_v2.yaml")
    for record in records:
        identity = record["identity"]
        if (
            identity["backend"] == "fastlio2"
            and identity["profile"] == "curve_1_5_right_1_0"
            and identity["arm"] == "smooth"
        ):
            record["metrics"]["fall"] = True
    report = ANALYZER.analyze_records(records, protocol, "pilot")
    assert report["decision"]["status"] == "FAIL"
    assert report["gate"]["simulation_safety"]["route_fail"]
    assert report["gate"]["simulation_safety"]["paired_excess_by_stratum"] == {
        "fastlio2/curve_1_5_right_1_0": [552, 553]
    }


def test_v2_runner_stops_only_after_second_complete_matched_triplet() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    schedule = VALIDATOR.build_stage_schedule(protocol, "pilot")
    visited = []

    def execute(row: dict) -> dict:
        visited.append((row["block_id"], row["arm"]))
        return {
            **row,
            "passed": True,
            "stop_required": False,
            "stop_reasons": [],
            "simulation_safety_event": row["arm"] == "smooth",
        }

    results = RUNNER.execute_schedule(schedule, execute, protocol)
    assert len(results) == 6
    assert set(visited) == {
        (block, arm)
        for block in (552, 553)
        for arm in ("smooth", "zero", "antismooth")
    }
    assert results[-1]["stop_required"]
    assert results[-1]["stop_reasons"] == [
        "repeated_paired_smooth_specific_safety_harm"
    ]


def test_v2_analyzer_accepts_predeclared_safety_terminated_partial_inventory() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_action_risk_intervention_pilot_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    records = [
        record
        for record in _pilot_records("slam_action_risk_intervention_pilot_v2.yaml")
        if record["identity"]["backend"] == "fastlio2"
        and record["identity"]["profile"] == "curve_1_5_right_1_0"
    ]
    for record in records:
        if record["identity"]["arm"] == "smooth":
            record["metrics"]["base_contact"] = True
    report = ANALYZER.analyze_records(records, protocol, "pilot")
    assert report["inventory"]["observed_run_count"] == 6
    assert report["inventory"]["safety_terminated_early"]
    assert report["integrity"]["passed"]
    assert report["decision"]["status"] == "FAIL"
    assert report["simulation_safety"]["route_fail"]


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
