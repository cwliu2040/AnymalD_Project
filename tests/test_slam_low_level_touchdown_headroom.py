from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
ROS_SOURCE = ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"
sys.path.insert(0, str(ROS_SOURCE))

from anymal_locomotion_ros2.touchdown_shaping_intervention_core import (  # noqa: E402
    TouchdownShapingConfig,
    apply_touchdown_shaping_intervention,
)
from anymal_locomotion_ros2.touchdown_phase_estimator_core import (  # noqa: E402
    estimate_touchdown_phase,
)
from anymal_locomotion_ros2.touchdown_phase_tracker_core import (  # noqa: E402
    initial_touchdown_phase_tracker_state,
    update_tracker_from_probabilities,
)
from anymal_locomotion_ros2.anymal_d_kinematics_core import (  # noqa: E402
    anymal_d_foot_kinematics,
)
from anymal_locomotion_ros2.touchdown_residual_pipeline_core import (  # noqa: E402
    apply_touchdown_residual_pipeline,
)


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _module(
    "validate_touchdown_headroom",
    ROOT / "scripts/validation/validate_slam_low_level_touchdown_headroom_protocol.py",
)
ANALYZER = _module(
    "analyze_touchdown_headroom",
    ROOT / "scripts/validation/analyze_slam_low_level_touchdown_headroom.py",
)
ENVELOPE_BUILDER = _module(
    "build_touchdown_baseline_envelope",
    ROOT / "scripts/validation/build_slam_low_level_model1450_envelope.py",
)
BASELINE_RECORD = _module(
    "build_touchdown_baseline_record",
    ROOT / "scripts/validation/build_slam_low_level_baseline_run_record.py",
)
PHASE_V2_VALIDATOR = _module(
    "validate_touchdown_phase_tracker_v2",
    ROOT / "scripts/validation/validate_slam_low_level_touchdown_phase_tracker_v2.py",
)


def _jacobian(scale: float = 0.10) -> np.ndarray:
    value = np.zeros((4, 3, 12), dtype=np.float32)
    for foot in range(4):
        value[foot, 2, foot] = scale
    return value


def _apply(**updates):
    values = {
        "requested_command": [1.0, -0.2, 0.6],
        "backbone_action": np.linspace(-0.2, 0.2, 12, dtype=np.float32),
        "swing_progress": [0.9, 0.2, 0.8, 0.1],
        "phase_confidence": [0.95, 0.95, 0.95, 0.95],
        "foot_vertical_velocity_mps": [-0.5, 0.2, -0.25, 0.1],
        "foot_position_jacobian_per_action": _jacobian(),
        "tracking_valid": True,
        "attenuation_fraction": 0.5,
    }
    values.update(updates)
    return apply_touchdown_shaping_intervention(**values)


def test_zero_and_invalid_paths_are_exact_backbone_without_command_scaling() -> None:
    baseline = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    for result in (
        _apply(backbone_action=baseline, attenuation_fraction=0.0),
        _apply(backbone_action=baseline, tracking_valid=False),
    ):
        np.testing.assert_array_equal(
            result.effective_command, np.asarray([1.0, -0.2, 0.6], dtype=np.float32)
        )
        np.testing.assert_array_equal(result.applied_action, baseline)
        np.testing.assert_array_equal(result.action_residual, np.zeros(12, np.float32))


def test_touchdown_solve_targets_only_eligible_downward_feet() -> None:
    result = _apply()
    np.testing.assert_array_equal(result.active_feet, [True, False, True, False])
    assert result.action_residual[0] > 0.0
    assert result.action_residual[2] > 0.0
    assert np.count_nonzero(result.action_residual[[1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]) == 0
    assert float(np.max(np.abs(result.action_residual))) <= 0.05
    np.testing.assert_array_equal(
        result.effective_command, np.asarray([1.0, -0.2, 0.6], dtype=np.float32)
    )


def test_residual_uses_uniform_bound_rescaling_without_changing_direction() -> None:
    unconstrained = _apply(
        foot_vertical_velocity_mps=[-20.0, 0.2, -10.0, 0.1],
        config=TouchdownShapingConfig(residual_linf_limit=10.0),
    )
    bounded = _apply(foot_vertical_velocity_mps=[-20.0, 0.2, -10.0, 0.1])
    assert bounded.residual_scale < 1.0
    assert np.isclose(np.max(np.abs(bounded.action_residual)), 0.05, atol=1.0e-7)
    active = np.flatnonzero(np.abs(unconstrained.action_residual) > 1.0e-9)
    ratios = bounded.action_residual[active] / unconstrained.action_residual[active]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1.0e-5, atol=1.0e-7)


def test_no_contact_truth_backend_or_profile_enters_core_and_random_commands_are_preserved() -> None:
    rng = np.random.default_rng(42)
    for command in rng.uniform([-2.0, -1.5, -2.0], [3.0, 1.5, 2.0], size=(32, 3)):
        result = _apply(requested_command=command)
        np.testing.assert_array_equal(result.effective_command, command.astype(np.float32))


def test_malformed_phase_or_jacobian_fails_closed_at_wiring_boundary() -> None:
    for updates in (
        {"swing_progress": [0.2] * 3},
        {"swing_progress": [0.2, 0.3, 1.2, 0.4]},
        {"phase_confidence": [float("nan")] * 4},
        {"foot_position_jacobian_per_action": np.zeros((4, 2, 12))},
    ):
        try:
            _apply(**updates)
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed mechanism input was accepted: {updates}")


def test_history_phase_estimator_uses_only_joint_velocity_and_previous_action_history() -> None:
    feature_count = 36
    artifact = {
        "frozen": True,
        "passed": True,
        "history_offsets_samples": [0],
        "feature_mean": [0.0] * feature_count,
        "feature_scale": [1.0] * feature_count,
        "classifier_weight": [[0.0] * feature_count for _ in range(4)],
        "classifier_intercept": [2.0, -2.0, 1.0, -1.0],
        "progress_weight": [[0.0] * feature_count for _ in range(4)],
        "progress_intercept": [0.9, 0.2, 0.8, 0.1],
    }
    zeros = np.zeros((1, 12), dtype=np.float32)
    result = estimate_touchdown_phase(zeros, zeros, zeros, artifact)
    assert result.tracking_valid is True
    np.testing.assert_allclose(result.swing_progress, [0.9, 0.2, 0.8, 0.1])
    assert result.confidence[0] > result.confidence[1]


def test_history_phase_estimator_malformed_or_nonfinite_history_is_invalid_zero() -> None:
    artifact = {
        "frozen": True,
        "passed": True,
        "history_offsets_samples": [0, 2],
        "feature_mean": [0.0] * 72,
        "feature_scale": [1.0] * 72,
        "classifier_weight": [[0.0] * 72 for _ in range(4)],
        "classifier_intercept": [0.0] * 4,
        "progress_weight": [[0.0] * 72 for _ in range(4)],
        "progress_intercept": [0.0] * 4,
    }
    short = np.zeros((2, 12), dtype=np.float32)
    result = estimate_touchdown_phase(short, short, short, artifact)
    assert result.tracking_valid is False
    np.testing.assert_array_equal(result.swing_progress, np.zeros(4))
    np.testing.assert_array_equal(result.confidence, np.zeros(4))


def test_rejected_phase_estimator_artifact_cannot_enter_runtime() -> None:
    artifact = {
        "frozen": False,
        "passed": False,
        "history_offsets_samples": [0],
    }
    zeros = np.zeros((1, 12), dtype=np.float32)
    result = estimate_touchdown_phase(zeros, zeros, zeros, artifact)
    assert result.tracking_valid is False
    np.testing.assert_array_equal(result.swing_progress, np.zeros(4))
    np.testing.assert_array_equal(result.confidence, np.zeros(4))


def _tracker_config() -> dict:
    return {
        "early_transition_confirmation_samples": 2,
        "late_transition_confirmation_samples": 1,
        "stance_reset_confirmation_samples": 2,
        "minimum_early_swing_samples_before_late": 3,
        "early_enter_probability": 0.55,
        "early_enter_maximum_progress": 0.45,
        "late_enter_probability": 0.80,
        "late_hold_probability": 0.65,
        "stance_reset_probability": 0.70,
        "maximum_progress_regression_per_sample": 0.08,
        "maximum_late_state_samples": 8,
        "late_swing_progress_minimum": 0.70,
    }


def test_temporal_tracker_forbids_direct_stance_to_late_swing_activation() -> None:
    state = initial_touchdown_phase_tracker_state()
    late = np.tile([0.01, 0.01, 0.98], (4, 1))
    for _ in range(5):
        result = update_tracker_from_probabilities(
            late, np.full(4, 0.9), state, _tracker_config()
        )
        assert result.tracking_valid is True
        assert not np.any(result.eligible_late_swing)
        state = result.state


def test_temporal_tracker_requires_confirmed_early_then_monotonic_late_evidence() -> None:
    state = initial_touchdown_phase_tracker_state()
    early = np.tile([0.05, 0.90, 0.05], (4, 1))
    late = np.tile([0.05, 0.05, 0.90], (4, 1))
    for progress in (0.1, 0.2, 0.4):
        result = update_tracker_from_probabilities(
            early, np.full(4, progress), state, _tracker_config()
        )
        state = result.state
    assert not np.any(result.eligible_late_swing)
    for progress in (0.75,):
        result = update_tracker_from_probabilities(
            late, np.full(4, progress), state, _tracker_config()
        )
        state = result.state
    assert np.all(result.eligible_late_swing)
    assert np.all(result.confidence >= 0.80)


def test_phase_v2_fresh_gate_cannot_promote_low_precision_candidate() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/slam_low_level_touchdown_phase_tracker_v2.yaml").read_text()
    )
    passing = {
        "late_swing_precision": 0.95,
        "late_swing_recall": 0.50,
        "false_trigger_fraction": 0.005,
        "swing_progress_mae": 0.10,
        "positive_predictions_by_foot": [2, 2, 2, 2],
    }
    metrics = {
        "aggregate": dict(passing),
        "by_profile": {profile: dict(passing) for profile in config["data_roles"]["fresh_validation"]["profiles"]},
    }
    assert PHASE_V2_VALIDATOR._metric_failures(metrics, config) == []
    metrics["aggregate"]["late_swing_precision"] = 0.89
    assert "aggregate_precision" in PHASE_V2_VALIDATOR._metric_failures(metrics, config)


def test_anymal_d_foot_jacobian_matches_central_finite_difference() -> None:
    rng = np.random.default_rng(20260825)
    position = rng.uniform(-0.8, 0.8, 12)
    result = anymal_d_foot_kinematics(position)
    epsilon = 1.0e-6
    for joint in range(12):
        delta = np.zeros(12)
        delta[joint] = epsilon
        numeric = (
            anymal_d_foot_kinematics(position + delta).foot_position_base_m
            - anymal_d_foot_kinematics(position - delta).foot_position_base_m
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            numeric, result.foot_jacobian_per_rad[:, :, joint], atol=1.0e-8
        )


def _synthetic_phase_artifact() -> dict:
    feature_count = 36
    weights = np.zeros((4, 3, feature_count))
    progress = np.zeros((4, feature_count))
    haa_indices = (0, 2, 1, 3)
    for foot, index in enumerate(haa_indices):
        weights[foot, 1, index] = -5.0
        weights[foot, 2, index] = 5.0
        progress[foot, index] = 0.4
    return {
        "frozen": True, "passed": True, "development_candidate": False,
        "history_offsets_samples": [0],
        "feature_mean": [0.0] * feature_count,
        "feature_scale": [1.0] * feature_count,
        "classifier_weight": weights.tolist(),
        "classifier_intercept": np.zeros((4, 3)).tolist(),
        "progress_weight": progress.tolist(),
        "progress_intercept": [0.5] * 4,
        "tracker": _tracker_config(),
    }


def test_complete_touchdown_pipeline_is_exact_zero_until_confirmed_late_swing() -> None:
    state = initial_touchdown_phase_tracker_state()
    artifact = _synthetic_phase_artifact()
    backbone = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    command = np.asarray([1.0, 0.0, -0.5], dtype=np.float32)
    early_position = np.zeros(12)
    early_position[[0, 1, 2, 3]] = -1.0
    zeros = np.zeros((1, 12))
    for _ in range(3):
        result = apply_touchdown_residual_pipeline(
            requested_command=command, backbone_action=backbone,
            joint_position_history=early_position[None, :],
            joint_velocity_history=zeros, previous_action_history=zeros,
            current_joint_position_rad=early_position,
            current_joint_velocity_radps=np.zeros(12),
            phase_artifact=artifact, phase_state=state,
            attenuation_fraction=0.5, tracking_valid=True,
        )
        state = result.phase.state
        np.testing.assert_array_equal(result.intervention.action_residual, np.zeros(12))
    late_position = np.zeros(12)
    late_position[[0, 1, 2, 3]] = 1.0
    velocity = np.zeros(12)
    velocity[[4, 8]] = 1.0
    velocity[[6, 10]] = -1.0
    result = apply_touchdown_residual_pipeline(
        requested_command=command, backbone_action=backbone,
        joint_position_history=late_position[None, :],
        joint_velocity_history=velocity[None, :], previous_action_history=zeros,
        current_joint_position_rad=late_position,
        current_joint_velocity_radps=velocity,
        phase_artifact=artifact, phase_state=state,
        attenuation_fraction=0.5, tracking_valid=True,
    )
    assert result.mode == "valid_touchdown_residual"
    assert np.any(result.intervention.active_feet)
    assert np.max(np.abs(result.intervention.action_residual)) <= 0.05
    np.testing.assert_array_equal(result.intervention.effective_command, command)


def test_complete_touchdown_pipeline_rejects_unfrozen_phase_artifact() -> None:
    artifact = _synthetic_phase_artifact()
    artifact["frozen"] = False
    artifact["passed"] = False
    zeros = np.zeros((1, 12))
    result = apply_touchdown_residual_pipeline(
        requested_command=[1.0, 0.0, 0.0], backbone_action=np.zeros(12),
        joint_position_history=zeros, joint_velocity_history=zeros,
        previous_action_history=zeros, current_joint_position_rad=np.zeros(12),
        current_joint_velocity_radps=np.zeros(12), phase_artifact=artifact,
        phase_state=initial_touchdown_phase_tracker_state(),
        attenuation_fraction=0.5, tracking_valid=True,
    )
    assert result.mode == "phase_invalid_exact_zero_residual"
    np.testing.assert_array_equal(result.intervention.action_residual, np.zeros(12))


def _protocol() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/slam_low_level_touchdown_headroom_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_protocol_freezes_mechanism_gates_and_disjoint_schedules() -> None:
    protocol = _protocol()
    result = VALIDATOR.validate_protocol(protocol)
    assert result["passed"], result["failures"]
    assert result["baseline_count"] == 36
    assert result["stage_counts"] == {
        "wiring_smoke": 6,
        "pilot": 72,
        "confirmation": 72,
    }
    all_blocks = {
        row["block_id"] for row in result["baseline_schedule"]
    } | {
        row["block_id"]
        for rows in result["schedules"].values()
        for row in rows
    }
    assert not all_blocks.intersection({581, 582, 583, 584})
    assert min(all_blocks) == 585
    assert protocol["boundaries"]["live_execution_authorized"] is False
    assert protocol["mechanism"]["phase_source"]["estimator_artifact_frozen"] is False


def _baseline_record(row: dict, metrics: tuple[str, ...]) -> tuple[Path, dict]:
    values = {
        metric: [float(row["block_id"]), float(row["block_id"]) + 0.5]
        for metric in metrics
    }
    return Path(f"missing/{row['profile']}_{row['block_id']}.json"), {
        "identity": {
            "stage": "baseline_envelope",
            "profile": row["profile"],
            "block_id": row["block_id"],
            "arm": "zero",
        },
        "dataset_role": "anti_collapse_calibration_only",
        "gate": {"passed": True},
        "trace": {"checks": {
            "requested_command_exact_original": True,
            "zero_residual_exact_model1450": True,
            "frame_samples_used_as_independent_replicates": False,
        }},
        "sample_units": {metric: "complete_stride" for metric in metrics},
        "metrics": values,
    }


def test_baseline_envelope_builder_is_complete_hierarchical_and_candidate_blind() -> None:
    protocol = _protocol()
    rows = VALIDATOR.build_baseline_schedule(protocol)
    metrics = tuple(protocol["baseline_envelope"]["metrics"])
    records = [_baseline_record(row, metrics) for row in rows]
    report = ENVELOPE_BUILDER.build_envelope(records, protocol)
    assert report["passed"], report["failures"]
    assert report["frozen"] is True
    assert report["candidate_outcomes_used"] is False
    assert report["effect_claim_allowed"] is False
    for profile in protocol["baseline_envelope"]["profiles"]:
        for metric in metrics:
            envelope = report["profiles"][profile][metric]
            assert envelope["complete_run_count"] == 12
            assert len(envelope["within_run_summaries"]) == 12

    incomplete = records[:-1]
    failed = ENVELOPE_BUILDER.build_envelope(incomplete, protocol)
    assert failed["passed"] is False
    assert "missing_baseline_identity" in failed["failures"]


def test_baseline_metric_reducer_uses_complete_gait_events_and_canonical_joint_data() -> None:
    samples = []
    for index in range(160):
        time_s = 0.02 * (index + 1)
        phase = index % 40
        feet = {}
        for foot, name in enumerate(BASELINE_RECORD.FOOT_NAMES):
            foot_phase = (phase + 10 * foot) % 40
            stance = foot_phase < 20
            z = 0.0 if stance else 0.08 * np.sin(np.pi * (foot_phase - 20) / 20.0)
            feet[name] = {
                "normal_force_n": 100.0 if stance else 0.0,
                "tangential_speed_mps": 0.02 if stance else 0.3,
                "position_w_m": [0.01 * index + 0.1 * foot, 0.2 * (1 if foot in (0, 2) else -1), z],
                "position_b_m": [0.2 * (1 if foot < 2 else -1), 0.2 * (1 if foot in (0, 2) else -1), -0.5 + z],
            }
        joint = (0.2 * np.sin(2.0 * np.pi * index / 40.0 + np.arange(12))).tolist()
        joint_velocity = (0.5 * np.cos(2.0 * np.pi * index / 40.0 + np.arange(12))).tolist()
        samples.append({
            "time_s": time_s,
            "command": [1.0, 0.0, 0.4],
            "actual_velocity": [0.9, 0.0, 0.39],
            "base_position_w_m": [0.01 * index, 0.0, 0.6],
            "base_height_m": 0.6,
            "yaw_rad": 0.005 * index,
            "feet": feet,
            "joint_position_rad": joint,
            "joint_velocity_radps": joint_velocity,
            "applied_joint_torque_nm": [5.0] * 12,
        })
    trace = {
        "metadata": {"joint_position_limits_rad": [[-2.0, 2.0]] * 12},
        "samples": samples,
    }
    metrics, units = BASELINE_RECORD.gait_and_posture_metrics(trace)
    assert metrics["stride_length_m"]
    assert metrics["swing_duration_s"]
    assert metrics["foot_clearance_m"]
    assert 0.0 < metrics["duty_factor"] < 1.0
    assert metrics["minimum_joint_margin_rad"] > 1.0
    assert metrics["energy_per_progress"] > 0.0
    assert metrics["stopped_fraction"] == 0.0
    assert units["stride_length_m"] == "complete_stride"
    assert units["torque_rms_nm"] == "complete_run"


def test_baseline_policy_trace_requires_original_command_and_48d_model1450() -> None:
    observation = np.zeros(48, dtype=np.float32)
    observation[9:12] = [1.0, 0.0, -0.5]
    policy = {"records": [{
        "observation": observation.tolist(),
        "received_command": [1.0, 0.0, -0.5],
        "effective_command": [1.0, 0.0, -0.5],
        "raw_action": [0.0] * 12,
        "watchdog_timed_out": False,
    }]}
    checks = BASELINE_RECORD.baseline_trace_checks(policy)
    assert checks["requested_command_exact_original"] is True
    assert checks["zero_residual_exact_model1450"] is True
    assert checks["frame_samples_used_as_independent_replicates"] is False
    policy["records"][0]["effective_command"][0] = 0.9
    assert BASELINE_RECORD.baseline_trace_checks(policy)["requested_command_exact_original"] is False


def _record(row: dict, *, primary_speed: float = 0.98) -> dict:
    arm = row["arm"]
    metrics = {
        "moving_linear_speed_mps": 1.0,
        "realized_yaw_rate_radps": 0.50,
        "normalized_progress": 10.0,
        "stopped_fraction": 0.0,
        "touchdown_vertical_speed_abs_mps": 1.0,
        "touchdown_contact_impulse_ns": 10.0,
        "touchdown_roll_pitch_rate_impulse_rad": 1.0,
        "lidar_scan_time_rotation_rad": 0.20,
        "valid_requested_usable_next_horizon_failure_fraction": 0.40,
        "tracking_restricted_mean_survival_time_s": 4.0,
        "fall": False,
        "base_contact": False,
    }
    if arm == "touchdown_soft_low":
        metrics.update({
            "touchdown_vertical_speed_abs_mps": 0.90,
            "touchdown_contact_impulse_ns": 9.5,
            "touchdown_roll_pitch_rate_impulse_rad": 0.95,
            "lidar_scan_time_rotation_rad": 0.19,
            "valid_requested_usable_next_horizon_failure_fraction": 0.35,
            "tracking_restricted_mean_survival_time_s": 4.1,
        })
    elif arm == "touchdown_soft":
        metrics.update({
            "moving_linear_speed_mps": primary_speed,
            "realized_yaw_rate_radps": 0.49,
            "normalized_progress": 9.9,
            "touchdown_vertical_speed_abs_mps": 0.80,
            "touchdown_contact_impulse_ns": 9.0,
            "touchdown_roll_pitch_rate_impulse_rad": 0.90,
            "lidar_scan_time_rotation_rad": 0.18,
            "valid_requested_usable_next_horizon_failure_fraction": 0.30,
            "tracking_restricted_mean_survival_time_s": 4.2,
        })
    checks = {name: True for name in ANALYZER.TRACE_CHECKS}
    checks["zero_arm_bit_exact_model1450"] = arm == "zero"
    return {
        "identity": {
            "stage": row["stage"],
            "backend": row["backend"],
            "profile": row["profile"],
            "block_id": row["block_id"],
            "arm": arm,
        },
        "dataset_role": row["dataset_role"],
        "gate": {"passed": True},
        "trace": {
            "checks": checks,
            "maximum_residual_linf": 0.0 if arm == "zero" else 0.04,
        },
        "anti_collapse": {name: True for name in ANALYZER.COLLAPSE_CHECKS},
        "metrics": metrics,
    }


def test_pilot_analyzer_requires_matched_motion_and_repeated_mechanism_slam_chain() -> None:
    protocol = _protocol()
    rows = VALIDATOR.build_stage_schedule(protocol, "pilot")
    passing = [_record(row) for row in rows]
    report = ANALYZER.analyze_records(passing, protocol, "pilot")
    assert report["decision"]["status"] == "PILOT_PASS", report
    assert report["decision"]["pooled_backend_rescue_used"] is False

    limiter_collapse = [_record(row, primary_speed=0.80) for row in rows]
    report = ANALYZER.analyze_records(limiter_collapse, protocol, "pilot")
    assert report["decision"]["status"] == "INCONCLUSIVE", report
    assert report["decision"]["supported_strata"] == []


def test_repeated_treatment_specific_safety_excess_is_fail() -> None:
    protocol = _protocol()
    rows = VALIDATOR.build_stage_schedule(protocol, "pilot")
    records = [_record(row) for row in rows]
    for record in records:
        identity = record["identity"]
        if (
            identity["backend"] == "fastlio2"
            and identity["profile"] == "mixed_curve_bounded"
            and identity["block_id"] in {598, 599}
            and identity["arm"] == "touchdown_soft"
        ):
            record["metrics"]["fall"] = True
    report = ANALYZER.analyze_records(records, protocol, "pilot")
    assert report["decision"]["status"] == "FAIL", report
    assert report["simulation_safety"]["repeated_treatment_specific_harm"] is True


def test_single_safety_or_anti_collapse_outcome_is_not_an_integrity_failure() -> None:
    protocol = _protocol()
    rows = VALIDATOR.build_stage_schedule(protocol, "pilot")
    records = [_record(row) for row in rows]
    one_primary = next(
        record for record in records
        if record["identity"]["arm"] == "touchdown_soft"
    )
    one_primary["metrics"]["fall"] = True
    report = ANALYZER.analyze_records(records, protocol, "pilot")
    assert report["integrity"]["passed"] is True
    assert report["decision"]["status"] == "INCONCLUSIVE"

    one_primary["metrics"]["fall"] = False
    one_primary["anti_collapse"]["posture_envelope_pass"] = False
    report = ANALYZER.analyze_records(records, protocol, "pilot")
    assert report["integrity"]["passed"] is True
    assert any(not row["anti_collapse_passed"] for row in report["block_reports"])
    assert report["decision"]["status"] == "INCONCLUSIVE"
