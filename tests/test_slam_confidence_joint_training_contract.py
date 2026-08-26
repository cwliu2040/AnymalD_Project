from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from anymal_locomotion.joint_training_contract import (
    FULL_POLICY_OBSERVATION_DIM,
    HISTORY_FRAME_DIM,
    HISTORY_LENGTH,
    NEUTRAL_LOCALIZATION,
    accumulated_command_progress,
    build_history_frame,
    compose_full_policy_observation,
    decide_joint_training_route,
    evaluate_body_lidar_mechanism,
    evaluate_directional_metrics,
    evaluate_matched_motion,
    evaluate_paired_noninferiority,
    event_aligned_body_lidar_summary,
    history_localization_channels,
)
from scripts.validation.evaluate_slam_confidence_joint_training_block import evaluate_block


ROOT = Path(__file__).resolve().parents[1]


def _legacy(command=(1.0, -0.2, 0.4), marker=0.0):
    value = np.zeros(48, dtype=np.float32)
    value[0] = marker
    value[9:12] = command
    return value


def test_j1_j2_have_equal_width_and_only_localization_channels_differ() -> None:
    frames = [
        build_history_frame(
            _legacy(marker=index),
            (0.8 - 0.1 * index, 1.0, 0.1 * index),
            requested_command=(1.0, -0.2, 0.4),
        )
        for index in range(3)
    ]
    j1 = compose_full_policy_observation(
        _legacy(), frames, current_requested_command=(1.0, -0.2, 0.4), arm="J1"
    )
    j2 = compose_full_policy_observation(
        _legacy(), frames, current_requested_command=(1.0, -0.2, 0.4), arm="J2"
    )
    assert j1.shape == j2.shape == (FULL_POLICY_OBSERVATION_DIM,)
    np.testing.assert_array_equal(history_localization_channels(j1), np.tile(NEUTRAL_LOCALIZATION, (20, 1)))
    nonlocal_mask = np.ones((HISTORY_LENGTH, HISTORY_FRAME_DIM), dtype=bool)
    nonlocal_mask[:, 48:51] = False
    np.testing.assert_array_equal(
        j1[48:].reshape(20, 51)[nonlocal_mask],
        j2[48:].reshape(20, 51)[nonlocal_mask],
    )
    np.testing.assert_array_equal(
        j1[9:12], np.asarray([1.0, -0.2, 0.4], dtype=np.float32)
    )
    assert history_localization_channels(j2)[-1].tolist() == pytest.approx([0.6, 1.0, 0.2])


def test_history_contract_rejects_scaled_current_command_future_width_and_bad_localization() -> None:
    frame = build_history_frame(_legacy(), (1.0, 1.0, 0.0), requested_command=(1.0, -0.2, 0.4))
    with pytest.raises(ValueError, match="original request"):
        compose_full_policy_observation(
            _legacy(command=(0.5, -0.1, 0.2)),
            [frame],
            current_requested_command=(1.0, -0.2, 0.4),
            arm="J2",
        )
    with pytest.raises(ValueError, match="more than"):
        compose_full_policy_observation(
            _legacy(), [frame] * 21, current_requested_command=(1.0, -0.2, 0.4), arm="J2"
        )
    with pytest.raises(ValueError, match="validity"):
        build_history_frame(_legacy(), (1.0, 0.5, 0.0), requested_command=(1.0, -0.2, 0.4))


def test_accumulated_progress_does_not_alias_on_a_closed_curve() -> None:
    dt = 0.02
    times = np.arange(0.0, 8.0 + dt, dt)
    commands = np.tile(np.asarray([1.5, 0.0, 1.0]), (len(times), 1))
    linear = np.tile(np.asarray([1.5, 0.0, 0.0]), (len(times), 1))
    yaw = np.ones(len(times))
    result = accumulated_command_progress(times, commands, linear, yaw)
    assert result["linear_progress_ratio"] == pytest.approx(1.0)
    assert result["yaw_progress_ratio"] == pytest.approx(1.0)
    assert result["stopped_fraction"] == pytest.approx(0.0)


def test_accumulated_progress_penalizes_wrong_direction_and_yaw_oscillation() -> None:
    times = np.asarray([0.0, 1.0, 2.0])
    commands = np.asarray([[1.0, 0.0, 1.0]] * 3)
    linear = np.asarray([[-1.0, 0.0, 0.0]] * 3)
    yaw = np.asarray([-1.0, 1.0, -1.0])
    result = accumulated_command_progress(times, commands, linear, yaw)
    assert result["linear_progress_ratio"] == pytest.approx(-1.0)
    assert result["yaw_progress_ratio"] == pytest.approx(0.0)


def test_paired_gate_requires_relative_and_absolute_safety() -> None:
    specs = {
        "cadence": {"mode": "upper_relative", "tolerance": 0.1},
        "height": {"mode": "lower_difference", "tolerance": 0.03, "absolute_minimum": 0.45},
    }
    passed = evaluate_paired_noninferiority(
        {"cadence": 3.1, "height": 0.53}, {"cadence": 3.0, "height": 0.55}, specs
    )
    assert passed["passed"] is True
    failed = evaluate_paired_noninferiority(
        {"cadence": 3.4, "height": 0.44}, {"cadence": 3.0, "height": 0.55}, specs
    )
    assert set(failed["failures"]) == {"cadence", "height"}
    assert failed["metrics"]["height"]["absolute_pass"] is False


def test_body_lidar_windows_follow_observed_localization_transitions() -> None:
    times = np.arange(0.0, 1.02, 0.02)
    localization = np.tile(np.asarray([1.0, 1.0, 0.0]), (len(times), 1))
    localization[20:, 0] = 0.6
    localization[30:, 1] = 0.0
    commands = np.tile(np.asarray([1.0, 0.0, 0.5]), (len(times), 1))
    linear = commands.copy()
    linear[:, 2] = 0.0
    angular = np.zeros_like(commands)
    angular[:, 2] = 0.5
    report = event_aligned_body_lidar_summary(times, localization, linear, angular, commands)
    assert report["available"] is True
    assert report["event_count"] == 2
    assert report["lidar_scan_rotation_error_rms_rad"] == pytest.approx(0.0)
    assert report["lidar_scan_translation_error_rms_m"] == pytest.approx(0.0)


def test_matched_motion_rejects_stop_and_speed_reduction() -> None:
    times = np.arange(0.0, 1.02, 0.02)
    commands = np.tile(np.asarray([1.0, 0.0, 0.5]), (len(times), 1))
    comparator_linear = np.tile(np.asarray([1.0, 0.0, 0.0]), (len(times), 1))
    comparator_yaw = np.full(len(times), 0.5)
    passed = evaluate_matched_motion(
        times,
        commands,
        commands.copy(),
        comparator_linear.copy(),
        comparator_linear,
        comparator_yaw.copy(),
        comparator_yaw,
    )
    assert passed["passed"] is True

    stopped = evaluate_matched_motion(
        times,
        commands,
        commands.copy(),
        np.zeros_like(comparator_linear),
        comparator_linear,
        np.zeros_like(comparator_yaw),
        comparator_yaw,
    )
    assert stopped["passed"] is False
    assert {"moving_linear_speed", "realized_yaw_rate", "linear_progress", "yaw_progress", "stopped_fraction"}.issubset(
        stopped["failures"]
    )


def _gait_metrics() -> dict[str, float]:
    return {
        "cadence_hz": 2.5,
        "contact_switch_rate_hz": 5.0,
        "duty_factor": 0.55,
        "body_height_m": 0.52,
        "stance_width_m": 0.45,
        "minimum_joint_margin_rad": 0.20,
        "foot_clearance_m": 0.08,
        "torque_rms_nm": 20.0,
        "energy_per_progress": 50.0,
        "stance_foot_slip_rms_mps": 0.03,
        "stopped_fraction": 0.0,
    }


def _block_record(arm: str, oscillation: float) -> dict:
    times = np.arange(0.0, 1.02, 0.02)
    commands = np.tile(np.asarray([1.0, 0.0, 0.5]), (len(times), 1))
    phase = np.linspace(0.0, 8.0 * np.pi, len(times))
    linear = np.zeros((len(times), 3))
    linear[:, 0] = 1.0
    linear[:, 2] = oscillation * np.sin(phase)
    angular = np.zeros((len(times), 3))
    angular[:, 0] = oscillation * np.sin(phase)
    angular[:, 2] = 0.5
    localization = np.tile(np.asarray([1.0, 1.0, 0.0]), (len(times), 1))
    localization[10:, 0] = 0.6
    return {
        "arm": arm,
        "backend": "liosam",
        "profile": "mixed",
        "block": 1,
        "times_s": times.tolist(),
        "requested_commands": commands.tolist(),
        "body_linear_velocity_mps": linear.tolist(),
        "body_angular_velocity_radps": angular.tolist(),
        "localization_state": localization.tolist(),
        "gait_safety_metrics": _gait_metrics(),
        "slam_metrics": {
            "future_unusable_fraction": 0.10 if arm == "J2" else 0.20,
            "ground_truth_drift_m": 0.10 if arm == "J2" else 0.20,
            "point_support": 110.0 if arm == "J2" else 100.0,
        },
        "safety_event_count": 0,
    }


def test_frozen_block_evaluator_requires_matched_motion_body_lidar_and_slam() -> None:
    protocol = yaml.safe_load((ROOT / "configs/slam_confidence_joint_training_v1.yaml").read_text())
    candidate = _block_record("J2", 0.05)
    comparator = _block_record("J1", 0.20)
    report = evaluate_block(candidate, comparator, protocol)
    assert report["integrity_passed"] is True
    assert report["matched_motion_passed"] is True
    assert report["body_lidar_passed"] is True
    assert report["slam_direction_passed"] is True
    assert report["anti_collapse_passed"] is True
    assert report["safety_event_excess"] is False

    candidate["requested_commands"][0][0] = 0.5
    failed = evaluate_block(candidate, comparator, protocol)
    assert failed["integrity_passed"] is False
    assert failed["matched_motion_passed"] is False


def _row(comparison: str, block: int, supported: bool) -> dict:
    return {
        "comparison": comparison,
        "backend": "liosam",
        "profile": "mixed",
        "block": block,
        "integrity_passed": True,
        "matched_motion_passed": True,
        "body_lidar_passed": supported,
        "slam_direction_passed": supported,
        "anti_collapse_passed": True,
        "safety_event_excess": False,
    }


def test_go_no_go_separates_generic_and_localization_aware_value() -> None:
    generic = [_row("J1-J0", block, True) for block in range(4)]
    no_localization = [_row("J2-J1", block, block < 2) for block in range(4)]
    report = decide_joint_training_route(generic + no_localization)
    assert report["status"] == "GENERIC_SMOOTHER_LOCOMOTION_ONLY"
    localization = [_row("J2-J1", block, block < 3) for block in range(4)]
    report = decide_joint_training_route(generic + localization)
    assert report["status"] == "LOCALIZATION_AWARE_HEADROOM_CONTINUE_TO_TEACHER_ADAPTATION"
    stopped = decide_joint_training_route(
        [_row("J1-J0", block, False) for block in range(4)]
        + [_row("J2-J1", block, False) for block in range(4)]
    )
    assert stopped["status"] == "STOP_FULL_POLICY_JOINT_TRAINING_ROUTE"


def test_protocol_forbids_old_limiter_semantics_and_keeps_execution_closed() -> None:
    config = yaml.safe_load((ROOT / "configs/slam_confidence_joint_training_v1.yaml").read_text())
    assert config["comparators"]["J1"]["full_observation_dimension"] == 1068
    assert config["comparators"]["J2"]["full_observation_dimension"] == 1068
    assert config["observation_contract"]["requested_command_scaling_allowed"] is False
    assert config["reward_contract"]["confidence_scaled_target_forbidden"] is True
    assert config["reward_contract"]["invalid_motion_penalty_that_overrides_requested_motion_forbidden"] is True
    assert config["training_budget"]["execution_authorized"] is False
    assert config["training_budget"]["independent_training_seeds"] == [1450, 1451, 1452]
    assert config["training_budget"]["cherry_pick_seed_for_formal_evaluation_forbidden"] is True
    assert config["execution_gates"]["ppo_training_authorized"] is False
    assert config["execution_gates"]["live_ros_wiring_authorized"] is False
    assert config["execution_gates"]["physical_robot_authorized"] is False
    assert config["execution_gates"]["blocks_581_584_allowed"] is False
    assert config["execution_gates"]["touchdown_blocks_602_605_allowed"] is False
