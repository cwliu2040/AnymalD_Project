"""Static integration checks for the first FAST-LIO2 confidence extractor."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

from anymal_locomotion_ros2.slam_confidence_core import (
    DegradationReason,
    TrackingState,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROS_PACKAGE = (
    PROJECT_ROOT
    / "deployment"
    / "ros2_ws"
    / "src"
    / "anymal_locomotion_ros2"
)
NODE_PATH = ROS_PACKAGE / "anymal_locomotion_ros2" / "fastlio_confidence_node.py"
BENCHMARK_LAUNCH = ROS_PACKAGE / "launch" / "fastlio2_benchmark.launch.py"
REPLAY_LAUNCH = ROS_PACKAGE / "launch" / "fastlio2_replay_benchmark.launch.py"
LOCOMOTION_LAUNCH = (
    ROS_PACKAGE / "launch" / "fastlio2_locomotion_benchmark.launch.py"
)
RUNTIME_CONFIG = ROS_PACKAGE / "config" / "slam_confidence_fastlio2.yaml"
CONTRACT_CONFIG = PROJECT_ROOT / "configs" / "slam_confidence_contract.yaml"
MESSAGE_PATH = (
    PROJECT_ROOT
    / "deployment"
    / "ros2_ws"
    / "src"
    / "anymal_locomotion_interfaces"
    / "msg"
    / "SlamConfidence.msg"
)


def _message_constants() -> dict[str, int]:
    constants: dict[str, int] = {}
    pattern = re.compile(r"^uint(?:8|16|32|64) ([A-Z][A-Z0-9_]*)=(\d+)$")
    for line in MESSAGE_PATH.read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            constants[match.group(1)] = int(match.group(2))
    return constants


def test_core_enums_are_wire_compatible() -> None:
    constants = _message_constants()
    for state in TrackingState:
        assert constants[f"STATE_{state.name}"] == int(state)
    for reason in DegradationReason:
        assert constants[f"REASON_{reason.name}"] == int(reason)


def test_contract_reason_classes_cover_every_nonzero_wire_reason() -> None:
    contract = yaml.safe_load(CONTRACT_CONFIG.read_text(encoding="utf-8"))
    configured = {
        reason
        for reasons in contract["reason_classes"].values()
        for reason in reasons
    }
    expected = {
        name
        for name in _message_constants()
        if name.startswith("REASON_") and name != "REASON_NONE"
    }
    assert configured == expected


def test_runtime_thresholds_match_the_contract_starting_points() -> None:
    contract = yaml.safe_load(CONTRACT_CONFIG.read_text(encoding="utf-8"))
    runtime = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))[
        "anymal_fastlio_confidence_extractor"
    ]["ros__parameters"]
    freshness = contract["timing"]["provisional_freshness_s"]["fastlio2"]
    hysteresis = contract["hysteresis"]
    assembly = contract["bundle_assembly"]["fastlio2"]

    assert runtime["publish_rate_hz"] == contract["timing"][
        "logical_publish_rate_hz"
    ]
    assert runtime["future_stamp_tolerance_s"] == contract["timing"][
        "future_stamp_tolerance_s"
    ]
    assert runtime["source_timeout_s"] == freshness["confidence_source"]
    assert runtime["odometry_timeout_s"] == freshness["canonical_odometry"]
    assert runtime["lidar_timeout_s"] == freshness["lidar_input"]
    assert runtime["imu_timeout_s"] == freshness["imu_input"]
    for name, value in hysteresis.items():
        assert runtime[name] == value
    assert runtime["diagnostic_match_timeout_s"] == assembly[
        "diagnostic_match_timeout_s"
    ]
    assert runtime["max_pending_bundles"] == assembly[
        "max_pending_bundles"
    ]
    assert assembly["timeout_anchor"] == (
        "first_required_input_arrival_logical_time"
    )


def test_runtime_topics_match_the_contract_and_fast_input_mapping() -> None:
    contract = yaml.safe_load(CONTRACT_CONFIG.read_text(encoding="utf-8"))
    runtime = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))[
        "anymal_fastlio_confidence_extractor"
    ]["ros__parameters"]

    assert runtime["canonical_odometry_topic"] == contract["topics"][
        "canonical_odometry"
    ]
    assert runtime["confidence_topic"] == contract["topics"]["authoritative"]
    assert runtime["tracking_valid_topic"] == contract["topics"][
        "tracking_valid_mirror"
    ]
    assert runtime["native_odometry_topic"] == "/Odometry"
    assert runtime["effective_points_topic"] == "/cloud_effected"
    assert runtime["lidar_input_topic"] == "/fastlio/points"
    assert runtime["imu_topic"] == "/imu/data"


def test_extractor_has_no_ground_truth_subscription() -> None:
    tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
    exact_strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "/odom" not in exact_strings
    assert "/slam/odom" in exact_strings
    assert "/Odometry" in exact_strings
    assert "/cloud_effected" in exact_strings
    assert "/fastlio/points" in exact_strings
    assert "/imu/data" in exact_strings


def test_extractor_loads_only_validated_artifact_and_otherwise_fail_closes() -> None:
    source = NODE_PATH.read_text(encoding="utf-8")
    assert '_UNCALIBRATED_ID = "uncalibrated"' in source
    assert "DegradationReason.UNCALIBRATED" in source
    assert "CalibratedConfidenceEstimator" in source
    assert "self._estimator.confidence if self._estimator else 0.0" in source
    assert "calibration_available" not in source


def test_package_installs_config_and_console_entry_point() -> None:
    setup_source = (ROS_PACKAGE / "setup.py").read_text(encoding="utf-8")
    assert 'glob("config/*.yaml")' in setup_source
    assert (
        "fastlio_confidence_extractor = "
        "anymal_locomotion_ros2.fastlio_confidence_node:main"
    ) in setup_source
    assert (
        "fastlio_confidence_fault_validation = "
        "anymal_locomotion_ros2.fastlio_confidence_fault_validation:main"
    ) in setup_source
    assert (
        "slam_confidence_dds_fault_validation = "
        "anymal_locomotion_ros2.slam_confidence_dds_fault_validation:main"
    ) in setup_source


def test_benchmark_gates_effect_output_and_extractor_together() -> None:
    source = BENCHMARK_LAUNCH.read_text(encoding="utf-8")
    assert '"enable_confidence",\n                default_value="false"' in source
    assert '"publish.effect_en": ParameterValue(' in source
    assert "condition=IfCondition(enable_confidence)" in source
    assert '("/Odometry", candidate_odom_topic)' in source
    assert '("/cloud_effected", effective_points_topic)' in source


def test_replay_and_locomotion_wrappers_keep_instrumentation_opt_in() -> None:
    for path in (REPLAY_LAUNCH, LOCOMOTION_LAUNCH):
        source = path.read_text(encoding="utf-8")
        assert '"enable_confidence": enable_confidence' in source
        assert '"enable_confidence",' in source
        assert 'default_value="false"' in source


def test_locomotion_wrapper_exposes_paired_seed_and_shared_degradation() -> None:
    source = LOCOMOTION_LAUNCH.read_text(encoding="utf-8")
    assert 'LaunchConfiguration("simulation_seed")' in source
    assert '"--seed",' in source
    for argument in (
        "point_density",
        "point_density_profile",
        "point_density_min",
    ):
        assert f'LaunchConfiguration("{argument}")' in source
        assert source.count(f'"{argument}": {argument}') == 2
    assert 'choices=["constant", "gradual_v1", "gradual_v2"]' in source
    assert 'LaunchConfiguration("record_bag")' in source
    assert 'LaunchConfiguration("confidence_loss_is_outcome")' in source
    assert '"allow_expected_tracking_loss": ParameterValue(' in source
    assert '"--disable-keyboard-controls"' not in source
    for topic in (
        "/lidar/points_raw",
        "/simulation/episode_reset_ack",
        "/slam_confidence",
    ):
        assert f'"{topic}"' in source


def test_fast_replay_evaluator_uses_fast_backend_signals() -> None:
    source = REPLAY_LAUNCH.read_text(encoding="utf-8")
    assert '"backend_kind": "fastlio2"' in source
    assert '"adapted_cloud_topic": "/fastlio/points"' in source


def test_replay_calibration_capture_declares_native_profile() -> None:
    source = REPLAY_LAUNCH.read_text(encoding="utf-8")
    assert '"deskew_mode": "native"' in source
    assert '"confidence_dataset_path"' in source
