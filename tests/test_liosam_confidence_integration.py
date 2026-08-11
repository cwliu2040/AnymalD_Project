"""Static integration checks for the LIO-SAM confidence extractor."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROS_PACKAGE = (
    PROJECT_ROOT
    / "deployment"
    / "ros2_ws"
    / "src"
    / "anymal_locomotion_ros2"
)
NODE_PATH = ROS_PACKAGE / "anymal_locomotion_ros2" / "liosam_confidence_node.py"
ADAPTER_PATH = ROS_PACKAGE / "anymal_locomotion_ros2" / "liosam_odom_adapter.py"
RUNTIME_CONFIG = ROS_PACKAGE / "config" / "slam_confidence_liosam.yaml"
CONTRACT_CONFIG = PROJECT_ROOT / "configs" / "slam_confidence_contract.yaml"
LIO_LAUNCH = ROS_PACKAGE / "launch" / "lio_sam.launch.py"


def _runtime() -> dict:
    return yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))[
        "anymal_liosam_confidence_extractor"
    ]["ros__parameters"]


def _contract() -> dict:
    return yaml.safe_load(CONTRACT_CONFIG.read_text(encoding="utf-8"))


def test_liosam_runtime_thresholds_match_common_contract() -> None:
    runtime = _runtime()
    contract = _contract()
    freshness = contract["timing"]["provisional_freshness_s"]["liosam"]
    assembly = contract["bundle_assembly"]["liosam"]

    assert runtime["publish_rate_hz"] == contract["timing"][
        "logical_publish_rate_hz"
    ]
    assert runtime["source_timeout_s"] == freshness["confidence_source"]
    assert runtime["odometry_timeout_s"] == freshness[
        "canonical_odometry"
    ]
    assert runtime["lidar_timeout_s"] == freshness["lidar_input"]
    assert runtime["imu_timeout_s"] == freshness["imu_input"]
    assert runtime["future_stamp_tolerance_s"] == contract["timing"][
        "future_stamp_tolerance_s"
    ]
    assert runtime["diagnostic_match_timeout_s"] == assembly[
        "diagnostic_match_timeout_s"
    ]
    assert runtime["max_pending_bundles"] == assembly[
        "max_pending_bundles"
    ]
    minima = assembly["conservative_feature_hard_minima"]
    assert runtime["edge_feature_min_valid"] == minima["corner_at_or_below"]
    assert runtime["surface_feature_min_valid"] == minima[
        "surface_at_or_below"
    ]
    for name, value in contract["hysteresis"].items():
        assert runtime[name] == value


def test_liosam_topics_use_public_backend_signals_and_no_ground_truth() -> None:
    runtime = _runtime()
    assert runtime["native_odometry_topic"] == (
        "/lio_sam/mapping/odometry"
    )
    assert runtime["canonical_odometry_topic"] == "/slam/odom"
    assert runtime["incremental_odometry_topic"] == (
        "/lio_sam/mapping/odometry_incremental"
    )
    assert runtime["feature_cloud_info_topic"] == (
        "/lio_sam/feature/cloud_info"
    )
    assert runtime["lidar_input_topic"] == "/lio_sam/points"
    assert runtime["imu_topic"] == "/imu/data"
    assert runtime["motion_deskew_topic"] == "/lio_sam/deskew/motion"
    assert "/odom" not in runtime.values()

    tree = ast.parse(NODE_PATH.read_text(encoding="utf-8"))
    exact_strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "/odom" not in exact_strings


def test_liosam_extractor_supports_artifact_but_defaults_fail_closed() -> None:
    source = NODE_PATH.read_text(encoding="utf-8")
    assert '_BACKEND_ID = "liosam"' in source
    assert '_UNCALIBRATED_ID = "uncalibrated"' in source
    assert "DegradationReason.UNCALIBRATED" in source
    assert "CalibratedConfidenceEstimator" in source
    assert "self._estimator.confidence if self._estimator else 0.0" in source


def test_native_liosam_subscriptions_match_best_effort_publishers() -> None:
    node_source = NODE_PATH.read_text(encoding="utf-8")
    adapter_source = ADAPTER_PATH.read_text(encoding="utf-8")
    assert "_BEST_EFFORT_DEPTH_20" in node_source
    assert "_BEST_EFFORT_DEPTH_200" in node_source
    assert "ReliabilityPolicy.BEST_EFFORT" in node_source
    assert "_SOURCE_QOS" in adapter_source
    assert "ReliabilityPolicy.BEST_EFFORT" in adapter_source
    assert "_OUTPUT_QOS" in adapter_source


def test_odometry_adapter_applies_lidar_to_body_translation() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    assert '"source_frame_id", "odom"' in source
    assert '"source_child_frame_id", "odom_mapping"' in source
    assert '"output_frame_id", "map"' in source
    assert '"output_child_frame_id", "base_link"' in source
    assert "body_pose_from_sensor_pose" in source
    assert "sensor_translation_in_body_xyz" in source


def test_package_installs_liosam_entry_points_and_config() -> None:
    setup_source = (ROS_PACKAGE / "setup.py").read_text(encoding="utf-8")
    assert (
        "liosam_odom_adapter = "
        "anymal_locomotion_ros2.liosam_odom_adapter:main"
    ) in setup_source
    assert (
        "liosam_confidence_extractor = "
        "anymal_locomotion_ros2.liosam_confidence_node:main"
    ) in setup_source
    assert 'glob("config/*.yaml")' in setup_source


def test_liosam_launch_keeps_confidence_opt_in_and_atomic() -> None:
    source = LIO_LAUNCH.read_text(encoding="utf-8")
    assert '"enable_confidence",\n                default_value="false"' in source
    assert 'executable="liosam_odom_adapter"' in source
    assert 'executable="liosam_confidence_extractor"' in source
    assert source.count("condition=IfCondition(enable_confidence)") == 2
    assert "slam_confidence_liosam.yaml" in source
    assert "slam_confidence_liosam_native_v3.json" in source
    assert '"motion_deskew_required": ParameterValue(' in source


def test_formal_wrappers_pass_through_opt_in_without_policy_consumption() -> None:
    paths_and_names = (
        (ROS_PACKAGE / "launch" / "lio_benchmark.launch.py", "enable_confidence"),
        (
            ROS_PACKAGE / "launch" / "lio_replay_benchmark.launch.py",
            "enable_confidence",
        ),
        (ROS_PACKAGE / "launch" / "bringup.launch.py", "enable_slam_confidence"),
    )
    for path, argument_name in paths_and_names:
        source = path.read_text(encoding="utf-8")
        assert f'"{argument_name}"' in source
        assert 'default_value="false"' in source
    policy_source = (
        ROS_PACKAGE / "anymal_locomotion_ros2" / "policy_node.py"
    ).read_text(encoding="utf-8")
    assert "/slam_confidence" not in policy_source


def test_motion_deskew_status_does_not_mutate_cloud_info_header() -> None:
    source = (
        ROS_PACKAGE / "anymal_locomotion_ros2" / "motion_deskew_node.py"
    ).read_text(encoding="utf-8")
    assert "status.header = copy.deepcopy(message.header)" in source
    assert "status.header = message.header\n" not in source


def test_formal_replay_calibration_defaults_to_native_deskew() -> None:
    source = (
        ROS_PACKAGE / "launch" / "lio_replay_benchmark.launch.py"
    ).read_text(encoding="utf-8")
    assert '"use_motion_deskew",\n                default_value="false"' in source
    assert 'default_value="/lio_sam/deskew/cloud_info"' in source
    assert '"deskew_mode": "native"' in source
    assert '"confidence_dataset_path"' in source
    # Calibration replay deliberately overrides the deployed launch default
    # with an empty artifact path, preventing holdout feature contamination.
    assert '"confidence_artifact_path",\n                default_value=""' in source
