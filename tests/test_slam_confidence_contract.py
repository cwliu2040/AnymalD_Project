"""Static checks for the backend-neutral SLAM confidence contract."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_CONFIG = PROJECT_ROOT / "configs" / "slam_confidence_contract.yaml"
INTERFACE_ROOT = (
    PROJECT_ROOT
    / "deployment"
    / "ros2_ws"
    / "src"
    / "anymal_locomotion_interfaces"
)
MESSAGE_PATH = INTERFACE_ROOT / "msg" / "SlamConfidence.msg"
RUNTIME_MANIFEST = (
    PROJECT_ROOT
    / "deployment"
    / "ros2_ws"
    / "src"
    / "anymal_locomotion_ros2"
    / "package.xml"
)
INTEGER_CONSTANT_PATTERN = re.compile(
    r"^uint(?:8|16|32|64) ([A-Z][A-Z0-9_]*)=(\d+)$"
)


def _contract() -> dict:
    return yaml.safe_load(CONTRACT_CONFIG.read_text(encoding="utf-8"))


def _message_lines() -> list[str]:
    return [
        line.strip()
        for line in MESSAGE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _integer_constants() -> dict[str, int]:
    constants: dict[str, int] = {}
    for line in _message_lines():
        match = INTEGER_CONSTANT_PATTERN.fullmatch(line)
        if match:
            constants[match.group(1)] = int(match.group(2))
    return constants


def test_message_layout_contains_atomic_v1_fields() -> None:
    constants = _integer_constants()
    assert constants["SCHEMA_VERSION"] == 1
    fields = [
        line
        for line in _message_lines()
        if INTEGER_CONSTANT_PATTERN.fullmatch(line) is None
    ]
    assert fields == [
        "uint8 schema_version",
        "string<=32 backend_id",
        "string<=128 calibration_id",
        "bool source_stamp_valid",
        "builtin_interfaces/Time source_stamp",
        "builtin_interfaces/Time evaluation_stamp",
        "builtin_interfaces/Duration confidence_age",
        "float32 slam_confidence",
        "bool slam_tracking_valid",
        "uint8 tracking_state",
        "uint32 degradation_reasons",
    ]


def test_state_and_reason_constants_are_unambiguous() -> None:
    constants = _integer_constants()
    assert {
        name: constants[name]
        for name in (
            "STATE_INITIALIZING",
            "STATE_TRACKING",
            "STATE_DEGRADED",
            "STATE_LOST",
        )
    } == {
        "STATE_INITIALIZING": 0,
        "STATE_TRACKING": 1,
        "STATE_DEGRADED": 2,
        "STATE_LOST": 3,
    }

    reason_values = {
        name: value for name, value in constants.items() if name.startswith("REASON_")
    }
    assert reason_values["REASON_NONE"] == 0
    nonzero = [value for name, value in reason_values.items() if name != "REASON_NONE"]
    assert len(nonzero) == len(set(nonzero))
    assert all(value > 0 and value & (value - 1) == 0 for value in nonzero)
    for required in (
        "REASON_SIGNAL_MISSING",
        "REASON_LOW_CONFIDENCE",
        "REASON_RECOVERY_PENDING",
        "REASON_CLOCK_RESET",
        "REASON_UNCALIBRATED",
    ):
        assert required in reason_values


def test_config_uses_one_atomic_topic_and_explicit_ground_truth_boundary() -> None:
    contract = _contract()
    assert contract["schema_version"] == _integer_constants()["SCHEMA_VERSION"]
    assert (
        contract["status"]
        == "both_backends_calibrated_native_artifacts_installed"
    )
    assert contract["topics"] == {
        "authoritative": "/slam_confidence",
        "tracking_valid_mirror": "/slam_tracking_valid",
        "canonical_odometry": "/slam/odom",
        "forbidden_runtime_ground_truth": "/odom",
    }
    assert contract["source_stamp"]["matches_canonical_odometry_header"] is True
    assert contract["source_stamp"]["advanced_by_imu_only"] is False
    assert contract["source_stamp"]["advanced_by_lidar_only"] is False


def test_hysteresis_and_freshness_starting_points_are_fail_closed() -> None:
    contract = _contract()
    hysteresis = contract["hysteresis"]
    assert 0.0 < hysteresis["invalidate_below"] < hysteresis["degrade_below"]
    assert hysteresis["degrade_below"] < hysteresis["recover_at_or_above"] <= 1.0
    assert hysteresis["degrade_dwell_s"] > 0.0
    assert hysteresis["invalidate_dwell_s"] >= hysteresis["degrade_dwell_s"]
    assert hysteresis["recover_dwell_s"] > hysteresis["invalidate_dwell_s"]

    freshness = contract["timing"]["provisional_freshness_s"]
    assert freshness["liosam"]["canonical_odometry"] > freshness["fastlio2"][
        "canonical_odometry"
    ]
    assert all(
        value > 0.0
        for backend in freshness.values()
        for value in backend.values()
    )


def test_stale_high_confidence_is_explicitly_invalid() -> None:
    example = _contract()["stale_high_confidence_example"]
    assert example["source_stamp_valid"] is True
    assert example["slam_confidence"] > 0.9
    assert example["slam_tracking_valid"] is False
    assert example["tracking_state"] == "STATE_LOST"
    assert "REASON_SOURCE_STALE" in example["required_reasons"]


def test_score_and_offline_forecast_share_the_evaluation_anchor() -> None:
    contract = _contract()
    score = contract["score_semantics"]
    target = contract["offline_target"]

    assert score["forecast_anchor"] == "evaluation_stamp"
    assert score["update_trigger"] == "logical_evaluation_tick"
    assert score["wire_quantization_before_threshold"] == "float32"
    assert score["missing_score_on_healthy_tick"] == "hard_signal_missing"
    assert target["forecast_anchor"] == "evaluation_stamp"
    assert target["pose_alignment_stamp"] == "source_stamp"


def test_fast_bundle_assembly_is_exact_bounded_and_arrival_anchored() -> None:
    assembly = _contract()["bundle_assembly"]["fastlio2"]

    assert assembly["required_exact_stamp_inputs"] == [
        "native_odometry",
        "canonical_odometry",
        "effective_points",
    ]
    assert assembly["diagnostic_match_timeout_s"] > 0.0
    assert assembly["timeout_anchor"] == (
        "first_required_input_arrival_logical_time"
    )
    assert assembly["timeout_clock"] == "ros_logical_time"
    assert assembly["max_pending_bundles"] >= 2
    assert assembly["pending_overflow"] == (
        "one_shot_signal_missing_then_evict_oldest"
    )
    assert assembly["commit_order"] == "monotonic_source_stamp"
    assert assembly["older_incomplete_crossed_by_new_complete"] == (
        "one_shot_signal_missing_then_evict"
    )


def test_validation_statistics_use_independent_capture_groups() -> None:
    contract = _contract()
    target = contract["offline_target"]
    validation = contract["validation"]
    split = contract["dataset_split"]
    assert target["prediction_horizon_s"] > validation[
        "gradual_event_median_lead_time_s_min"
    ]
    assert validation["minimum_independent_gradual_events"] >= 20
    assert validation["minimum_capture_groups"] >= 5
    assert validation["bootstrap_unit"] == "bag_or_episode"
    assert validation["bootstrap_confidence_level"] == 0.95
    assert split["unit"] == "capture_group"
    assert sum(
        split[key]
        for key in (
            "train_fraction",
            "probability_calibration_fraction",
            "threshold_validation_fraction",
            "final_holdout_fraction",
        )
    ) == 1.0


def test_interface_package_is_in_the_deployment_build_closure() -> None:
    interface_manifest = ET.parse(INTERFACE_ROOT / "package.xml").getroot()
    buildtools = {node.text for node in interface_manifest.findall("buildtool_depend")}
    dependencies = {node.text for node in interface_manifest.findall("depend")}
    assert {"ament_cmake", "rosidl_default_generators"} <= buildtools
    assert "builtin_interfaces" in dependencies

    runtime_manifest = ET.parse(RUNTIME_MANIFEST).getroot()
    runtime_dependencies = {
        node.text for node in runtime_manifest.findall("exec_depend")
    }
    assert "anymal_locomotion_interfaces" in runtime_dependencies
    assert "builtin_interfaces" in runtime_dependencies

    setup_source = (PROJECT_ROOT / "scripts" / "setup_deployment.sh").read_text(
        encoding="utf-8"
    )
    assert "install/anymal_locomotion_interfaces" in setup_source
    assert "SlamConfidence.msg" in setup_source
