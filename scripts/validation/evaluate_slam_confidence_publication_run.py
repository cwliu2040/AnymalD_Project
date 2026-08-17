#!/usr/bin/env python3
"""Generate offline usability labels and false-stop metrics from one live bag."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS_PACKAGE_SOURCE = PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"
sys.path.insert(0, str(ROS_PACKAGE_SOURCE))

from anymal_locomotion_ros2.slam_confidence_label_core import (
    EvaluationRecord,
    OfflineLabelConfig,
    OfflineLabelSample,
    OfflinePoseSample,
    generate_offline_labels,
)
from anymal_locomotion_ros2.lio_benchmark_core import PoseSample, evaluate_trajectory


def _project_file(value: Path, *, must_exist: bool = True) -> Path:
    resolved = value.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path escapes project: {resolved}")
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _nanoseconds(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _pose_sample(message: Any) -> OfflinePoseSample:
    quaternion = message.pose.pose.orientation
    x, y, z, w = (
        float(quaternion.x), float(quaternion.y),
        float(quaternion.z), float(quaternion.w),
    )
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    position = message.pose.pose.position
    return OfflinePoseSample(
        stamp_ns=_nanoseconds(message.header.stamp),
        x=float(position.x), y=float(position.y), z=float(position.z),
        yaw=yaw, roll=roll, pitch=pitch,
    )


def _deduplicate_poses(samples: list[OfflinePoseSample]) -> list[OfflinePoseSample]:
    return [value for _, value in sorted({sample.stamp_ns: sample for sample in samples}.items())]


def read_publication_bag(bag_dir: Path) -> dict[str, Any]:
    """Deserialize only the offline evaluation topics from a rosbag2 directory."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError("source the ROS 2 publication overlay before reading bags") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    required = {"/odom", "/slam/odom", "/slam_confidence"}
    missing = sorted(required - set(topic_types))
    if missing:
        raise ValueError(f"bag lacks offline evaluation topics: {missing}")
    message_types = {topic: get_message(topic_types[topic]) for topic in required}
    truth: list[OfflinePoseSample] = []
    estimates: list[OfflinePoseSample] = []
    evaluations: list[EvaluationRecord] = []
    confidence: list[dict[str, Any]] = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in required:
            continue
        message = deserialize_message(data, message_types[topic])
        if topic == "/odom":
            truth.append(_pose_sample(message))
        elif topic == "/slam/odom":
            estimates.append(_pose_sample(message))
        else:
            evaluation_ns = _nanoseconds(message.evaluation_stamp)
            source_ns = _nanoseconds(message.source_stamp) if message.source_stamp_valid else None
            evaluations.append(EvaluationRecord(evaluation_ns, source_ns))
            confidence.append(
                {
                    "evaluation_stamp_ns": evaluation_ns,
                    "source_stamp_ns": source_ns,
                    "confidence": float(message.slam_confidence),
                    "tracking_valid": bool(message.slam_tracking_valid),
                    "tracking_state": int(message.tracking_state),
                    "degradation_reasons": int(message.degradation_reasons),
                }
            )
    evaluation_by_stamp = {
        value.evaluation_stamp_ns: (value, sample)
        for value, sample in zip(evaluations, confidence, strict=True)
    }
    ordered = [evaluation_by_stamp[key] for key in sorted(evaluation_by_stamp)]
    return {
        "truth": _deduplicate_poses(truth),
        "estimates": _deduplicate_poses(estimates),
        "evaluations": [pair[0] for pair in ordered],
        "confidence": [pair[1] for pair in ordered],
    }


def label_config(contract: dict[str, Any]) -> OfflineLabelConfig:
    target = contract["offline_target"]
    return OfflineLabelConfig(
        prediction_horizon_ns=round(float(target["prediction_horizon_s"]) * 1e9),
        odometry_outage_ns=round(float(target["odometry_outage_s"]) * 1e9),
        pose_error_sustain_ns=round(float(target["pose_error_sustain_s"]) * 1e9),
        translation_error_m=float(target["translation_error_m"]),
        yaw_error_deg=float(target["yaw_error_deg"]),
        translation_jump_residual_m=float(target["translation_jump_residual_m"]),
        yaw_jump_residual_deg=float(target["yaw_jump_residual_deg"]),
    )


def evaluate_false_stops(
    *, labels: list[OfflineLabelSample], policy_records: list[dict[str, Any]], arm: str,
) -> dict[str, Any]:
    """Evaluate policy ticks against the latest causal evaluation-grid label."""
    normalized_arm = arm.upper()
    if normalized_arm not in {"A", "B", "C", "D"}:
        raise ValueError("arm must be A, B, C, or D")
    stamps = [label.evaluation_stamp_ns for label in labels]
    known = matched = requested = confidence_stops = false_stops = 0
    records: list[dict[str, Any]] = []
    for record in policy_records:
        evaluation_ns = round(float(record["clock_s"]) * 1e9)
        index = bisect.bisect_right(stamps, evaluation_ns) - 1
        if index < 0:
            continue
        matched += 1
        label = labels[index]
        command = record["received_command"]
        requested_motion = math.hypot(float(command[0]), float(command[1])) >= 0.25
        if normalized_arm == "A":
            safe_scale = 1.0
        else:
            observation = record["observation"]
            if len(observation) != 51:
                raise ValueError(f"arm {normalized_arm} requires 51-D diagnostics")
            confidence = float(observation[48])
            valid = max(0.0, min(1.0, float(observation[49])))
            safe_scale = valid * max(0.0, min(1.0, (confidence - 0.2) / 0.8))
        is_confidence_stop = requested_motion and safe_scale < 0.20
        usable = label.usable_next_horizon
        if usable is not None:
            known += 1
        if requested_motion:
            requested += 1
        if is_confidence_stop and usable is not None:
            confidence_stops += 1
            false_stops += int(usable)
        records.append(
            {
                "policy_clock_ns": evaluation_ns,
                "label_evaluation_stamp_ns": label.evaluation_stamp_ns,
                "safe_scale": safe_scale,
                "requested_motion": requested_motion,
                "confidence_stop": is_confidence_stop,
                "usable_next_horizon": usable,
            }
        )
    return {
        "arm": normalized_arm,
        "policy_record_count": len(policy_records),
        "matched_policy_record_count": matched,
        "known_label_policy_record_count": known,
        "requested_motion_record_count": requested,
        "confidence_stop_known_label_count": confidence_stops,
        "false_stop_count": false_stops,
        "false_stop_fraction": (
            false_stops / confidence_stops if confidence_stops else None
        ),
        "records": records,
    }


def evaluate_tracking_survival(
    confidence: list[dict[str, Any]], *, sustained_loss_ns: int = 250_000_000,
) -> dict[str, Any]:
    """Summarize survival after the first valid tracking sample."""
    ordered = sorted(confidence, key=lambda value: value["evaluation_stamp_ns"])
    first_valid = next(
        (index for index, value in enumerate(ordered) if value["tracking_valid"]), None,
    )
    if first_valid is None:
        return {
            "tracking_ever_valid": False,
            "event_observed": True,
            "survival_time_s": 0.0,
            "observation_time_s": 0.0,
            "tracking_valid_fraction": 0.0,
        }
    active = ordered[first_valid:]
    start_ns = active[0]["evaluation_stamp_ns"]
    end_ns = active[-1]["evaluation_stamp_ns"]
    invalid_start: int | None = None
    event_ns: int | None = None
    for value in active:
        stamp_ns = value["evaluation_stamp_ns"]
        if value["tracking_valid"]:
            invalid_start = None
        else:
            invalid_start = stamp_ns if invalid_start is None else invalid_start
            if stamp_ns - invalid_start >= sustained_loss_ns:
                event_ns = invalid_start
                break
    return {
        "tracking_ever_valid": True,
        "event_observed": event_ns is not None,
        "survival_time_s": ((event_ns if event_ns is not None else end_ns) - start_ns) / 1e9,
        "observation_time_s": (end_ns - start_ns) / 1e9,
        "tracking_valid_fraction": sum(value["tracking_valid"] for value in active) / len(active),
    }


def evaluate_trajectory_metrics(
    ground_truth: list[OfflinePoseSample], estimates: list[OfflinePoseSample],
) -> dict[str, Any]:
    def convert(sample: OfflinePoseSample) -> PoseSample:
        return PoseSample(
            stamp_s=sample.stamp_ns / 1e9,
            x=sample.x, y=sample.y, z=sample.z, yaw=sample.yaw,
            roll=sample.roll, pitch=sample.pitch,
        )

    return evaluate_trajectory(
        [convert(sample) for sample in ground_truth],
        [convert(sample) for sample in estimates],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--policy-diagnostics", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--contract", type=Path, default=PROJECT_ROOT / "configs/slam_confidence_contract.yaml")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bag_dir = args.bag.expanduser().resolve()
    if not bag_dir.is_relative_to(PROJECT_ROOT) or not (bag_dir / "metadata.yaml").is_file():
        raise ValueError("bag must be a project-local rosbag2 directory")
    diagnostics_path = _project_file(args.policy_diagnostics)
    contract_path = _project_file(args.contract)
    output_path = _project_file(args.output, must_exist=False)
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    bag = read_publication_bag(bag_dir)
    config = label_config(contract)
    labels = generate_offline_labels(
        evaluations=bag["evaluations"], estimates=bag["estimates"],
        ground_truth=bag["truth"], config=config,
    )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    false_stops = evaluate_false_stops(
        labels=labels, policy_records=diagnostics["records"], arm=args.arm,
    )
    tracking = evaluate_tracking_survival(bag["confidence"])
    trajectory = evaluate_trajectory_metrics(bag["truth"], bag["estimates"])
    matched_fraction = false_stops["matched_policy_record_count"] / max(
        1, false_stops["policy_record_count"]
    )
    gate = {
        "passed": bool(len(labels) >= 25 and matched_fraction >= 0.95),
        "label_count": len(labels),
        "matched_policy_fraction": matched_fraction,
        "ground_truth_role": "offline_label_only",
    }
    output = {
        "schema_version": 1,
        "kind": "slam_confidence_publication_offline_usability",
        "arm": args.arm,
        "config": asdict(config),
        "gate": gate,
        "false_stop": false_stops,
        "tracking": tracking,
        "trajectory": trajectory,
        "labels": [asdict(label) for label in labels],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "false_stop": {key: value for key, value in false_stops.items() if key != "records"}}, indent=2))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
