#!/usr/bin/env python3
"""Replay estimator15 from recorded IMU/joints/foot contacts and verify parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))

from anymal_locomotion_ros2.onnx_backend import OnnxBackend
from anymal_locomotion_ros2.policy_core import (
    PolicyContract,
    canonical_joint_state,
    projected_gravity_from_quaternion,
)
from anymal_locomotion_ros2.proprioceptive_velocity_estimator_core import (
    EstimatorInputBundle,
    EstimatorInputSynchronizer,
    EstimatorRuntime,
    assemble_step,
    load_estimator_metadata,
    reorder_foot_contacts,
)


def _stamp_ns(message: Any) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_estimates(
    replay: dict[int, np.ndarray], recorded: dict[int, np.ndarray], *, atol: float,
) -> dict[str, Any]:
    common = sorted(set(replay) & set(recorded))
    errors = [float(np.max(np.abs(replay[stamp] - recorded[stamp]))) for stamp in common]
    matched_fraction = len(common) / max(1, len(replay))
    maximum = max(errors, default=None)
    return {
        "replay_output_count": len(replay),
        "recorded_output_count": len(recorded),
        "exact_stamp_match_count": len(common),
        "matched_replay_fraction": matched_fraction,
        "maximum_absolute_velocity_error_mps": maximum,
        "atol_mps": atol,
        "passed": bool(
            replay and recorded and matched_fraction >= 0.95
            and maximum is not None and maximum <= atol
        ),
    }


def replay_bag(
    bag_dir: Path, estimator_metadata: Path, policy_metadata: Path,
    *, sync_tolerance_s: float = 0.025,
) -> tuple[dict[str, int], dict[int, np.ndarray], dict[int, np.ndarray]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError("source the ROS 2 publication overlay before reading bags") from exc
    _, model_path = load_estimator_metadata(estimator_metadata)
    contract = PolicyContract.from_metadata(policy_metadata)
    defaults = np.asarray(contract.default_joint_positions, dtype=np.float32)
    runtime = EstimatorRuntime(OnnxBackend(str(model_path)))
    synchronizer = EstimatorInputSynchronizer(sync_tolerance_s)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    reset_ack_topic = "/simulation/episode_reset_ack"
    required = {
        "/imu/data", "/joint_states", "/foot_contacts",
        "/locomotion/estimated_odom", reset_ack_topic,
    }
    missing = sorted(required - set(types))
    if missing:
        raise ValueError(f"bag lacks estimator replay topics: {missing}")
    message_types = {topic: get_message(types[topic]) for topic in required}
    counts = {topic: 0 for topic in required}
    replay: dict[int, np.ndarray] = {}
    recorded: dict[int, np.ndarray] = {}

    def consume(bundles: list[EstimatorInputBundle]) -> None:
        for bundle in bundles:
            if not bundle.synchronized:
                runtime.reset()
                continue
            message = bundle.joint
            angular, acceleration, gravity = bundle.imu
            try:
                positions, velocities = canonical_joint_state(
                    message.name, message.position, message.velocity, contract,
                )
                step = assemble_step(
                    angular, acceleration, gravity,
                    positions - defaults, velocities, bundle.contacts,
                )
                estimate = runtime.step(
                    bundle.joint_stamp_ns * 1.0e-9, step
                )
                if estimate is not None:
                    replay[bundle.joint_stamp_ns] = estimate
            except ValueError:
                runtime.reset()

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in required:
            continue
        message = deserialize_message(data, message_types[topic])
        counts[topic] += 1
        if topic == reset_ack_topic:
            # The acknowledgement is emitted by policy_node only after its
            # estimator and input synchronizer have been reset.  Replaying
            # this marker preserves runtime episode-boundary semantics while
            # retaining rosbag serialization order across all input topics.
            runtime.reset()
            synchronizer.reset()
            continue
        stamp_ns = _stamp_ns(message)
        if topic == "/imu/data":
            if message.header.frame_id != "base_link":
                runtime.reset()
                synchronizer.reset()
                continue
            try:
                gravity = projected_gravity_from_quaternion(
                    message.orientation.x, message.orientation.y,
                    message.orientation.z, message.orientation.w,
                )
                angular = np.asarray(
                    (message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z),
                    dtype=np.float32,
                )
                acceleration = np.asarray(
                    (message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z),
                    dtype=np.float32,
                )
                if not np.isfinite(angular).all() or not np.isfinite(acceleration).all():
                    raise ValueError
                consume(
                    synchronizer.push_imu(
                        stamp_ns, (angular, acceleration, gravity)
                    )
                )
            except ValueError:
                runtime.reset()
                synchronizer.reset()
        elif topic == "/foot_contacts":
            if message.header.frame_id != "base_link":
                runtime.reset()
                synchronizer.reset()
                continue
            try:
                consume(
                    synchronizer.push_contacts(
                        stamp_ns,
                        reorder_foot_contacts(
                            message.foot_names, message.contact_probabilities
                        ),
                    )
                )
            except ValueError:
                runtime.reset()
                synchronizer.reset()
        elif topic == "/locomotion/estimated_odom":
            recorded[stamp_ns] = np.asarray(
                (message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.linear.z),
                dtype=np.float32,
            )
        else:
            try:
                consume(synchronizer.push_joint(stamp_ns, message))
            except ValueError:
                runtime.reset()
                synchronizer.reset()
    return counts, replay, recorded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--estimator-metadata", type=Path, required=True)
    parser.add_argument("--policy-metadata", type=Path, required=True)
    parser.add_argument("--sync-tolerance-s", type=float, default=0.025)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = [value.expanduser().resolve() for value in (args.bag, args.estimator_metadata, args.policy_metadata, args.output)]
    bag, estimator_metadata, policy_metadata, output = paths
    if not all(path.is_relative_to(PROJECT_ROOT) for path in paths):
        raise ValueError("estimator replay paths must remain inside the project")
    if not (bag / "metadata.yaml").is_file():
        raise ValueError("bag is not a rosbag2 directory")
    counts, replay, recorded = replay_bag(
        bag, estimator_metadata, policy_metadata,
        sync_tolerance_s=args.sync_tolerance_s,
    )
    parity = compare_estimates(replay, recorded, atol=args.atol)
    report = {
        "schema_version": 1,
        "kind": "proprioceptive_velocity_estimator_live_replay_parity",
        "estimator_metadata_path": str(estimator_metadata),
        "estimator_metadata_sha256": _sha256(estimator_metadata),
        "policy_metadata_path": str(policy_metadata),
        "policy_metadata_sha256": _sha256(policy_metadata),
        "ground_truth_used_by_runtime_or_replay": False,
        "synchronization": {
            "method": "source_stamp_nearest_after_per_topic_watermark",
            "tolerance_s": args.sync_tolerance_s,
            "cross_topic_callback_order_independent": True,
            "episode_boundary": "policy_episode_reset_ack",
        },
        "topic_counts": counts,
        "parity": parity,
        "gate": {"passed": parity["passed"]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"topic_counts": counts, "parity": parity, "gate": report["gate"]}, indent=2))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
