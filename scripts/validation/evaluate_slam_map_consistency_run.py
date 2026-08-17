#!/usr/bin/env python3
"""Build GT/SLAM surfaces from identical raw scans and score map consistency."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS_PACKAGE_SOURCE = PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"
sys.path.insert(0, str(ROS_PACKAGE_SOURCE))

from anymal_locomotion_ros2.map_consistency_core import (
    SurfacePose,
    build_observed_surface,
    evaluate_map_consistency,
)


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _pose(message: Any) -> SurfacePose:
    orientation = message.pose.pose.orientation
    sinr_cosp = 2.0 * (orientation.w * orientation.x + orientation.y * orientation.z)
    cosr_cosp = 1.0 - 2.0 * (orientation.x * orientation.x + orientation.y * orientation.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(
        -1.0,
        min(1.0, 2.0 * (orientation.w * orientation.y - orientation.z * orientation.x)),
    )
    pitch = math.asin(sinp)
    yaw = math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )
    position = message.pose.pose.position
    return SurfacePose(
        stamp_ns=_stamp_ns(message.header.stamp),
        x=float(position.x), y=float(position.y), z=float(position.z), yaw=yaw,
        roll=roll, pitch=pitch,
    )


def _read_bag(
    bag_dir: Path, points_per_scan: int,
) -> tuple[list[tuple[int, np.ndarray]], list[SurfacePose], list[SurfacePose]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs_py.point_cloud2 import read_points_numpy
    except ImportError as exc:
        raise RuntimeError("source the ROS 2 publication overlay before reading bags") from exc
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    required = {"/odom", "/slam/odom", "/lidar/points_raw"}
    missing = sorted(required - set(types))
    if missing:
        raise ValueError(f"bag lacks map evaluation topics: {missing}")
    message_types = {topic: get_message(types[topic]) for topic in required}
    scans: list[tuple[int, np.ndarray]] = []
    truth: dict[int, SurfacePose] = {}
    estimates: dict[int, SurfacePose] = {}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in required:
            continue
        message = deserialize_message(data, message_types[topic])
        if topic == "/odom":
            sample = _pose(message)
            truth[sample.stamp_ns] = sample
        elif topic == "/slam/odom":
            sample = _pose(message)
            estimates[sample.stamp_ns] = sample
        else:
            points = np.asarray(
                read_points_numpy(message, field_names=("x", "y", "z"), skip_nans=True),
                dtype=np.float64,
            ).reshape(-1, 3)
            finite = points[np.isfinite(points).all(axis=1)]
            if len(finite) >= 20:
                indices = np.linspace(
                    0, len(finite) - 1, min(points_per_scan, len(finite)), dtype=np.int64,
                )
                scans.append((_stamp_ns(message.header.stamp), finite[indices]))
    return scans, list(truth.values()), list(estimates.values())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-scan", type=int, default=32)
    parser.add_argument("--maximum-points", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bag_dir = args.bag.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not bag_dir.is_relative_to(PROJECT_ROOT) or not (bag_dir / "metadata.yaml").is_file():
        raise ValueError("bag must be a project-local rosbag2 directory")
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("output must remain inside the project")
    scans, truth, estimates = _read_bag(bag_dir, args.points_per_scan)
    reference = build_observed_surface(scans, truth, points_per_scan=args.points_per_scan)
    estimated = build_observed_surface(scans, estimates, points_per_scan=args.points_per_scan)
    result = evaluate_map_consistency(estimated, reference, maximum_points=args.maximum_points)
    payload = {
        "schema_version": 1,
        "kind": "slam_confidence_publication_map_consistency",
        "ground_truth_role": "offline_reference_surface_only",
        "surface_contract": {
            "same_raw_scan_samples_for_estimate_and_reference": True,
            "base_to_lidar_translation_xyz_m": [0.20, 0.0, 0.35],
            "points_per_scan": args.points_per_scan,
            "raw_scan_count": len(scans),
            "reference_surface_point_count": len(reference),
            "estimated_surface_point_count": len(estimated),
        },
        "metrics": asdict(result),
        "gate": {
            "passed": len(scans) >= 10 and result.evaluated_point_count >= 500,
            "role": "measurement_completeness_only",
            "efficacy_threshold_applied": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
