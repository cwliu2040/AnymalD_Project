"""Run one deterministic LIO-SAM motion profile and write compact metrics."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from lio_sam.msg import CloudInfo
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from anymal_locomotion_ros2.lio_benchmark_core import (
    PoseSample,
    evaluate_trajectory,
    get_motion_profile,
    interpolate_pose_sample,
    registered_overlap_separation,
)


def _stamp_seconds(message) -> float:
    return (
        float(message.header.stamp.sec)
        + float(message.header.stamp.nanosec) * 1.0e-9
    )


def _rpy_from_odometry(message: Odometry) -> tuple[float, float, float]:
    orientation = message.pose.pose.orientation
    norm = math.sqrt(
        orientation.x**2
        + orientation.y**2
        + orientation.z**2
        + orientation.w**2
    )
    if norm < 1.0e-9:
        raise ValueError("Odometry quaternion has near-zero norm")
    x = orientation.x / norm
    y = orientation.y / norm
    z = orientation.z / norm
    w = orientation.w / norm
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(
        max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    )
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return roll, pitch, yaw


def _pose_sample(message: Odometry) -> PoseSample:
    position = message.pose.pose.position
    linear = message.twist.twist.linear
    angular = message.twist.twist.angular
    roll, pitch, yaw = _rpy_from_odometry(message)
    return PoseSample(
        stamp_s=_stamp_seconds(message),
        x=float(position.x),
        y=float(position.y),
        z=float(position.z),
        yaw=yaw,
        linear_speed_mps=math.sqrt(
            linear.x**2 + linear.y**2 + linear.z**2
        ),
        angular_speed_rps=math.sqrt(
            angular.x**2 + angular.y**2 + angular.z**2
        ),
        yaw_rate_rps=float(angular.z),
        roll=roll,
        pitch=pitch,
    )


class LioBenchmarkNode(Node):
    """Publish a smooth command profile and evaluate LIO against sim truth."""

    def __init__(self) -> None:
        super().__init__("anymal_lio_benchmark")
        self.declare_parameter("profile", "stationary")
        self.declare_parameter("output_path", "")
        self.declare_parameter("project_root", "")
        self.declare_parameter("readiness_timeout_s", 12.0)
        self.declare_parameter("loop_closure_expectation", "disabled")

        self._profile = get_motion_profile(
            str(self.get_parameter("profile").value)
        )
        project_root = Path(
            str(self.get_parameter("project_root").value)
        ).expanduser().resolve()
        output_path = Path(
            str(self.get_parameter("output_path").value)
        ).expanduser().resolve()
        if not project_root.is_dir():
            raise ValueError(f"project_root is not a directory: {project_root}")
        if not output_path.is_relative_to(project_root):
            raise ValueError(
                f"Benchmark output must remain inside {project_root}: {output_path}"
            )
        self._output_path = output_path
        self._readiness_timeout_s = float(
            self.get_parameter("readiness_timeout_s").value
        )
        if self._readiness_timeout_s <= 0.0:
            raise ValueError("readiness_timeout_s must be positive")
        self._loop_closure_expectation = str(
            self.get_parameter("loop_closure_expectation").value
        )
        if self._loop_closure_expectation not in {
            "disabled",
            "required",
            "forbidden",
        }:
            raise ValueError(
                "loop_closure_expectation must be disabled, required, "
                f"or forbidden; received {self._loop_closure_expectation!r}"
            )

        self._ground_truth: list[PoseSample] = []
        self._estimate: list[PoseSample] = []
        self._first_sim_time_s: float | None = None
        self._start_time_s: float | None = None
        self._finished = False
        self.exit_code = 1

        self._counts = {
            "ground_truth_odometry": 0,
            "mapping_odometry": 0,
            "raw_cloud": 0,
            "adapted_cloud": 0,
            "cloud_info": 0,
            "imu": 0,
            "imu_available_after_ready": 0,
            "imu_unavailable_after_ready": 0,
            "odom_available_after_ready": 0,
            "odom_unavailable_after_ready": 0,
            "motion_deskew_applied": 0,
            "motion_deskew_unavailable": 0,
            "motion_deskew_unavailable_after_ready": 0,
            "loop_closure_marker": 0,
        }
        self._last_stamps: dict[str, float] = {}
        self._timestamp_violations: dict[str, int] = {}
        self._raw_intervals_s: list[float] = []
        self._adapted_intervals_s: list[float] = []
        self._cloud_info_intervals_s: list[float] = []
        self._adapted_time_spans_ns: list[int] = []
        self._last_cloud_info: CloudInfo | None = None
        self._last_motion_deskew_applied = False
        self._motion_deskew_norms_m: list[float] = []
        self._previous_truth_registered_xyz: np.ndarray | None = None
        self._registered_overlap_separations_m: list[float] = []
        self._registered_overlap_samples: list[dict[str, float]] = []
        self._loop_constraint_edge_count_max = 0

        self._command_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(
            Odometry,
            "/odom",
            self._on_ground_truth,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            "/lio_sam/mapping/odometry",
            self._on_mapping_odometry,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            "/lidar/points_raw",
            self._on_raw_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            "/lio_sam/points",
            self._on_adapted_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CloudInfo,
            "/lio_sam/deskew/cloud_info_motion_corrected",
            self._on_cloud_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Vector3Stamped,
            "/lio_sam/deskew/motion",
            self._on_motion_deskew,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            "/imu/data",
            self._on_imu,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            MarkerArray,
            "/lio_sam/mapping/loop_closure_constraints",
            self._on_loop_closure_constraints,
            10,
        )
        self.create_timer(0.05, self._on_timer)

    def _observe_stamp(self, name: str, stamp_s: float) -> None:
        previous = self._last_stamps.get(name)
        if previous is not None and stamp_s <= previous:
            self._timestamp_violations[name] = (
                self._timestamp_violations.get(name, 0) + 1
            )
        self._last_stamps[name] = stamp_s

    def _on_ground_truth(self, message: Odometry) -> None:
        sample = _pose_sample(message)
        self._observe_stamp("ground_truth_odometry", sample.stamp_s)
        self._ground_truth.append(sample)
        self._counts["ground_truth_odometry"] += 1

    def _on_mapping_odometry(self, message: Odometry) -> None:
        sample = _pose_sample(message)
        self._observe_stamp("mapping_odometry", sample.stamp_s)
        self._estimate.append(sample)
        self._counts["mapping_odometry"] += 1

    def _observe_cloud(
        self,
        name: str,
        message: PointCloud2,
        intervals: list[float],
    ) -> None:
        stamp_s = _stamp_seconds(message)
        previous = self._last_stamps.get(name)
        self._observe_stamp(name, stamp_s)
        if (
            previous is not None
            and stamp_s > previous
            and self._start_time_s is not None
        ):
            intervals.append(stamp_s - previous)
        self._counts[name] += 1

    def _on_raw_cloud(self, message: PointCloud2) -> None:
        self._observe_cloud("raw_cloud", message, self._raw_intervals_s)

    def _on_adapted_cloud(self, message: PointCloud2) -> None:
        self._observe_cloud(
            "adapted_cloud",
            message,
            self._adapted_intervals_s,
        )
        time_field = next(
            (field for field in message.fields if field.name == "t"),
            None,
        )
        point_count = int(message.width) * int(message.height)
        if (
            time_field is None
            or point_count == 0
            or int(message.point_step) <= int(time_field.offset)
        ):
            return
        times = np.ndarray(
            shape=(point_count,),
            dtype="<u4",
            buffer=message.data,
            offset=int(time_field.offset),
            strides=(int(message.point_step),),
        )
        if self._start_time_s is not None:
            self._adapted_time_spans_ns.append(
                int(times.max()) - int(times.min())
            )

    def _on_cloud_info(self, message: CloudInfo) -> None:
        stamp_s = _stamp_seconds(message)
        previous = self._last_stamps.get("cloud_info")
        self._observe_stamp("cloud_info", stamp_s)
        if (
            previous is not None
            and stamp_s > previous
            and self._start_time_s is not None
        ):
            self._cloud_info_intervals_s.append(stamp_s - previous)
        self._counts["cloud_info"] += 1
        self._last_cloud_info = message
        if self._start_time_s is not None:
            if bool(message.imu_available):
                self._counts["imu_available_after_ready"] += 1
            else:
                self._counts["imu_unavailable_after_ready"] += 1
            if bool(message.odom_available):
                self._counts["odom_available_after_ready"] += 1
            else:
                self._counts["odom_unavailable_after_ready"] += 1
            # Simulator truth is used only to register corrected scans for this
            # metric; it is never published back into the LIO pipeline.
            self._measure_wall_separation(message)

    def _on_imu(self, message: Imu) -> None:
        self._observe_stamp("imu", _stamp_seconds(message))
        self._counts["imu"] += 1

    def _on_loop_closure_constraints(
        self,
        message: MarkerArray,
    ) -> None:
        edge_count = max(
            (
                len(marker.points) // 2
                for marker in message.markers
                if marker.type == Marker.LINE_LIST
                and marker.ns == "loop_edges"
            ),
            default=0,
        )
        self._counts["loop_closure_marker"] += 1
        self._loop_constraint_edge_count_max = max(
            self._loop_constraint_edge_count_max,
            edge_count,
        )

    def _measure_wall_separation(self, message: CloudInfo) -> None:
        cloud = message.cloud_deskewed
        field_offsets = {
            field.name: int(field.offset) for field in cloud.fields
        }
        point_count = int(cloud.width) * int(cloud.height)
        if (
            cloud.is_bigendian
            or point_count <= 0
            or not {"x", "y", "z"}.issubset(field_offsets)
        ):
            return
        lidar_xyz = np.column_stack(
            [
                np.ndarray(
                    shape=(point_count,),
                    dtype="<f4",
                    buffer=cloud.data,
                    offset=field_offsets[name],
                    strides=(int(cloud.point_step),),
                )
                for name in ("x", "y", "z")
            ]
        )
        try:
            base_pose = interpolate_pose_sample(
                self._ground_truth,
                _stamp_seconds(message),
            )
        except ValueError:
            return

        sr, cr = math.sin(base_pose.roll), math.cos(base_pose.roll)
        sp, cp = math.sin(base_pose.pitch), math.cos(base_pose.pitch)
        sy, cy = math.sin(base_pose.yaw), math.cos(base_pose.yaw)
        rotation_world_from_base = np.asarray(
            (
                (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                (-sp, cp * sr, cp * cr),
            ),
            dtype=np.float64,
        )
        lidar_origin_world = (
            np.asarray((base_pose.x, base_pose.y, base_pose.z))
            + rotation_world_from_base @ np.asarray((0.20, 0.0, 0.35))
        )
        truth_registered_xyz = (
            lidar_xyz @ rotation_world_from_base.T + lidar_origin_world
        )
        if self._previous_truth_registered_xyz is not None:
            separation = registered_overlap_separation(
                self._previous_truth_registered_xyz,
                truth_registered_xyz,
            )
            if separation is not None:
                self._registered_overlap_separations_m.append(separation)
                self._registered_overlap_samples.append(
                    {
                        "stamp_s": _stamp_seconds(message),
                        "separation_p95_m": separation,
                        "linear_speed_mps": base_pose.linear_speed_mps,
                        "angular_speed_rps": base_pose.angular_speed_rps,
                        "roll_deg": math.degrees(base_pose.roll),
                        "pitch_deg": math.degrees(base_pose.pitch),
                    }
                )
        self._previous_truth_registered_xyz = truth_registered_xyz

    def _on_motion_deskew(
        self,
        message: Vector3Stamped,
    ) -> None:
        applied = message.header.frame_id == "motion_deskew_applied"
        self._last_motion_deskew_applied = applied
        if applied:
            self._counts["motion_deskew_applied"] += 1
            self._motion_deskew_norms_m.append(
                math.sqrt(
                    message.vector.x**2
                    + message.vector.y**2
                    + message.vector.z**2
                )
            )
        else:
            self._counts["motion_deskew_unavailable"] += 1
            if self._start_time_s is not None:
                self._counts["motion_deskew_unavailable_after_ready"] += 1

    def _ready(self) -> bool:
        return (
            bool(self._ground_truth)
            and len(self._estimate) >= 2
            and self._counts["raw_cloud"] >= 2
            and self._counts["adapted_cloud"] >= 2
            and self._last_cloud_info is not None
            and bool(self._last_cloud_info.imu_available)
            and self._last_motion_deskew_applied
        )

    def _publish_command(
        self,
        values: tuple[float, float, float],
    ) -> None:
        message = Twist()
        message.linear.x = values[0]
        message.linear.y = values[1]
        message.angular.z = values[2]
        self._command_publisher.publish(message)

    def _on_timer(self) -> None:
        if self._finished:
            return
        now_s = self.get_clock().now().nanoseconds * 1.0e-9
        if now_s <= 0.0:
            self._publish_command((0.0, 0.0, 0.0))
            return
        if self._first_sim_time_s is None:
            self._first_sim_time_s = now_s

        if self._start_time_s is None:
            self._publish_command((0.0, 0.0, 0.0))
            if self._ready():
                self._start_time_s = now_s
                self.get_logger().info(
                    f"Benchmark profile {self._profile.name!r} is ready"
                )
                return
            if now_s - self._first_sim_time_s > self._readiness_timeout_s:
                self._finish(
                    forced_failure=(
                        "LIO-SAM did not become ready within "
                        f"{self._readiness_timeout_s:.1f} simulation seconds"
                    )
                )
            return

        elapsed_s = now_s - self._start_time_s
        if self._profile.should_publish_command(elapsed_s):
            self._publish_command(self._profile.command_at(elapsed_s))
        if elapsed_s >= self._profile.duration_s:
            self._finish()

    @staticmethod
    def _interval_summary(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean_s": float(array.mean()),
            "min_s": float(array.min()),
            "max_s": float(array.max()),
            "max_abs_error_from_0_1_s": float(
                np.max(np.abs(array - 0.1))
            ),
        }

    def _finish(self, forced_failure: str | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        self._publish_command((0.0, 0.0, 0.0))

        failures: list[str] = []
        metrics: dict[str, float | int] = {}
        finish_time_s = self.get_clock().now().nanoseconds * 1.0e-9
        evaluation_start_s = (
            self._start_time_s
            if self._start_time_s is not None
            else self._first_sim_time_s
        )
        evaluation_start_s = evaluation_start_s or 0.0
        evaluation_ground_truth = [
            sample
            for sample in self._ground_truth
            if evaluation_start_s - 0.1 <= sample.stamp_s <= finish_time_s
        ]
        evaluation_estimate = [
            sample
            for sample in self._estimate
            if evaluation_start_s <= sample.stamp_s <= finish_time_s
        ]
        try:
            metrics = evaluate_trajectory(
                evaluation_ground_truth,
                evaluation_estimate,
                ground_truth_sensor_offset_xyz=(0.20, 0.0, 0.35),
            )
        except ValueError as error:
            failures.append(str(error))
        if forced_failure is not None:
            failures.append(forced_failure)

        if metrics:
            translation_limit = max(
                0.10,
                0.01 * float(metrics["path_length_m"]),
            )
            if float(metrics["translation_ate_rmse_m"]) > translation_limit:
                failures.append(
                    "translation ATE RMSE exceeds "
                    f"{translation_limit:.3f} m"
                )
            if float(metrics["yaw_rmse_deg"]) > 1.0:
                failures.append("yaw RMSE exceeds 1 degree")
            if float(metrics["translation_jump_residual_max_m"]) > 0.20:
                failures.append("translation pose jump exceeds 0.20 m")
            if float(metrics["yaw_jump_residual_max_deg"]) > 2.0:
                failures.append("yaw pose jump exceeds 2 degrees")
            if (
                self._profile.name == "forward_3_0"
                and int(metrics["ground_truth_high_speed_scan_count"]) < 5
            ):
                failures.append(
                    "fewer than 5 mapping scans reached 2.5 m/s"
                )
            if (
                self._profile.name == "yaw_2_0"
                and int(metrics["ground_truth_high_yaw_rate_scan_count"]) < 5
            ):
                failures.append(
                    "fewer than 5 mapping scans reached 1.5 rad/s yaw rate"
                )
        if self._counts["imu_unavailable_after_ready"] != 0:
            failures.append("IMU deskew became unavailable after readiness")
        if (
            self._counts["motion_deskew_unavailable_after_ready"]
            != 0
        ):
            failures.append(
                "motion deskew became unavailable after readiness"
            )
        if sum(self._timestamp_violations.values()) != 0:
            failures.append("one or more topic timestamps were non-monotonic")
        interval_groups = {
            "raw LiDAR": self._raw_intervals_s,
            "adapted LiDAR": self._adapted_intervals_s,
            "LIO-SAM cloud info": self._cloud_info_intervals_s,
        }
        for label, intervals in interval_groups.items():
            if not intervals:
                failures.append(f"no {label} scan intervals were observed")
            elif max(intervals) > 0.15:
                failures.append(
                    f"{label} dropped one or more 10 Hz scans "
                    f"(maximum interval {max(intervals):.3f} s)"
                )
        if not self._adapted_time_spans_ns:
            failures.append("no adapted LiDAR time spans were observed")
        elif min(self._adapted_time_spans_ns) < 90_000_000:
            failures.append("adapted LiDAR scan spans less than 90 ms")
        if len(self._registered_overlap_separations_m) < 10:
            failures.append(
                "fewer than 10 truth-registered vertical wall overlaps observed"
            )
        elif max(self._registered_overlap_separations_m) > 0.15:
            failures.append(
                "truth-registered vertical wall separation exceeds 0.15 m"
            )
        if (
            self._loop_closure_expectation == "required"
            and self._loop_constraint_edge_count_max < 1
        ):
            failures.append("no LIO-SAM loop constraint edge was observed")
        if (
            self._loop_closure_expectation == "forbidden"
            and self._loop_constraint_edge_count_max != 0
        ):
            failures.append(
                "an unexpected LIO-SAM loop constraint edge was observed"
            )

        invalid_reasons: list[str] = []
        if metrics:
            if max(
                float(metrics["ground_truth_roll_delta_max_deg"]),
                float(metrics["ground_truth_pitch_delta_max_deg"]),
            ) > 15.0:
                invalid_reasons.append(
                    "ground-truth body tilt exceeded 15 degrees"
                )
            if float(metrics["ground_truth_height_drop_max_m"]) > 0.15:
                invalid_reasons.append(
                    "ground-truth body height dropped more than 0.15 m"
                )

        if invalid_reasons:
            status = "invalid"
        elif failures:
            status = "failed"
        else:
            status = "passed"

        report = {
            "schema_version": 2,
            "profile": self._profile.name,
            "target_command": list(self._profile.target),
            "profile_duration_s": self._profile.duration_s,
            "status": status,
            "failures": failures,
            "invalid_reasons": invalid_reasons,
            "trajectory": metrics,
            "counts": self._counts,
            "timestamp_violations": self._timestamp_violations,
            "raw_cloud_intervals": self._interval_summary(
                self._raw_intervals_s
            ),
            "adapted_cloud_intervals": self._interval_summary(
                self._adapted_intervals_s
            ),
            "cloud_info_intervals": self._interval_summary(
                self._cloud_info_intervals_s
            ),
            "adapted_time_span_ns": {
                "count": len(self._adapted_time_spans_ns),
                "min": (
                    min(self._adapted_time_spans_ns)
                    if self._adapted_time_spans_ns
                    else None
                ),
                "max": (
                    max(self._adapted_time_spans_ns)
                    if self._adapted_time_spans_ns
                    else None
                ),
            },
            "motion_deskew_displacement_norm_m": {
                "count": len(self._motion_deskew_norms_m),
                "max": (
                    max(self._motion_deskew_norms_m)
                    if self._motion_deskew_norms_m
                    else None
                ),
            },
            "loop_closure": {
                "expectation": self._loop_closure_expectation,
                "marker_message_count": self._counts[
                    "loop_closure_marker"
                ],
                "max_constraint_edge_count": (
                    self._loop_constraint_edge_count_max
                ),
            },
            "truth_registered_vertical_wall_separation_p95_m": {
                "count": len(self._registered_overlap_separations_m),
                "max": (
                    max(self._registered_overlap_separations_m)
                    if self._registered_overlap_separations_m
                    else None
                ),
                "mean": (
                    float(np.mean(self._registered_overlap_separations_m))
                    if self._registered_overlap_separations_m
                    else None
                ),
            },
            "registered_wall_samples": self._registered_overlap_samples,
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        log_report = dict(report)
        log_report.pop("registered_wall_samples")
        self.get_logger().info(json.dumps(log_report, sort_keys=True))
        self.exit_code = (
            0 if status == "passed" else 2 if status == "invalid" else 1
        )


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LioBenchmarkNode()
    try:
        while rclpy.ok() and not node._finished:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        exit_code = node.exit_code
        try:
            node.destroy_node()
        except (KeyboardInterrupt, RuntimeError):
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv)
