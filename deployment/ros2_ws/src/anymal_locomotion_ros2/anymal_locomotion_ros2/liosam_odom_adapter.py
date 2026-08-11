"""Adapt LIO-SAM lidar odometry to the canonical body odometry contract."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from anymal_locomotion_ros2.slam_odom_adapter_core import (
    body_pose_from_sensor_pose,
    body_velocity_from_pose_delta,
)


_SOURCE_QOS = QoSProfile(
    depth=20,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
_OUTPUT_QOS = QoSProfile(
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _stamp_nanoseconds(message: Odometry) -> int:
    nanosec = int(message.header.stamp.nanosec)
    if not 0 <= nanosec < 1_000_000_000:
        raise ValueError("timestamp nanosec field is outside [0, 1e9)")
    return int(message.header.stamp.sec) * 1_000_000_000 + nanosec


class LiosamOdometryAdapter(Node):
    """Convert LIO-SAM's lidar pose into canonical map/base_link odometry."""

    def __init__(self) -> None:
        super().__init__("anymal_liosam_odom_adapter")
        self.declare_parameter(
            "source_topic",
            "/lio_sam/mapping/odometry",
        )
        self.declare_parameter("output_topic", "/slam/odom")
        self.declare_parameter("source_frame_id", "odom")
        self.declare_parameter("source_child_frame_id", "odom_mapping")
        self.declare_parameter("output_frame_id", "map")
        self.declare_parameter("output_child_frame_id", "base_link")
        self.declare_parameter(
            "sensor_translation_in_body_xyz",
            [0.20, 0.0, 0.35],
        )
        self.declare_parameter("derive_twist_from_pose", True)

        self._source_frame_id = str(
            self.get_parameter("source_frame_id").value
        )
        self._source_child_frame_id = str(
            self.get_parameter("source_child_frame_id").value
        )
        self._output_frame_id = str(
            self.get_parameter("output_frame_id").value
        )
        self._output_child_frame_id = str(
            self.get_parameter("output_child_frame_id").value
        )
        translation = np.asarray(
            self.get_parameter("sensor_translation_in_body_xyz").value,
            dtype=np.float64,
        )
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(
                "sensor_translation_in_body_xyz must contain finite XYZ"
            )
        self._sensor_translation_in_body = translation
        self._derive_twist = bool(
            self.get_parameter("derive_twist_from_pose").value
        )
        self._previous_stamp_ns: int | None = None
        self._previous_position: np.ndarray | None = None
        self._previous_quaternion: np.ndarray | None = None

        self._publisher = self.create_publisher(
            Odometry,
            str(self.get_parameter("output_topic").value),
            _OUTPUT_QOS,
        )
        self._subscription = self.create_subscription(
            Odometry,
            str(self.get_parameter("source_topic").value),
            self._on_odometry,
            _SOURCE_QOS,
        )

    def _on_odometry(self, message: Odometry) -> None:
        if message.header.frame_id != self._source_frame_id:
            self.get_logger().error(
                "LIO-SAM odometry frame mismatch: "
                f"{message.header.frame_id!r} != {self._source_frame_id!r}"
            )
            return
        if message.child_frame_id != self._source_child_frame_id:
            self.get_logger().error(
                "LIO-SAM odometry child frame mismatch: "
                f"{message.child_frame_id!r} != "
                f"{self._source_child_frame_id!r}"
            )
            return
        try:
            stamp_ns = _stamp_nanoseconds(message)
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        if (
            self._previous_stamp_ns is not None
            and stamp_ns <= self._previous_stamp_ns
        ):
            self.get_logger().error(
                "LIO-SAM odometry timestamp is not increasing: "
                f"{self._previous_stamp_ns} -> {stamp_ns}"
            )
            return

        pose = message.pose.pose
        try:
            position, quaternion = body_pose_from_sensor_pose(
                (
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ),
                (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ),
                self._sensor_translation_in_body,
            )
        except ValueError as error:
            self.get_logger().error(str(error))
            return

        output = copy.deepcopy(message)
        output.header.frame_id = self._output_frame_id
        output.child_frame_id = self._output_child_frame_id
        output.pose.pose.position.x = float(position[0])
        output.pose.pose.position.y = float(position[1])
        output.pose.pose.position.z = float(position[2])
        output.pose.pose.orientation.x = float(quaternion[0])
        output.pose.pose.orientation.y = float(quaternion[1])
        output.pose.pose.orientation.z = float(quaternion[2])
        output.pose.pose.orientation.w = float(quaternion[3])

        if self._derive_twist:
            if (
                self._previous_stamp_ns is None
                or self._previous_position is None
                or self._previous_quaternion is None
            ):
                linear_body = np.zeros(3, dtype=np.float64)
                angular_body = np.zeros(3, dtype=np.float64)
            else:
                try:
                    linear_body, angular_body = body_velocity_from_pose_delta(
                        self._previous_position,
                        self._previous_quaternion,
                        position,
                        quaternion,
                        (stamp_ns - self._previous_stamp_ns) * 1.0e-9,
                    )
                except ValueError as error:
                    self.get_logger().error(str(error))
                    return
            output.twist.twist.linear.x = float(linear_body[0])
            output.twist.twist.linear.y = float(linear_body[1])
            output.twist.twist.linear.z = float(linear_body[2])
            output.twist.twist.angular.x = float(angular_body[0])
            output.twist.twist.angular.y = float(angular_body[1])
            output.twist.twist.angular.z = float(angular_body[2])
            output.twist.covariance = [0.0] * 36

        self._publisher.publish(output)
        self._previous_stamp_ns = stamp_ns
        self._previous_position = position
        self._previous_quaternion = quaternion


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LiosamOdometryAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass


if __name__ == "__main__":
    main()
