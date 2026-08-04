"""Adapt FAST-LIO2 odometry to the project SLAM odometry contract."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from anymal_locomotion_ros2.slam_odom_adapter_core import (
    body_velocity_from_pose_delta,
    normalized_quaternion_xyzw,
)

_RELIABLE_QOS = QoSProfile(
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _stamp_seconds(message: Odometry) -> float:
    return (
        float(message.header.stamp.sec)
        + float(message.header.stamp.nanosec) * 1.0e-9
    )


class FastlioOdometryAdapter(Node):
    """Relabel validated FAST-LIO2 body pose and derive body-frame twist."""

    def __init__(self) -> None:
        super().__init__("anymal_fastlio_odom_adapter")
        self.declare_parameter("source_topic", "/Odometry")
        self.declare_parameter("output_topic", "/slam/odom")
        self.declare_parameter("source_frame_id", "camera_init")
        self.declare_parameter("source_child_frame_id", "body")
        self.declare_parameter("output_frame_id", "map")
        self.declare_parameter("output_child_frame_id", "base_link")
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
        self._derive_twist = bool(
            self.get_parameter("derive_twist_from_pose").value
        )
        self._previous_stamp_s: float | None = None
        self._previous_position: np.ndarray | None = None
        self._previous_quaternion: np.ndarray | None = None

        self._publisher = self.create_publisher(
            Odometry,
            str(self.get_parameter("output_topic").value),
            _RELIABLE_QOS,
        )
        self._subscription = self.create_subscription(
            Odometry,
            str(self.get_parameter("source_topic").value),
            self._on_odometry,
            _RELIABLE_QOS,
        )

    def _on_odometry(self, message: Odometry) -> None:
        if message.header.frame_id != self._source_frame_id:
            self.get_logger().error(
                "FAST-LIO2 odometry frame mismatch: "
                f"{message.header.frame_id!r} != {self._source_frame_id!r}"
            )
            return
        if message.child_frame_id != self._source_child_frame_id:
            self.get_logger().error(
                "FAST-LIO2 odometry child frame mismatch: "
                f"{message.child_frame_id!r} != "
                f"{self._source_child_frame_id!r}"
            )
            return

        stamp_s = _stamp_seconds(message)
        position = np.asarray(
            (
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(position).all():
            self.get_logger().error("FAST-LIO2 odometry position is not finite")
            return
        try:
            quaternion = normalized_quaternion_xyzw(
                (
                    message.pose.pose.orientation.x,
                    message.pose.pose.orientation.y,
                    message.pose.pose.orientation.z,
                    message.pose.pose.orientation.w,
                )
            )
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        if self._previous_stamp_s is not None and stamp_s <= self._previous_stamp_s:
            self.get_logger().error(
                "FAST-LIO2 odometry timestamp is not increasing: "
                f"{self._previous_stamp_s} -> {stamp_s}"
            )
            return

        output = copy.deepcopy(message)
        output.header.frame_id = self._output_frame_id
        output.child_frame_id = self._output_child_frame_id
        output.pose.pose.orientation.x = float(quaternion[0])
        output.pose.pose.orientation.y = float(quaternion[1])
        output.pose.pose.orientation.z = float(quaternion[2])
        output.pose.pose.orientation.w = float(quaternion[3])

        if self._derive_twist:
            if (
                self._previous_stamp_s is None
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
                        stamp_s - self._previous_stamp_s,
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
            # The candidate does not publish a calibrated twist covariance;
            # do not copy an unrelated source value into the derived twist.
            output.twist.covariance = [0.0] * 36

        self._publisher.publish(output)
        self._previous_stamp_s = stamp_s
        self._previous_position = position
        self._previous_quaternion = quaternion


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FastlioOdometryAdapter()
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
