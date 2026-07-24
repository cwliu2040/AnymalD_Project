"""External ROS 2 node that runs the ANYmal-D locomotion policy at 50 Hz."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState

from anymal_locomotion_ros2.policy_core import (
    PolicyContract,
    PolicyRuntime,
    RobotState,
    canonical_joint_state,
    projected_gravity_from_quaternion,
)
from anymal_locomotion_ros2.onnx_backend import OnnxBackend
from anymal_locomotion_ros2.torchscript_backend import TorchScriptBackend

DEFAULT_EXPORT_DIR = (
    "/home/ros/anymal_locomotion/exported/"
    "anymal_d_locomotion_v1/high_speed_v0.2.0"
)


def create_inference_backend(backend_name: str, policy_path: str) -> Any:
    """Create the selected external-process inference backend."""
    normalized_name = backend_name.strip().lower()
    if normalized_name == "onnx":
        return OnnxBackend(policy_path)
    if normalized_name == "torchscript":
        return TorchScriptBackend(policy_path)
    raise ValueError(
        "backend must be 'onnx' or 'torchscript', "
        f"received {backend_name!r}"
    )


class AnymalPolicyNode(Node):
    """Assemble ROS state, execute the policy, and publish joint targets."""

    def __init__(
        self,
        backend_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__("anymal_locomotion_policy")
        self.declare_parameter("backend", "onnx")
        self.declare_parameter("policy_path", f"{DEFAULT_EXPORT_DIR}/policy.onnx")
        self.declare_parameter("metadata_path", f"{DEFAULT_EXPORT_DIR}/policy_metadata.yaml")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("odometry_topic", "/odom")
        self.declare_parameter("command_topic", "/cmd_vel")
        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter("expected_imu_frame", "base_link")
        self.declare_parameter("expected_odometry_child_frame", "base_link")
        self.declare_parameter("state_timeout_s", 0.1)
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("max_abs_policy_action", 10.0)

        policy_path = self.get_parameter("policy_path").value
        metadata_path = self.get_parameter("metadata_path").value
        backend_name = str(self.get_parameter("backend").value)
        if backend_factory is None:
            backend = create_inference_backend(backend_name, policy_path)
        else:
            backend = backend_factory(policy_path)
        self._contract = PolicyContract.from_metadata(metadata_path)
        self._runtime = PolicyRuntime(
            self._contract,
            backend,
            max_abs_policy_action=float(self.get_parameter("max_abs_policy_action").value),
        )
        self._state_timeout = float(self.get_parameter("state_timeout_s").value)
        self._command_timeout = float(self.get_parameter("command_timeout_s").value)
        self._expected_imu_frame = str(self.get_parameter("expected_imu_frame").value)
        self._expected_odom_child_frame = str(
            self.get_parameter("expected_odometry_child_frame").value
        )
        if self._state_timeout <= 0.0 or self._command_timeout <= 0.0:
            raise ValueError("State and command timeouts must be positive")

        self._joint_positions: np.ndarray | None = None
        self._joint_velocities: np.ndarray | None = None
        self._base_linear_velocity: np.ndarray | None = None
        self._base_angular_velocity: np.ndarray | None = None
        self._projected_gravity: np.ndarray | None = None
        self._command = np.zeros(3, dtype=np.float32)
        self._receipt_times: dict[str, float] = {}
        self._warning_times: dict[str, float] = {}

        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter("imu_topic").value),
            self._on_imu,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odometry_topic").value),
            self._on_odometry,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("command_topic").value),
            self._on_command,
            10,
        )
        self._joint_command_publisher = self.create_publisher(
            JointState,
            str(self.get_parameter("joint_command_topic").value),
            10,
        )
        self.create_timer(self._contract.control_period_s, self._on_policy_tick)
        self.get_logger().info(
            f"Loaded {backend_name} 48-D -> 12-D policy; "
            f"control period={self._contract.control_period_s:.3f} s"
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _warn_throttled(self, key: str, message: str) -> None:
        now = self._now_seconds()
        if now - self._warning_times.get(key, float("-inf")) >= 1.0:
            self.get_logger().warning(message)
            self._warning_times[key] = now

    def _on_joint_state(self, message: JointState) -> None:
        try:
            positions, velocities = canonical_joint_state(
                message.name,
                message.position,
                message.velocity,
                self._contract,
            )
        except ValueError as error:
            self._warn_throttled("joint_state_invalid", str(error))
            return
        self._joint_positions = positions
        self._joint_velocities = velocities
        self._receipt_times["joint_state"] = self._now_seconds()

    def _on_imu(self, message: Imu) -> None:
        if message.header.frame_id != self._expected_imu_frame:
            self._warn_throttled(
                "imu_frame",
                f"IMU frame_id must be '{self._expected_imu_frame}', "
                f"received '{message.header.frame_id}'",
            )
            return
        try:
            angular_velocity = np.asarray(
                [
                    message.angular_velocity.x,
                    message.angular_velocity.y,
                    message.angular_velocity.z,
                ],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(angular_velocity)):
                raise ValueError("IMU angular velocity contains NaN or Inf")
            gravity = projected_gravity_from_quaternion(
                message.orientation.x,
                message.orientation.y,
                message.orientation.z,
                message.orientation.w,
            )
        except ValueError as error:
            self._warn_throttled("imu_invalid", str(error))
            return
        self._base_angular_velocity = angular_velocity
        self._projected_gravity = gravity
        self._receipt_times["imu"] = self._now_seconds()

    def _on_odometry(self, message: Odometry) -> None:
        if message.child_frame_id != self._expected_odom_child_frame:
            self._warn_throttled(
                "odometry_frame",
                f"Odometry child_frame_id must be '{self._expected_odom_child_frame}', "
                f"received '{message.child_frame_id}'",
            )
            return
        velocity = np.asarray(
            [
                message.twist.twist.linear.x,
                message.twist.twist.linear.y,
                message.twist.twist.linear.z,
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(velocity)):
            self._warn_throttled(
                "odometry_invalid",
                "Odometry linear velocity contains NaN or Inf",
            )
            return
        self._base_linear_velocity = velocity
        self._receipt_times["odometry"] = self._now_seconds()

    def _on_command(self, message: Twist) -> None:
        command = np.asarray(
            [message.linear.x, message.linear.y, message.angular.z],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(command)):
            self._warn_throttled("command_invalid", "Velocity command contains NaN or Inf")
            return
        self._command = command
        self._receipt_times["command"] = self._now_seconds()

    def _on_policy_tick(self) -> None:
        now = self._now_seconds()
        required = ("joint_state", "imu", "odometry")
        missing = [name for name in required if name not in self._receipt_times]
        stale = [
            name
            for name in required
            if name in self._receipt_times
            and now - self._receipt_times[name] > self._state_timeout
        ]
        if missing or stale:
            self._warn_throttled(
                "state_not_ready",
                f"No policy output: missing={missing}, stale={stale}",
            )
            return

        command = self._command
        if now - self._receipt_times.get("command", float("-inf")) > self._command_timeout:
            command = np.zeros(3, dtype=np.float32)

        assert self._base_linear_velocity is not None
        assert self._base_angular_velocity is not None
        assert self._projected_gravity is not None
        assert self._joint_positions is not None
        assert self._joint_velocities is not None
        state = RobotState(
            base_linear_velocity=self._base_linear_velocity,
            base_angular_velocity=self._base_angular_velocity,
            projected_gravity=self._projected_gravity,
            joint_positions=self._joint_positions,
            joint_velocities=self._joint_velocities,
        )
        try:
            result = self._runtime.step(state, command)
        except (RuntimeError, ValueError) as error:
            self._warn_throttled("inference_error", f"No policy output: {error}")
            return

        output = JointState()
        output.header.stamp = self.get_clock().now().to_msg()
        output.name = list(self._contract.joint_order)
        output.position = result.joint_targets.astype(float).tolist()
        self._joint_command_publisher.publish(output)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: AnymalPolicyNode | None = None
    try:
        node = AnymalPolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
