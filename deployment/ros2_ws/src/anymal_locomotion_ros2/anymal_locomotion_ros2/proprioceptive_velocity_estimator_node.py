"""Estimate body linear velocity from IMU, joint, and foot-contact history."""

from __future__ import annotations

import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState

from anymal_locomotion_interfaces.msg import FootContactState
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


def _stamp_ns(message) -> int:
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


class ProprioceptiveVelocityEstimatorNode(Node):
    def __init__(self) -> None:
        super().__init__("proprioceptive_velocity_estimator")
        self.declare_parameter("metadata_path", "")
        self.declare_parameter("policy_metadata_path", "")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("foot_contact_topic", "/foot_contacts")
        self.declare_parameter("output_topic", "/locomotion/estimated_odom")
        self.declare_parameter("expected_frame", "base_link")
        self.declare_parameter("synchronization_tolerance_s", 0.025)
        self.declare_parameter("receipt_timeout_s", 0.10)
        metadata_path = str(self.get_parameter("metadata_path").value)
        policy_metadata_path = str(self.get_parameter("policy_metadata_path").value)
        if not metadata_path or not policy_metadata_path:
            raise ValueError("metadata_path and policy_metadata_path are required")
        _, model_path = load_estimator_metadata(metadata_path)
        contract = PolicyContract.from_metadata(policy_metadata_path)
        self._joint_contract = contract
        self._defaults = np.asarray(contract.default_joint_positions, dtype=np.float32)
        self._runtime = EstimatorRuntime(OnnxBackend(str(model_path)))
        self._frame = str(self.get_parameter("expected_frame").value)
        self._sync_tolerance = float(self.get_parameter("synchronization_tolerance_s").value)
        self._receipt_timeout = float(self.get_parameter("receipt_timeout_s").value)
        self._uses_sim_time = bool(self.get_parameter("use_sim_time").value)
        if self._sync_tolerance < 0.0 or self._receipt_timeout <= 0.0:
            raise ValueError("synchronization tolerance must be non-negative and timeout positive")
        self._synchronizer = EstimatorInputSynchronizer(self._sync_tolerance)
        self._publisher = self.create_publisher(
            Odometry, str(self.get_parameter("output_topic").value), qos_profile_sensor_data
        )
        self.create_subscription(Imu, str(self.get_parameter("imu_topic").value), self._on_imu, qos_profile_sensor_data)
        self.create_subscription(FootContactState, str(self.get_parameter("foot_contact_topic").value), self._on_contacts, qos_profile_sensor_data)
        self.create_subscription(JointState, str(self.get_parameter("joint_state_topic").value), self._on_joints, qos_profile_sensor_data)

    def _on_imu(self, message: Imu) -> None:
        if message.header.frame_id != self._frame:
            self._runtime.reset()
            self._synchronizer.reset()
            return
        try:
            gravity = projected_gravity_from_quaternion(
                message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w
            )
            angular = np.asarray([message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z], dtype=np.float32)
            acceleration = np.asarray([message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z], dtype=np.float32)
            if not np.all(np.isfinite(angular)) or not np.all(np.isfinite(acceleration)):
                raise ValueError
        except ValueError:
            self._runtime.reset()
            self._synchronizer.reset()
            return
        quaternion = (message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w)
        try:
            bundles = self._synchronizer.push_imu(
                _stamp_ns(message),
                (
                    time.monotonic_ns(), angular, acceleration,
                    gravity, quaternion,
                ),
            )
        except ValueError:
            self._runtime.reset()
            self._synchronizer.reset()
            return
        self._consume(bundles)

    def _on_contacts(self, message: FootContactState) -> None:
        if message.header.frame_id != self._frame:
            self._runtime.reset()
            self._synchronizer.reset()
            return
        try:
            values = reorder_foot_contacts(message.foot_names, message.contact_probabilities)
        except ValueError:
            self._runtime.reset()
            self._synchronizer.reset()
            return
        try:
            bundles = self._synchronizer.push_contacts(
                _stamp_ns(message), (time.monotonic_ns(), values)
            )
        except ValueError:
            self._runtime.reset()
            self._synchronizer.reset()
            return
        self._consume(bundles)

    def _on_joints(self, message: JointState) -> None:
        try:
            bundles = self._synchronizer.push_joint(
                _stamp_ns(message), message
            )
        except ValueError:
            self._runtime.reset()
            self._synchronizer.reset()
            return
        self._consume(bundles)

    def _consume(self, bundles: list[EstimatorInputBundle]) -> None:
        for bundle in bundles:
            self._process_bundle(bundle)

    def _process_bundle(self, bundle: EstimatorInputBundle) -> None:
        if not bundle.synchronized:
            self._runtime.reset()
            return
        message = bundle.joint
        now_ns = time.monotonic_ns()
        imu_receipt, angular, acceleration, gravity, quaternion = bundle.imu
        contact_receipt, contacts = bundle.contacts
        if (
            not self._uses_sim_time
            and max(now_ns - imu_receipt, now_ns - contact_receipt) * 1.0e-9
            > self._receipt_timeout
        ):
            self._runtime.reset()
            return
        try:
            positions, velocities = canonical_joint_state(
                message.name, message.position, message.velocity, self._joint_contract
            )
            sample = assemble_step(
                angular,
                acceleration,
                gravity,
                positions - self._defaults,
                velocities,
                contacts,
            )
            estimate = self._runtime.step(
                bundle.joint_stamp_ns * 1.0e-9, sample
            )
        except ValueError:
            self._runtime.reset()
            return
        if estimate is None:
            return
        output = Odometry()
        output.header = message.header
        output.header.frame_id = "odom"
        output.child_frame_id = self._frame
        output.pose.pose.orientation.x = quaternion[0]
        output.pose.pose.orientation.y = quaternion[1]
        output.pose.pose.orientation.z = quaternion[2]
        output.pose.pose.orientation.w = quaternion[3]
        output.twist.twist.linear.x = float(estimate[0])
        output.twist.twist.linear.y = float(estimate[1])
        output.twist.twist.linear.z = float(estimate[2])
        output.twist.twist.angular.x = float(angular[0])
        output.twist.twist.angular.y = float(angular[1])
        output.twist.twist.angular.z = float(angular[2])
        self._publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ProprioceptiveVelocityEstimatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
