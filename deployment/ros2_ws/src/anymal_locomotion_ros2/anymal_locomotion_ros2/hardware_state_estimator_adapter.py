"""Adapt a vendor nav_msgs/Odometry state estimate to the policy velocity topic."""

from __future__ import annotations

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from anymal_locomotion_ros2.hardware_state_estimator_adapter_core import adapt_linear_velocity


class HardwareStateEstimatorAdapter(Node):
    def __init__(self) -> None:
        super().__init__("hardware_state_estimator_adapter")
        self.declare_parameter("input_topic", "")
        self.declare_parameter("output_topic", "/locomotion/estimated_odom")
        self.declare_parameter("input_velocity_frame", "body")
        self.declare_parameter("expected_child_frame", "base_link")
        input_topic = str(self.get_parameter("input_topic").value)
        if not input_topic:
            raise ValueError("input_topic is required and must be supplied from the ANYmal SDK contract")
        self._input_velocity_frame = str(self.get_parameter("input_velocity_frame").value)
        self._expected_child_frame = str(self.get_parameter("expected_child_frame").value)
        self._publisher = self.create_publisher(
            Odometry, str(self.get_parameter("output_topic").value), qos_profile_sensor_data
        )
        self.create_subscription(Odometry, input_topic, self._on_odometry, qos_profile_sensor_data)

    def _on_odometry(self, message: Odometry) -> None:
        if message.child_frame_id != self._expected_child_frame:
            return
        velocity = message.twist.twist.linear
        orientation = message.pose.pose.orientation
        try:
            body_velocity = adapt_linear_velocity(
                [velocity.x, velocity.y, velocity.z],
                [orientation.x, orientation.y, orientation.z, orientation.w],
                input_velocity_frame=self._input_velocity_frame,
            )
        except ValueError:
            return
        output = Odometry()
        output.header = message.header
        output.child_frame_id = self._expected_child_frame
        output.pose = message.pose
        output.twist = message.twist
        output.twist.twist.linear.x = float(body_velocity[0])
        output.twist.twist.linear.y = float(body_velocity[1])
        output.twist.twist.linear.z = float(body_velocity[2])
        self._publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HardwareStateEstimatorAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
