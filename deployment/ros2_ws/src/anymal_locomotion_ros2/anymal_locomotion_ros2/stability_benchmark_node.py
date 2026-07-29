"""Publish one deterministic locomotion profile without starting LIO-SAM."""

from __future__ import annotations

import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from anymal_locomotion_ros2.lio_benchmark_core import get_motion_profile


def _stamp_seconds(message: Odometry) -> float:
    return (
        float(message.header.stamp.sec)
        + float(message.header.stamp.nanosec) * 1.0e-9
    )


class StabilityBenchmarkNode(Node):
    """Drive a smooth profile and stop after its settle interval."""

    def __init__(self) -> None:
        super().__init__("anymal_stability_benchmark")
        self.declare_parameter("profile", "stationary")
        self.declare_parameter("output_path", "")
        self.declare_parameter("project_root", "")
        self.declare_parameter("readiness_timeout_s", 10.0)

        self._profile = get_motion_profile(
            str(self.get_parameter("profile").value)
        )
        project_root = Path(
            str(self.get_parameter("project_root").value)
        ).expanduser().resolve()
        self._output_path = Path(
            str(self.get_parameter("output_path").value)
        ).expanduser().resolve()
        if not project_root.is_dir():
            raise ValueError(f"project_root is not a directory: {project_root}")
        if not self._output_path.is_relative_to(project_root):
            raise ValueError(
                f"benchmark output must remain inside {project_root}: "
                f"{self._output_path}"
            )
        self._readiness_timeout_s = float(
            self.get_parameter("readiness_timeout_s").value
        )
        if not math.isfinite(self._readiness_timeout_s) or self._readiness_timeout_s <= 0.0:
            raise ValueError("readiness_timeout_s must be positive")

        self._first_sim_time_s: float | None = None
        self._start_time_s: float | None = None
        self._last_odometry_stamp_s: float | None = None
        self._odometry_count = 0
        self._finished = False
        self.exit_code = 1

        self._command_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(
            Odometry,
            "/odom",
            self._on_odometry,
            qos_profile_sensor_data,
        )
        self.create_timer(0.02, self._on_timer)

    def _on_odometry(self, message: Odometry) -> None:
        stamp_s = _stamp_seconds(message)
        if (
            self._last_odometry_stamp_s is not None
            and stamp_s <= self._last_odometry_stamp_s
        ):
            self._finish(
                failure=(
                    "ground-truth odometry timestamp did not increase: "
                    f"{self._last_odometry_stamp_s} -> {stamp_s}"
                )
            )
            return
        self._last_odometry_stamp_s = stamp_s
        self._odometry_count += 1

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
            if self._odometry_count >= 5:
                self._start_time_s = now_s
                self.get_logger().info(
                    f"Stability profile {self._profile.name!r} is ready"
                )
                return
            if now_s - self._first_sim_time_s > self._readiness_timeout_s:
                self._finish(
                    failure=(
                        "ground-truth odometry did not become ready within "
                        f"{self._readiness_timeout_s:.1f} simulation seconds"
                    )
                )
            return

        elapsed_s = now_s - self._start_time_s
        if self._profile.should_publish_command(elapsed_s):
            self._publish_command(self._profile.command_at(elapsed_s))
        if elapsed_s >= self._profile.duration_s:
            self._finish()

    def _finish(self, *, failure: str | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        self._publish_command((0.0, 0.0, 0.0))
        result = {
            "schema_version": 1,
            "profile": self._profile.name,
            "target": list(self._profile.target),
            "duration_s": self._profile.duration_s,
            "odometry_count": self._odometry_count,
            "failure": failure,
            "passed": failure is None,
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if failure is None:
            self.get_logger().info(
                f"Stability profile completed: {self._profile.name}"
            )
            self.exit_code = 0
        else:
            self.get_logger().error(failure)
            self.exit_code = 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StabilityBenchmarkNode()
    try:
        while rclpy.ok() and not node._finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        exit_code = node.exit_code
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
