"""Publish one deterministic locomotion profile without starting LIO-SAM."""

from __future__ import annotations

import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from anymal_locomotion_interfaces.msg import SlamConfidence

from anymal_locomotion_ros2.lio_benchmark_core import get_motion_profile


_CONFIDENCE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
_FRESHNESS_REASONS = (
    SlamConfidence.REASON_SOURCE_STALE
    | SlamConfidence.REASON_ODOMETRY_STALE
    | SlamConfidence.REASON_LIDAR_STALE
    | SlamConfidence.REASON_IMU_STALE
)
_KNOWN_REASONS = (
    SlamConfidence.REASON_INITIALIZING
    | _FRESHNESS_REASONS
    | SlamConfidence.REASON_INSUFFICIENT_SUPPORT
    | SlamConfidence.REASON_DEGENERATE_GEOMETRY
    | SlamConfidence.REASON_HIGH_RESIDUAL
    | SlamConfidence.REASON_NOT_CONVERGED
    | SlamConfidence.REASON_ESTIMATOR_RESET
    | SlamConfidence.REASON_TIMESTAMP_INVALID
    | SlamConfidence.REASON_NUMERIC_INVALID
    | SlamConfidence.REASON_BACKEND_ERROR
    | SlamConfidence.REASON_SIGNAL_MISSING
    | SlamConfidence.REASON_LOW_CONFIDENCE
    | SlamConfidence.REASON_RECOVERY_PENDING
    | SlamConfidence.REASON_CLOCK_RESET
    | SlamConfidence.REASON_UNCALIBRATED
)


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
        self.declare_parameter("require_slam_confidence", False)
        self.declare_parameter("confidence_topic", "/slam_confidence")
        self.declare_parameter("expected_confidence_backend", "")
        self.declare_parameter("expected_calibration_id", "")

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
        self._require_slam_confidence = bool(
            self.get_parameter("require_slam_confidence").value
        )
        self._expected_confidence_backend = str(
            self.get_parameter("expected_confidence_backend").value
        )
        self._expected_calibration_id = str(
            self.get_parameter("expected_calibration_id").value
        )

        self._first_sim_time_s: float | None = None
        self._start_time_s: float | None = None
        self._last_odometry_stamp_s: float | None = None
        self._odometry_count = 0
        self._confidence_count = 0
        self._confidence_valid_count = 0
        self._confidence_invalid_after_tracking_count = 0
        self._confidence_freshness_invalid_count = 0
        self._confidence_unexplained_invalid_count = 0
        self._confidence_uncalibrated_count = 0
        self._confidence_identity_mismatch_count = 0
        self._confidence_timestamp_violation_count = 0
        self._confidence_unknown_reason_count = 0
        self._confidence_distinct_scores: set[float] = set()
        self._confidence_reason_counts: dict[int, int] = {}
        self._confidence_reached_tracking = False
        self._last_confidence_evaluation_ns: int | None = None
        self._confidence_max_evaluation_gap_s = 0.0
        self._finished = False
        self.exit_code = 1

        self._command_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(
            Odometry,
            "/odom",
            self._on_odometry,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            SlamConfidence,
            str(self.get_parameter("confidence_topic").value),
            self._on_confidence,
            _CONFIDENCE_QOS,
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

    def _on_confidence(self, message: SlamConfidence) -> None:
        self._confidence_count += 1
        self._confidence_distinct_scores.add(round(float(message.slam_confidence), 7))
        reasons = int(message.degradation_reasons)
        self._confidence_reason_counts[reasons] = (
            self._confidence_reason_counts.get(reasons, 0) + 1
        )
        if reasons & ~_KNOWN_REASONS:
            self._confidence_unknown_reason_count += 1
        if reasons & SlamConfidence.REASON_UNCALIBRATED:
            self._confidence_uncalibrated_count += 1
        if (
            self._expected_confidence_backend
            and message.backend_id != self._expected_confidence_backend
        ) or (
            self._expected_calibration_id
            and message.calibration_id != self._expected_calibration_id
        ):
            self._confidence_identity_mismatch_count += 1

        evaluation_ns = (
            int(message.evaluation_stamp.sec) * 1_000_000_000
            + int(message.evaluation_stamp.nanosec)
        )
        if (
            self._last_confidence_evaluation_ns is not None
            and evaluation_ns <= self._last_confidence_evaluation_ns
        ):
            self._confidence_timestamp_violation_count += 1
        elif self._last_confidence_evaluation_ns is not None:
            self._confidence_max_evaluation_gap_s = max(
                self._confidence_max_evaluation_gap_s,
                (evaluation_ns - self._last_confidence_evaluation_ns) * 1.0e-9,
            )
        self._last_confidence_evaluation_ns = evaluation_ns
        if message.source_stamp_valid:
            source_ns = (
                int(message.source_stamp.sec) * 1_000_000_000
                + int(message.source_stamp.nanosec)
            )
            if source_ns > evaluation_ns:
                self._confidence_timestamp_violation_count += 1

        if message.slam_tracking_valid:
            self._confidence_valid_count += 1
            self._confidence_reached_tracking = True
        elif self._confidence_reached_tracking:
            self._confidence_invalid_after_tracking_count += 1
            if reasons & _FRESHNESS_REASONS:
                self._confidence_freshness_invalid_count += 1
            if reasons == SlamConfidence.REASON_NONE:
                self._confidence_unexplained_invalid_count += 1

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
            self._publish_command(
                self._profile.command_at(0.0)
                if self._profile.start_immediately
                else (0.0, 0.0, 0.0)
            )
            return
        if self._first_sim_time_s is None:
            self._first_sim_time_s = now_s

        if self._start_time_s is None:
            if self._profile.start_immediately:
                self._start_time_s = now_s
                self._publish_command(self._profile.command_at(0.0))
                self.get_logger().info(
                    f"Stability profile {self._profile.name!r} started immediately"
                )
                return
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
        confidence_failures: list[str] = []
        if self._require_slam_confidence:
            if self._confidence_count == 0:
                confidence_failures.append("no confidence messages received")
            if not self._confidence_reached_tracking:
                confidence_failures.append("confidence never reached TRACKING")
            if self._confidence_identity_mismatch_count:
                confidence_failures.append("confidence backend/calibration ID mismatch")
            if self._confidence_uncalibrated_count:
                confidence_failures.append("UNCALIBRATED confidence observed")
            if self._confidence_timestamp_violation_count:
                confidence_failures.append("confidence timestamp violation")
            if self._confidence_unknown_reason_count:
                confidence_failures.append("unknown confidence reason bit")
            if self._confidence_freshness_invalid_count:
                confidence_failures.append("confidence freshness loss after TRACKING")
            if self._confidence_unexplained_invalid_count:
                confidence_failures.append("unexplained confidence invalid after TRACKING")
            if self._confidence_max_evaluation_gap_s > 0.075:
                confidence_failures.append("confidence logical publication gap exceeded 75 ms")
        if failure is None and confidence_failures:
            failure = "; ".join(confidence_failures)
        result = {
            "schema_version": 1,
            "profile": self._profile.name,
            "target": list(self._profile.target),
            "duration_s": self._profile.duration_s,
            "odometry_count": self._odometry_count,
            "slam_confidence": {
                "required": self._require_slam_confidence,
                "message_count": self._confidence_count,
                "tracking_valid_count": self._confidence_valid_count,
                "invalid_after_tracking_count": (
                    self._confidence_invalid_after_tracking_count
                ),
                "freshness_invalid_count": self._confidence_freshness_invalid_count,
                "unexplained_invalid_count": (
                    self._confidence_unexplained_invalid_count
                ),
                "uncalibrated_count": self._confidence_uncalibrated_count,
                "identity_mismatch_count": (
                    self._confidence_identity_mismatch_count
                ),
                "timestamp_violation_count": (
                    self._confidence_timestamp_violation_count
                ),
                "unknown_reason_count": self._confidence_unknown_reason_count,
                "distinct_score_count": len(self._confidence_distinct_scores),
                "max_evaluation_gap_s": self._confidence_max_evaluation_gap_s,
                "reason_mask_counts": {
                    str(mask): count
                    for mask, count in sorted(self._confidence_reason_counts.items())
                },
                "failures": confidence_failures,
            },
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
