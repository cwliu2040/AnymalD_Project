"""Fail-closed FAST-LIO2 extractor for backend-neutral SLAM confidence."""

from __future__ import annotations

import math
from collections.abc import Sequence

import rclpy
from builtin_interfaces.msg import Duration, Time
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Bool

from anymal_locomotion_interfaces.msg import SlamConfidence
from anymal_locomotion_ros2.fastlio_confidence_core import (
    FastlioAssemblerConfig,
    FastlioBundle,
    FastlioSignalAssembler,
    PoseTuple,
)
from anymal_locomotion_ros2.slam_confidence_core import (
    DegradationReason,
    HealthAssessment,
    SlamConfidenceStateMachine,
    SourceObservation,
    StateMachineConfig,
    seconds_to_nanoseconds,
)
from anymal_locomotion_ros2.slam_confidence_estimator import (
    CalibratedConfidenceEstimator,
    CausalPoseMotion,
)


_BACKEND_ID = "fastlio2"
_UNCALIBRATED_ID = "uncalibrated"

_RELIABLE_DEPTH_20 = QoSProfile(
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
_RELIABLE_DEPTH_10 = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
_AUTHORITATIVE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _stamp_nanoseconds(stamp: Time) -> int:
    if stamp.nanosec >= 1_000_000_000:
        raise ValueError("timestamp nanosec field is outside [0, 1e9)")
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _time_from_nanoseconds(value_ns: int) -> Time:
    seconds, nanoseconds = divmod(value_ns, 1_000_000_000)
    return Time(sec=seconds, nanosec=nanoseconds)


def _duration_from_nanoseconds(value_ns: int) -> Duration:
    if value_ns < 0:
        raise ValueError("confidence age cannot be negative")
    seconds, nanoseconds = divmod(value_ns, 1_000_000_000)
    return Duration(sec=seconds, nanosec=nanoseconds)


def _pose_tuple(message: Odometry) -> PoseTuple:
    pose = message.pose.pose
    return (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )


def _point_cloud_layout_valid(message: PointCloud2) -> bool:
    point_count = int(message.width) * int(message.height)
    if point_count == 0:
        return int(message.row_step) == 0 and len(message.data) == 0
    if message.point_step <= 0:
        return False
    minimum_row_step = int(message.point_step) * int(message.width)
    if message.row_step < minimum_row_step:
        return False
    return len(message.data) >= int(message.row_step) * int(message.height)


class FastlioConfidenceExtractor(Node):
    """Assemble deployable FAST-LIO2 signals and publish a fail-closed snapshot."""

    def __init__(self, **node_kwargs: object) -> None:
        super().__init__("anymal_fastlio_confidence_extractor", **node_kwargs)
        self.declare_parameter("native_odometry_topic", "/Odometry")
        self.declare_parameter("canonical_odometry_topic", "/slam/odom")
        self.declare_parameter("effective_points_topic", "/cloud_effected")
        self.declare_parameter("lidar_input_topic", "/fastlio/points")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("confidence_topic", "/slam_confidence")
        self.declare_parameter(
            "tracking_valid_topic",
            "/slam_tracking_valid",
        )
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("source_timeout_s", 0.30)
        self.declare_parameter("odometry_timeout_s", 0.30)
        self.declare_parameter("lidar_timeout_s", 0.30)
        self.declare_parameter("imu_timeout_s", 0.05)
        self.declare_parameter("future_stamp_tolerance_s", 0.005)
        self.declare_parameter("diagnostic_match_timeout_s", 0.05)
        self.declare_parameter("max_pending_bundles", 64)
        self.declare_parameter("degrade_below", 0.45)
        self.declare_parameter("degrade_dwell_s", 0.10)
        self.declare_parameter("invalidate_below", 0.25)
        self.declare_parameter("invalidate_dwell_s", 0.20)
        self.declare_parameter("recover_at_or_above", 0.55)
        self.declare_parameter("recover_dwell_s", 0.50)
        self.declare_parameter("calibration_artifact_path", "")

        artifact_path = str(
            self.get_parameter("calibration_artifact_path").value
        ).strip()
        self._estimator = (
            CalibratedConfidenceEstimator(artifact_path, backend_id=_BACKEND_ID)
            if artifact_path
            else None
        )
        self._motion = CausalPoseMotion()
        thresholds = self._estimator.thresholds if self._estimator else {}

        publish_rate_hz = self._positive_float_parameter("publish_rate_hz")
        future_tolerance_ns = seconds_to_nanoseconds(
            self._nonnegative_float_parameter("future_stamp_tolerance_s")
        )
        state_config = StateMachineConfig(
            source_timeout_ns=seconds_to_nanoseconds(
                self._positive_float_parameter("source_timeout_s")
            ),
            future_tolerance_ns=future_tolerance_ns,
            degrade_below=float(
                thresholds.get("degrade_below", self.get_parameter("degrade_below").value)
            ),
            degrade_dwell_ns=seconds_to_nanoseconds(
                self._positive_float_parameter("degrade_dwell_s")
            ),
            invalidate_below=float(thresholds.get(
                "invalidate_below", self.get_parameter("invalidate_below").value
            )),
            invalidate_dwell_ns=seconds_to_nanoseconds(
                self._positive_float_parameter("invalidate_dwell_s")
            ),
            recover_at_or_above=float(thresholds.get(
                "recover_at_or_above", self.get_parameter("recover_at_or_above").value
            )),
            recover_dwell_ns=seconds_to_nanoseconds(
                self._positive_float_parameter("recover_dwell_s")
            ),
        )
        assembler_config = FastlioAssemblerConfig(
            odometry_timeout_ns=seconds_to_nanoseconds(
                self._positive_float_parameter("odometry_timeout_s")
            ),
            lidar_timeout_ns=seconds_to_nanoseconds(
                self._positive_float_parameter("lidar_timeout_s")
            ),
            imu_timeout_ns=seconds_to_nanoseconds(
                self._positive_float_parameter("imu_timeout_s")
            ),
            diagnostic_match_timeout_ns=seconds_to_nanoseconds(
                self._positive_float_parameter("diagnostic_match_timeout_s")
            ),
            future_tolerance_ns=future_tolerance_ns,
            max_pending_bundles=int(
                self.get_parameter("max_pending_bundles").value
            ),
        )
        self._assembler = FastlioSignalAssembler(assembler_config)
        self._state_machine = SlamConfidenceStateMachine(
            state_config,
            backend_id=_BACKEND_ID,
            calibration_id=(
                self._estimator.calibration_id
                if self._estimator
                else _UNCALIBRATED_ID
            ),
        )
        self._last_clock_ns: int | None = None
        self._last_log_signature: tuple[int, int] | None = None
        self._last_effective_point_count: int | None = None
        self._pending_estimator_row: dict | None = None
        self._pending_estimator_holds_score = False

        self._confidence_publisher = self.create_publisher(
            SlamConfidence,
            str(self.get_parameter("confidence_topic").value),
            _AUTHORITATIVE_QOS,
        )
        self._valid_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("tracking_valid_topic").value),
            _AUTHORITATIVE_QOS,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("native_odometry_topic").value),
            self._on_native_odometry,
            _RELIABLE_DEPTH_20,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("canonical_odometry_topic").value),
            self._on_canonical_odometry,
            _RELIABLE_DEPTH_20,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("effective_points_topic").value),
            self._on_effective_points,
            _RELIABLE_DEPTH_20,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("lidar_input_topic").value),
            self._on_lidar_input,
            _RELIABLE_DEPTH_20,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter("imu_topic").value),
            self._on_imu,
            _RELIABLE_DEPTH_10,
        )
        self.create_timer(1.0 / publish_rate_hz, self._on_timer)
        if self._estimator:
            self.get_logger().info(
                f"Loaded native FAST-LIO2 confidence artifact {self._estimator.calibration_id}"
            )
        else:
            self.get_logger().warning(
                "FAST-LIO2 confidence extractor has no calibration artifact; "
                "tracking remains UNCALIBRATED and fail-closed"
            )

    def _on_native_odometry(self, message: Odometry) -> None:
        now_ns = self._prepare_clock()
        stamp_ns = self._safe_stamp(message.header.stamp, now_ns)
        if stamp_ns is None:
            return
        self._assembler.observe_native_odometry(
            stamp_ns=stamp_ns,
            observed_at_ns=now_ns,
            pose=_pose_tuple(message),
            frame_id=message.header.frame_id,
            child_frame_id=message.child_frame_id,
        )
        self._flush_assembler(now_ns)

    def _on_canonical_odometry(self, message: Odometry) -> None:
        now_ns = self._prepare_clock()
        stamp_ns = self._safe_stamp(message.header.stamp, now_ns)
        if stamp_ns is None:
            return
        self._assembler.observe_canonical_odometry(
            stamp_ns=stamp_ns,
            observed_at_ns=now_ns,
            pose=_pose_tuple(message),
            frame_id=message.header.frame_id,
            child_frame_id=message.child_frame_id,
        )
        self._flush_assembler(now_ns)

    def _on_effective_points(self, message: PointCloud2) -> None:
        now_ns = self._prepare_clock()
        stamp_ns = self._safe_stamp(message.header.stamp, now_ns)
        if stamp_ns is None:
            return
        point_count = int(message.width) * int(message.height)
        self._assembler.observe_effective_points(
            stamp_ns=stamp_ns,
            observed_at_ns=now_ns,
            point_count=point_count,
            layout_valid=_point_cloud_layout_valid(message),
        )
        self._flush_assembler(now_ns)

    def _on_lidar_input(self, message: PointCloud2) -> None:
        now_ns = self._prepare_clock()
        stamp_ns = self._safe_stamp(message.header.stamp, now_ns)
        if stamp_ns is None:
            return
        self._assembler.observe_lidar_input(
            stamp_ns=stamp_ns,
            has_points=int(message.width) * int(message.height) > 0,
            layout_valid=_point_cloud_layout_valid(message),
        )
        self._flush_assembler(now_ns)

    def _on_imu(self, message: Imu) -> None:
        now_ns = self._prepare_clock()
        stamp_ns = self._safe_stamp(message.header.stamp, now_ns)
        if stamp_ns is None:
            return
        finite = all(
            math.isfinite(value)
            for value in (
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            )
        )
        self._assembler.observe_imu_input(stamp_ns=stamp_ns, finite=finite)
        self._flush_assembler(now_ns)

    def _on_timer(self) -> None:
        now_ns = self._prepare_clock()
        self._flush_assembler(now_ns)
        assessment = self._assembler.assess(evaluation_ns=now_ns)
        if self._estimator is None:
            assessment = HealthAssessment(
                hard_reasons=(assessment.hard_reasons | DegradationReason.UNCALIBRATED),
                quality_reasons=assessment.quality_reasons,
            )
        elif self._pending_estimator_row is not None:
            row = self._pending_estimator_row
            self._pending_estimator_row = None
            row["source_age_s"] = max(
                0, now_ns - int(row["source_stamp_ns"])
            ) * 1.0e-9
            try:
                self._estimator.observe(
                    row,
                    update_confidence=not self._pending_estimator_holds_score,
                )
            except ValueError as error:
                self.get_logger().error(str(error))
                self._state_machine.observe_hard_event(
                    DegradationReason.SIGNAL_MISSING,
                    ros_now_ns=now_ns,
                )
            self._pending_estimator_holds_score = False
        snapshot = self._state_machine.evaluate(
            evaluation_ns=now_ns,
            health=assessment,
            confidence=(self._estimator.confidence if self._estimator else 0.0),
        )

        message = SlamConfidence()
        message.schema_version = SlamConfidence.SCHEMA_VERSION
        message.backend_id = snapshot.backend_id
        message.calibration_id = snapshot.calibration_id
        message.source_stamp_valid = snapshot.source_stamp_ns is not None
        if snapshot.source_stamp_ns is not None:
            message.source_stamp = _time_from_nanoseconds(
                snapshot.source_stamp_ns
            )
        message.evaluation_stamp = _time_from_nanoseconds(
            snapshot.evaluation_stamp_ns
        )
        message.confidence_age = _duration_from_nanoseconds(
            snapshot.confidence_age_ns
        )
        message.slam_confidence = snapshot.confidence
        message.slam_tracking_valid = snapshot.tracking_valid
        message.tracking_state = int(snapshot.state)
        message.degradation_reasons = int(snapshot.reasons)
        self._confidence_publisher.publish(message)
        self._valid_publisher.publish(Bool(data=snapshot.tracking_valid))
        self._log_transition(snapshot.state, snapshot.reasons)

    def _flush_assembler(self, now_ns: int) -> None:
        events = self._assembler.take_events()
        if events:
            self._state_machine.observe_hard_event(events, ros_now_ns=now_ns)
        for bundle in self._assembler.take_completed():
            self._observe_bundle(bundle, now_ns=now_ns)

    def _observe_bundle(self, bundle: FastlioBundle, *, now_ns: int) -> None:
        self._last_effective_point_count = bundle.effective_point_count
        if self._estimator is not None:
            motion = self._motion.observe(bundle.source_stamp_ns, bundle.native_pose)
            # Calibration consumes the first 20 Hz evaluation row for each
            # source stamp. Defer inference to that same logical timer grid so
            # callback scheduling cannot change the source-age feature.
            self._pending_estimator_row = {
                "source_stamp_ns": bundle.source_stamp_ns,
                "features": {
                    "effective_features": bundle.effective_point_count,
                    **motion,
                }
            }
            self._pending_estimator_holds_score = bool(bundle.hard_reasons)
        self._state_machine.observe_source(
            SourceObservation(
                source_stamp_ns=bundle.source_stamp_ns,
                content_fingerprint=bundle.content_fingerprint,
                hard_reasons=(
                    bundle.hard_reasons
                    | (
                        DegradationReason.UNCALIBRATED
                        if self._estimator is None
                        else DegradationReason.NONE
                    )
                ),
                quality_reasons=bundle.quality_reasons,
            ),
            ros_now_ns=now_ns,
        )

    def _safe_stamp(self, stamp: Time, now_ns: int) -> int | None:
        try:
            return _stamp_nanoseconds(stamp)
        except ValueError as error:
            self.get_logger().error(str(error))
            self._state_machine.observe_hard_event(
                DegradationReason.TIMESTAMP_INVALID,
                ros_now_ns=now_ns,
            )
            return None

    def _prepare_clock(self) -> int:
        now_ns = int(self.get_clock().now().nanoseconds)
        if self._last_clock_ns is not None and now_ns < self._last_clock_ns:
            self._assembler.reset()
            self._state_machine.reset_for_clock(ros_now_ns=now_ns)
            self._motion.reset()
            if self._estimator is not None:
                self._estimator.reset()
            self._pending_estimator_row = None
            self._pending_estimator_holds_score = False
            self._last_effective_point_count = None
            self.get_logger().warning(
                "ROS clock moved backwards; cleared FAST-LIO2 confidence state"
            )
        self._last_clock_ns = now_ns
        return now_ns

    def _positive_float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def _nonnegative_float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def _log_transition(
        self,
        state: int,
        reasons: DegradationReason,
    ) -> None:
        signature = (int(state), int(reasons))
        if signature == self._last_log_signature:
            return
        self._last_log_signature = signature
        self.get_logger().info(
            "SLAM confidence transition: "
            f"state={int(state)} reasons=0x{int(reasons):08x} "
            f"effective_points={self._last_effective_point_count}"
        )


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FastlioConfidenceExtractor()
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
