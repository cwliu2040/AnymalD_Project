"""Fail-closed LIO-SAM extractor for backend-neutral SLAM confidence."""

from __future__ import annotations

import math
from collections.abc import Sequence

import rclpy
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Vector3Stamped
from lio_sam.msg import CloudInfo
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Bool

from anymal_locomotion_interfaces.msg import SlamConfidence
from anymal_locomotion_ros2.liosam_confidence_core import (
    LiosamAssemblerConfig,
    LiosamBundle,
    LiosamSignalAssembler,
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


_BACKEND_ID = "liosam"
_UNCALIBRATED_ID = "uncalibrated"

_BEST_EFFORT_DEPTH_20 = QoSProfile(
    depth=20,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
_BEST_EFFORT_DEPTH_200 = QoSProfile(
    depth=200,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
_RELIABLE_DEPTH_20 = QoSProfile(
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
_AUTHORITATIVE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _stamp_nanoseconds(stamp: Time) -> int:
    if not 0 <= int(stamp.nanosec) < 1_000_000_000:
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


def _pose_disagreement(mapping_pose: PoseTuple, incremental_pose: PoseTuple) -> tuple[float, float]:
    translation = math.dist(mapping_pose[:3], incremental_pose[:3])

    def yaw(pose: PoseTuple) -> float:
        qx, qy, qz, qw = pose[3:]
        return math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    delta = math.atan2(
        math.sin(yaw(mapping_pose) - yaw(incremental_pose)),
        math.cos(yaw(mapping_pose) - yaw(incremental_pose)),
    )
    return translation, abs(math.degrees(delta))


class LiosamConfidenceExtractor(Node):
    """Assemble public LIO-SAM signals into the common fail-closed schema."""

    def __init__(self) -> None:
        super().__init__("anymal_liosam_confidence_extractor")
        self.declare_parameter(
            "native_odometry_topic",
            "/lio_sam/mapping/odometry",
        )
        self.declare_parameter("canonical_odometry_topic", "/slam/odom")
        self.declare_parameter(
            "incremental_odometry_topic",
            "/lio_sam/mapping/odometry_incremental",
        )
        self.declare_parameter(
            "feature_cloud_info_topic",
            "/lio_sam/feature/cloud_info",
        )
        self.declare_parameter("lidar_input_topic", "/lio_sam/points")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter(
            "motion_deskew_topic",
            "/lio_sam/deskew/motion",
        )
        self.declare_parameter("motion_deskew_required", True)
        self.declare_parameter("confidence_topic", "/slam_confidence")
        self.declare_parameter(
            "tracking_valid_topic",
            "/slam_tracking_valid",
        )
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("source_timeout_s", 0.50)
        self.declare_parameter("odometry_timeout_s", 0.50)
        self.declare_parameter("lidar_timeout_s", 0.50)
        self.declare_parameter("imu_timeout_s", 0.05)
        self.declare_parameter("future_stamp_tolerance_s", 0.005)
        self.declare_parameter("diagnostic_match_timeout_s", 0.20)
        self.declare_parameter("max_pending_bundles", 64)
        self.declare_parameter("edge_feature_min_valid", 10)
        self.declare_parameter("surface_feature_min_valid", 100)
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
        assembler_config = LiosamAssemblerConfig(
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
            edge_feature_min_valid=self._nonnegative_int_parameter(
                "edge_feature_min_valid"
            ),
            surface_feature_min_valid=self._nonnegative_int_parameter(
                "surface_feature_min_valid"
            ),
            motion_deskew_required=bool(
                self.get_parameter("motion_deskew_required").value
            ),
            max_pending_bundles=self._positive_int_parameter(
                "max_pending_bundles"
            ),
        )
        self._assembler = LiosamSignalAssembler(assembler_config)
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
        self._last_feature_counts: tuple[int, int] | None = None
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
            _BEST_EFFORT_DEPTH_20,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("canonical_odometry_topic").value),
            self._on_canonical_odometry,
            _RELIABLE_DEPTH_20,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("incremental_odometry_topic").value),
            self._on_incremental_odometry,
            _BEST_EFFORT_DEPTH_20,
        )
        self.create_subscription(
            CloudInfo,
            str(self.get_parameter("feature_cloud_info_topic").value),
            self._on_feature_cloud_info,
            _BEST_EFFORT_DEPTH_20,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("lidar_input_topic").value),
            self._on_lidar_input,
            _BEST_EFFORT_DEPTH_20,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter("imu_topic").value),
            self._on_imu,
            _BEST_EFFORT_DEPTH_200,
        )
        self.create_subscription(
            Vector3Stamped,
            str(self.get_parameter("motion_deskew_topic").value),
            self._on_motion_deskew,
            _RELIABLE_DEPTH_20,
        )
        self.create_timer(1.0 / publish_rate_hz, self._on_timer)
        if self._estimator:
            self.get_logger().info(
                f"Loaded native LIO-SAM confidence artifact {self._estimator.calibration_id}"
            )
        else:
            self.get_logger().warning(
                "LIO-SAM confidence extractor has no calibration artifact; "
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

    def _on_incremental_odometry(self, message: Odometry) -> None:
        now_ns = self._prepare_clock()
        stamp_ns = self._safe_stamp(message.header.stamp, now_ns)
        if stamp_ns is None:
            return
        flag = float(message.pose.covariance[0])
        if not math.isfinite(flag):
            self._state_machine.observe_hard_event(
                DegradationReason.NUMERIC_INVALID,
                ros_now_ns=now_ns,
            )
            return
        self._assembler.observe_incremental_odometry(
            stamp_ns=stamp_ns,
            observed_at_ns=now_ns,
            pose=_pose_tuple(message),
            degenerate=flag >= 0.5,
            frame_id=message.header.frame_id,
            child_frame_id=message.child_frame_id,
        )
        self._flush_assembler(now_ns)

    def _on_feature_cloud_info(self, message: CloudInfo) -> None:
        now_ns = self._prepare_clock()
        stamp_ns = self._safe_stamp(message.header.stamp, now_ns)
        if stamp_ns is None:
            return
        corner_count = int(message.cloud_corner.width) * int(
            message.cloud_corner.height
        )
        surface_count = int(message.cloud_surface.width) * int(
            message.cloud_surface.height
        )
        self._assembler.observe_feature_info(
            stamp_ns=stamp_ns,
            observed_at_ns=now_ns,
            corner_count=corner_count,
            surface_count=surface_count,
            imu_available=bool(message.imu_available),
            odom_available=bool(message.odom_available),
            frame_id=message.header.frame_id,
            layout_valid=(
                _point_cloud_layout_valid(message.cloud_corner)
                and _point_cloud_layout_valid(message.cloud_surface)
            ),
        )
        self._flush_assembler(now_ns)

    def _on_motion_deskew(self, message: Vector3Stamped) -> None:
        now_ns = self._prepare_clock()
        stamp_ns = self._safe_stamp(message.header.stamp, now_ns)
        if stamp_ns is None:
            return
        status = message.header.frame_id
        if status not in {
            "motion_deskew_applied",
            "motion_deskew_unavailable",
        }:
            self._state_machine.observe_hard_event(
                DegradationReason.BACKEND_ERROR,
                ros_now_ns=now_ns,
            )
            return
        self._assembler.observe_motion_deskew(
            stamp_ns=stamp_ns,
            observed_at_ns=now_ns,
            applied=status == "motion_deskew_applied",
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

    def _observe_bundle(self, bundle: LiosamBundle, *, now_ns: int) -> None:
        self._last_feature_counts = (
            bundle.corner_count,
            bundle.surface_count,
        )
        if self._estimator is not None:
            motion = self._motion.observe(
                bundle.source_stamp_ns, bundle.native_pose
            )
            translation_delta, yaw_delta = _pose_disagreement(
                bundle.native_pose,
                bundle.incremental_pose,
            )
            self._pending_estimator_row = {
                "source_stamp_ns": bundle.source_stamp_ns,
                "features": {
                    "corner_features": bundle.corner_count,
                    "surface_features": bundle.surface_count,
                    "degenerate": bundle.degenerate,
                    "odom_available": bundle.odom_available,
                    "mapping_incremental_translation_m": translation_delta,
                    "mapping_incremental_yaw_delta_deg": yaw_delta,
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
            self._last_feature_counts = None
            self.get_logger().warning(
                "ROS clock moved backwards; cleared LIO-SAM confidence state"
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

    def _positive_int_parameter(self, name: str) -> int:
        value = self.get_parameter(name).value
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _nonnegative_int_parameter(self, name: str) -> int:
        value = self.get_parameter(name).value
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    def _log_transition(self, state: int, reasons: int) -> None:
        signature = (int(state), int(reasons))
        if signature == self._last_log_signature:
            return
        self._last_log_signature = signature
        self.get_logger().info(
            "LIO-SAM confidence transition: "
            f"state={int(state)} reasons={int(reasons)} "
            f"features={self._last_feature_counts}"
        )


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LiosamConfidenceExtractor()
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
