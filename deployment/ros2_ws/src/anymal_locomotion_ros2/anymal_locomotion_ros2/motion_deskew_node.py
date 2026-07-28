"""Complete LIO-SAM motion deskew without modifying upstream sources."""

from __future__ import annotations

import math
import sys
from collections import deque
from collections.abc import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from lio_sam.msg import CloudInfo
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2

from anymal_locomotion_ros2.motion_deskew_core import (
    TimedAngularVelocity,
    TimedPose,
    deskew_points_from_imu_and_odometry,
    select_range_image_points,
    translational_offsets_from_odometry,
)


def _stamp_seconds(message) -> float:
    return (
        float(message.header.stamp.sec)
        + float(message.header.stamp.nanosec) * 1.0e-9
    )


class MotionDeskewNode(Node):
    """Add missing translation while retaining upstream raw-IMU rotation."""

    def __init__(self) -> None:
        super().__init__("anymal_lio_motion_deskew")
        self.declare_parameter(
            "input_topic",
            "/lio_sam/deskew/cloud_info",
        )
        self.declare_parameter(
            "output_topic",
            "/lio_sam/deskew/cloud_info_motion_corrected",
        )
        self.declare_parameter(
            "corrected_cloud_topic",
            "/lio_sam/deskew/cloud_deskewed_motion_corrected",
        )
        self.declare_parameter(
            "odometry_topic",
            "/lio_sam/odometry/imu_incremental",
        )
        self.declare_parameter("raw_cloud_topic", "/lio_sam/points")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter(
            "diagnostics_topic",
            "/lio_sam/deskew/motion",
        )
        self.declare_parameter("scan_rate_hz", 10.0)
        self.declare_parameter("n_scan", 32)
        self.declare_parameter("horizon_scan", 1024)
        self.declare_parameter("downsample_rate", 1)
        self.declare_parameter("min_range_m", 0.5)
        self.declare_parameter("max_range_m", 100.0)
        self.declare_parameter("max_pending_scans", 5)
        self.declare_parameter("apply_translation", True)
        self.declare_parameter("replace_upstream_rotation", False)

        scan_rate_hz = float(self.get_parameter("scan_rate_hz").value)
        self._scan_period_s = 1.0 / scan_rate_hz
        self._n_scan = int(self.get_parameter("n_scan").value)
        self._horizon_scan = int(self.get_parameter("horizon_scan").value)
        self._downsample_rate = int(
            self.get_parameter("downsample_rate").value
        )
        self._min_range_m = float(self.get_parameter("min_range_m").value)
        self._max_range_m = float(self.get_parameter("max_range_m").value)
        self._max_pending_scans = int(
            self.get_parameter("max_pending_scans").value
        )
        self._replace_upstream_rotation = bool(
            self.get_parameter("replace_upstream_rotation").value
        )
        self._apply_translation = bool(
            self.get_parameter("apply_translation").value
        )
        if (
            not math.isfinite(scan_rate_hz)
            or scan_rate_hz <= 0.0
            or self._n_scan <= 0
            or self._horizon_scan <= 0
            or self._downsample_rate <= 0
            or not math.isfinite(self._min_range_m)
            or not math.isfinite(self._max_range_m)
            or self._min_range_m < 0.0
            or self._max_range_m <= self._min_range_m
            or self._max_pending_scans <= 0
        ):
            raise ValueError("invalid SE(3) deskew configuration")

        self._odometry: deque[TimedPose] = deque()
        self._imu: deque[TimedAngularVelocity] = deque()
        self._raw_clouds: deque[PointCloud2] = deque()
        self._pending: deque[CloudInfo] = deque()
        self._publisher = self.create_publisher(
            CloudInfo,
            str(self.get_parameter("output_topic").value),
            qos_profile_sensor_data,
        )
        self._cloud_publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("corrected_cloud_topic").value),
            qos_profile_sensor_data,
        )
        self._diagnostics_publisher = self.create_publisher(
            Vector3Stamped,
            str(self.get_parameter("diagnostics_topic").value),
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odometry_topic").value),
            self._on_odometry,
            qos_profile_sensor_data,
        )
        if self._replace_upstream_rotation:
            self.create_subscription(
                Imu,
                str(self.get_parameter("imu_topic").value),
                self._on_imu,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                PointCloud2,
                str(self.get_parameter("raw_cloud_topic").value),
                self._on_raw_cloud,
                qos_profile_sensor_data,
            )
        self.create_subscription(
            CloudInfo,
            str(self.get_parameter("input_topic").value),
            self._on_cloud_info,
            qos_profile_sensor_data,
        )

    def _on_odometry(self, message: Odometry) -> None:
        orientation = message.pose.pose.orientation
        position = message.pose.pose.position
        sample = TimedPose(
            stamp_s=_stamp_seconds(message),
            position_xyz=(position.x, position.y, position.z),
            orientation_xyzw=(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
            state_id=int(round(message.pose.covariance[0])),
        )
        if self._odometry and sample.stamp_s <= self._odometry[-1].stamp_s:
            return
        self._odometry.append(sample)
        while (
            len(self._odometry) > 2
            and self._odometry[1].stamp_s < sample.stamp_s - 3.0
        ):
            self._odometry.popleft()
        self._flush_pending()

    def _on_imu(self, message: Imu) -> None:
        angular = message.angular_velocity
        sample = TimedAngularVelocity(
            stamp_s=_stamp_seconds(message),
            angular_velocity_xyz=(angular.x, angular.y, angular.z),
        )
        if self._imu and sample.stamp_s <= self._imu[-1].stamp_s:
            return
        self._imu.append(sample)
        while (
            len(self._imu) > 2
            and self._imu[1].stamp_s < sample.stamp_s - 3.0
        ):
            self._imu.popleft()
        self._flush_pending()

    def _on_raw_cloud(self, message: PointCloud2) -> None:
        stamp_s = _stamp_seconds(message)
        if self._raw_clouds and stamp_s <= _stamp_seconds(
            self._raw_clouds[-1]
        ):
            return
        self._raw_clouds.append(message)
        while (
            len(self._raw_clouds) > 2
            and _stamp_seconds(self._raw_clouds[1]) < stamp_s - 3.0
        ):
            self._raw_clouds.popleft()
        self._flush_pending()

    def _on_cloud_info(self, message: CloudInfo) -> None:
        if not self._odometry:
            self._publish(message, None)
            return
        self._pending.append(message)
        self._flush_pending()
        while len(self._pending) > self._max_pending_scans:
            self._publish(self._pending.popleft(), None)

    def _flush_pending(self) -> None:
        while (
            self._pending
            and self._odometry
            and (self._raw_clouds or not self._replace_upstream_rotation)
            and (self._imu or not self._replace_upstream_rotation)
        ):
            message = self._pending[0]
            scan_start_s = _stamp_seconds(message)
            scan_end_s = scan_start_s + self._scan_period_s
            if self._odometry[-1].stamp_s < scan_end_s:
                return
            if (
                self._replace_upstream_rotation
                and self._imu[-1].stamp_s < scan_end_s
            ):
                return
            raw_cloud = None
            if self._replace_upstream_rotation:
                raw_cloud = next(
                    (
                        cloud
                        for cloud in self._raw_clouds
                        if math.isclose(
                            _stamp_seconds(cloud),
                            scan_start_s,
                            rel_tol=0.0,
                            abs_tol=1.0e-7,
                        )
                    ),
                    None,
                )
            if self._replace_upstream_rotation and raw_cloud is None:
                if _stamp_seconds(self._raw_clouds[-1]) <= scan_start_s:
                    return
                self._pending.popleft()
                self.get_logger().warning(
                    f"SE(3) deskew unavailable for scan {scan_start_s:.3f}: "
                    "matching raw cloud was not retained"
                )
                self._publish(message, None)
                continue
            self._pending.popleft()
            try:
                samples = list(self._odometry)
                translation = self._apply_deskew(
                    raw_cloud,
                    message.cloud_deskewed,
                    message,
                    scan_start_s,
                    samples,
                )
            except (ValueError, TypeError) as error:
                self.get_logger().warning(
                    f"SE(3) deskew unavailable for scan "
                    f"{scan_start_s:.3f}: {error}"
                )
                self._publish(message, None)
                continue
            self._publish(message, translation)

    def _apply_deskew(
        self,
        raw_cloud: PointCloud2 | None,
        cloud: PointCloud2,
        cloud_info: CloudInfo,
        scan_start_s: float,
        samples: list[TimedPose],
    ) -> np.ndarray:
        if cloud.is_bigendian:
            raise ValueError("big-endian PointCloud2 is unsupported")
        point_count = int(cloud.width) * int(cloud.height)
        if point_count <= 0:
            raise ValueError("deskewed cloud contains no points")
        cloud_info_columns = np.asarray(
            cloud_info.point_col_ind[:point_count],
            dtype=np.int32,
        )
        if cloud_info_columns.shape != (point_count,):
            raise ValueError("cloud_info has fewer column indices than points")
        if self._replace_upstream_rotation:
            if raw_cloud is None or raw_cloud.is_bigendian:
                raise ValueError(
                    "full SE(3) mode requires a little-endian raw cloud"
                )
            raw_dtype = point_cloud2.dtype_from_fields(
                raw_cloud.fields,
                point_step=raw_cloud.point_step,
            )
            raw = np.frombuffer(
                raw_cloud.data,
                dtype=raw_dtype,
                count=int(raw_cloud.width) * int(raw_cloud.height),
            )
            missing = {"x", "y", "z", "ring", "t"} - set(
                raw.dtype.names or ()
            )
            if missing:
                raise ValueError(
                    f"raw PointCloud2 lacks fields {sorted(missing)}"
                )
            raw_xyz = np.column_stack((raw["x"], raw["y"], raw["z"]))
            selected_xyz, relative_times_s, selected_columns = (
                select_range_image_points(
                    raw_xyz,
                    raw["ring"],
                    np.asarray(raw["t"], dtype=np.float64) * 1.0e-9,
                    n_scan=self._n_scan,
                    horizon_scan=self._horizon_scan,
                    downsample_rate=self._downsample_rate,
                    min_range_m=self._min_range_m,
                    max_range_m=self._max_range_m,
                )
            )
            if len(selected_xyz) != point_count or not np.array_equal(
                selected_columns,
                cloud_info_columns,
            ):
                raise ValueError(
                    "raw range-image reconstruction does not match cloud_info "
                    f"({len(selected_xyz)} selected, {point_count} expected)"
                )
            corrected_xyz, translation = (
                deskew_points_from_imu_and_odometry(
                    selected_xyz,
                    scan_start_s + relative_times_s,
                    list(self._imu),
                    samples,
                    apply_translation=self._apply_translation,
                )
            )
            corrected_xyz = corrected_xyz.astype(np.float32)
        else:
            if self._apply_translation:
                offsets, translation = translational_offsets_from_odometry(
                    cloud_info_columns,
                    horizon_scan=self._horizon_scan,
                    scan_start_s=scan_start_s,
                    scan_period_s=self._scan_period_s,
                    samples=samples,
                )
            else:
                offsets = np.zeros((point_count, 3), dtype=np.float64)
                translation = np.zeros(3, dtype=np.float64)
            corrected_xyz = offsets.astype(np.float32)
        field_offsets = {
            field.name: int(field.offset) for field in cloud.fields
        }
        if not {"x", "y", "z"}.issubset(field_offsets):
            raise ValueError("deskewed PointCloud2 lacks x/y/z fields")
        data = bytearray(cloud.data)
        for axis, name in enumerate(("x", "y", "z")):
            values = np.ndarray(
                shape=(point_count,),
                dtype="<f4",
                buffer=data,
                offset=field_offsets[name],
                strides=(int(cloud.point_step),),
            )
            if self._replace_upstream_rotation:
                values[:] = corrected_xyz[:, axis]
            else:
                values += corrected_xyz[:, axis]
        cloud.data = bytes(data)
        return translation

    def _publish(
        self,
        message: CloudInfo,
        translation: np.ndarray | None,
    ) -> None:
        status = Vector3Stamped()
        status.header = message.header
        if translation is None:
            status.header.frame_id = "motion_deskew_unavailable"
            status.vector.x = math.nan
            status.vector.y = math.nan
            status.vector.z = math.nan
        else:
            status.header.frame_id = "motion_deskew_applied"
            status.vector.x = float(translation[0])
            status.vector.y = float(translation[1])
            status.vector.z = float(translation[2])
        self._diagnostics_publisher.publish(status)
        self._cloud_publisher.publish(message.cloud_deskewed)
        self._publisher.publish(message)


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MotionDeskewNode()
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
    main(sys.argv)
