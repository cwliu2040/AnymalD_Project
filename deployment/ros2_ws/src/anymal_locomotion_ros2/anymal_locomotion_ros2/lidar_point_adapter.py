"""Convert Isaac RTX full scans into the PointCloud2 layout required by LIO-SAM."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from anymal_locomotion_ros2.lidar_adapter_core import (
    OUSTER_POINT_DTYPE,
    convert_rtx_points_to_ouster,
    scan_start_nanoseconds,
)

_OUTPUT_FIELDS = (
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="t", offset=16, datatype=PointField.UINT32, count=1),
    PointField(name="reflectivity", offset=20, datatype=PointField.UINT16, count=1),
    PointField(name="ring", offset=22, datatype=PointField.UINT8, count=1),
    PointField(name="noise", offset=24, datatype=PointField.UINT16, count=1),
    PointField(name="range", offset=28, datatype=PointField.UINT32, count=1),
)


class LidarPointAdapter(Node):
    """Validate and enrich complete OS1 scans without entering Isaac Sim Python."""

    def __init__(self) -> None:
        super().__init__("anymal_lidar_point_adapter")
        self.declare_parameter("input_topic", "/lidar/points_raw")
        self.declare_parameter("output_topic", "/lio_sam/points")
        self.declare_parameter("frame_id", "lidar_link")
        self.declare_parameter("scan_rate_hz", 10.0)
        self.declare_parameter("raw_stamp_is_scan_end", True)

        self._frame_id = str(self.get_parameter("frame_id").value)
        scan_rate_hz = float(self.get_parameter("scan_rate_hz").value)
        if not np.isfinite(scan_rate_hz) or scan_rate_hz <= 0.0:
            raise ValueError("scan_rate_hz must be finite and positive")
        self._scan_period_s = 1.0 / scan_rate_hz
        self._raw_stamp_is_scan_end = bool(
            self.get_parameter("raw_stamp_is_scan_end").value
        )
        self._logged_contract = False

        self._publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("output_topic").value),
            qos_profile_sensor_data,
        )
        self._subscription = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_topic").value),
            self._on_cloud,
            qos_profile_sensor_data,
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        if message.is_bigendian:
            self.get_logger().error("Big-endian PointCloud2 is not supported")
            return
        if message.height != 1:
            self.get_logger().error(
                f"Expected one unorganized full scan, received height={message.height}"
            )
            return
        if message.row_step != message.point_step * message.width:
            self.get_logger().error("PointCloud2 row padding is not supported")
            return
        field_names = {field.name for field in message.fields}
        missing = {"x", "y", "z"} - field_names
        if missing:
            self.get_logger().error(
                f"RTX PointCloud2 is missing required fields: {sorted(missing)}"
            )
            return
        if message.header.frame_id and message.header.frame_id != self._frame_id:
            self.get_logger().error(
                "RTX PointCloud2 frame mismatch: "
                f"{message.header.frame_id!r} != {self._frame_id!r}"
            )
            return

        input_dtype = point_cloud2.dtype_from_fields(
            message.fields,
            point_step=message.point_step,
        )
        raw = np.frombuffer(message.data, dtype=input_dtype, count=message.width)
        xyz = np.column_stack((raw["x"], raw["y"], raw["z"]))
        intensity = raw["intensity"] if "intensity" in raw.dtype.names else None
        converted = convert_rtx_points_to_ouster(
            xyz,
            intensity,
            scan_period_s=self._scan_period_s,
        )
        if converted.size == 0:
            self.get_logger().warning("RTX scan contained no finite points")
            return

        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        output_stamp_ns = scan_start_nanoseconds(
            stamp_ns,
            scan_period_s=self._scan_period_s,
            stamp_is_scan_end=self._raw_stamp_is_scan_end,
        )
        output = PointCloud2()
        output.header.stamp.sec = output_stamp_ns // 1_000_000_000
        output.header.stamp.nanosec = output_stamp_ns % 1_000_000_000
        output.header.frame_id = self._frame_id
        output.height = 1
        output.width = int(converted.size)
        output.fields = list(_OUTPUT_FIELDS)
        output.is_bigendian = False
        output.point_step = OUSTER_POINT_DTYPE.itemsize
        output.row_step = output.point_step * output.width
        output.data = converted.tobytes()
        output.is_dense = True
        self._publisher.publish(output)

        if not self._logged_contract:
            self.get_logger().info(
                "Publishing LIO-SAM Ouster scans with ring/t fields: "
                f"points={output.width}, frame={output.header.frame_id}, "
                f"scan_period={self._scan_period_s:.3f}s"
            )
            self._logged_contract = True


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LidarPointAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
