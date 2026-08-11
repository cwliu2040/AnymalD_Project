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
    OS1_32_HORIZONTAL_RESOLUTION,
    OUSTER_POINT_DTYPE,
    convert_rtx_points_to_ouster,
    deterministic_point_indices,
    gradual_density_ratio,
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
        self.declare_parameter("point_density", 1.0)
        self.declare_parameter("point_density_profile", "constant")
        self.declare_parameter("point_density_min", 0.01)
        self.declare_parameter("time_source", "sensor_order")
        self.declare_parameter("point_order", "destaggered")

        self._frame_id = str(self.get_parameter("frame_id").value)
        scan_rate_hz = float(self.get_parameter("scan_rate_hz").value)
        if not np.isfinite(scan_rate_hz) or scan_rate_hz <= 0.0:
            raise ValueError("scan_rate_hz must be finite and positive")
        self._scan_period_s = 1.0 / scan_rate_hz
        self._raw_stamp_is_scan_end = bool(
            self.get_parameter("raw_stamp_is_scan_end").value
        )
        self._point_density = float(
            self.get_parameter("point_density").value
        )
        if not 0.0 < self._point_density <= 1.0:
            raise ValueError("point_density must be in (0, 1]")
        self._point_density_profile = str(
            self.get_parameter("point_density_profile").value
        ).strip().lower()
        if self._point_density_profile not in {"constant", "gradual_v1", "gradual_v2"}:
            raise ValueError("unsupported point_density_profile")
        self._point_density_min = float(
            self.get_parameter("point_density_min").value
        )
        if not 0.0 < self._point_density_min <= self._point_density:
            raise ValueError("point_density_min must be in (0, point_density]")
        self._first_stamp_ns: int | None = None
        self._time_source = str(
            self.get_parameter("time_source").value
        ).strip().lower()
        if self._time_source not in {
            "sensor_order",
            "sensor_order_fire_time",
            "sensor_order_destaggered",
            "azimuth",
        }:
            raise ValueError(
                "time_source must be sensor_order, sensor_order_fire_time, "
                "sensor_order_destaggered, or azimuth"
            )
        self._point_order = str(
            self.get_parameter("point_order").value
        ).strip().lower()
        if self._point_order not in {
            "destaggered",
            "staggered",
            "column",
            "column_shift1",
        }:
            raise ValueError(
                "point_order must be destaggered, staggered, column, or "
                "column_shift1"
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
            time_source=self._time_source,
            point_order=self._point_order,
            horizontal_resolution=OS1_32_HORIZONTAL_RESOLUTION,
        )
        if converted.size == 0:
            self.get_logger().warning("RTX scan contained no finite points")
            return
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        if self._first_stamp_ns is None:
            self._first_stamp_ns = stamp_ns
        elapsed_s = max(0, stamp_ns - self._first_stamp_ns) * 1.0e-9
        density = self._point_density
        if self._point_density_profile in {"gradual_v1", "gradual_v2"}:
            phases = (
                {"healthy_s": 3.0, "ramp_down_s": 1.5, "hold_s": 4.5, "ramp_up_s": 3.0}
                if self._point_density_profile == "gradual_v2"
                else {}
            )
            density = gradual_density_ratio(
                elapsed_s,
                nominal_ratio=self._point_density,
                minimum_ratio=self._point_density_min,
                **phases,
            )
        if self._point_density_profile == "gradual_v2":
            indices = np.arange(
                max(1, int(round(converted.size * density))), dtype=np.int64
            )
        else:
            indices = deterministic_point_indices(converted.size, density)
        converted = converted[indices]
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
                f"point_density={density:.3f}, "
                f"density_profile={self._point_density_profile}, "
                f"time_source={self._time_source}, "
                f"point_order={self._point_order}, "
                f"scan_period={self._scan_period_s:.3f}s, "
                f"rings={int(converted['ring'].min())}.."
                f"{int(converted['ring'].max())}, "
                f"relative_time_ns={int(converted['t'].min())}.."
                f"{int(converted['t'].max())}, "
                f"raw_stamp_ns={stamp_ns}, output_stamp_ns={output_stamp_ns}"
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
        except (KeyboardInterrupt, RuntimeError):
            pass


if __name__ == "__main__":
    main()
