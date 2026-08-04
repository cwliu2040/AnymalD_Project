"""Publish the project LiDAR scan in the FAST-LIO2 Ouster field contract."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from anymal_locomotion_ros2.lidar_adapter_core import (
    FASTLIO_POINT_DTYPE,
    convert_rtx_points_to_ouster,
    deterministic_point_indices,
    scan_start_nanoseconds,
)

_OUTPUT_FIELDS = (
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(
        name="intensity",
        offset=12,
        datatype=PointField.FLOAT32,
        count=1,
    ),
    PointField(name="t", offset=16, datatype=PointField.UINT32, count=1),
    PointField(
        name="reflectivity",
        offset=20,
        datatype=PointField.UINT16,
        count=1,
    ),
    PointField(name="ring", offset=22, datatype=PointField.UINT8, count=1),
    PointField(
        name="ambient",
        offset=24,
        datatype=PointField.UINT16,
        count=1,
    ),
    PointField(name="range", offset=28, datatype=PointField.UINT32, count=1),
)

_RELIABLE_OUTPUT_QOS = QoSProfile(
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


class FastlioPointAdapter(Node):
    """Convert the same raw scan as LIO-SAM without using project deskew."""

    def __init__(self) -> None:
        super().__init__("anymal_fastlio_point_adapter")
        self.declare_parameter("input_topic", "/lidar/points_raw")
        self.declare_parameter("output_topic", "/fastlio/points")
        self.declare_parameter("frame_id", "lidar_link")
        self.declare_parameter("scan_rate_hz", 10.0)
        self.declare_parameter("raw_stamp_is_scan_end", True)
        self.declare_parameter("point_density", 1.0)

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
        self._logged_contract = False

        # The candidate fork uses the default reliable PointCloud2
        # subscription.  Keep the raw simulator input sensor-data QoS, but
        # offer the adapted cloud as reliable so DDS can connect to it.
        self._publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("output_topic").value),
            _RELIABLE_OUTPUT_QOS,
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
                "Expected one unorganized full scan, received "
                f"height={message.height}"
            )
            return
        if message.row_step != message.point_step * message.width:
            self.get_logger().error("PointCloud2 row padding is not supported")
            return
        field_names = {field.name for field in message.fields}
        missing = {"x", "y", "z"} - field_names
        if missing:
            self.get_logger().error(
                "Raw PointCloud2 is missing required fields: "
                f"{sorted(missing)}"
            )
            return
        if message.header.frame_id and message.header.frame_id != self._frame_id:
            self.get_logger().error(
                "Raw PointCloud2 frame mismatch: "
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
        indices = deterministic_point_indices(
            xyz.shape[0],
            self._point_density,
        )
        xyz = xyz[indices]
        if intensity is not None:
            intensity = intensity[indices]
        converted = convert_rtx_points_to_ouster(
            xyz,
            intensity,
            scan_period_s=self._scan_period_s,
        )
        if converted.size == 0:
            self.get_logger().warning("Raw scan contained no finite points")
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
        output.point_step = FASTLIO_POINT_DTYPE.itemsize
        output.row_step = output.point_step * output.width
        # noise and ambient occupy the same bytes in the two contracts.  The
        # candidate only sees the field names above, not the source dtype.
        output.data = converted.tobytes()
        output.is_dense = True
        self._publisher.publish(output)

        if not self._logged_contract:
            self.get_logger().info(
                "Publishing FAST-LIO2 Ouster scans with ring/t/ambient "
                f"fields: points={output.width}, "
                f"point_density={self._point_density:.2f}, "
                f"frame={output.header.frame_id}, "
                f"scan_period={self._scan_period_s:.3f}s, "
                f"rings={int(converted['ring'].min())}.."
                f"{int(converted['ring'].max())}, "
                f"relative_time_ns={int(converted['t'].min())}.."
                f"{int(converted['t'].max())}, reliable_output=true"
            )
            self._logged_contract = True


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FastlioPointAdapter()
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
