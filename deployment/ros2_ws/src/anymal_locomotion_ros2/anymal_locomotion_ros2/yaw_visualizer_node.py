"""Render registered clouds into an anonymous fixed-view MP4."""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Empty


class YawVisualizerNode(Node):
    def __init__(self) -> None:
        super().__init__("anymal_yaw_visualizer")
        self.declare_parameter("cloud_topic", "/cloud_registered")
        self.declare_parameter("output_path", "")
        self.declare_parameter("project_root", "")
        self.declare_parameter("view_half_extent_m", 15.0)
        self.declare_parameter("decay_s", 30.0)
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("image_size_px", 720)
        self.declare_parameter("max_points_per_frame", 120000)

        project_root = Path(
            str(self.get_parameter("project_root").value)
        ).expanduser().resolve()
        output_path = Path(
            str(self.get_parameter("output_path").value)
        ).expanduser().resolve()
        if not project_root.is_dir() or not output_path.is_relative_to(project_root):
            raise ValueError("visual output must remain inside the project")
        self._output_path = output_path
        self._extent = float(self.get_parameter("view_half_extent_m").value)
        self._decay_s = float(self.get_parameter("decay_s").value)
        self._fps = float(self.get_parameter("fps").value)
        self._size = int(self.get_parameter("image_size_px").value)
        self._max_points = int(
            self.get_parameter("max_points_per_frame").value
        )
        if (
            self._extent <= 0.0
            or self._decay_s <= 0.0
            or self._fps <= 0.0
            or self._size < 64
            or self._max_points < 100
        ):
            raise ValueError("invalid yaw visualizer parameters")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps,
            (self._size, self._size),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"cannot open MP4 writer: {output_path}")
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._frame_count = 0
        self._finished = False
        self.exit_code = 1
        topic = str(self.get_parameter("cloud_topic").value)
        self.create_subscription(
            PointCloud2, topic, self._on_cloud, qos_profile_sensor_data
        )
        self.create_subscription(Empty, "/lio_replay/finish", self._on_finish, 10)

    @staticmethod
    def _stamp_s(message: PointCloud2) -> float:
        return (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        points = point_cloud2.read_points_numpy(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if not xyz.size:
            return
        stamp_s = self._stamp_s(message)
        self._history.append((stamp_s, xyz))
        while self._history and stamp_s - self._history[0][0] > self._decay_s:
            self._history.popleft()
        combined = np.concatenate([sample for _, sample in self._history])
        if len(combined) > self._max_points:
            indices = np.linspace(
                0, len(combined) - 1, self._max_points, dtype=np.int64
            )
            combined = combined[indices]
        frame = self._render(combined)
        self._writer.write(frame)
        self._frame_count += 1

    def _render(self, xyz: np.ndarray) -> np.ndarray:
        frame = np.full((self._size, self._size, 3), 18, dtype=np.uint8)
        scale = (self._size - 1) / (2.0 * self._extent)
        px = np.rint((xyz[:, 0] + self._extent) * scale).astype(np.int32)
        py = np.rint((self._extent - xyz[:, 1]) * scale).astype(np.int32)
        valid = (
            (px >= 0) & (px < self._size) & (py >= 0) & (py < self._size)
        )
        if np.any(valid):
            z = np.clip((xyz[valid, 2] + 1.0) / 4.0, 0.0, 1.0)
            colors = np.column_stack(
                (
                    255.0 * (1.0 - z),
                    180.0 + 75.0 * z,
                    255.0 * z,
                )
            ).astype(np.uint8)
            frame[py[valid], px[valid]] = colors
        center = self._size // 2
        cv2.drawMarker(
            frame, (center, center), (255, 255, 255), cv2.MARKER_CROSS, 12, 1
        )
        return frame

    def _on_finish(self, _message: Empty) -> None:
        if self._finished:
            return
        self._finished = True
        self._writer.release()
        self.exit_code = 0 if self._frame_count > 0 else 1
        self.get_logger().info(
            f"visual MP4 frames={self._frame_count} path={self._output_path}"
        )

    def close(self) -> None:
        if self._writer.isOpened():
            self._writer.release()


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YawVisualizerNode()
    try:
        while rclpy.ok() and not node._finished:
            rclpy.spin_once(node)
    finally:
        exit_code = node.exit_code
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv)
