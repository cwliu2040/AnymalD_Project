"""Evaluate one deterministic LIO-SAM replay, including loop factors."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray

from anymal_locomotion_ros2.lio_benchmark_core import (
    PoseSample,
    evaluate_trajectory,
)
from anymal_locomotion_ros2.lio_benchmark_node import _pose_sample


class LioReplayEvaluatorNode(Node):
    """Compare replayed mapping output with recorded simulator truth."""

    def __init__(self) -> None:
        super().__init__("anymal_lio_replay_evaluator")
        self.declare_parameter("output_path", "")
        self.declare_parameter("project_root", "")
        self.declare_parameter("estimate_topic", "/lio_sam/mapping/odometry")
        self.declare_parameter(
            "ground_truth_sensor_offset_xyz",
            [0.20, 0.0, 0.35],
        )
        self.declare_parameter("loop_closure_expectation", "disabled")

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
                "Replay output must remain inside the project: "
                f"{self._output_path}"
            )
        self._expectation = str(
            self.get_parameter("loop_closure_expectation").value
        )
        if self._expectation not in {"required", "forbidden"}:
            raise ValueError(
                "loop_closure_expectation must be required or forbidden"
            )
        self._estimate_topic = str(
            self.get_parameter("estimate_topic").value
        )
        offset = np.asarray(
            self.get_parameter("ground_truth_sensor_offset_xyz").value,
            dtype=np.float64,
        )
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError(
                "ground_truth_sensor_offset_xyz must contain three "
                "finite values"
            )
        self._ground_truth_sensor_offset_xyz = tuple(
            float(value) for value in offset
        )

        self._truth: list[PoseSample] = []
        self._estimate: list[PoseSample] = []
        self._truth_violations = 0
        self._estimate_violations = 0
        self._loop_marker_messages = 0
        self._max_constraint_edges = 0
        self._last_constraint_stamp_s: float | None = None
        self._finished = False
        self.exit_code = 1

        self.create_subscription(
            Odometry,
            "/odom",
            self._on_truth,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            self._estimate_topic,
            self._on_estimate,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            MarkerArray,
            "/lio_sam/mapping/loop_closure_constraints",
            self._on_loop_constraints,
            10,
        )
        self.create_subscription(
            Empty,
            "/lio_replay/finish",
            self._on_finish,
            10,
        )

    @staticmethod
    def _append_monotonic(
        samples: list[PoseSample],
        sample: PoseSample,
    ) -> bool:
        violation = bool(samples and sample.stamp_s <= samples[-1].stamp_s)
        samples.append(sample)
        return violation

    def _on_truth(self, message: Odometry) -> None:
        self._truth_violations += int(
            self._append_monotonic(self._truth, _pose_sample(message))
        )

    def _on_estimate(self, message: Odometry) -> None:
        self._estimate_violations += int(
            self._append_monotonic(self._estimate, _pose_sample(message))
        )

    def _on_loop_constraints(self, message: MarkerArray) -> None:
        edge_markers = [
            marker
            for marker in message.markers
            if marker.type == Marker.LINE_LIST
            and marker.ns == "loop_edges"
        ]
        edge_count = max(
            (len(marker.points) // 2 for marker in edge_markers),
            default=0,
        )
        self._loop_marker_messages += 1
        if edge_count > self._max_constraint_edges:
            self._max_constraint_edges = edge_count
            stamps = [
                float(marker.header.stamp.sec)
                + float(marker.header.stamp.nanosec) * 1.0e-9
                for marker in edge_markers
                if marker.points
            ]
            if stamps:
                self._last_constraint_stamp_s = max(stamps)

    def _on_finish(self, _message: Empty) -> None:
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        failures: list[str] = []
        metrics: dict[str, float | int] = {}
        try:
            metrics = evaluate_trajectory(
                self._truth,
                self._estimate,
                ground_truth_sensor_offset_xyz=(
                    self._ground_truth_sensor_offset_xyz
                ),
            )
        except ValueError as error:
            failures.append(str(error))

        if self._truth_violations or self._estimate_violations:
            failures.append("one or more replay timestamps were non-monotonic")
        if metrics:
            translation_limit = max(
                0.10,
                0.01 * float(metrics["path_length_m"]),
            )
            if float(metrics["translation_ate_rmse_m"]) > translation_limit:
                failures.append(
                    "translation ATE RMSE exceeds "
                    f"{translation_limit:.3f} m"
                )
            if float(metrics["yaw_rmse_deg"]) > 1.0:
                failures.append("yaw RMSE exceeds 1 degree")
            if float(metrics["translation_jump_residual_max_m"]) > 0.20:
                failures.append("translation pose jump exceeds 0.20 m")
            if float(metrics["yaw_jump_residual_max_deg"]) > 2.0:
                failures.append("yaw pose jump exceeds 2 degrees")

        post_constraint_mapping_samples = 0
        if self._last_constraint_stamp_s is not None:
            post_constraint_mapping_samples = sum(
                sample.stamp_s > self._last_constraint_stamp_s
                for sample in self._estimate
            )
        if self._expectation == "required":
            if self._max_constraint_edges < 1:
                failures.append("no LIO-SAM loop constraint edge was observed")
            elif post_constraint_mapping_samples < 1:
                failures.append(
                    "no mapping update followed the accepted loop constraint"
                )
        elif self._max_constraint_edges != 0:
            failures.append(
                "an unexpected LIO-SAM loop constraint edge was observed"
            )

        report = {
            "schema_version": 1,
            "status": "passed" if not failures else "failed",
            "failures": failures,
            "trajectory": metrics,
            "estimate_topic": self._estimate_topic,
            "ground_truth_sensor_offset_xyz": list(
                self._ground_truth_sensor_offset_xyz
            ),
            "counts": {
                "ground_truth_odometry": len(self._truth),
                "mapping_odometry": len(self._estimate),
                "loop_closure_marker": self._loop_marker_messages,
            },
            "timestamp_violations": {
                "ground_truth_odometry": self._truth_violations,
                "mapping_odometry": self._estimate_violations,
            },
            "loop_closure": {
                "expectation": self._expectation,
                "max_constraint_edge_count": self._max_constraint_edges,
                "last_constraint_stamp_s": self._last_constraint_stamp_s,
                "post_constraint_mapping_samples": (
                    post_constraint_mapping_samples
                ),
            },
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.get_logger().info(json.dumps(report, sort_keys=True))
        self.exit_code = 0 if not failures else 1


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LioReplayEvaluatorNode()
    try:
        while rclpy.ok() and not node._finished:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        exit_code = node.exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv)
