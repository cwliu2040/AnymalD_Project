"""Evaluate one deterministic LIO-SAM replay, including loop factors."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from lio_sam.msg import CloudInfo
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray

from anymal_locomotion_ros2.lio_benchmark_core import (
    PoseSample,
    evaluate_trajectory,
)
from anymal_locomotion_interfaces.msg import SlamConfidence
from anymal_locomotion_ros2.yaw_stress_core import (
    evaluate_yaw_stress_phases,
    output_gap_metrics,
)
from anymal_locomotion_ros2.lio_benchmark_node import _pose_sample
from anymal_locomotion_ros2.slam_confidence_label_core import (
    EvaluationRecord,
    OfflineLabelConfig,
    OfflinePoseSample,
    generate_offline_labels,
)


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
        self.declare_parameter("yaw_stress_mode", False)
        self.declare_parameter("adapted_cloud_topic", "/lio_sam/points")
        self.declare_parameter("backend_kind", "liosam")
        self.declare_parameter("confidence_expected", False)
        self.declare_parameter(
            "expected_confidence_calibration_id",
            "uncalibrated",
        )
        self.declare_parameter("confidence_topic", "/slam_confidence")
        self.declare_parameter("confidence_dataset_path", "")
        self.declare_parameter("capture_group", "")
        self.declare_parameter("deskew_mode", "native")

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
        self._yaw_stress_mode = bool(
            self.get_parameter("yaw_stress_mode").value
        )
        self._adapted_cloud_topic = str(
            self.get_parameter("adapted_cloud_topic").value
        )
        self._backend_kind = str(self.get_parameter("backend_kind").value)
        self._confidence_expected = bool(
            self.get_parameter("confidence_expected").value
        )
        self._expected_confidence_calibration_id = str(
            self.get_parameter("expected_confidence_calibration_id").value
        ).strip()
        dataset_path = str(
            self.get_parameter("confidence_dataset_path").value
        ).strip()
        self._confidence_dataset_path = (
            Path(dataset_path).expanduser().resolve()
            if dataset_path
            else None
        )
        self._capture_group = str(
            self.get_parameter("capture_group").value
        ).strip()
        self._deskew_mode = str(self.get_parameter("deskew_mode").value)
        if self._confidence_dataset_path is not None:
            if not self._confidence_dataset_path.is_relative_to(project_root):
                raise ValueError("confidence dataset must remain inside project")
            if not self._capture_group:
                raise ValueError("capture_group is required for dataset capture")
            if self._deskew_mode != "native":
                raise ValueError("confidence calibration capture requires native deskew")
        if self._backend_kind not in {"liosam", "fastlio2"}:
            raise ValueError("backend_kind must be liosam or fastlio2")
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
        self._command_stamps_s: list[float] = []
        self._raw_cloud_stamps_s: list[float] = []
        self._adapted_cloud_stamps_s: list[float] = []
        self._adapted_cloud_widths: list[int] = []
        self._estimate_receipt_ages_s: list[float] = []
        self._effective_support: list[dict[str, int | float]] = []
        self._truth_violations = 0
        self._estimate_violations = 0
        self._loop_marker_messages = 0
        self._max_constraint_edges = 0
        self._last_constraint_stamp_s: float | None = None
        self._confidence_count = 0
        self._confidence_source_valid_count = 0
        self._confidence_tracking_valid_count = 0
        self._confidence_uncalibrated_count = 0
        self._confidence_backend_ids: set[str] = set()
        self._confidence_calibration_ids: set[str] = set()
        self._confidence_states: dict[int, int] = {}
        self._confidence_reason_or = 0
        self._confidence_values: list[float] = []
        self._confidence_samples: list[dict[str, int | float | bool | None]] = []
        self._support_by_stamp_ns: dict[int, dict[str, int | float | bool]] = {}
        self._incremental_by_stamp_ns: dict[
            int, dict[str, int | float | bool]
        ] = {}
        self._native_pose_by_stamp_ns: dict[int, PoseSample] = {}
        self._confidence_evaluation_violations = 0
        self._confidence_source_violations = 0
        self._last_confidence_evaluation_ns: int | None = None
        self._last_confidence_source_ns: int | None = None
        self._last_confidence_age_s: float | None = None
        self._confidence_signal_stamps: dict[str, set[int]] = {
            "native_mapping_odometry": set(),
            "canonical_odometry": set(),
            "incremental_mapping_odometry": set(),
            "feature_cloud_info": set(),
            "motion_deskew": set(),
        }
        self._confidence_signal_contracts: dict[str, set[str]] = {
            key: set() for key in self._confidence_signal_stamps
        }
        self._finished = False
        self.exit_code = 1

        self.create_subscription(
            Odometry,
            "/odom",
            self._on_truth,
            qos_profile_sensor_data,
        )
        if self._backend_kind == "liosam":
            self.create_subscription(
                CloudInfo,
                "/lio_sam/feature/cloud_info",
                self._on_lio_cloud_info,
                qos_profile_sensor_data,
            )
        else:
            self.create_subscription(
                PointCloud2,
                "/cloud_effected",
                self._on_fastlio_effected,
                qos_profile_sensor_data,
            )
        self.create_subscription(Twist, "/cmd_vel", self._on_command, 10)
        self.create_subscription(
            PointCloud2,
            "/lidar/points_raw",
            self._on_raw_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            self._adapted_cloud_topic,
            self._on_adapted_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            self._estimate_topic,
            self._on_estimate,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            "/slam/odom",
            self._on_canonical_odometry,
            10,
        )
        self.create_subscription(
            Odometry,
            "/lio_sam/mapping/odometry_incremental",
            self._on_incremental_odometry,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Vector3Stamped,
            "/lio_sam/deskew/motion",
            self._on_motion_deskew_status,
            10,
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
        self.create_subscription(
            SlamConfidence,
            str(self.get_parameter("confidence_topic").value),
            self._on_confidence,
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
        stamp_s = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )
        receipt_s = float(self.get_clock().now().nanoseconds) * 1.0e-9
        self._estimate_receipt_ages_s.append(max(0.0, receipt_s - stamp_s))
        self._estimate_violations += int(
            self._append_monotonic(self._estimate, _pose_sample(message))
        )
        if self._backend_kind == "liosam":
            self._native_pose_by_stamp_ns[round(stamp_s * 1_000_000_000)] = (
                _pose_sample(message)
            )
            self._record_confidence_signal(
                "native_mapping_odometry",
                message.header.stamp,
                f"{message.header.frame_id}->{message.child_frame_id}",
            )

    def _on_canonical_odometry(self, message: Odometry) -> None:
        self._record_confidence_signal(
            "canonical_odometry",
            message.header.stamp,
            f"{message.header.frame_id}->{message.child_frame_id}",
        )

    def _on_incremental_odometry(self, message: Odometry) -> None:
        self._record_confidence_signal(
            "incremental_mapping_odometry",
            message.header.stamp,
            f"{message.header.frame_id}->{message.child_frame_id}",
        )
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        pose = _pose_sample(message)
        self._incremental_by_stamp_ns[stamp_ns] = {
            "degenerate": float(message.pose.covariance[0]) >= 0.5,
            "x": pose.x,
            "y": pose.y,
            "z": pose.z,
            "yaw": pose.yaw,
        }

    def _on_motion_deskew_status(self, message: Vector3Stamped) -> None:
        self._record_confidence_signal(
            "motion_deskew",
            message.header.stamp,
            message.header.frame_id,
        )

    def _record_confidence_signal(
        self,
        stream: str,
        stamp,
        contract: str,
    ) -> None:
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        self._confidence_signal_stamps[stream].add(stamp_ns)
        self._confidence_signal_contracts[stream].add(contract)

    def _on_command(self, _message: Twist) -> None:
        self._command_stamps_s.append(
            float(self.get_clock().now().nanoseconds) * 1.0e-9
        )

    @staticmethod
    def _message_stamp_s(message: PointCloud2) -> float:
        return (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )

    def _on_raw_cloud(self, message: PointCloud2) -> None:
        self._raw_cloud_stamps_s.append(self._message_stamp_s(message))

    def _on_adapted_cloud(self, message: PointCloud2) -> None:
        self._adapted_cloud_stamps_s.append(self._message_stamp_s(message))
        self._adapted_cloud_widths.append(int(message.width) * int(message.height))

    def _on_lio_cloud_info(self, message: CloudInfo) -> None:
        corner = int(message.cloud_corner.width) * int(message.cloud_corner.height)
        surface = int(message.cloud_surface.width) * int(message.cloud_surface.height)
        self._effective_support.append(
            {
                "stamp_s": self._message_stamp_s(message.cloud_surface),
                "corner_features": corner,
                "surface_features": surface,
                "effective_features": corner + surface,
            }
        )
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        self._support_by_stamp_ns[stamp_ns] = {
            "corner_features": corner,
            "surface_features": surface,
            "effective_features": corner + surface,
            "imu_available": bool(message.imu_available),
            "odom_available": bool(message.odom_available),
        }
        self._record_confidence_signal(
            "feature_cloud_info",
            message.header.stamp,
            message.header.frame_id,
        )

    def _on_fastlio_effected(self, message: PointCloud2) -> None:
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        effective = int(message.width) * int(message.height)
        self._effective_support.append(
            {
                "stamp_s": self._message_stamp_s(message),
                "effective_features": effective,
            }
        )
        self._support_by_stamp_ns[stamp_ns] = {
            "effective_features": effective,
        }

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

    def _on_confidence(self, message: SlamConfidence) -> None:
        evaluation_ns = (
            int(message.evaluation_stamp.sec) * 1_000_000_000
            + int(message.evaluation_stamp.nanosec)
        )
        if (
            self._last_confidence_evaluation_ns is not None
            and evaluation_ns <= self._last_confidence_evaluation_ns
        ):
            self._confidence_evaluation_violations += 1
        self._last_confidence_evaluation_ns = evaluation_ns
        source_ns = None
        if message.source_stamp_valid:
            source_ns = (
                int(message.source_stamp.sec) * 1_000_000_000
                + int(message.source_stamp.nanosec)
            )
            if (
                self._last_confidence_source_ns is not None
                and source_ns < self._last_confidence_source_ns
            ):
                self._confidence_source_violations += 1
            self._last_confidence_source_ns = source_ns
            self._confidence_source_valid_count += 1
        self._confidence_count += 1
        self._confidence_tracking_valid_count += int(
            message.slam_tracking_valid
        )
        self._confidence_backend_ids.add(message.backend_id)
        self._confidence_calibration_ids.add(message.calibration_id)
        state = int(message.tracking_state)
        self._confidence_states[state] = self._confidence_states.get(state, 0) + 1
        reasons = int(message.degradation_reasons)
        self._confidence_reason_or |= reasons
        self._confidence_values.append(float(message.slam_confidence))
        self._confidence_uncalibrated_count += int(
            bool(reasons & SlamConfidence.REASON_UNCALIBRATED)
        )
        self._last_confidence_age_s = (
            float(message.confidence_age.sec)
            + float(message.confidence_age.nanosec) * 1.0e-9
        )
        self._confidence_samples.append(
            {
                "evaluation_stamp_ns": evaluation_ns,
                "source_stamp_ns": source_ns,
                "confidence_age_s": self._last_confidence_age_s,
                "degradation_reasons": reasons,
            }
        )

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
        if metrics and not self._yaw_stress_mode:
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

        yaw_stress: dict = {}
        if self._yaw_stress_mode:
            if not self._command_stamps_s:
                failures.append("yaw stress replay observed no /cmd_vel messages")
            else:
                experiment_start_s = min(self._command_stamps_s)
                yaw_stress = {
                    "experiment_start_s": experiment_start_s,
                    "evaluation": evaluate_yaw_stress_phases(
                        self._truth,
                        self._estimate,
                        experiment_start_s=experiment_start_s,
                        ground_truth_sensor_offset_xyz=(
                            self._ground_truth_sensor_offset_xyz
                        ),
                    ),
                    "output_gaps": output_gap_metrics(
                        sample.stamp_s for sample in self._estimate
                    ),
                    "output_receipt_age_s": {
                        "sample_count": len(self._estimate_receipt_ages_s),
                        "median": float(np.median(self._estimate_receipt_ages_s))
                        if self._estimate_receipt_ages_s else None,
                        "p95": float(np.percentile(self._estimate_receipt_ages_s, 95.0))
                        if self._estimate_receipt_ages_s else None,
                        "max": max(self._estimate_receipt_ages_s)
                        if self._estimate_receipt_ages_s else None,
                    },
                    "transport": {
                        "raw_cloud": output_gap_metrics(
                            self._raw_cloud_stamps_s
                        ),
                        "adapted_cloud": output_gap_metrics(
                            self._adapted_cloud_stamps_s
                        ),
                        "adapted_cloud_topic": self._adapted_cloud_topic,
                        "adapted_point_count": {
                            "sample_count": len(self._adapted_cloud_widths),
                            "min": min(self._adapted_cloud_widths)
                            if self._adapted_cloud_widths else None,
                            "median": float(np.median(self._adapted_cloud_widths))
                            if self._adapted_cloud_widths else None,
                            "max": max(self._adapted_cloud_widths)
                            if self._adapted_cloud_widths else None,
                        },
                    },
                    "native_effective_support": self._effective_support_summary(),
                    "tracking_ready_before_ramp": bool(
                        self._estimate
                        and self._estimate[0].stamp_s
                        <= experiment_start_s + 5.0
                    ),
                }
                if not yaw_stress["tracking_ready_before_ramp"]:
                    failures.append(
                        "backend did not publish odometry before yaw ramp"
                    )

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

        expected_backend_id = self._backend_kind
        if self._confidence_expected:
            if self._confidence_count == 0:
                failures.append("confidence instrumentation published no messages")
            if self._confidence_source_valid_count == 0:
                failures.append("confidence never assembled a valid source")
            if self._confidence_backend_ids != {expected_backend_id}:
                failures.append("confidence backend_id did not match replay backend")
            expected_calibration_ids = {
                self._expected_confidence_calibration_id
            }
            if self._confidence_calibration_ids != expected_calibration_ids:
                failures.append("instrumentation used an unexpected calibration_id")
            if self._expected_confidence_calibration_id == "uncalibrated":
                if self._confidence_uncalibrated_count != self._confidence_count:
                    failures.append("uncalibrated instrumentation omitted its reason")
                if self._confidence_tracking_valid_count:
                    failures.append("uncalibrated instrumentation became tracking-valid")
            else:
                if self._confidence_uncalibrated_count:
                    failures.append("calibrated instrumentation reported uncalibrated")
                if not self._confidence_tracking_valid_count:
                    failures.append("calibrated instrumentation never became tracking-valid")
                if not self._confidence_values or max(self._confidence_values) <= 0.0:
                    failures.append("calibrated instrumentation never published a nonzero score")
            if self._confidence_evaluation_violations:
                failures.append("confidence evaluation stamps were not increasing")
            if self._confidence_source_violations:
                failures.append("confidence source stamps regressed")
        elif self._confidence_count:
            failures.append("confidence published while instrumentation was disabled")

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
                "raw_cloud": len(self._raw_cloud_stamps_s),
                "adapted_cloud": len(self._adapted_cloud_stamps_s),
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
            "yaw_stress": yaw_stress,
            "slam_confidence": {
                "expected": self._confidence_expected,
                "message_count": self._confidence_count,
                "source_valid_count": self._confidence_source_valid_count,
                "tracking_valid_count": self._confidence_tracking_valid_count,
                "uncalibrated_count": self._confidence_uncalibrated_count,
                "backend_ids": sorted(self._confidence_backend_ids),
                "calibration_ids": sorted(self._confidence_calibration_ids),
                "state_counts": {
                    str(key): value
                    for key, value in sorted(self._confidence_states.items())
                },
                "reason_or": self._confidence_reason_or,
                "score": {
                    "min": min(self._confidence_values)
                    if self._confidence_values else None,
                    "max": max(self._confidence_values)
                    if self._confidence_values else None,
                    "distinct_float32_count": len(set(self._confidence_values)),
                },
                "evaluation_stamp_violations": (
                    self._confidence_evaluation_violations
                ),
                "source_stamp_violations": self._confidence_source_violations,
                "last_confidence_age_s": self._last_confidence_age_s,
                "signal_coverage": self._confidence_signal_summary(),
            },
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self._confidence_dataset_path is not None:
            self._write_confidence_dataset()
        self.get_logger().info(json.dumps(report, sort_keys=True))
        self.exit_code = 0 if not failures else 1

    @staticmethod
    def _offline_pose(sample: PoseSample) -> OfflinePoseSample:
        return OfflinePoseSample(
            stamp_ns=round(sample.stamp_s * 1_000_000_000),
            x=sample.x,
            y=sample.y,
            z=sample.z,
            yaw=sample.yaw,
            roll=sample.roll,
            pitch=sample.pitch,
        )

    def _write_confidence_dataset(self) -> None:
        """Write an offline-only labelled capture; GT never enters extractor."""

        assert self._confidence_dataset_path is not None
        lifecycle_mask = (
            SlamConfidence.REASON_INITIALIZING
            | SlamConfidence.REASON_LOW_CONFIDENCE
            | SlamConfidence.REASON_RECOVERY_PENDING
            | SlamConfidence.REASON_UNCALIBRATED
        )
        abrupt_mask = (
            SlamConfidence.REASON_SOURCE_STALE
            | SlamConfidence.REASON_ODOMETRY_STALE
            | SlamConfidence.REASON_LIDAR_STALE
            | SlamConfidence.REASON_IMU_STALE
            | SlamConfidence.REASON_ESTIMATOR_RESET
            | SlamConfidence.REASON_TIMESTAMP_INVALID
            | SlamConfidence.REASON_NUMERIC_INVALID
            | SlamConfidence.REASON_BACKEND_ERROR
            | SlamConfidence.REASON_CLOCK_RESET
        )
        evaluations = [
            EvaluationRecord(
                evaluation_stamp_ns=int(row["evaluation_stamp_ns"]),
                source_stamp_ns=(
                    int(row["source_stamp_ns"])
                    if row["source_stamp_ns"] is not None
                    else None
                ),
                hard_failure=bool(
                    int(row["degradation_reasons"]) & abrupt_mask
                ),
            )
            for row in self._confidence_samples
        ]
        labels = generate_offline_labels(
            evaluations=evaluations,
            estimates=[self._offline_pose(sample) for sample in self._estimate],
            ground_truth=[self._offline_pose(sample) for sample in self._truth],
            config=OfflineLabelConfig(
                prediction_horizon_ns=500_000_000,
                odometry_outage_ns=300_000_000,
                pose_error_sustain_ns=200_000_000,
                translation_error_m=0.10,
                yaw_error_deg=1.0,
                translation_jump_residual_m=0.20,
                yaw_jump_residual_deg=2.0,
            ),
            ground_truth_sensor_offset_xyz=(
                self._ground_truth_sensor_offset_xyz
            ),
        )
        motion_by_stamp_ns = {}
        previous = None
        for sample in self._estimate:
            stamp_ns = round(sample.stamp_s * 1_000_000_000)
            linear_speed = yaw_rate = 0.0
            if previous is not None:
                dt = sample.stamp_s - previous.stamp_s
                if dt > 1.0e-6:
                    linear_speed = math.dist(
                        (sample.x, sample.y, sample.z),
                        (previous.x, previous.y, previous.z),
                    ) / dt
                    yaw_delta = math.atan2(
                        math.sin(sample.yaw - previous.yaw),
                        math.cos(sample.yaw - previous.yaw),
                    )
                    yaw_rate = abs(math.degrees(yaw_delta) / dt)
            motion_by_stamp_ns[stamp_ns] = {
                "linear_speed_mps": linear_speed,
                "yaw_rate_deg_s": yaw_rate,
            }
            previous = sample
        rows = []
        for raw, label in zip(self._confidence_samples, labels, strict=True):
            source_ns = raw["source_stamp_ns"]
            support = (
                self._support_by_stamp_ns.get(int(source_ns), {})
                if source_ns is not None
                else {}
            )
            incremental = (
                self._incremental_by_stamp_ns.get(int(source_ns), {})
                if source_ns is not None
                else {}
            )
            native_pose = (
                self._native_pose_by_stamp_ns.get(int(source_ns))
                if source_ns is not None
                else None
            )
            disagreement = {}
            if native_pose is not None and incremental:
                yaw_delta = math.atan2(
                    math.sin(native_pose.yaw - float(incremental["yaw"])),
                    math.cos(native_pose.yaw - float(incremental["yaw"])),
                )
                disagreement = {
                    "mapping_incremental_translation_m": math.dist(
                        (native_pose.x, native_pose.y, native_pose.z),
                        (
                            float(incremental["x"]),
                            float(incremental["y"]),
                            float(incremental["z"]),
                        ),
                    ),
                    "mapping_incremental_yaw_delta_deg": abs(
                        math.degrees(yaw_delta)
                    ),
                }
            operational_reasons = int(raw["degradation_reasons"]) & ~lifecycle_mask
            rows.append(
                {
                    "evaluation_stamp_ns": label.evaluation_stamp_ns,
                    "source_stamp_ns": label.source_stamp_ns,
                    "source_age_s": (
                        label.source_age_ns * 1.0e-9
                        if label.source_age_ns is not None
                        else None
                    ),
                    "features": {
                        **support,
                        **incremental,
                        **disagreement,
                        **(
                            motion_by_stamp_ns.get(int(source_ns), {})
                            if source_ns is not None
                            else {}
                        ),
                    },
                    "operational_reasons": operational_reasons,
                    "abrupt_hard_fault": bool(operational_reasons & abrupt_mask),
                    "translation_error_m": label.translation_error_m,
                    "yaw_error_deg": label.yaw_error_deg,
                    "usable_next_0_5s": label.usable_next_horizon,
                    "failure_reasons": list(label.failure_reasons),
                }
            )
        payload = {
            "schema_version": 2,
            "kind": "slam_confidence_offline_capture",
            "backend_id": self._backend_kind,
            "capture_group": self._capture_group,
            "deskew_mode": self._deskew_mode,
            "forecast_anchor": "evaluation_stamp",
            "prediction_horizon_s": 0.5,
            "odometry_outage_semantics": "source_advance_gap",
            "runtime_ground_truth_subscription": False,
            "offline_label_ground_truth_topic": "/odom",
            "rows": rows,
        }
        self._confidence_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self._confidence_dataset_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _effective_support_summary(self) -> dict:
        counts = [
            int(sample["effective_features"])
            for sample in self._effective_support
        ]
        return {
            "semantics": (
                "LIO-SAM extracted corner+surface features"
                if self._backend_kind == "liosam"
                else "FAST-LIO2 selected point-to-plane measurement features"
            ),
            "sample_count": len(counts),
            "min": min(counts) if counts else None,
            "median": float(np.median(counts)) if counts else None,
            "p10": float(np.percentile(counts, 10.0)) if counts else None,
            "max": max(counts) if counts else None,
            "zero_count": sum(count == 0 for count in counts),
            "samples": self._effective_support,
        }

    def _confidence_signal_summary(self) -> dict:
        required = [
            "native_mapping_odometry",
            "canonical_odometry",
            "incremental_mapping_odometry",
            "feature_cloud_info",
        ]
        if self._deskew_mode == "project":
            required.append("motion_deskew")
        matched = set.intersection(
            *(self._confidence_signal_stamps[name] for name in required)
        )
        return {
            "counts": {
                name: len(self._confidence_signal_stamps[name])
                for name in required
            },
            "contracts": {
                name: sorted(self._confidence_signal_contracts[name])
                for name in required
            },
            "required_streams": required,
            "exact_required_match_count": len(matched),
            "pairwise_with_native": {
                name: len(
                    self._confidence_signal_stamps[
                        "native_mapping_odometry"
                    ]
                    & self._confidence_signal_stamps[name]
                )
                for name in required[1:]
            },
        }


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
