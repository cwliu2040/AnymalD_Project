"""ROS 2/DDS hard-fault qualification for both SLAM confidence extractors."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import rclpy
from builtin_interfaces.msg import Time
from lio_sam.msg import CloudInfo
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2

from anymal_locomotion_interfaces.msg import SlamConfidence
from anymal_locomotion_ros2.fastlio_confidence_node import (
    FastlioConfidenceExtractor,
)
from anymal_locomotion_ros2.liosam_confidence_node import (
    LiosamConfidenceExtractor,
)
from anymal_locomotion_ros2.slam_confidence_observation_core import (
    ppo_confidence_observation,
)


_RELIABLE = QoSProfile(
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
_OUTPUT_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
_SOURCE_PERIOD_S = 0.10
_IMU_PERIOD_S = 0.02
_RECEIPT_TIMEOUT_S = 0.15


@dataclass(frozen=True)
class DdsFaultResult:
    backend_id: str
    name: str
    fault_kind: str
    expected_reason: str
    deadline_s: float
    detection_latency_s: float | None
    confidence_at_detection: float | None
    tracking_valid_at_detection: bool | None
    detected_reasons: int
    recovery_latency_s: float | None
    watchdog_confidence: float | None
    watchdog_tracking_valid: float | None
    passed: bool
    failures: tuple[str, ...]


def _stamp_from_nanoseconds(value_ns: int) -> Time:
    seconds, nanoseconds = divmod(value_ns, 1_000_000_000)
    return Time(sec=seconds, nanosec=nanoseconds)


def _point_cloud(stamp: Time, count: int) -> PointCloud2:
    message = PointCloud2()
    message.header.stamp = stamp
    message.header.frame_id = "lidar_link"
    message.height = 1
    message.width = count
    message.point_step = 4
    message.row_step = 4 * count
    message.data = bytes(message.row_step)
    message.is_dense = True
    return message


def _odometry(
    stamp: Time,
    *,
    frame_id: str,
    child_frame_id: str,
    x: float,
) -> Odometry:
    message = Odometry()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.child_frame_id = child_frame_id
    message.pose.pose.position.x = x
    message.pose.pose.orientation.w = 1.0
    return message


class _DdsInjector(Node):
    def __init__(self, backend_id: str, prefix: str) -> None:
        super().__init__(f"{backend_id}_confidence_dds_injector")
        self.backend_id = backend_id
        self.prefix = prefix
        self.publishers_by_name: dict[str, object] = {}
        self._create_all_publishers()

    def _create_all_publishers(self) -> None:
        if self.backend_id == "fastlio2":
            specifications = {
                "native": (Odometry, "native_odom"),
                "canonical": (Odometry, "canonical_odom"),
                "diagnostic": (PointCloud2, "effective_points"),
                "lidar": (PointCloud2, "lidar"),
                "imu": (Imu, "imu"),
            }
        else:
            specifications = {
                "native": (Odometry, "native_odom"),
                "canonical": (Odometry, "canonical_odom"),
                "incremental": (Odometry, "incremental_odom"),
                "diagnostic": (CloudInfo, "cloud_info"),
                "lidar": (PointCloud2, "lidar"),
                "imu": (Imu, "imu"),
            }
        for name, (message_type, suffix) in specifications.items():
            if name not in self.publishers_by_name:
                self.publishers_by_name[name] = self.create_publisher(
                    message_type, f"{self.prefix}/{suffix}", _RELIABLE
                )

    def kill_publishers(self, *names: str) -> None:
        for name in names:
            publisher = self.publishers_by_name.pop(name, None)
            if publisher is not None:
                self.destroy_publisher(publisher)

    def restore_publishers(self) -> None:
        self._create_all_publishers()

    def publish_imu(self, stamp_ns: int) -> None:
        publisher = self.publishers_by_name.get("imu")
        if publisher is None:
            return
        message = Imu()
        message.header.stamp = _stamp_from_nanoseconds(stamp_ns)
        message.header.frame_id = "imu_link"
        message.orientation.w = 1.0
        message.linear_acceleration.z = 9.81
        publisher.publish(message)

    def publish_source(self, stamp_ns: int, *, zero_support: bool = False) -> None:
        stamp = _stamp_from_nanoseconds(stamp_ns)
        x = float(stamp_ns) * 1.0e-10
        native = self.publishers_by_name.get("native")
        canonical = self.publishers_by_name.get("canonical")
        lidar = self.publishers_by_name.get("lidar")
        diagnostic = self.publishers_by_name.get("diagnostic")
        if self.backend_id == "fastlio2":
            if native is not None:
                native.publish(_odometry(
                    stamp, frame_id="camera_init", child_frame_id="body", x=x
                ))
            if canonical is not None:
                canonical.publish(_odometry(
                    stamp, frame_id="map", child_frame_id="base_link", x=x
                ))
            if diagnostic is not None:
                diagnostic.publish(_point_cloud(stamp, 0 if zero_support else 1000))
        else:
            if native is not None:
                native.publish(_odometry(
                    stamp, frame_id="odom", child_frame_id="odom_mapping", x=x
                ))
            if canonical is not None:
                canonical.publish(_odometry(
                    stamp, frame_id="map", child_frame_id="base_link", x=x
                ))
            incremental = self.publishers_by_name.get("incremental")
            if incremental is not None:
                incremental.publish(_odometry(
                    stamp, frame_id="odom", child_frame_id="odom_mapping", x=x
                ))
            if diagnostic is not None:
                info = CloudInfo()
                info.header.stamp = stamp
                info.header.frame_id = "lidar_link"
                info.imu_available = 1
                info.odom_available = 1
                info.cloud_corner = _point_cloud(
                    stamp, 0 if zero_support else 100
                )
                info.cloud_surface = _point_cloud(
                    stamp, 0 if zero_support else 1000
                )
                diagnostic.publish(info)
        if lidar is not None:
            lidar.publish(_point_cloud(stamp, 1000))


class _ConfidenceMonitor(Node):
    def __init__(self, backend_id: str, topic: str) -> None:
        super().__init__(f"{backend_id}_confidence_dds_monitor")
        self.samples: list[tuple[int, SlamConfidence]] = []
        self.create_subscription(
            SlamConfidence, topic, self._on_confidence, _OUTPUT_QOS
        )

    def _on_confidence(self, message: SlamConfidence) -> None:
        self.samples.append((time.monotonic_ns(), message))


class _BackendHarness:
    def __init__(self, backend_id: str, project_root: Path) -> None:
        self.backend_id = backend_id
        self.prefix = f"/slam_confidence_dds_validation/{backend_id}"
        self.injector = _DdsInjector(backend_id, self.prefix)
        self.monitor = _ConfidenceMonitor(
            backend_id, f"{self.prefix}/slam_confidence"
        )
        artifact_name = (
            "slam_confidence_fastlio2_native_v2.json"
            if backend_id == "fastlio2"
            else "slam_confidence_liosam_native_v3.json"
        )
        artifact = project_root / (
            "deployment/ros2_ws/src/anymal_locomotion_ros2/config/" + artifact_name
        )
        common = [
            Parameter("native_odometry_topic", value=f"{self.prefix}/native_odom"),
            Parameter("canonical_odometry_topic", value=f"{self.prefix}/canonical_odom"),
            Parameter("lidar_input_topic", value=f"{self.prefix}/lidar"),
            Parameter("imu_topic", value=f"{self.prefix}/imu"),
            Parameter("confidence_topic", value=f"{self.prefix}/slam_confidence"),
            Parameter("tracking_valid_topic", value=f"{self.prefix}/tracking_valid"),
            Parameter("calibration_artifact_path", value=str(artifact)),
        ]
        if backend_id == "fastlio2":
            overrides = common + [Parameter(
                "effective_points_topic", value=f"{self.prefix}/effective_points"
            )]
            self.extractor = FastlioConfidenceExtractor(
                parameter_overrides=overrides
            )
            self.timeouts = {"lidar": 0.30, "odom": 0.30, "diagnostic": 0.05}
        else:
            overrides = common + [
                Parameter(
                    "incremental_odometry_topic",
                    value=f"{self.prefix}/incremental_odom",
                ),
                Parameter(
                    "feature_cloud_info_topic", value=f"{self.prefix}/cloud_info"
                ),
                Parameter("motion_deskew_required", value=False),
            ]
            self.extractor = LiosamConfidenceExtractor(
                parameter_overrides=overrides
            )
            self.timeouts = {"lidar": 0.50, "odom": 0.50, "diagnostic": 0.20}
        self.executor = SingleThreadedExecutor()
        for node in (self.injector, self.monitor, self.extractor):
            self.executor.add_node(node)
        self._next_source_ns = 0
        self._next_imu_ns = 0

    def drive(
        self,
        duration_s: float,
        *,
        zero_support: bool = False,
    ) -> None:
        end_ns = time.monotonic_ns() + int(duration_s * 1.0e9)
        while time.monotonic_ns() < end_ns:
            steady_ns = time.monotonic_ns()
            ros_ns = int(self.injector.get_clock().now().nanoseconds)
            if steady_ns >= self._next_imu_ns:
                self.injector.publish_imu(ros_ns)
                self._next_imu_ns = steady_ns + int(_IMU_PERIOD_S * 1.0e9)
            if steady_ns >= self._next_source_ns:
                self.injector.publish_source(ros_ns, zero_support=zero_support)
                self._next_source_ns = steady_ns + int(_SOURCE_PERIOD_S * 1.0e9)
            self.executor.spin_once(timeout_sec=0.005)

    def wait_for(
        self,
        predicate: Callable[[SlamConfidence], bool],
        timeout_s: float,
        *,
        zero_support: bool = False,
        after_index: int = 0,
    ) -> tuple[int, SlamConfidence] | None:
        end_ns = time.monotonic_ns() + int(timeout_s * 1.0e9)
        checked = after_index
        while time.monotonic_ns() < end_ns:
            self.drive(0.025, zero_support=zero_support)
            while checked < len(self.monitor.samples):
                sample = self.monitor.samples[checked]
                checked += 1
                if predicate(sample[1]):
                    return sample
        return None

    def seed_tracking(self) -> SlamConfidence:
        sample = self.wait_for(
            lambda message: message.slam_tracking_valid,
            12.0,
            after_index=len(self.monitor.samples),
        )
        if sample is None:
            raise RuntimeError(
                f"{self.backend_id} did not reach TRACKING on healthy DDS inputs"
            )
        return sample[1]

    def recover(self, started_ns: int) -> float | None:
        self.injector.restore_publishers()
        sample = self.wait_for(
            lambda message: message.slam_tracking_valid,
            12.0,
            after_index=len(self.monitor.samples),
        )
        if sample is None:
            return None
        return (sample[0] - started_ns) * 1.0e-9

    def close(self) -> None:
        for node in (self.extractor, self.monitor, self.injector):
            try:
                self.executor.remove_node(node)
                node.destroy_node()
            except Exception:
                pass
        self.executor.shutdown()


def _fault_case(
    harness: _BackendHarness,
    *,
    name: str,
    fault_kind: str,
    expected_reason: int,
    expected_reason_name: str,
    deadline_s: float,
    publisher_names: tuple[str, ...],
    zero_support: bool = False,
) -> DdsFaultResult:
    harness.seed_tracking()
    failures: list[str] = []
    baseline_index = len(harness.monitor.samples)
    if publisher_names:
        harness.injector.kill_publishers(*publisher_names)
    if zero_support:
        # Payload-fault latency starts at the first faulty source publication,
        # not at an arbitrary point in the 10 Hz source schedule.
        harness._next_source_ns = 0
    injected_ns = time.monotonic_ns()
    detected = harness.wait_for(
        lambda message: (
            not message.slam_tracking_valid
            and bool(message.degradation_reasons & expected_reason)
        ),
        deadline_s + 0.20,
        zero_support=zero_support,
        after_index=baseline_index,
    )
    if detected is None:
        failures.append("expected fail-closed reason was not observed")
        latency_s = None
        message = None
    else:
        detected_ns, message = detected
        latency_s = (detected_ns - injected_ns) * 1.0e-9
        if latency_s > deadline_s + 0.10:
            failures.append("fault detection exceeded timeout plus one 20 Hz tick")
        if message.slam_confidence < 0.45:
            failures.append("hard fault did not preserve the prior usable score")
    recovery_started_ns = time.monotonic_ns()
    recovery_latency_s = harness.recover(recovery_started_ns)
    if recovery_latency_s is None:
        failures.append("healthy DDS publishers did not recover TRACKING")
    return DdsFaultResult(
        backend_id=harness.backend_id,
        name=name,
        fault_kind=fault_kind,
        expected_reason=expected_reason_name,
        deadline_s=deadline_s,
        detection_latency_s=latency_s,
        confidence_at_detection=(
            float(message.slam_confidence) if message is not None else None
        ),
        tracking_valid_at_detection=(
            bool(message.slam_tracking_valid) if message is not None else None
        ),
        detected_reasons=(
            int(message.degradation_reasons) if message is not None else 0
        ),
        recovery_latency_s=recovery_latency_s,
        watchdog_confidence=None,
        watchdog_tracking_valid=None,
        passed=not failures,
        failures=tuple(failures),
    )


def _extractor_death_case(harness: _BackendHarness) -> DdsFaultResult:
    baseline = harness.seed_tracking()
    failures: list[str] = []
    harness.executor.remove_node(harness.extractor)
    harness.extractor.destroy_node()
    killed_ns = time.monotonic_ns()
    observation = None
    detection_latency_s = None
    receipt_ns = None
    deadline_ns = killed_ns + int((_RECEIPT_TIMEOUT_S + 0.10) * 1.0e9)
    while time.monotonic_ns() < deadline_ns:
        harness.drive(0.01)
        if not harness.monitor.samples:
            continue
        receipt_ns, last = harness.monitor.samples[-1]
        now_ns = time.monotonic_ns()
        age_s = max(0.0, (now_ns - receipt_ns) * 1.0e-9)
        confidence_age_s = (
            float(last.confidence_age.sec)
            + float(last.confidence_age.nanosec) * 1.0e-9
        )
        observation = ppo_confidence_observation(
            confidence=float(last.slam_confidence),
            tracking_valid=bool(last.slam_tracking_valid),
            source_stamp_valid=bool(last.source_stamp_valid),
            confidence_age_s=confidence_age_s,
            receipt_age_s=age_s,
            receipt_timeout_s=_RECEIPT_TIMEOUT_S,
        )
        if not observation.receipt_watchdog_valid:
            detection_latency_s = age_s
            break
    if not harness.monitor.samples:
        failures.append("no authoritative confidence sample was received")
    else:
        if observation is None or observation.receipt_watchdog_valid:
            failures.append("steady receipt watchdog remained valid after publisher death")
        if detection_latency_s is None:
            failures.append("steady receipt watchdog did not trip by its deadline")
        elif detection_latency_s > _RECEIPT_TIMEOUT_S + 0.03:
            failures.append("steady receipt watchdog exceeded scheduling tolerance")
        if observation is not None and observation.slam_confidence != 0.0:
            failures.append("watchdog did not force confidence to zero")
        if observation is not None and observation.slam_tracking_valid != 0.0:
            failures.append("watchdog did not force tracking-valid to zero")
    return DdsFaultResult(
        backend_id=harness.backend_id,
        name="extractor_publisher_death",
        fault_kind="dds_node_death",
        expected_reason="CONSUMER_RECEIPT_WATCHDOG",
        deadline_s=_RECEIPT_TIMEOUT_S,
        detection_latency_s=detection_latency_s,
        confidence_at_detection=float(baseline.slam_confidence),
        tracking_valid_at_detection=False,
        detected_reasons=0,
        recovery_latency_s=None,
        watchdog_confidence=(
            observation.slam_confidence if observation is not None else None
        ),
        watchdog_tracking_valid=(
            observation.slam_tracking_valid if observation is not None else None
        ),
        passed=not failures,
        failures=tuple(failures),
    )


def validate_backend(backend_id: str, project_root: Path) -> list[DdsFaultResult]:
    harness = _BackendHarness(backend_id, project_root)
    try:
        cases = [
            _fault_case(
                harness,
                name="imu_topic_freeze",
                fault_kind="topic_freeze",
                expected_reason=SlamConfidence.REASON_IMU_STALE,
                expected_reason_name="IMU_STALE",
                deadline_s=0.05,
                publisher_names=("imu",),
            ),
            _fault_case(
                harness,
                name="lidar_topic_freeze",
                fault_kind="topic_freeze",
                expected_reason=SlamConfidence.REASON_LIDAR_STALE,
                expected_reason_name="LIDAR_STALE",
                deadline_s=harness.timeouts["lidar"],
                publisher_names=("lidar",),
            ),
            _fault_case(
                harness,
                name="odometry_publishers_death",
                fault_kind="dds_publisher_death",
                expected_reason=SlamConfidence.REASON_ODOMETRY_STALE,
                expected_reason_name="ODOMETRY_STALE",
                deadline_s=harness.timeouts["odom"],
                publisher_names=("native", "canonical"),
            ),
            _fault_case(
                harness,
                name="diagnostic_publisher_death",
                fault_kind="dds_publisher_death",
                expected_reason=SlamConfidence.REASON_SIGNAL_MISSING,
                expected_reason_name="SIGNAL_MISSING",
                deadline_s=harness.timeouts["diagnostic"] + _SOURCE_PERIOD_S,
                publisher_names=("diagnostic",),
            ),
            _fault_case(
                harness,
                name="zero_support_topic_payload",
                fault_kind="topic_payload_fault",
                expected_reason=SlamConfidence.REASON_INSUFFICIENT_SUPPORT,
                expected_reason_name="INSUFFICIENT_SUPPORT",
                deadline_s=0.05,
                publisher_names=(),
                zero_support=True,
            ),
        ]
        cases.append(_extractor_death_case(harness))
        return cases
    finally:
        harness.close()


def validate_dds_faults(project_root: Path) -> dict:
    results: list[DdsFaultResult] = []
    for backend_id in ("fastlio2", "liosam"):
        results.extend(validate_backend(backend_id, project_root))
    return {
        "schema_version": 1,
        "kind": "slam_confidence_ros2_dds_fault_qualification",
        "transport": rclpy.get_rmw_implementation_identifier(),
        "receipt_watchdog_timeout_s": _RECEIPT_TIMEOUT_S,
        "case_count": len(results),
        "passed_case_count": sum(result.passed for result in results),
        "passed": all(result.passed for result in results),
        "results": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/slam_confidence/dds_fault_qualification.json"),
    )
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = project_root / output
    output = output.resolve()
    if not (project_root / "configs/slam_confidence_contract.yaml").is_file():
        parser.error(f"not an anymal_locomotion project root: {project_root}")
    if not output.is_relative_to(project_root):
        parser.error(f"output must remain inside project root: {output}")

    rclpy.init()
    try:
        report = validate_dds_faults(project_root)
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
