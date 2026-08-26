"""External ROS 2 node that runs the ANYmal-D locomotion policy at 50 Hz."""

from __future__ import annotations

import json
import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import UInt64

from anymal_locomotion_interfaces.msg import FootContactState, SlamConfidence

from anymal_locomotion_ros2.policy_core import (
    GaitModeGovernor,
    PolicyContract,
    PolicyRuntime,
    RobotState,
    canonical_joint_state,
    decelerate_command_toward_zero,
    projected_gravity_from_quaternion,
)
from anymal_locomotion_ros2.onnx_backend import OnnxBackend
from anymal_locomotion_ros2.proprioceptive_velocity_estimator_core import (
    EstimatorInputBundle,
    EstimatorInputSynchronizer,
    EstimatorRuntime,
    assemble_step,
    load_estimator_metadata,
    reorder_foot_contacts,
)
from anymal_locomotion_ros2.slam_confidence_observation_core import (
    confidence_identity_valid,
    ppo_confidence_observation,
)
from anymal_locomotion_ros2.torchscript_backend import TorchScriptBackend
from anymal_locomotion_ros2.touchdown_phase_tracker_core import (
    initial_touchdown_phase_tracker_state,
)
from anymal_locomotion_ros2.touchdown_residual_pipeline_core import (
    apply_touchdown_residual_pipeline,
)


def create_inference_backend(backend_name: str, policy_path: str) -> Any:
    """Create the selected external-process inference backend."""
    normalized_name = backend_name.strip().lower()
    if normalized_name == "onnx":
        return OnnxBackend(policy_path)
    if normalized_name == "torchscript":
        return TorchScriptBackend(policy_path)
    raise ValueError(
        "backend must be 'onnx' or 'torchscript', "
        f"received {backend_name!r}"
    )


class AnymalPolicyNode(Node):
    """Assemble ROS state, execute the policy, and publish joint targets."""

    def __init__(
        self,
        backend_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__("anymal_locomotion_policy")
        self.declare_parameter("backend", "onnx")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("metadata_path", "")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("odometry_topic", "/odom")
        self.declare_parameter("foot_contact_topic", "/foot_contacts")
        self.declare_parameter("enable_velocity_estimator", False)
        self.declare_parameter("velocity_estimator_metadata_path", "")
        self.declare_parameter(
            "velocity_estimator_output_topic",
            "/locomotion/estimated_odom",
        )
        self.declare_parameter("velocity_estimator_sync_tolerance_s", 0.025)
        self.declare_parameter("velocity_estimator_receipt_timeout_s", 0.10)
        self.declare_parameter("command_topic", "/cmd_vel")
        self.declare_parameter("slam_confidence_topic", "/slam_confidence")
        self.declare_parameter("slam_confidence_receipt_timeout_s", 0.15)
        self.declare_parameter("expected_slam_confidence_backend", "")
        self.declare_parameter("expected_slam_confidence_calibration_id", "")
        self.declare_parameter("enable_gait_mode_governor", False)
        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter(
            "episode_reset_topic", "/simulation/episode_reset"
        )
        self.declare_parameter(
            "episode_reset_ack_topic", "/simulation/episode_reset_ack"
        )
        self.declare_parameter("expected_imu_frame", "base_link")
        self.declare_parameter("expected_odometry_child_frame", "base_link")
        self.declare_parameter("state_timeout_s", 0.1)
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("watchdog_linear_deceleration", 0.0)
        self.declare_parameter("watchdog_angular_deceleration", 0.0)
        self.declare_parameter(
            "inference_trigger",
            "synchronized_state",
        )
        self.declare_parameter("state_sync_tolerance_s", 0.01)
        self.declare_parameter("max_abs_policy_action", 10.0)
        self.declare_parameter("diagnostics_path", "")
        self.declare_parameter("state_transplant_manifest_path", "")
        self.declare_parameter("enable_touchdown_residual_experiment", False)
        self.declare_parameter("touchdown_residual_arm", "zero")
        self.declare_parameter("touchdown_phase_artifact_path", "")
        self.declare_parameter("touchdown_phase_artifact_sha256", "")

        policy_path = str(self.get_parameter("policy_path").value)
        metadata_path = str(self.get_parameter("metadata_path").value)
        backend_name = str(self.get_parameter("backend").value)
        if not policy_path or not metadata_path:
            raise ValueError(
                "policy_path and metadata_path are required; use the project "
                "bringup launch or pass both ROS parameters explicitly"
            )
        if backend_factory is None:
            backend = create_inference_backend(backend_name, policy_path)
        else:
            backend = backend_factory(policy_path)
        self._contract = PolicyContract.from_metadata(metadata_path)
        enable_gait_mode_governor = bool(
            self.get_parameter("enable_gait_mode_governor").value
        )
        if enable_gait_mode_governor and self._contract.observation_dimension != 51:
            raise ValueError("gait-mode governor requires a 51-D policy")
        self._runtime = PolicyRuntime(
            self._contract,
            backend,
            max_abs_policy_action=float(self.get_parameter("max_abs_policy_action").value),
            gait_mode_governor=(
                GaitModeGovernor() if enable_gait_mode_governor else None
            ),
        )
        self._touchdown_experiment_enabled = bool(
            self.get_parameter("enable_touchdown_residual_experiment").value
        )
        self._touchdown_arm = str(
            self.get_parameter("touchdown_residual_arm").value
        ).strip()
        attenuation_by_arm = {
            "zero": 0.0,
            "touchdown_soft_low": 0.25,
            "touchdown_soft": 0.50,
        }
        if self._touchdown_arm not in attenuation_by_arm:
            raise ValueError("unsupported touchdown residual experiment arm")
        self._touchdown_attenuation = attenuation_by_arm[self._touchdown_arm]
        self._touchdown_phase_artifact: dict[str, Any] | None = None
        self._touchdown_phase_state = initial_touchdown_phase_tracker_state()
        self._touchdown_joint_position_history: list[np.ndarray] = []
        self._touchdown_joint_velocity_history: list[np.ndarray] = []
        self._touchdown_previous_action_history: list[np.ndarray] = []
        self._touchdown_history_length = 0
        if self._touchdown_experiment_enabled:
            if self._contract.observation_dimension != 48:
                raise ValueError("touchdown experiment requires frozen 48-D model1450")
            artifact_path = Path(str(
                self.get_parameter("touchdown_phase_artifact_path").value
            )).expanduser().resolve()
            expected_sha256 = str(
                self.get_parameter("touchdown_phase_artifact_sha256").value
            ).strip()
            if not artifact_path.is_file() or len(expected_sha256) != 64:
                raise ValueError("touchdown experiment requires hash-locked phase artifact")
            actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError("touchdown phase artifact SHA-256 mismatch")
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if artifact.get("frozen") is not True or artifact.get("passed") is not True:
                raise ValueError("touchdown experiment requires frozen PASS phase artifact")
            self._touchdown_phase_artifact = artifact
            self._touchdown_history_length = max(
                int(value) for value in artifact["history_offsets_samples"]
            ) + 1
        self._velocity_estimator_enabled = bool(
            self.get_parameter("enable_velocity_estimator").value
        )
        self._velocity_estimator: EstimatorRuntime | None = None
        self._velocity_estimator_sync_tolerance_s = float(
            self.get_parameter("velocity_estimator_sync_tolerance_s").value
        )
        self._velocity_estimator_receipt_timeout_s = float(
            self.get_parameter("velocity_estimator_receipt_timeout_s").value
        )
        self._velocity_estimator_uses_sim_time = bool(
            self.get_parameter("use_sim_time").value
        )
        self._velocity_estimator_synchronizer = EstimatorInputSynchronizer(
            self._velocity_estimator_sync_tolerance_s
        )
        self._velocity_estimator_publisher = None
        if self._velocity_estimator_enabled:
            estimator_metadata_path = str(
                self.get_parameter("velocity_estimator_metadata_path").value
            ).strip()
            if not estimator_metadata_path:
                raise ValueError(
                    "enabled velocity estimator requires metadata path"
                )
            if (
                self._velocity_estimator_sync_tolerance_s < 0.0
                or self._velocity_estimator_receipt_timeout_s <= 0.0
            ):
                raise ValueError(
                    "velocity estimator synchronization tolerance must be "
                    "non-negative and receipt timeout positive"
                )
            _, estimator_model_path = load_estimator_metadata(
                estimator_metadata_path
            )
            self._velocity_estimator = EstimatorRuntime(
                OnnxBackend(str(estimator_model_path))
            )
            self._velocity_estimator_publisher = self.create_publisher(
                Odometry,
                str(
                    self.get_parameter(
                        "velocity_estimator_output_topic"
                    ).value
                ),
                qos_profile_sensor_data,
            )
        self._uses_slam_confidence = self._contract.observation_dimension == 51
        self._expected_slam_confidence_backend = str(
            self.get_parameter("expected_slam_confidence_backend").value
        ).strip()
        self._expected_slam_confidence_calibration_id = str(
            self.get_parameter(
                "expected_slam_confidence_calibration_id"
            ).value
        ).strip()
        if self._uses_slam_confidence and not (
            self._expected_slam_confidence_backend
            and self._expected_slam_confidence_calibration_id
        ):
            raise ValueError(
                "51-D policy requires expected_slam_confidence_backend and "
                "expected_slam_confidence_calibration_id"
            )
        self._slam_confidence_receipt_timeout = float(
            self.get_parameter("slam_confidence_receipt_timeout_s").value
        )
        if self._slam_confidence_receipt_timeout <= 0.0:
            raise ValueError("slam_confidence_receipt_timeout_s must be positive")
        self._latest_slam_confidence: SlamConfidence | None = None
        self._slam_confidence_receipt_steady_ns: int | None = None
        transplant_manifest_path = str(
            self.get_parameter("state_transplant_manifest_path").value
        ).strip()
        self._transplant_expected_observation: np.ndarray | None = None
        self._transplant_observation_checked = False
        self._transplant_observation_parity: dict[str, float] | None = None
        self._transplant_action_replay = False
        self._transplant_initial_command_pending = False
        transplant_initial_command: np.ndarray | None = None
        if transplant_manifest_path:
            transplant_document = json.loads(
                Path(transplant_manifest_path)
                .expanduser()
                .resolve()
                .read_text(encoding="utf-8")
            )
            transplant_state = transplant_document.get("state")
            if (
                transplant_document.get("schema_version") != 1
                or not isinstance(transplant_state, dict)
            ):
                raise ValueError(
                    "state transplant manifest must contain schema_version 1 "
                    "and a state mapping"
                )
            action_replay_available = bool(
                transplant_document.get("deterministic_action_replay")
            )
            self._transplant_action_replay = action_replay_available
            self._transplant_initial_command_pending = (
                action_replay_available
            )
            self._runtime.seed_previous_action(
                (
                    transplant_document["deterministic_action_replay"][
                        "expected_observation"
                    ][36:48]
                    if action_replay_available
                    else transplant_state.get("bootstrap_action")
                )
            )
            expected_observation = np.asarray(
                (
                    transplant_document["deterministic_action_replay"][
                        "expected_observation"
                    ]
                    if action_replay_available
                    else transplant_state.get("expected_observation")
                ),
                dtype=np.float32,
            )
            if (
                expected_observation.shape
                != (self._contract.observation_dimension,)
                or not np.all(np.isfinite(expected_observation))
            ):
                raise ValueError(
                    "state transplant expected_observation must contain "
                    f"{self._contract.observation_dimension} finite values"
                )
            self._transplant_expected_observation = expected_observation
            transplant_initial_command = np.asarray(
                transplant_state.get("effective_command"),
                dtype=np.float32,
            )
            if (
                transplant_initial_command.shape != (3,)
                or not np.all(np.isfinite(transplant_initial_command))
            ):
                raise ValueError(
                    "state transplant effective_command must contain "
                    "3 finite values"
                )
        self._state_timeout = float(self.get_parameter("state_timeout_s").value)
        self._command_timeout = float(self.get_parameter("command_timeout_s").value)
        self._watchdog_linear_deceleration = float(
            self.get_parameter("watchdog_linear_deceleration").value
        )
        self._watchdog_angular_deceleration = float(
            self.get_parameter("watchdog_angular_deceleration").value
        )
        self._inference_trigger = str(
            self.get_parameter("inference_trigger").value
        ).strip().lower()
        self._state_sync_tolerance = float(
            self.get_parameter("state_sync_tolerance_s").value
        )
        self._expected_imu_frame = str(self.get_parameter("expected_imu_frame").value)
        self._expected_odom_child_frame = str(
            self.get_parameter("expected_odometry_child_frame").value
        )
        if self._state_timeout <= 0.0 or self._command_timeout <= 0.0:
            raise ValueError("State and command timeouts must be positive")
        watchdog_decelerations = (
            self._watchdog_linear_deceleration,
            self._watchdog_angular_deceleration,
        )
        if any(value < 0.0 for value in watchdog_decelerations):
            raise ValueError("Watchdog decelerations must be non-negative")
        if (watchdog_decelerations[0] == 0.0) != (
            watchdog_decelerations[1] == 0.0
        ):
            raise ValueError(
                "Watchdog linear and angular decelerations must both be zero "
                "or both be positive"
            )
        if self._inference_trigger not in (
            "synchronized_state",
            "timer",
            "estimator_joint_state",
        ):
            raise ValueError(
                "inference_trigger must be 'synchronized_state', 'timer', "
                "or 'estimator_joint_state'"
            )
        if (
            self._inference_trigger == "estimator_joint_state"
            and not self._velocity_estimator_enabled
        ):
            raise ValueError(
                "estimator_joint_state trigger requires the velocity estimator"
            )
        if self._state_sync_tolerance < 0.0:
            raise ValueError("state_sync_tolerance_s must be non-negative")

        self._joint_positions: np.ndarray | None = None
        self._joint_velocities: np.ndarray | None = None
        self._base_linear_velocity: np.ndarray | None = None
        self._base_angular_velocity: np.ndarray | None = None
        self._projected_gravity: np.ndarray | None = None
        self._command = np.zeros(3, dtype=np.float32)
        self._effective_command = np.zeros(3, dtype=np.float32)
        self._last_policy_time: float | None = None
        self._receipt_times: dict[str, float] = {}
        if transplant_initial_command is not None:
            self._command = transplant_initial_command
            # Sim time is zero until the bridge publishes its first /clock.
            # This keeps the restored command fresh for the first inference.
            self._receipt_times["command"] = 0.0
        self._state_stamps: dict[str, float] = {}
        self._last_inference_joint_stamp = float("-inf")
        self._warning_times: dict[str, float] = {}
        diagnostics_path = str(
            self.get_parameter("diagnostics_path").value
        ).strip()
        self._diagnostics_path = (
            Path(diagnostics_path).expanduser().resolve()
            if diagnostics_path
            else None
        )
        self._diagnostic_records: list[dict[str, Any]] = []
        self._reset_events: list[dict[str, Any]] = []
        self._last_reset_sequence = 0

        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter("imu_topic").value),
            self._on_imu,
            qos_profile_sensor_data,
        )
        if self._velocity_estimator_enabled:
            self.create_subscription(
                FootContactState,
                str(self.get_parameter("foot_contact_topic").value),
                self._on_foot_contacts,
                qos_profile_sensor_data,
            )
        else:
            self.create_subscription(
                Odometry,
                str(self.get_parameter("odometry_topic").value),
                self._on_odometry,
                qos_profile_sensor_data,
            )
        self.create_subscription(
            Twist,
            str(self.get_parameter("command_topic").value),
            self._on_command,
            10,
        )
        if self._uses_slam_confidence:
            confidence_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.create_subscription(
                SlamConfidence,
                str(self.get_parameter("slam_confidence_topic").value),
                self._on_slam_confidence,
                confidence_qos,
            )
        self.create_subscription(
            UInt64,
            str(self.get_parameter("episode_reset_topic").value),
            self._on_episode_reset,
            10,
        )
        self._episode_reset_ack_publisher = self.create_publisher(
            UInt64,
            str(self.get_parameter("episode_reset_ack_topic").value),
            10,
        )
        self._joint_command_publisher = self.create_publisher(
            JointState,
            str(self.get_parameter("joint_command_topic").value),
            10,
        )
        if self._inference_trigger == "timer":
            self.create_timer(
                self._contract.control_period_s,
                self._on_policy_tick,
            )
        self.get_logger().info(
            f"Loaded {backend_name} "
            f"{self._contract.observation_dimension}-D -> 12-D policy; "
            f"control period={self._contract.control_period_s:.3f} s; "
            f"trigger={self._inference_trigger}"
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _warn_throttled(self, key: str, message: str) -> None:
        now = self._now_seconds()
        if now - self._warning_times.get(key, float("-inf")) >= 1.0:
            self.get_logger().warning(message)
            self._warning_times[key] = now

    @staticmethod
    def _stamp_seconds(message) -> float:
        return (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )

    @staticmethod
    def _stamp_nanoseconds(message) -> int:
        return (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )

    def _state_updated(self, name: str, message) -> None:
        self._receipt_times[name] = self._now_seconds()
        self._state_stamps[name] = self._stamp_seconds(message)
        if self._inference_trigger == "synchronized_state":
            self._try_synchronized_policy_step()

    def _on_joint_state(self, message: JointState) -> None:
        try:
            positions, velocities = canonical_joint_state(
                message.name,
                message.position,
                message.velocity,
                self._contract,
            )
        except ValueError as error:
            self._warn_throttled("joint_state_invalid", str(error))
            return
        self._joint_positions = positions
        self._joint_velocities = velocities
        self._state_updated("joint_state", message)
        if self._velocity_estimator_enabled:
            try:
                bundles = self._velocity_estimator_synchronizer.push_joint(
                    self._stamp_nanoseconds(message),
                    (message, positions, velocities),
                )
            except ValueError as error:
                self._invalidate_velocity_estimator(str(error))
            else:
                self._consume_velocity_estimator_bundles(bundles)
        if (
            self._inference_trigger == "estimator_joint_state"
            and not self._velocity_estimator_enabled
            and self._base_linear_velocity is not None
        ):
            self._run_policy(output_stamp_s=self._stamp_seconds(message))

    def _on_imu(self, message: Imu) -> None:
        if message.header.frame_id != self._expected_imu_frame:
            if self._velocity_estimator_enabled:
                self._velocity_estimator_synchronizer.reset()
            self._warn_throttled(
                "imu_frame",
                f"IMU frame_id must be '{self._expected_imu_frame}', "
                f"received '{message.header.frame_id}'",
            )
            return
        try:
            angular_velocity = np.asarray(
                [
                    message.angular_velocity.x,
                    message.angular_velocity.y,
                    message.angular_velocity.z,
                ],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(angular_velocity)):
                raise ValueError("IMU angular velocity contains NaN or Inf")
            linear_acceleration = np.asarray(
                [
                    message.linear_acceleration.x,
                    message.linear_acceleration.y,
                    message.linear_acceleration.z,
                ],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(linear_acceleration)):
                raise ValueError("IMU linear acceleration contains NaN or Inf")
            gravity = projected_gravity_from_quaternion(
                message.orientation.x,
                message.orientation.y,
                message.orientation.z,
                message.orientation.w,
            )
        except ValueError as error:
            if self._velocity_estimator_enabled:
                self._velocity_estimator_synchronizer.reset()
            self._warn_throttled("imu_invalid", str(error))
            return
        self._base_angular_velocity = angular_velocity
        self._projected_gravity = gravity
        self._state_updated("imu", message)
        if self._velocity_estimator_enabled:
            try:
                bundles = self._velocity_estimator_synchronizer.push_imu(
                    self._stamp_nanoseconds(message),
                    (
                        time.monotonic_ns(), angular_velocity,
                        linear_acceleration, gravity,
                    ),
                )
            except ValueError as error:
                self._invalidate_velocity_estimator(str(error))
            else:
                self._consume_velocity_estimator_bundles(bundles)

    def _on_foot_contacts(self, message: FootContactState) -> None:
        try:
            contacts = reorder_foot_contacts(
                message.foot_names,
                message.contact_probabilities,
            )
        except ValueError as error:
            self._velocity_estimator_synchronizer.reset()
            self._invalidate_velocity_estimator(str(error))
            return
        try:
            bundles = self._velocity_estimator_synchronizer.push_contacts(
                self._stamp_nanoseconds(message),
                (time.monotonic_ns(), contacts),
            )
        except ValueError as error:
            self._invalidate_velocity_estimator(str(error))
        else:
            self._consume_velocity_estimator_bundles(bundles)

    def _invalidate_velocity_estimator(self, reason: str) -> None:
        if self._velocity_estimator is not None:
            self._velocity_estimator.reset()
        self._base_linear_velocity = None
        self._receipt_times.pop("odometry", None)
        self._state_stamps.pop("odometry", None)
        self._warn_throttled("velocity_estimator_invalid", reason)

    def _consume_velocity_estimator_bundles(
        self, bundles: list[EstimatorInputBundle]
    ) -> None:
        for bundle in bundles:
            estimate_ready = self._update_velocity_estimate(bundle)
            if (
                estimate_ready
                and self._inference_trigger == "estimator_joint_state"
            ):
                self._run_policy(output_stamp_s=bundle.joint_stamp_ns * 1.0e-9)

    def _update_velocity_estimate(
        self, bundle: EstimatorInputBundle
    ) -> bool:
        if self._velocity_estimator is None:
            self._invalidate_velocity_estimator(
                "Velocity estimator inputs are not ready"
            )
            return False
        if not bundle.synchronized:
            self._invalidate_velocity_estimator(
                "Velocity estimator input timestamps are not synchronized"
            )
            return False
        message, joint_positions, joint_velocities = bundle.joint
        imu_receipt_ns, angular_velocity, linear_acceleration, gravity = bundle.imu
        contact_receipt_ns, contacts = bundle.contacts
        joint_stamp = bundle.joint_stamp_ns * 1.0e-9
        now_ns = time.monotonic_ns()
        if (
            not self._velocity_estimator_uses_sim_time
            and max(
                now_ns - imu_receipt_ns,
                now_ns - contact_receipt_ns,
            )
            * 1.0e-9
            > self._velocity_estimator_receipt_timeout_s
        ):
            self._invalidate_velocity_estimator(
                "Velocity estimator input receipt watchdog expired"
            )
            return False
        self._base_angular_velocity = angular_velocity
        self._projected_gravity = gravity
        self._joint_positions = joint_positions
        self._joint_velocities = joint_velocities
        try:
            sample = assemble_step(
                self._base_angular_velocity,
                linear_acceleration,
                self._projected_gravity,
                joint_positions
                - np.asarray(
                    self._contract.default_joint_positions,
                    dtype=np.float32,
                ),
                joint_velocities,
                contacts,
            )
            estimate = self._velocity_estimator.step(joint_stamp, sample)
        except ValueError as error:
            self._invalidate_velocity_estimator(str(error))
            return False
        if estimate is None:
            self._base_linear_velocity = None
            self._receipt_times.pop("odometry", None)
            self._state_stamps.pop("odometry", None)
            return False
        self._base_linear_velocity = estimate
        self._receipt_times["odometry"] = self._now_seconds()
        self._state_stamps["odometry"] = joint_stamp
        self._state_stamps["joint_state"] = joint_stamp
        self._state_stamps["imu"] = bundle.imu_stamp_ns * 1.0e-9
        if self._velocity_estimator_publisher is not None:
            output = Odometry()
            output.header = message.header
            output.header.frame_id = "odom"
            output.child_frame_id = self._expected_odom_child_frame
            output.twist.twist.linear.x = float(estimate[0])
            output.twist.twist.linear.y = float(estimate[1])
            output.twist.twist.linear.z = float(estimate[2])
            output.twist.twist.angular.x = float(self._base_angular_velocity[0])
            output.twist.twist.angular.y = float(self._base_angular_velocity[1])
            output.twist.twist.angular.z = float(self._base_angular_velocity[2])
            self._velocity_estimator_publisher.publish(output)
        return True

    def _on_odometry(self, message: Odometry) -> None:
        if message.child_frame_id != self._expected_odom_child_frame:
            self._warn_throttled(
                "odometry_frame",
                f"Odometry child_frame_id must be '{self._expected_odom_child_frame}', "
                f"received '{message.child_frame_id}'",
            )
            return
        velocity = np.asarray(
            [
                message.twist.twist.linear.x,
                message.twist.twist.linear.y,
                message.twist.twist.linear.z,
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(velocity)):
            self._warn_throttled(
                "odometry_invalid",
                "Odometry linear velocity contains NaN or Inf",
            )
            return
        self._base_linear_velocity = velocity
        self._state_updated("odometry", message)

    def _on_command(self, message: Twist) -> None:
        command = np.asarray(
            [message.linear.x, message.linear.y, message.angular.z],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(command)):
            self._warn_throttled("command_invalid", "Velocity command contains NaN or Inf")
            return
        self._command = command
        self._receipt_times["command"] = self._now_seconds()

    def _on_slam_confidence(self, message: SlamConfidence) -> None:
        self._latest_slam_confidence = message
        self._slam_confidence_receipt_steady_ns = time.monotonic_ns()

    def _slam_confidence_observation(self) -> np.ndarray | None:
        if not self._uses_slam_confidence:
            return None
        message = self._latest_slam_confidence
        receipt_ns = self._slam_confidence_receipt_steady_ns
        if message is None or receipt_ns is None:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        receipt_age_s = max(0.0, (time.monotonic_ns() - receipt_ns) * 1.0e-9)
        confidence_age_s = (
            float(message.confidence_age.sec)
            + float(message.confidence_age.nanosec) * 1.0e-9
        )
        identity_valid = confidence_identity_valid(
            schema_version=int(message.schema_version),
            expected_schema_version=SlamConfidence.SCHEMA_VERSION,
            backend_id=message.backend_id,
            calibration_id=message.calibration_id,
            expected_backend_id=self._expected_slam_confidence_backend,
            expected_calibration_id=(
                self._expected_slam_confidence_calibration_id
            ),
        )
        try:
            observation = ppo_confidence_observation(
                confidence=(
                    float(message.slam_confidence) if identity_valid else 0.0
                ),
                tracking_valid=(
                    bool(message.slam_tracking_valid) and identity_valid
                ),
                source_stamp_valid=(
                    bool(message.source_stamp_valid) and identity_valid
                ),
                confidence_age_s=(
                    confidence_age_s if identity_valid else 0.5
                ),
                receipt_age_s=receipt_age_s,
                receipt_timeout_s=self._slam_confidence_receipt_timeout,
            )
        except ValueError:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return np.asarray(
            [
                observation.slam_confidence,
                observation.slam_tracking_valid,
                observation.confidence_age_normalized,
            ],
            dtype=np.float32,
        )

    def _on_episode_reset(self, message: UInt64) -> None:
        sequence = int(message.data)
        if sequence <= 0:
            self._warn_throttled(
                "episode_reset_invalid",
                f"Ignoring non-positive episode reset sequence {sequence}",
            )
            return
        if sequence > self._last_reset_sequence:
            self._runtime.reset()
            self._touchdown_phase_state = initial_touchdown_phase_tracker_state()
            self._touchdown_joint_position_history.clear()
            self._touchdown_joint_velocity_history.clear()
            self._touchdown_previous_action_history.clear()
            if self._velocity_estimator is not None:
                self._velocity_estimator.reset()
                self._velocity_estimator_synchronizer.reset()
                self._base_linear_velocity = None
                self._receipt_times.pop("odometry", None)
                self._state_stamps.pop("odometry", None)
            self._effective_command.fill(0.0)
            self._last_policy_time = None
            self._last_reset_sequence = sequence
            self._reset_events.append(
                {
                    "clock_s": self._now_seconds(),
                    "sequence": sequence,
                }
            )
            self.get_logger().info(
                f"Reset policy action history for simulator episode {sequence}"
            )
        acknowledgement = UInt64()
        acknowledgement.data = sequence
        self._episode_reset_ack_publisher.publish(acknowledgement)

    def _try_synchronized_policy_step(self) -> None:
        required = ("joint_state", "imu", "odometry")
        if any(name not in self._state_stamps for name in required):
            return
        stamps = [self._state_stamps[name] for name in required]
        joint_stamp = self._state_stamps["joint_state"]
        if joint_stamp <= self._last_inference_joint_stamp:
            return
        if max(stamps) - min(stamps) > self._state_sync_tolerance:
            return
        self._last_inference_joint_stamp = joint_stamp
        self._run_policy(output_stamp_s=joint_stamp)

    def _on_policy_tick(self) -> None:
        self._run_policy()

    def _run_policy(self, *, output_stamp_s: float | None = None) -> None:
        now = self._now_seconds()
        if self._transplant_initial_command_pending:
            self._receipt_times["command"] = now
            self._transplant_initial_command_pending = False
        required = ("joint_state", "imu", "odometry")
        missing = [name for name in required if name not in self._receipt_times]
        stale = [
            name
            for name in required
            if name in self._receipt_times
            and now - self._receipt_times[name] > self._state_timeout
        ]
        if missing or stale:
            self._warn_throttled(
                "state_not_ready",
                f"No policy output: missing={missing}, stale={stale}",
            )
            return

        received_command = self._command
        command_age_s = now - self._receipt_times.get(
            "command", float("-inf")
        )
        watchdog_timed_out = command_age_s > self._command_timeout
        policy_elapsed_s = (
            0.0
            if self._last_policy_time is None
            else max(now - self._last_policy_time, 0.0)
        )
        self._last_policy_time = now
        watchdog_deceleration_enabled = (
            self._watchdog_linear_deceleration > 0.0
        )
        if not watchdog_timed_out:
            effective_command = received_command.copy()
        elif watchdog_deceleration_enabled:
            effective_command = decelerate_command_toward_zero(
                self._effective_command,
                policy_elapsed_s,
                self._watchdog_linear_deceleration,
                self._watchdog_angular_deceleration,
            )
        else:
            effective_command = np.zeros(3, dtype=np.float32)
        self._effective_command = effective_command

        assert self._base_linear_velocity is not None
        assert self._base_angular_velocity is not None
        assert self._projected_gravity is not None
        assert self._joint_positions is not None
        assert self._joint_velocities is not None
        state = RobotState(
            base_linear_velocity=self._base_linear_velocity,
            base_angular_velocity=self._base_angular_velocity,
            projected_gravity=self._projected_gravity,
            joint_positions=self._joint_positions,
            joint_velocities=self._joint_velocities,
        )
        try:
            result = self._runtime.step(
                state,
                effective_command,
                self._slam_confidence_observation(),
            )
        except (RuntimeError, ValueError) as error:
            self._warn_throttled("inference_error", f"No policy output: {error}")
            return
        adapted_action = result.raw_action
        adapted_joint_targets = result.joint_targets
        touchdown_diagnostic: dict[str, Any] | None = None
        if self._touchdown_experiment_enabled:
            assert self._touchdown_phase_artifact is not None
            self._touchdown_joint_position_history.append(self._joint_positions.copy())
            self._touchdown_joint_velocity_history.append(self._joint_velocities.copy())
            self._touchdown_previous_action_history.append(
                result.observation[36:48].copy()
            )
            for history in (
                self._touchdown_joint_position_history,
                self._touchdown_joint_velocity_history,
                self._touchdown_previous_action_history,
            ):
                del history[:-self._touchdown_history_length]
            pipeline = apply_touchdown_residual_pipeline(
                requested_command=effective_command,
                backbone_action=result.raw_action,
                joint_position_history=self._touchdown_joint_position_history,
                joint_velocity_history=self._touchdown_joint_velocity_history,
                previous_action_history=self._touchdown_previous_action_history,
                current_joint_position_rad=self._joint_positions,
                current_joint_velocity_radps=self._joint_velocities,
                phase_artifact=self._touchdown_phase_artifact,
                phase_state=self._touchdown_phase_state,
                attenuation_fraction=self._touchdown_attenuation,
                tracking_valid=True,
            )
            self._touchdown_phase_state = pipeline.phase.state
            adapted_action = pipeline.intervention.applied_action
            defaults = np.asarray(
                self._contract.default_joint_positions, dtype=np.float32
            )
            adapted_joint_targets = (
                defaults + self._contract.action_scale * adapted_action
            )
            self._runtime.seed_previous_action(adapted_action)
            touchdown_diagnostic = {
                "arm": self._touchdown_arm,
                "attenuation_fraction": self._touchdown_attenuation,
                "mode": pipeline.mode,
                "wiring_valid": pipeline.wiring_valid,
                "tracking_valid": pipeline.phase.tracking_valid,
                "eligible_feet": pipeline.intervention.active_feet.astype(bool).tolist(),
                "phase_progress": pipeline.phase.swing_progress.astype(float).tolist(),
                "phase_confidence": pipeline.phase.confidence.astype(float).tolist(),
                "action_residual": pipeline.intervention.action_residual.astype(float).tolist(),
                "adapted_action": adapted_action.astype(float).tolist(),
                "horizontal_first_order_correction_m": (
                    np.einsum(
                        "fij,j->fi",
                        pipeline.kinematics.foot_jacobian_per_policy_action,
                        pipeline.intervention.action_residual,
                    )[:, :2].astype(float).tolist()
                    if pipeline.kinematics is not None
                    else [[0.0, 0.0]] * 4
                ),
            }
        if (
            self._transplant_expected_observation is not None
            and not self._transplant_observation_checked
        ):
            errors = np.abs(
                result.observation - self._transplant_expected_observation
            )
            term_slices = {
                "base_linear_velocity": slice(0, 3),
                "base_angular_velocity": slice(3, 6),
                "projected_gravity": slice(6, 9),
                "velocity_command": slice(9, 12),
                "joint_position": slice(12, 24),
                "joint_velocity": slice(24, 36),
                "previous_action": slice(36, 48),
            }
            if self._uses_slam_confidence:
                term_slices["slam_confidence"] = slice(48, 51)
            term_errors = {
                name: float(np.max(errors[term_slice]))
                for name, term_slice in term_slices.items()
            }
            self._transplant_observation_checked = True
            self._transplant_observation_parity = term_errors
            self.get_logger().info(
                (
                    "Action-history replay first external observation max errors: "
                    if self._transplant_action_replay
                    else (
                        "State-transplant post-bootstrap observation max errors "
                        "(physics contact cache unavailable): "
                    )
                )
                + f"{term_errors}"
            )
        if self._diagnostics_path is not None:
            self._diagnostic_records.append(
                {
                    "clock_s": now,
                    "state_stamps_s": dict(self._state_stamps),
                    "received_command": received_command.astype(float).tolist(),
                    "effective_command": effective_command.astype(float).tolist(),
                    "command_age_s": (
                        None
                        if not np.isfinite(command_age_s)
                        else float(command_age_s)
                    ),
                    "watchdog_timed_out": watchdog_timed_out,
                    "watchdog_deceleration_enabled": (
                        watchdog_deceleration_enabled
                    ),
                    "observation": result.observation.astype(float).tolist(),
                    "raw_action": result.raw_action.astype(float).tolist(),
                    **(
                        {"touchdown_residual": touchdown_diagnostic}
                        if touchdown_diagnostic is not None else {}
                    ),
                }
            )

        output = JointState()
        if output_stamp_s is None:
            output.header.stamp = self.get_clock().now().to_msg()
        else:
            seconds = int(np.floor(output_stamp_s))
            nanoseconds = int(round((output_stamp_s - seconds) * 1.0e9))
            if nanoseconds >= 1_000_000_000:
                seconds += 1
                nanoseconds -= 1_000_000_000
            output.header.stamp.sec = seconds
            output.header.stamp.nanosec = nanoseconds
        output.name = list(self._contract.joint_order)
        output.position = adapted_joint_targets.astype(float).tolist()
        self._joint_command_publisher.publish(output)

    def write_diagnostics(self) -> None:
        if self._diagnostics_path is None:
            return
        self._diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        self._diagnostics_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "records": self._diagnostic_records,
                    "reset_events": self._reset_events,
                    "state_transplant_post_bootstrap_observation_max_errors": (
                        self._transplant_observation_parity
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: AnymalPolicyNode | None = None
    try:
        node = AnymalPolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.write_diagnostics()
                node.destroy_node()
            except (KeyboardInterrupt, RuntimeError):
                pass
        try:
            rclpy.try_shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass


if __name__ == "__main__":
    main()
