"""Static dependency and artifact-boundary tests."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PACKAGE = PROJECT_ROOT / "source" / "anymal_locomotion"
EVALUATION_SCRIPT = PROJECT_ROOT / "scripts" / "rsl_rl" / "evaluate.py"
ROS2_POLICY_NODE = (
    PROJECT_ROOT
    / "deployment"
    / "ros2_ws"
    / "src"
    / "anymal_locomotion_ros2"
    / "anymal_locomotion_ros2"
    / "policy_node.py"
)
ROS2_BRIDGE = (
    PROJECT_ROOT
    / "source"
    / "anymal_locomotion"
    / "anymal_locomotion"
    / "simulation"
    / "ros2_bridge.py"
)
ROS2_BRIDGE_HOST = PROJECT_ROOT / "scripts" / "validation" / "validate_ros2_bridge.py"
PHYSICS_IMU = (
    PROJECT_ROOT
    / "source"
    / "anymal_locomotion"
    / "anymal_locomotion"
    / "simulation"
    / "physics_imu.py"
)
RTX_LIDAR = (
    PROJECT_ROOT
    / "source"
    / "anymal_locomotion"
    / "anymal_locomotion"
    / "simulation"
    / "rtx_lidar.py"
)
BRINGUP_LAUNCH = (
    PROJECT_ROOT
    / "deployment"
    / "ros2_ws"
    / "src"
    / "anymal_locomotion_ros2"
    / "launch"
    / "bringup.launch.py"
)
KEYBOARD_TELEOP = (
    PROJECT_ROOT
    / "deployment"
    / "ros2_ws"
    / "src"
    / "anymal_locomotion_ros2"
    / "anymal_locomotion_ros2"
    / "keyboard_teleop.py"
)
ONNX_BACKEND = (
    PROJECT_ROOT
    / "deployment"
    / "ros2_ws"
    / "src"
    / "anymal_locomotion_ros2"
    / "anymal_locomotion_ros2"
    / "onnx_backend.py"
)
SETUP_DEPLOYMENT = PROJECT_ROOT / "scripts" / "setup_deployment.sh"
DEPLOYMENT_POLICY_ROOT = (
    PROJECT_ROOT
    / "exported"
    / "anymal_d_locomotion_v1"
    / "high_speed_v0.2.0"
)


def test_training_package_does_not_import_rclpy() -> None:
    violations: list[str] = []
    for path in TRAINING_PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name == "rclpy" for alias in node.names):
                violations.append(str(path))
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "rclpy":
                violations.append(str(path))
    assert violations == []


def test_artifact_configuration_is_project_local() -> None:
    config_path = PROJECT_ROOT / "configs" / "artifacts.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = (config_path.parent / config["project_root"]).resolve()
    assert root == PROJECT_ROOT
    for key in ("logs_root", "checkpoints_root", "exports_root"):
        path = (root / config[key]).resolve()
        assert root in path.parents


def test_evaluation_uses_public_fixed_command_configuration() -> None:
    source = EVALUATION_SCRIPT.read_text(encoding="utf-8")
    assert "command_cfg.ranges.lin_vel_x" in source
    assert "command_cfg.ranges.lin_vel_y" in source
    assert "command_cfg.ranges.ang_vel_z" in source
    assert "command_manager._" not in source
    assert "rclpy" not in source
    assert 'LOG_ROOT / "evaluation"' in source


def test_recovery_v05_samples_high_combined_stop_and_long_horizons() -> None:
    command_source = (
        PROJECT_ROOT
        / "source"
        / "anymal_locomotion"
        / "anymal_locomotion"
        / "tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "config"
        / "anymal_d"
        / "mdp"
        / "commands.py"
    ).read_text(encoding="utf-8")
    env_source = (
        PROJECT_ROOT
        / "source"
        / "anymal_locomotion"
        / "anymal_locomotion"
        / "tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "config"
        / "anymal_d"
        / "flat_env_cfg.py"
    ).read_text(encoding="utf-8")

    assert "RecoveryV05VelocityCommand" in command_source
    assert "previous_high_combined" in command_source
    assert "previous_high_straight" in command_source
    assert "high_combined_stop_probability" in command_source
    assert "high_combined_straight_probability" in command_source
    assert "combined_to_reverse" in command_source
    assert "straight_to_burst" in command_source
    assert "warehouse_sequence_probability" in command_source
    assert "warehouse_durations_s" in command_source
    assert "turning_regression_probability" in command_source
    assert "turning_profiles" in command_source
    assert "low_yaw_profile_probability" in command_source
    assert "low_curve_profile_probability" in command_source
    assert "high_curve_profile_probability" in command_source
    assert "self.episode_length_s = 40.0" in env_source
    assert "resampling_time_range=(6.0, 12.0)" in env_source
    assert "AnymalDLocomotionRobustEnvCfg" in env_source
    assert "high_combined_flat_orientation_l2" in env_source
    assert "high_combined_feet_slide" in env_source
    assert "low_yaw_track_ang_vel_z_exp" in env_source
    assert "low_curve_track_lin_vel_xy_exp" in env_source
    assert "high_curve_track_lin_vel_xy_exp" in env_source


def test_ros2_policy_node_stays_outside_training_and_uses_bridge_topics() -> None:
    source = ROS2_POLICY_NODE.read_text(encoding="utf-8")
    assert "import rclpy" in source
    assert "Odometry" in source
    assert "Imu" in source
    assert "JointState" in source
    assert '"/imu/data"' in source
    assert '"/joint_command"' in source
    assert '"/simulation/episode_reset"' in source
    assert '"/simulation/episode_reset_ack"' in source
    assert "socket" not in source
    assert "udp" not in source.lower()


def test_isaac_ros2_bridge_uses_action_graph_and_name_based_commands() -> None:
    source = ROS2_BRIDGE.read_text(encoding="utf-8")
    host_source = ROS2_BRIDGE_HOST.read_text(encoding="utf-8")
    for node_type in (
        "ConstantQuatd",
        "ConstantMatrix4d",
        "SetMatrix4Rotation",
        "OgnInvertMatrix",
        "TransformVector",
        "OnImpulseEvent",
        "OnPhysicsStep",
        "IsaacReadIMU",
        "ROS2PublishClock",
        "ROS2Publisher",
        "ROS2Subscriber",
        "ROS2PublishOdometry",
        "ROS2PublishRawTransformTree",
        "ROS2PublishImu",
        "ROS2SubscribeTwist",
        "ROS2SubscribeJointState",
        "IsaacArticulationController",
    ):
        assert node_type in source
    assert "SubscribeJointState.outputs:jointNames" in source
    assert "ComputeOdometry.outputs:linearVelocity" in source
    assert "BodyAngularVelocity.outputs:result" in source
    assert "ReadImuSensor.outputs:sensorTime" in source
    assert "ReadImuSensor.outputs:angVel" in source
    assert "ReadImuSensor.outputs:linAcc" in source
    assert (
        '"ExtractImuOrientation.outputs:quaternion",\n'
        '            "PublishImu.inputs:orientation"'
    ) in source
    assert (
        '"ComposeImuOrientation.outputs:output",\n'
        '            "ImuInverseOrientation.inputs:matrix"'
    ) in source
    assert "ComputeImuOrientation" in source
    assert "InitialImuOrientation" in source
    assert "imu_reset_grace_steps_remaining = 1" in host_source
    assert "imu_reset_transient_angular_velocity_max_error" in host_source
    assert "MatrixMultiply" in source
    assert "GetMatrix4Quaternion" in source
    assert 'imu_topic: str = "imu/data"' in source
    assert "imu_update_period_s: float = 0.005" in source
    assert '("PublishOdometry.inputs:publishRawVelocities", True)' in source
    assert '("PublishTransform.inputs:parentFrameId", "odom")' in source
    assert '("PublishTransform.inputs:childFrameId", "base_link")' in source
    assert "ComputeOdometry.outputs:position" in source
    assert "PublishTransform.inputs:translation" in source
    assert "ComputeOdometry.outputs:orientation" in source
    assert "PublishTransform.inputs:rotation" in source
    assert "Context.inputs:domain_id" in source
    assert "read_joint_position_command" in source
    assert "read_velocity_command" in source
    assert "publish_episode_reset" in source
    assert "read_episode_reset_ack" in source
    assert "read_imu_state" in source
    assert "write_base_orientation" in source
    assert "write_joint_state" in source
    assert "trigger_policy_step" in source
    assert '"sensor_msgs"' in source
    assert '"JointState"' in source
    assert "outputs:timeStamp" in source
    assert "rclpy" not in source
    assert "socket" not in source
    assert "udp" not in source.lower()


def test_bridge_host_preserves_manager_based_actuator_path() -> None:
    source = ROS2_BRIDGE_HOST.read_text(encoding="utf-8")
    assert "PhysicsImuSpawnerCfg(sensor_period=0.005)" in source
    assert '"assets" / "maps" / "factory" / "Factory_Layout.usd"' in source
    assert "terrain_type=\"usd\"" in source
    assert 'prim_path="/World/Factory"' in source
    assert "create_lio_validation_landmarks" not in source
    assert '"{ENV_REGEX_NS}/Robot/base/imu_sensor"' in source
    assert "connect_articulation_controller=not args_cli.external_control" in source
    assert "read_joint_position_command" in source
    assert "validate_runtime_joint_names" in source
    assert "env.step(action_for_step)" in source
    assert "joint_command_timeout_steps" in source
    assert "actions.zero_()" in source
    assert "rclpy" not in source


def test_physics_imu_is_project_owned_and_authored_before_startup() -> None:
    source = PHYSICS_IMU.read_text(encoding="utf-8")
    assert '"IsaacSensorCreateImuSensor"' in source
    assert "sensor_period: float = 0.005" in source
    assert "linear_acceleration_filter_size: int = 1" in source
    assert "angular_velocity_filter_size: int = 1" in source
    assert "orientation_filter_size: int = 1" in source
    assert "rclpy" not in source
    assert "socket" not in source
    assert "udp" not in source.lower()


def test_rtx_lidar_is_project_owned_and_uses_official_ros2_bridge_writer() -> None:
    source = RTX_LIDAR.read_text(encoding="utf-8")
    host_source = ROS2_BRIDGE_HOST.read_text(encoding="utf-8")
    assert '"IsaacSensorCreateRtxLidar"' in source
    assert 'variant: str = "OS1_REV6_32ch10hz1024res"' in source
    assert '"omni:sensor:Core:outputFrameOfReference": "SENSOR"' in source
    assert "rep.create.render_product" in source
    assert '"RtxLidarROS2PublishPointCloudBuffer"' in source
    assert "ros2_writer.attach([render_product])" in source
    assert "publish_ground_truth_tf=not args_cli.enable_lio_sam" in host_source
    assert "args_cli.enable_cameras = True" in host_source
    assert '" --/renderer/raytracingMotion/enabled=true"' in host_source
    assert "enableHydraEngineMasking=true" in host_source
    assert "enabledForHydraEngines=0,1,2,3,4" in host_source
    assert "base_env.sim.render()" in host_source
    assert "rclpy" not in source


def test_complete_bringup_uses_project_defaults_and_official_teleop() -> None:
    source = BRINGUP_LAUNCH.read_text(encoding="utf-8")
    host_source = ROS2_BRIDGE_HOST.read_text(encoding="utf-8")
    teleop_source = KEYBOARD_TELEOP.read_text(encoding="utf-8")
    factory_map = PROJECT_ROOT / "assets" / "maps" / "factory" / "Factory_Layout.usd"

    assert factory_map.is_file()
    assert "_find_project_root" in source
    assert 'os.environ.get("ISAACLAB_ROOT"' in source
    assert 'Path.home() / "IsaacLab"' in source
    assert "/home/ros/" not in source
    assert 'default_value="cuda:0"' in source
    assert '"ROS_DOMAIN_ID"' in source
    assert 'default_value="1"' in source
    assert '"teleop_twist_keyboard"' in source
    assert '"--enable-lio-sam"' in source
    assert '"--disable-episode-timeout"' in source
    assert '"Factory_Layout.usd"' in source
    assert '"--spawn-x"' in host_source
    assert '"--spawn-y"' in host_source
    assert '"--spawn-yaw"' in host_source
    assert '"cmd_vel:=/cmd_vel"' in source
    assert '"enable_locomotion_diagnostics"' in source
    assert 'default_value="false"' in source
    assert '"--locomotion-diagnostics-output"' in source
    assert '"locomotion_diagnostics.json"' in source
    assert '"diagnostics_path": PathJoinSubstitution' in source
    assert '"policy_diagnostics.json"' in source
    assert "UnlessCondition(enable_locomotion_diagnostics)" in source
    assert "IfCondition(enable_locomotion_diagnostics)" in source
    assert "command_safety_node" not in source
    assert "cmd_vel_raw" not in source
    assert "_validate_factory_physics_material" in host_source
    ground_probe_source = (
        PROJECT_ROOT
        / "scripts"
        / "validation"
        / "validate_factory_ground_collision.py"
    ).read_text(encoding="utf-8")
    assert 'prim.GetTypeName() == "Plane"' in ground_probe_source
    assert "prim.HasAPI(UsdPhysics.CollisionAPI)" in ground_probe_source
    assert "RigidObjectCollection" in ground_probe_source
    factory_source = (
        PROJECT_ROOT / "assets" / "maps" / "factory" / "Factory_Layout.usda"
    ).read_text(encoding="utf-8")
    assert 'bindMaterialAs = "strongerThanDescendants"' in factory_source
    assert "float physics:staticFriction = 1" in factory_source
    assert "float physics:dynamicFriction = 1" in factory_source
    assert 'token physxMaterial:frictionCombineMode = "multiply"' in factory_source
    assert "source_without_coplanar_floor.usdc" in factory_source
    refinery_builder = (
        PROJECT_ROOT
        / "scripts"
        / "validation"
        / "build_factory_refinery_fix.py"
    ).read_text(encoding="utf-8")
    assert "REMOVED_FACE_INDICES = (46822, 46823)" in refinery_builder
    assert "corrected_face_count != 50344" in refinery_builder
    assert "from teleop_twist_keyboard import main as teleop_main" in teleop_source
    assert "tkinter" not in teleop_source


def test_fresh_clone_contains_policy_and_reproducible_setup_entrypoint() -> None:
    for filename in ("policy.onnx", "policy.pt", "policy_metadata.yaml"):
        assert (DEPLOYMENT_POLICY_ROOT / filename).is_file()

    setup_source = SETUP_DEPLOYMENT.read_text(encoding="utf-8")
    assert "git -C \"${PROJECT_ROOT}\" lfs pull" in setup_source
    assert "vcs import" in setup_source
    assert 'rmdir "${LIO_SAM_ROOT}"' in setup_source
    assert "LIO-SAM path is non-empty but is not a Git checkout" in setup_source
    assert "set +u" in setup_source
    assert 'source "${ROS_SETUP}"' in setup_source
    assert "set -u" in setup_source
    assert "rosdep install" in setup_source
    assert "python3 -m pip install" in setup_source
    assert "colcon build" in setup_source
    assert 'ISAACLAB_ROOT:-${HOME}/IsaacLab' in setup_source


def test_lio_sam_multi_node_executable_keeps_its_internal_node_names() -> None:
    source = (
        PROJECT_ROOT
        / "deployment"
        / "ros2_ws"
        / "src"
        / "anymal_locomotion_ros2"
        / "launch"
        / "lio_sam.launch.py"
    ).read_text(encoding="utf-8")
    assert 'executable="lio_sam_imuPreintegration"' in source
    assert 'name="lio_sam_imuPreintegration"' not in source


def test_onnx_backend_is_external_and_cpu_only() -> None:
    source = ONNX_BACKEND.read_text(encoding="utf-8")
    assert "ReferenceEvaluator" in source
    assert "np.float32" in source
    assert "rclpy" not in source
