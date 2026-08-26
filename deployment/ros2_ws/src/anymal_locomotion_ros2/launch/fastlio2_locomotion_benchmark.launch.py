"""Run candidate locomotion from selected native SLAM odometry/confidence."""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _shutdown_if_benchmark_failed(event, _context):
    if event.returncode == 0:
        return []
    return [
        EmitEvent(
            event=Shutdown(
                reason=(
                    "SLAM-confidence locomotion benchmark driver failed with "
                    f"exit code {event.returncode}"
                )
            )
        )
    ]


def _find_project_root(package_share: Path) -> Path:
    for start in (Path(__file__).resolve(), package_share.resolve()):
        for candidate in (start, *start.parents):
            if (
                (candidate / "pyproject.toml").is_file()
                and (
                    candidate
                    / "assets"
                    / "maps"
                    / "factory"
                    / "Factory_Layout.usd"
                ).is_file()
                and (candidate / "deployment" / "ros2_ws").is_dir()
            ):
                return candidate
    raise RuntimeError(
        "Cannot locate the anymal_locomotion repository from "
        f"{package_share}"
    )


def _simulation_actions(context, *_) -> list[object]:
    interactive = (
        LaunchConfiguration("interactive").perform(context).strip().lower()
        == "true"
    )
    headless = (
        LaunchConfiguration("headless").perform(context).strip().lower()
        == "true"
    )
    project_root = LaunchConfiguration("project_root")
    output_dir = LaunchConfiguration("output_dir")
    command = [
        PathJoinSubstitution(
            [LaunchConfiguration("isaaclab_root"), "isaaclab.sh"]
        ),
        "-p",
        PathJoinSubstitution(
            [project_root, "scripts", "validation", "validate_ros2_bridge.py"]
        ),
        "--device",
        LaunchConfiguration("device"),
        "--seed",
        LaunchConfiguration("simulation_seed"),
    ]
    if headless:
        command.append("--headless")
    command.extend(
        [
            "--steps",
            LaunchConfiguration("simulation_steps"),
            "--real-time",
            "--external-control",
            "--joint-command-wait-timeout-s",
            LaunchConfiguration("joint_command_wait_timeout_s"),
            "--disable-episode-timeout",
            "--enhanced-determinism",
            "--imu-observation-parity-atol",
            LaunchConfiguration("imu_observation_parity_atol"),
            "--imu-angular-velocity-parity-atol",
            LaunchConfiguration("imu_angular_velocity_parity_atol"),
            "--factory-usd-path",
            LaunchConfiguration("factory_usd_path"),
            "--factory-friction",
            LaunchConfiguration("factory_friction"),
            "--spawn-x",
            LaunchConfiguration("spawn_x"),
            "--spawn-y",
            LaunchConfiguration("spawn_y"),
            "--spawn-yaw",
            LaunchConfiguration("spawn_yaw"),
            "--state-transplant-manifest",
            LaunchConfiguration("state_transplant_manifest"),
            "--enable-lio-sam",
            "--locomotion-diagnostics-output",
            PathJoinSubstitution([output_dir, "locomotion_diagnostics.json"]),
            "--locomotion-profile",
            LaunchConfiguration("profile"),
        ]
    )
    if not interactive:
        command.extend(
            [
                "--benchmark-completion-file",
                PathJoinSubstitution([output_dir, "driver.json"]),
            ]
        )
    simulation = ExecuteProcess(
        cmd=command,
        output="screen" if interactive else "log",
        sigterm_timeout="30",
        sigkill_timeout="10",
    )
    return [
        simulation,
        RegisterEventHandler(
            OnProcessExit(
                target_action=simulation,
                on_exit=[
                    EmitEvent(
                        event=Shutdown(
                            reason=(
                                "SLAM-confidence locomotion simulation completed"
                            )
                        )
                    )
                ],
            )
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("anymal_locomotion_ros2"))
    project_root_default = _find_project_root(package_share)
    default_isaaclab_root = Path(
        os.environ.get("ISAACLAB_ROOT", Path.home() / "IsaacLab")
    ).expanduser()
    local_cyclonedds_uri = (
        "file://" + str(package_share / "config" / "cyclonedds_local.xml")
    )

    project_root = LaunchConfiguration("project_root")
    isaaclab_root = LaunchConfiguration("isaaclab_root")
    device = LaunchConfiguration("device")
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    rmw_implementation = LaunchConfiguration("rmw_implementation")
    profile = LaunchConfiguration("profile")
    output_dir = LaunchConfiguration("output_dir")
    policy_path = LaunchConfiguration("policy_path")
    metadata_path = LaunchConfiguration("metadata_path")
    factory_usd_path = LaunchConfiguration("factory_usd_path")
    factory_friction = LaunchConfiguration("factory_friction")
    spawn_x = LaunchConfiguration("spawn_x")
    spawn_y = LaunchConfiguration("spawn_y")
    spawn_yaw = LaunchConfiguration("spawn_yaw")
    simulation_steps = LaunchConfiguration("simulation_steps")
    state_transplant_manifest = LaunchConfiguration(
        "state_transplant_manifest"
    )
    fastlio_config_path = LaunchConfiguration("fastlio_config_path")
    fastlio_blind = LaunchConfiguration("fastlio_blind")
    fastlio_point_filter_num = LaunchConfiguration(
        "fastlio_point_filter_num"
    )
    fastlio_max_iteration = LaunchConfiguration("fastlio_max_iteration")
    fastlio_filter_size_surf = LaunchConfiguration(
        "fastlio_filter_size_surf"
    )
    fastlio_filter_size_map = LaunchConfiguration("fastlio_filter_size_map")
    fastlio_cube_side_length = LaunchConfiguration(
        "fastlio_cube_side_length"
    )
    fastlio_time_source = LaunchConfiguration("fastlio_time_source")
    fastlio_point_order = LaunchConfiguration("fastlio_point_order")
    fastlio_time_sync_en = LaunchConfiguration("fastlio_time_sync_en")
    fastlio_time_offset_lidar_to_imu = LaunchConfiguration(
        "fastlio_time_offset_lidar_to_imu"
    )
    point_density = LaunchConfiguration("point_density")
    point_density_profile = LaunchConfiguration("point_density_profile")
    point_density_min = LaunchConfiguration("point_density_min")
    point_density_healthy_s = LaunchConfiguration("point_density_healthy_s")
    point_density_ramp_down_s = LaunchConfiguration("point_density_ramp_down_s")
    point_density_hold_s = LaunchConfiguration("point_density_hold_s")
    point_density_ramp_up_s = LaunchConfiguration("point_density_ramp_up_s")
    command_scale_pulse_enabled = LaunchConfiguration(
        "command_scale_pulse_enabled"
    )
    command_scale_pulse_start_s = LaunchConfiguration(
        "command_scale_pulse_start_s"
    )
    command_scale_pulse_duration_s = LaunchConfiguration(
        "command_scale_pulse_duration_s"
    )
    command_scale_pulse_scale = LaunchConfiguration(
        "command_scale_pulse_scale"
    )
    command_scale_pulse_scale_x = LaunchConfiguration(
        "command_scale_pulse_scale_x"
    )
    command_scale_pulse_scale_y = LaunchConfiguration(
        "command_scale_pulse_scale_y"
    )
    command_scale_pulse_scale_z = LaunchConfiguration(
        "command_scale_pulse_scale_z"
    )
    enable_confidence = LaunchConfiguration("enable_confidence")
    slam_odom_topic = LaunchConfiguration("slam_odom_topic")
    policy_odometry_topic = LaunchConfiguration("policy_odometry_topic")
    enable_velocity_estimator = LaunchConfiguration(
        "enable_velocity_estimator"
    )
    velocity_estimator_metadata_path = LaunchConfiguration(
        "velocity_estimator_metadata_path"
    )
    velocity_estimator_sync_tolerance_s = LaunchConfiguration(
        "velocity_estimator_sync_tolerance_s"
    )
    imu_observation_parity_atol = LaunchConfiguration(
        "imu_observation_parity_atol"
    )
    imu_angular_velocity_parity_atol = LaunchConfiguration(
        "imu_angular_velocity_parity_atol"
    )
    policy_inference_trigger = LaunchConfiguration(
        "policy_inference_trigger"
    )
    policy_state_timeout_s = LaunchConfiguration("policy_state_timeout_s")
    joint_command_wait_timeout_s = LaunchConfiguration(
        "joint_command_wait_timeout_s"
    )
    slam_backend = LaunchConfiguration("slam_backend")
    expected_confidence_backend = LaunchConfiguration(
        "expected_confidence_backend"
    )
    expected_calibration_id = LaunchConfiguration(
        "expected_calibration_id"
    )
    interactive = LaunchConfiguration("interactive")
    use_rviz = LaunchConfiguration("use_rviz")
    open_teleop_terminal = LaunchConfiguration("open_teleop_terminal")
    enable_touchdown_residual_experiment = LaunchConfiguration(
        "enable_touchdown_residual_experiment"
    )
    touchdown_residual_arm = LaunchConfiguration("touchdown_residual_arm")
    touchdown_phase_artifact_path = LaunchConfiguration(
        "touchdown_phase_artifact_path"
    )
    touchdown_phase_artifact_sha256 = LaunchConfiguration(
        "touchdown_phase_artifact_sha256"
    )

    project_python_path = [
        PathJoinSubstitution([project_root, "deployment", "python_vendor"]),
        ":",
        PathJoinSubstitution([project_root, "source", "anymal_locomotion"]),
        ":",
        EnvironmentVariable("PYTHONPATH", default_value=""),
    ]

    policy = Node(
        package="anymal_locomotion_ros2",
        executable="policy_node",
        name="anymal_policy",
        parameters=[
            {
                "use_sim_time": True,
                "backend": "onnx",
                "policy_path": policy_path,
                "metadata_path": metadata_path,
                "odometry_topic": policy_odometry_topic,
                "enable_velocity_estimator": ParameterValue(
                    enable_velocity_estimator,
                    value_type=bool,
                ),
                "velocity_estimator_metadata_path": (
                    velocity_estimator_metadata_path
                ),
                "velocity_estimator_sync_tolerance_s": ParameterValue(
                    velocity_estimator_sync_tolerance_s,
                    value_type=float,
                ),
                "expected_odometry_child_frame": "base_link",
                "inference_trigger": policy_inference_trigger,
                "state_timeout_s": ParameterValue(
                    policy_state_timeout_s,
                    value_type=float,
                ),
                "expected_slam_confidence_backend": (
                    expected_confidence_backend
                ),
                "expected_slam_confidence_calibration_id": (
                    expected_calibration_id
                ),
                "diagnostics_path": PathJoinSubstitution(
                    [output_dir, "policy_diagnostics.json"]
                ),
                "enable_touchdown_residual_experiment": ParameterValue(
                    enable_touchdown_residual_experiment, value_type=bool,
                ),
                "touchdown_residual_arm": touchdown_residual_arm,
                "touchdown_phase_artifact_path": touchdown_phase_artifact_path,
                "touchdown_phase_artifact_sha256": touchdown_phase_artifact_sha256,
            }
        ],
        output="screen",
    )
    stability_driver = Node(
        package="anymal_locomotion_ros2",
        executable="stability_benchmark",
        name="anymal_stability_benchmark",
        parameters=[
            {
                "use_sim_time": True,
                "profile": profile,
                "project_root": project_root,
                "require_slam_confidence": ParameterValue(
                    enable_confidence,
                    value_type=bool,
                ),
                "allow_expected_tracking_loss": ParameterValue(
                    LaunchConfiguration("confidence_loss_is_outcome"),
                    value_type=bool,
                ),
                "expected_confidence_backend": expected_confidence_backend,
                "expected_calibration_id": expected_calibration_id,
                "command_scale_pulse_enabled": ParameterValue(
                    command_scale_pulse_enabled, value_type=bool,
                ),
                "command_scale_pulse_start_s": ParameterValue(
                    command_scale_pulse_start_s, value_type=float,
                ),
                "command_scale_pulse_duration_s": ParameterValue(
                    command_scale_pulse_duration_s, value_type=float,
                ),
                "command_scale_pulse_scale": ParameterValue(
                    command_scale_pulse_scale, value_type=float,
                ),
                "command_scale_pulse_scale_x": ParameterValue(
                    command_scale_pulse_scale_x, value_type=float,
                ),
                "command_scale_pulse_scale_y": ParameterValue(
                    command_scale_pulse_scale_y, value_type=float,
                ),
                "command_scale_pulse_scale_z": ParameterValue(
                    command_scale_pulse_scale_z, value_type=float,
                ),
                "output_path": PathJoinSubstitution(
                    [output_dir, "driver.json"]
                ),
            }
        ],
        condition=UnlessCondition(interactive),
        output="screen",
    )
    fastlio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(package_share / "launch" / "fastlio2_benchmark.launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
            "config_path": fastlio_config_path,
            "raw_cloud_topic": "/lidar/points_raw",
            "fastlio_cloud_topic": "/fastlio/points",
            "candidate_odom_topic": "/Odometry",
            "slam_odom_topic": slam_odom_topic,
            "blind": fastlio_blind,
            "point_filter_num": fastlio_point_filter_num,
            "max_iteration": fastlio_max_iteration,
            "filter_size_surf": fastlio_filter_size_surf,
            "filter_size_map": fastlio_filter_size_map,
            "cube_side_length": fastlio_cube_side_length,
            "time_source": fastlio_time_source,
            "point_order": fastlio_point_order,
            "time_sync_en": fastlio_time_sync_en,
            "time_offset_lidar_to_imu": fastlio_time_offset_lidar_to_imu,
            "point_density": point_density,
            "point_density_profile": point_density_profile,
            "point_density_min": point_density_min,
            "point_density_healthy_s": point_density_healthy_s,
            "point_density_ramp_down_s": point_density_ramp_down_s,
            "point_density_hold_s": point_density_hold_s,
            "point_density_ramp_up_s": point_density_ramp_up_s,
            "enable_confidence": enable_confidence,
            "enable_visual_outputs": use_rviz,
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", slam_backend, "' == 'fastlio2'"])
        ),
    )
    liosam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(package_share / "launch" / "lio_sam.launch.py")
        ),
        launch_arguments={
            "use_rviz": use_rviz,
            "enable_confidence": enable_confidence,
            "use_motion_deskew": "false",
            "feature_cloud_info_topic": "/lio_sam/deskew/cloud_info",
            "static_transform_cyclonedds_uri": (
                "file://"
                + str(package_share / "config" / "cyclonedds_static_tf.xml")
            ),
            "loop_closure_enable": "false",
            "point_density": point_density,
            "point_density_profile": point_density_profile,
            "point_density_min": point_density_min,
            "point_density_healthy_s": point_density_healthy_s,
            "point_density_ramp_down_s": point_density_ramp_down_s,
            "point_density_hold_s": point_density_hold_s,
            "point_density_ramp_up_s": point_density_ramp_up_s,
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", slam_backend, "' == 'liosam'"])
        ),
    )
    fastlio_rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="fastlio2_rviz",
        arguments=[
            "-d",
            str(
                Path(get_package_share_directory("fast_lio"))
                / "rviz_cfg"
                / "fastlio.rviz"
            ),
        ],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    slam_backend,
                    "' == 'fastlio2' and '",
                    use_rviz,
                    "' == 'true'",
                ]
            )
        ),
        output="screen",
    )
    teleop = ExecuteProcess(
        cmd=[
            "gnome-terminal",
            "--wait",
            "--title=ANYmal-D confidence-aware PPO teleop",
            "--",
            "ros2",
            "run",
            "teleop_twist_keyboard",
            "teleop_twist_keyboard",
            "--ros-args",
            "-p",
            "speed:=0.5",
            "-p",
            "turn:=0.5",
            "-r",
            "cmd_vel:=/cmd_vel",
        ],
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    interactive,
                    "' == 'true' and '",
                    open_teleop_terminal,
                    "' == 'true'",
                ]
            )
        ),
        output="screen",
    )
    prepare_output = ExecuteProcess(
        cmd=["mkdir", "-p", output_dir],
        output="screen",
    )
    bag = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-o",
            PathJoinSubstitution([output_dir, "raw_bag"]),
            "/clock",
            "/odom",
            "/imu/data",
            "/lidar/points_raw",
            "/cmd_vel",
            "/joint_states",
            "/foot_contacts",
            "/locomotion/estimated_odom",
            "/tf",
            "/tf_static",
            "/simulation/episode_reset",
            "/simulation/episode_reset_ack",
            "/slam/odom",
            "/slam_confidence",
        ],
        condition=IfCondition(LaunchConfiguration("record_bag")),
        output="log",
    )
    roudi = ExecuteProcess(
        cmd=[
            FindExecutable(name="iox-roudi"),
            "--log-level",
            "warning",
        ],
        output="log",
        condition=None,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "project_root",
                default_value=str(project_root_default),
            ),
            DeclareLaunchArgument(
                "isaaclab_root",
                default_value=str(default_isaaclab_root),
            ),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument(
                "interactive",
                default_value="false",
                description=(
                    "Disable the scripted benchmark driver and accept live "
                    "/cmd_vel commands"
                ),
            ),
            DeclareLaunchArgument("headless", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument(
                "record_bag",
                default_value="false",
                description=(
                    "Record raw sensors, commands, reset handshake, SLAM and "
                    "confidence for matched publication replay"
                ),
            ),
            DeclareLaunchArgument(
                "confidence_loss_is_outcome",
                default_value="false",
                description=(
                    "Keep expected controlled tracking/freshness loss as an "
                    "experimental outcome instead of a driver failure"
                ),
            ),
            DeclareLaunchArgument(
                "open_teleop_terminal", default_value="false"
            ),
            DeclareLaunchArgument(
                "enable_touchdown_residual_experiment",
                default_value="false",
                description="Opt-in block597-only touchdown residual experiment",
            ),
            DeclareLaunchArgument(
                "touchdown_residual_arm",
                default_value="zero",
                choices=["zero", "touchdown_soft_low", "touchdown_soft"],
            ),
            DeclareLaunchArgument("touchdown_phase_artifact_path", default_value=""),
            DeclareLaunchArgument("touchdown_phase_artifact_sha256", default_value=""),
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value=EnvironmentVariable(
                    "ROS_DOMAIN_ID",
                    default_value="1",
                ),
            ),
            DeclareLaunchArgument(
                "rmw_implementation",
                default_value="rmw_cyclonedds_cpp",
            ),
            DeclareLaunchArgument(
                "slam_backend",
                default_value="fastlio2",
                choices=["fastlio2", "liosam"],
                description="Native SLAM backend that owns /slam/odom",
            ),
            DeclareLaunchArgument(
                "policy_odometry_topic",
                default_value="/slam/odom",
                description=(
                    "Validated body-state odometry consumed by the policy; "
                    "LIO-SAM uses /slam/policy_odom while /slam/odom remains "
                    "the confidence exact-stamp authority"
                ),
            ),
            DeclareLaunchArgument(
                "enable_velocity_estimator",
                default_value="false",
                description=(
                    "Use the project proprioceptive body-velocity estimator "
                    "as the policy odometry source"
                ),
            ),
            DeclareLaunchArgument(
                "velocity_estimator_metadata_path",
                default_value=PathJoinSubstitution(
                    [
                        project_root,
                        "exported",
                        "proprioceptive_velocity_estimator",
                        "v1",
                        "clean_candidate_08",
                        "velocity_estimator_metadata.json",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "velocity_estimator_sync_tolerance_s",
                default_value="0.025",
            ),
            DeclareLaunchArgument(
                "expected_confidence_backend",
                default_value="fastlio2",
                description="Exact confidence backend identity required",
            ),
            DeclareLaunchArgument(
                "expected_calibration_id",
                default_value="native-v1-1e6cf8347be1",
                description="Exact confidence calibration identity required",
            ),
            DeclareLaunchArgument("profile", default_value="stationary"),
            DeclareLaunchArgument(
                "output_dir",
                default_value=PathJoinSubstitution(
                    [
                        project_root,
                        "logs",
                        "stability_benchmarks",
                        "fastlio2",
                        profile,
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "policy_path",
                default_value=PathJoinSubstitution(
                    [
                        project_root,
                        "exported",
                        "anymal_d_locomotion_v1",
                        "recovery_v0.4.0",
                        "policy.onnx",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "metadata_path",
                default_value=PathJoinSubstitution(
                    [
                        project_root,
                        "exported",
                        "anymal_d_locomotion_v1",
                        "recovery_v0.4.0",
                        "policy_metadata.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "factory_usd_path",
                default_value=PathJoinSubstitution(
                    [
                        project_root,
                        "assets",
                        "maps",
                        "factory",
                        "Factory_Layout.usd",
                    ]
                ),
            ),
            DeclareLaunchArgument("factory_friction", default_value="1.0"),
            DeclareLaunchArgument("spawn_x", default_value="0.0"),
            DeclareLaunchArgument("spawn_y", default_value="-18.0"),
            DeclareLaunchArgument("spawn_yaw", default_value="0.0"),
            DeclareLaunchArgument("simulation_steps", default_value="4000"),
            DeclareLaunchArgument(
                "simulation_seed",
                default_value="42",
                description=(
                    "Isaac Lab environment seed; publication blocks must pass "
                    "the frozen paired block ID explicitly"
                ),
            ),
            DeclareLaunchArgument(
                "state_transplant_manifest",
                default_value="",
            ),
            DeclareLaunchArgument(
                "fastlio_config_path",
                default_value=str(
                    package_share / "config" / "fastlio2_anymal_ouster32.yaml"
                ),
            ),
            DeclareLaunchArgument("fastlio_blind", default_value="0.5"),
            DeclareLaunchArgument(
                "fastlio_point_filter_num",
                default_value="2",
            ),
            DeclareLaunchArgument(
                "fastlio_max_iteration",
                default_value="4",
            ),
            DeclareLaunchArgument(
                "fastlio_filter_size_surf",
                default_value="0.3",
            ),
            DeclareLaunchArgument(
                "fastlio_filter_size_map",
                default_value="0.6",
            ),
            DeclareLaunchArgument(
                "fastlio_cube_side_length",
                default_value="1000.0",
            ),
            DeclareLaunchArgument(
                "fastlio_time_source",
                default_value="sensor_order",
                description="Official reconstructed Ouster column time",
            ),
            DeclareLaunchArgument(
                "fastlio_point_order",
                default_value="staggered",
                description=(
                    "FAST-LIO2 input packing; staggered keeps the scan-end "
                    "point at the end of the cloud"
                ),
            ),
            DeclareLaunchArgument(
                "fastlio_time_sync_en",
                default_value="false",
                description="Use FAST-LIO2 native online LiDAR--IMU time sync",
            ),
            DeclareLaunchArgument(
                "fastlio_time_offset_lidar_to_imu",
                default_value="0.0",
                description="Native LiDAR-to-IMU timestamp offset in seconds",
            ),
            DeclareLaunchArgument(
                "point_density",
                default_value="1.0",
                description="Nominal deterministic LiDAR support fraction",
            ),
            DeclareLaunchArgument(
                "point_density_profile",
                default_value="constant",
                choices=["constant", "gradual_v1", "gradual_v2"],
                description=(
                    "Shared sensor-order-preserving support schedule used by "
                    "both native SLAM backends"
                ),
            ),
            DeclareLaunchArgument(
                "point_density_min",
                default_value="0.01",
                description="Minimum support fraction during gradual profiles",
            ),
            DeclareLaunchArgument("point_density_healthy_s", default_value="3.0"),
            DeclareLaunchArgument("point_density_ramp_down_s", default_value="1.5"),
            DeclareLaunchArgument("point_density_hold_s", default_value="4.5"),
            DeclareLaunchArgument("point_density_ramp_up_s", default_value="3.0"),
            DeclareLaunchArgument(
                "command_scale_pulse_enabled",
                default_value="false",
                description="Apply one deterministic command-scale pulse",
            ),
            DeclareLaunchArgument("command_scale_pulse_start_s", default_value="7.25"),
            DeclareLaunchArgument("command_scale_pulse_duration_s", default_value="0.75"),
            DeclareLaunchArgument("command_scale_pulse_scale", default_value="1.0"),
            DeclareLaunchArgument("command_scale_pulse_scale_x", default_value="-1.0"),
            DeclareLaunchArgument("command_scale_pulse_scale_y", default_value="-1.0"),
            DeclareLaunchArgument("command_scale_pulse_scale_z", default_value="-1.0"),
            DeclareLaunchArgument(
                "enable_confidence",
                default_value="false",
                description=(
                    "Enable selected native calibrated confidence authority; "
                    "a 51-D policy consumes the atomic topic"
                ),
            ),
            DeclareLaunchArgument(
                "slam_odom_topic",
                default_value="/slam/odom",
            ),
            DeclareLaunchArgument(
                "imu_observation_parity_atol",
                default_value="0.01",
                description=(
                    "Projected-gravity bridge tolerance for the LiDAR-enabled "
                    "live simulator"
                ),
            ),
            DeclareLaunchArgument(
                "imu_angular_velocity_parity_atol",
                default_value="0.03",
                description=(
                    "Physics-IMU angular-velocity tolerance. High-yaw live "
                    "evidence bounds the 5 ms sensor-vs-root sample error at "
                    "0.02694 rad/s; gravity remains at 0.01."
                ),
            ),
            DeclareLaunchArgument(
                "policy_inference_trigger",
                default_value="timer",
                description=(
                    "Policy trigger for low-rate SLAM odometry; timer uses "
                    "the latest SLAM state at the locomotion control rate"
                ),
            ),
            DeclareLaunchArgument(
                "policy_state_timeout_s",
                default_value="0.25",
                description=(
                    "Maximum receipt age for low-rate SLAM state before the "
                    "policy stops publishing"
                ),
            ),
            DeclareLaunchArgument(
                "joint_command_wait_timeout_s",
                default_value="0.03",
                description=(
                    "Maximum asynchronous DDS command wait; validation still "
                    "requires zero timeouts and records the measured maximum"
                ),
            ),
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable("RMW_IMPLEMENTATION", rmw_implementation),
            SetEnvironmentVariable("CYCLONEDDS_URI", local_cyclonedds_uri),
            SetEnvironmentVariable("TERM", "xterm-256color"),
            SetEnvironmentVariable("PYTHONPATH", project_python_path),
            prepare_output,
            roudi,
            bag,
            TimerAction(
                period=1.0,
                actions=[
                    fastlio,
                    liosam,
                    policy,
                    stability_driver,
                    fastlio_rviz,
                    teleop,
                    TimerAction(
                        period=2.0,
                        actions=[OpaqueFunction(function=_simulation_actions)],
                    ),
                ],
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=stability_driver,
                    on_exit=_shutdown_if_benchmark_failed,
                ),
                condition=UnlessCondition(interactive),
            ),
        ]
    )
