"""Run v0.4.0 locomotion with FAST-LIO2 odometry, without confidence/PPO changes."""

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
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
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
                    "FAST-LIO2 locomotion benchmark driver failed with "
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
    enable_confidence = LaunchConfiguration("enable_confidence")
    slam_odom_topic = LaunchConfiguration("slam_odom_topic")
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
                "odometry_topic": slam_odom_topic,
                "expected_odometry_child_frame": "base_link",
                "inference_trigger": policy_inference_trigger,
                "state_timeout_s": ParameterValue(
                    policy_state_timeout_s,
                    value_type=float,
                ),
                "diagnostics_path": PathJoinSubstitution(
                    [output_dir, "policy_diagnostics.json"]
                ),
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
                "expected_confidence_backend": "fastlio2",
                "expected_calibration_id": "native-v1-1e6cf8347be1",
                "output_path": PathJoinSubstitution(
                    [output_dir, "driver.json"]
                ),
            }
        ],
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
            "enable_confidence": enable_confidence,
        }.items(),
    )
    simulation = ExecuteProcess(
        cmd=[
            PathJoinSubstitution([isaaclab_root, "isaaclab.sh"]),
            "-p",
            PathJoinSubstitution(
                [project_root, "scripts", "validation", "validate_ros2_bridge.py"]
            ),
            "--device",
            device,
            "--headless",
            "--steps",
            simulation_steps,
            "--real-time",
            "--external-control",
            "--disable-episode-timeout",
            "--enhanced-determinism",
            "--imu-observation-parity-atol",
            imu_observation_parity_atol,
            "--imu-angular-velocity-parity-atol",
            imu_angular_velocity_parity_atol,
            "--factory-usd-path",
            factory_usd_path,
            "--factory-friction",
            factory_friction,
            "--spawn-x",
            spawn_x,
            "--spawn-y",
            spawn_y,
            "--spawn-yaw",
            spawn_yaw,
            "--state-transplant-manifest",
            state_transplant_manifest,
            "--enable-lio-sam",
            "--locomotion-diagnostics-output",
            PathJoinSubstitution([output_dir, "locomotion_diagnostics.json"]),
            "--locomotion-profile",
            profile,
            "--benchmark-completion-file",
            PathJoinSubstitution([output_dir, "driver.json"]),
        ],
        output="log",
        sigterm_timeout="30",
        sigkill_timeout="10",
    )
    prepare_output = ExecuteProcess(
        cmd=["mkdir", "-p", output_dir],
        output="screen",
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
                "enable_confidence",
                default_value="false",
                description=(
                    "Enable calibrated FAST-LIO2 confidence instrumentation; "
                    "the locomotion policy does not consume it"
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
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable("RMW_IMPLEMENTATION", rmw_implementation),
            SetEnvironmentVariable("CYCLONEDDS_URI", local_cyclonedds_uri),
            SetEnvironmentVariable("TERM", "xterm-256color"),
            SetEnvironmentVariable("PYTHONPATH", project_python_path),
            prepare_output,
            roudi,
            TimerAction(
                period=1.0,
                actions=[
                    fastlio,
                    policy,
                    stability_driver,
                    TimerAction(period=2.0, actions=[simulation]),
                ],
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=stability_driver,
                    on_exit=_shutdown_if_benchmark_failed,
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=simulation,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(
                                reason=(
                                    "FAST-LIO2 locomotion simulation completed"
                                )
                            )
                        )
                    ],
                )
            ),
        ]
    )
