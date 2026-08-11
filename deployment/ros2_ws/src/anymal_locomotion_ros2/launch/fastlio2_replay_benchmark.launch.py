"""Replay the same recorded sensor bag through the FAST-LIO2 candidate."""

from __future__ import annotations

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


def _find_project_root(package_share: Path) -> Path:
    for start in (Path(__file__).resolve(), package_share.resolve()):
        for candidate in (start, *start.parents):
            if (
                (candidate / "pyproject.toml").is_file()
                and (candidate / "deployment" / "ros2_ws").is_dir()
            ):
                return candidate
    raise RuntimeError(
        "Cannot locate the anymal_locomotion repository from "
        f"{package_share}"
    )


def _finish_replay(_event, _context):
    return [
        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2",
                        "topic",
                        "pub",
                        "--once",
                        "/lio_replay/finish",
                        "std_msgs/msg/Empty",
                        "{}",
                    ],
                    output="screen",
                )
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("anymal_locomotion_ros2"))
    project_root_default = _find_project_root(package_share)
    local_cyclonedds_uri = (
        "file://" + str(package_share / "config" / "cyclonedds_local.xml")
    )

    project_root = LaunchConfiguration("project_root")
    bag_path = LaunchConfiguration("bag_path")
    output_path = LaunchConfiguration("output_path")
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    config_path = LaunchConfiguration("config_path")
    blind = LaunchConfiguration("blind")
    point_filter_num = LaunchConfiguration("point_filter_num")
    max_iteration = LaunchConfiguration("max_iteration")
    filter_size_surf = LaunchConfiguration("filter_size_surf")
    filter_size_map = LaunchConfiguration("filter_size_map")
    cube_side_length = LaunchConfiguration("cube_side_length")
    bag_rate = LaunchConfiguration("bag_rate")
    rmw_implementation = LaunchConfiguration("rmw_implementation")
    point_density = LaunchConfiguration("point_density")
    point_density_profile = LaunchConfiguration("point_density_profile")
    point_density_min = LaunchConfiguration("point_density_min")
    time_source = LaunchConfiguration("time_source")
    point_order = LaunchConfiguration("point_order")
    time_sync_en = LaunchConfiguration("time_sync_en")
    time_offset_lidar_to_imu = LaunchConfiguration(
        "time_offset_lidar_to_imu"
    )
    enable_confidence = LaunchConfiguration("enable_confidence")
    confidence_dataset_path = LaunchConfiguration("confidence_dataset_path")
    confidence_artifact_path = LaunchConfiguration(
        "confidence_artifact_path"
    )
    expected_confidence_calibration_id = LaunchConfiguration(
        "expected_confidence_calibration_id"
    )
    capture_group = LaunchConfiguration("capture_group")

    fastlio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(package_share / "launch" / "fastlio2_benchmark.launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
            "config_path": config_path,
            "raw_cloud_topic": "/lidar/points_raw",
            "fastlio_cloud_topic": "/fastlio/points",
            "candidate_odom_topic": "/Odometry",
            "slam_odom_topic": "/slam/odom",
            "blind": blind,
            "point_filter_num": point_filter_num,
            "max_iteration": max_iteration,
            "filter_size_surf": filter_size_surf,
            "filter_size_map": filter_size_map,
            "cube_side_length": cube_side_length,
            "point_density": point_density,
            "point_density_profile": point_density_profile,
            "point_density_min": point_density_min,
            "time_source": time_source,
            "point_order": point_order,
            "time_sync_en": time_sync_en,
            "time_offset_lidar_to_imu": time_offset_lidar_to_imu,
            "enable_confidence": enable_confidence,
            "confidence_artifact_path": confidence_artifact_path,
        }.items(),
    )
    evaluator = Node(
        package="anymal_locomotion_ros2",
        executable="lio_replay_evaluator",
        name="anymal_fastlio_replay_evaluator",
        parameters=[
            {
                "use_sim_time": True,
                "project_root": project_root,
                "output_path": output_path,
                "estimate_topic": "/slam/odom",
                "backend_kind": "fastlio2",
                "adapted_cloud_topic": "/fastlio/points",
                "ground_truth_sensor_offset_xyz": [0.0, 0.0, 0.0],
                "loop_closure_expectation": "forbidden",
                "confidence_expected": ParameterValue(
                    enable_confidence,
                    value_type=bool,
                ),
                "confidence_dataset_path": confidence_dataset_path,
                "expected_confidence_calibration_id": (
                    expected_confidence_calibration_id
                ),
                "capture_group": capture_group,
                "deskew_mode": "native",
            }
        ],
        output="screen",
    )
    replay = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            bag_path,
            "--read-ahead-queue-size",
            "1000",
            "--disable-keyboard-controls",
            "--rate",
            bag_rate,
            "--topics",
            "/clock",
            "/odom",
            "/imu/data",
            "/lidar/points_raw",
        ],
        output="screen",
    )
    roudi = ExecuteProcess(
        cmd=[
            FindExecutable(name="iox-roudi"),
            "--log-level",
            "warning",
        ],
        output="log",
    )

    project_python_path = [
        PathJoinSubstitution([project_root, "deployment", "python_vendor"]),
        ":",
        PathJoinSubstitution([project_root, "source", "anymal_locomotion"]),
        ":",
        EnvironmentVariable("PYTHONPATH", default_value=""),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "project_root",
                default_value=str(project_root_default),
            ),
            DeclareLaunchArgument("bag_path"),
            DeclareLaunchArgument(
                "output_path",
                default_value=PathJoinSubstitution(
                    [
                        project_root,
                        "outputs",
                        "fastlio2_benchmark",
                        "replay.json",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value=str(
                    package_share / "config" / "fastlio2_anymal_ouster32.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value=EnvironmentVariable(
                    "ROS_DOMAIN_ID",
                    default_value="1",
                ),
            ),
            DeclareLaunchArgument("blind", default_value="0.5"),
            DeclareLaunchArgument("point_filter_num", default_value="2"),
            DeclareLaunchArgument("max_iteration", default_value="4"),
            DeclareLaunchArgument("filter_size_surf", default_value="0.3"),
            DeclareLaunchArgument("filter_size_map", default_value="0.6"),
            DeclareLaunchArgument("cube_side_length", default_value="1000.0"),
            DeclareLaunchArgument(
                "bag_rate",
                default_value="1.0",
                description=(
                    "Rosbag wall-clock replay rate; simulation timestamps "
                    "remain unchanged"
                ),
            ),
            DeclareLaunchArgument(
                "rmw_implementation",
                default_value="rmw_cyclonedds_cpp",
                description="ROS 2 middleware used by this experiment",
            ),
            DeclareLaunchArgument(
                "point_density",
                default_value="1.0",
                description="Deterministic fraction of raw scan points retained",
            ),
            DeclareLaunchArgument(
                "point_density_profile",
                default_value="constant",
                description="constant or causal gradual_v1 replay degradation",
            ),
            DeclareLaunchArgument("point_density_min", default_value="0.01"),
            DeclareLaunchArgument(
                "time_source",
                default_value="sensor_order",
                description="Official reconstructed Ouster column time",
            ),
            DeclareLaunchArgument(
                "point_order",
                default_value="staggered",
                description="FAST-LIO2 input packing",
            ),
            DeclareLaunchArgument(
                "time_sync_en",
                default_value="false",
                description="Use FAST-LIO2 native online LiDAR--IMU time sync",
            ),
            DeclareLaunchArgument(
                "time_offset_lidar_to_imu",
                default_value="0.0",
                description="Native LiDAR-to-IMU timestamp offset in seconds",
            ),
            DeclareLaunchArgument(
                "enable_confidence",
                default_value="false",
                description=(
                    "Enable uncalibrated confidence instrumentation; false "
                    "preserves the qualified replay baseline"
                ),
            ),
            DeclareLaunchArgument(
                "confidence_dataset_path",
                default_value="",
                description=(
                    "Offline labelled dataset output; requires confidence "
                    "instrumentation and a capture_group"
                ),
            ),
            DeclareLaunchArgument(
                "confidence_artifact_path",
                default_value="",
                description=(
                    "Optional installed confidence artifact; formal calibration "
                    "capture leaves this empty"
                ),
            ),
            DeclareLaunchArgument(
                "expected_confidence_calibration_id",
                default_value="uncalibrated",
                description=(
                    "Exact artifact ID expected by replay evaluation"
                ),
            ),
            DeclareLaunchArgument("capture_group", default_value=""),
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION",
                rmw_implementation,
            ),
            SetEnvironmentVariable("CYCLONEDDS_URI", local_cyclonedds_uri),
            SetEnvironmentVariable("PYTHONPATH", project_python_path),
            roudi,
            TimerAction(
                period=1.0,
                actions=[
                    fastlio,
                    evaluator,
                    TimerAction(period=2.0, actions=[replay]),
                ],
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=replay,
                    on_exit=_finish_replay,
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=evaluator,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(
                                reason=(
                                    "FAST-LIO2 replay evaluation completed"
                                )
                            )
                        )
                    ],
                )
            ),
        ]
    )
