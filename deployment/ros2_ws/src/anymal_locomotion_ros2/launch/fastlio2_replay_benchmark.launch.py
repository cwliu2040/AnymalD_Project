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
    filter_size_surf = LaunchConfiguration("filter_size_surf")
    filter_size_map = LaunchConfiguration("filter_size_map")
    bag_rate = LaunchConfiguration("bag_rate")
    rmw_implementation = LaunchConfiguration("rmw_implementation")
    point_density = LaunchConfiguration("point_density")

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
            "filter_size_surf": filter_size_surf,
            "filter_size_map": filter_size_map,
            "point_density": point_density,
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
                "ground_truth_sensor_offset_xyz": [0.0, 0.0, 0.0],
                "loop_closure_expectation": "forbidden",
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
            DeclareLaunchArgument("filter_size_surf", default_value="0.5"),
            DeclareLaunchArgument("filter_size_map", default_value="0.5"),
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
