"""Replay one recorded Factory sensor stream through a fresh LIO-SAM graph."""

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
    loop_closure_enable = LaunchConfiguration("loop_closure_enable")
    loop_closure_expectation = LaunchConfiguration(
        "loop_closure_expectation"
    )
    loop_search_radius = LaunchConfiguration("loop_search_radius")
    loop_search_time_diff = LaunchConfiguration("loop_search_time_diff")
    loop_search_keyframes = LaunchConfiguration("loop_search_keyframes")
    loop_fitness_score = LaunchConfiguration("loop_fitness_score")

    lio_sam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(package_share / "launch" / "lio_sam.launch.py")
        ),
        launch_arguments={
            "use_rviz": "false",
            "use_motion_deskew": "true",
            "feature_cloud_info_topic": (
                "/lio_sam/deskew/cloud_info_motion_corrected"
            ),
            "motion_deskew_apply_translation": "true",
            "motion_deskew_replace_upstream_rotation": "true",
            "static_transform_cyclonedds_uri": (
                "file://"
                + str(
                    package_share
                    / "config"
                    / "cyclonedds_static_tf.xml"
                )
            ),
            "loop_closure_enable": loop_closure_enable,
            "loop_search_radius": loop_search_radius,
            "loop_search_time_diff": loop_search_time_diff,
            "loop_search_keyframes": loop_search_keyframes,
            "loop_fitness_score": loop_fitness_score,
        }.items(),
    )
    evaluator = Node(
        package="anymal_locomotion_ros2",
        executable="lio_replay_evaluator",
        name="anymal_lio_replay_evaluator",
        parameters=[
            {
                "use_sim_time": True,
                "project_root": project_root,
                "output_path": output_path,
                "loop_closure_expectation": loop_closure_expectation,
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
                        "lio_sam_loop_closure",
                        "replay.json",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value=EnvironmentVariable(
                    "ROS_DOMAIN_ID",
                    default_value="1",
                ),
            ),
            DeclareLaunchArgument(
                "loop_closure_enable",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "loop_closure_expectation",
                default_value="forbidden",
            ),
            DeclareLaunchArgument(
                "loop_search_radius",
                default_value="1.5",
            ),
            DeclareLaunchArgument(
                "loop_search_time_diff",
                default_value="15.0",
            ),
            DeclareLaunchArgument(
                "loop_search_keyframes",
                default_value="25",
            ),
            DeclareLaunchArgument(
                "loop_fitness_score",
                default_value="0.3",
            ),
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION",
                "rmw_cyclonedds_cpp",
            ),
            SetEnvironmentVariable("CYCLONEDDS_URI", local_cyclonedds_uri),
            SetEnvironmentVariable("PYTHONPATH", project_python_path),
            roudi,
            TimerAction(
                period=1.0,
                actions=[
                    lio_sam,
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
                                reason="LIO replay evaluation completed"
                            )
                        )
                    ],
                )
            ),
        ]
    )
