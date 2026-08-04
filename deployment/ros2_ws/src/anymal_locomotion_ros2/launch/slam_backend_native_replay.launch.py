"""Replay one raw sensor bag through one native SLAM backend.

The selected backend publishes its native odometry topic directly to the
offline evaluator.  FAST-LIO2 therefore does not pass through the project
``/slam/odom`` adapter in this qualification launch.
"""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
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


def _backend_actions(context, *_) -> list[object]:
    package_share = Path(
        get_package_share_directory("anymal_locomotion_ros2")
    )
    backend = LaunchConfiguration("slam_backend").perform(context).strip()
    deskew_mode = LaunchConfiguration("deskew_mode").perform(context).strip()
    point_density = LaunchConfiguration("point_density")
    project_root = LaunchConfiguration("project_root")
    output_path = LaunchConfiguration("output_path")

    if backend not in {"liosam", "fastlio2"}:
        raise RuntimeError(
            "slam_backend must be 'liosam' or 'fastlio2', "
            f"received {backend!r}"
        )
    if deskew_mode not in {"native", "project"}:
        raise RuntimeError(
            "deskew_mode must be 'native' or 'project', "
            f"received {deskew_mode!r}"
        )
    if backend == "fastlio2" and deskew_mode != "native":
        raise RuntimeError(
            "FAST-LIO2 in this launch always uses its native deskew; "
            "use deskew_mode:=native"
        )

    if backend == "liosam":
        if deskew_mode == "native":
            use_motion_deskew = "false"
            feature_cloud_info_topic = "/lio_sam/deskew/cloud_info"
            apply_translation = "false"
            replace_rotation = "false"
        else:
            use_motion_deskew = "true"
            feature_cloud_info_topic = (
                "/lio_sam/deskew/cloud_info_motion_corrected"
            )
            apply_translation = "true"
            replace_rotation = "true"
        backend_action = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(package_share / "launch" / "lio_sam.launch.py")
            ),
            launch_arguments={
                "use_rviz": "false",
                "use_motion_deskew": use_motion_deskew,
                "feature_cloud_info_topic": feature_cloud_info_topic,
                "motion_deskew_apply_translation": apply_translation,
                "motion_deskew_replace_upstream_rotation": replace_rotation,
                "point_density": point_density,
                "static_transform_cyclonedds_uri": (
                    "file://"
                    + str(package_share / "config" / "cyclonedds_static_tf.xml")
                ),
                "loop_closure_enable": "false",
            }.items(),
        )
        estimate_topic = "/lio_sam/mapping/odometry"
        sensor_offset = [0.20, 0.0, 0.35]
    else:
        try:
            get_package_share_directory("fast_lio")
        except PackageNotFoundError as error:
            raise RuntimeError(
                "FAST-LIO2 is not in the sourced ROS 2 overlay.  Source "
                "./scripts/setup_deployment.sh first."
            ) from error
        backend_action = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(
                    package_share
                    / "launch"
                    / "fastlio2_native.launch.py"
                )
            ),
            launch_arguments={
                "use_sim_time": "true",
                "config_path": LaunchConfiguration("config_path"),
                "raw_cloud_topic": "/lidar/points_raw",
                "fastlio_cloud_topic": "/fastlio/points",
                "candidate_odom_topic": "/Odometry",
                "point_density": point_density,
                "enable_visual_outputs": "false",
            }.items(),
        )
        estimate_topic = "/Odometry"
        # FAST-LIO2's native state_point.pos is the IMU/base position.  LIO-SAM
        # mapping odometry is evaluated at the LiDAR pose, hence the distinct
        # ground-truth offset above.
        sensor_offset = [0.0, 0.0, 0.0]

    evaluator = Node(
        package="anymal_locomotion_ros2",
        executable="lio_replay_evaluator",
        name=f"anymal_{backend}_native_replay_evaluator",
        parameters=[
            {
                "use_sim_time": True,
                "project_root": project_root,
                "output_path": output_path,
                "estimate_topic": estimate_topic,
                "ground_truth_sensor_offset_xyz": sensor_offset,
                "loop_closure_expectation": "forbidden",
            }
        ],
        output="screen",
    )
    return [
        backend_action,
        evaluator,
        RegisterEventHandler(
            OnProcessExit(
                target_action=evaluator,
                on_exit=[
                    EmitEvent(
                        event=Shutdown(
                            reason="native SLAM replay evaluation completed"
                        )
                    )
                ],
            )
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    package_share = Path(
        get_package_share_directory("anymal_locomotion_ros2")
    )
    project_root_default = _find_project_root(package_share)
    local_cyclonedds_uri = (
        "file://" + str(package_share / "config" / "cyclonedds_local.xml")
    )
    project_root = LaunchConfiguration("project_root")
    bag_path = LaunchConfiguration("bag_path")
    output_path = LaunchConfiguration("output_path")
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    bag_rate = LaunchConfiguration("bag_rate")

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
        cmd=[FindExecutable(name="iox-roudi"), "--log-level", "warning"],
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
                    [project_root, "logs", "slam_backend_replay", "replay.json"]
                ),
            ),
            DeclareLaunchArgument("slam_backend", default_value="liosam"),
            DeclareLaunchArgument("deskew_mode", default_value="native"),
            DeclareLaunchArgument(
                "point_density",
                default_value="1.0",
                description="Deterministic fraction of raw scan points retained",
            ),
            DeclareLaunchArgument(
                "bag_rate",
                default_value="1.0",
                description="Wall-clock replay multiplier; timestamps are unchanged",
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
                    OpaqueFunction(function=_backend_actions),
                    TimerAction(period=2.0, actions=[replay]),
                ],
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=replay,
                    on_exit=_finish_replay,
                )
            ),
        ]
    )
