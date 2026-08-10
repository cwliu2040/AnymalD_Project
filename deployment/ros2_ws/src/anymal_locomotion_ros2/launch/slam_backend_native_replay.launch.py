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


def _backend_actions(context, *_) -> list[object]:
    package_share = Path(
        get_package_share_directory("anymal_locomotion_ros2")
    )
    backend = LaunchConfiguration("slam_backend").perform(context).strip()
    deskew_mode = LaunchConfiguration("deskew_mode").perform(context).strip()
    point_density = LaunchConfiguration("point_density")
    project_root = LaunchConfiguration("project_root")
    output_path = LaunchConfiguration("output_path")
    yaw_stress_mode = LaunchConfiguration("yaw_stress_mode")
    enable_effect_diagnostics = LaunchConfiguration(
        "enable_effect_diagnostics"
    )
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
    fastlio_acc_cov = LaunchConfiguration("fastlio_acc_cov")
    fastlio_gyr_cov = LaunchConfiguration("fastlio_gyr_cov")
    fastlio_b_acc_cov = LaunchConfiguration("fastlio_b_acc_cov")
    fastlio_b_gyr_cov = LaunchConfiguration("fastlio_b_gyr_cov")
    fastlio_time_direction = LaunchConfiguration("fastlio_time_direction")
    fastlio_time_source = LaunchConfiguration("fastlio_time_source")
    fastlio_point_order = LaunchConfiguration("fastlio_point_order")
    fastlio_time_sync_en = LaunchConfiguration("fastlio_time_sync_en")
    fastlio_time_offset_lidar_to_imu = LaunchConfiguration(
        "fastlio_time_offset_lidar_to_imu"
    )
    fastlio_raw_stamp_is_scan_end = LaunchConfiguration(
        "fastlio_raw_stamp_is_scan_end"
    )
    visual_video_path = LaunchConfiguration("visual_video_path").perform(
        context
    ).strip()

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
        adapted_cloud_topic = "/lio_sam/points"
        registered_cloud_topic = "/lio_sam/mapping/cloud_registered"
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
                "enable_visual_outputs": "true" if visual_video_path else "false",
                "enable_effect_diagnostics": enable_effect_diagnostics,
                "fastlio_blind": fastlio_blind,
                "point_filter_num": fastlio_point_filter_num,
                "max_iteration": fastlio_max_iteration,
                "filter_size_surf": fastlio_filter_size_surf,
                "filter_size_map": fastlio_filter_size_map,
                "cube_side_length": fastlio_cube_side_length,
                "acc_cov": fastlio_acc_cov,
                "gyr_cov": fastlio_gyr_cov,
                "b_acc_cov": fastlio_b_acc_cov,
                "b_gyr_cov": fastlio_b_gyr_cov,
                "time_direction": fastlio_time_direction,
                "time_source": fastlio_time_source,
                "point_order": fastlio_point_order,
                "time_sync_en": fastlio_time_sync_en,
                "time_offset_lidar_to_imu": fastlio_time_offset_lidar_to_imu,
                "raw_stamp_is_scan_end": fastlio_raw_stamp_is_scan_end,
            }.items(),
        )
        estimate_topic = "/Odometry"
        adapted_cloud_topic = "/fastlio/points"
        registered_cloud_topic = "/cloud_registered"
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
                "adapted_cloud_topic": adapted_cloud_topic,
                "backend_kind": backend,
                "ground_truth_sensor_offset_xyz": sensor_offset,
                "loop_closure_expectation": "forbidden",
                "yaw_stress_mode": ParameterValue(
                    yaw_stress_mode, value_type=bool
                ),
            }
        ],
        output="screen",
    )
    actions = [
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
    if visual_video_path:
        actions.insert(
            2,
            Node(
                package="anymal_locomotion_ros2",
                executable="yaw_visualizer",
                name=f"anymal_{backend}_yaw_visualizer",
                parameters=[
                    {
                        "use_sim_time": True,
                        "project_root": project_root,
                        "cloud_topic": registered_cloud_topic,
                        "output_path": visual_video_path,
                    }
                ],
                output="screen",
            ),
        )
    return actions


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
            "/cmd_vel",
            "/joint_states",
            "/tf",
            "/tf_static",
            "/simulation/episode_reset",
            "/simulation/episode_reset_ack",
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
            DeclareLaunchArgument("yaw_stress_mode", default_value="false"),
            DeclareLaunchArgument(
                "enable_effect_diagnostics", default_value="false"
            ),
            DeclareLaunchArgument("fastlio_blind", default_value="0.5"),
            DeclareLaunchArgument(
                "fastlio_point_filter_num", default_value="2"
            ),
            DeclareLaunchArgument(
                "fastlio_max_iteration", default_value="4"
            ),
            DeclareLaunchArgument(
                "fastlio_filter_size_surf", default_value="0.3"
            ),
            DeclareLaunchArgument(
                "fastlio_filter_size_map", default_value="0.6"
            ),
            DeclareLaunchArgument(
                "fastlio_cube_side_length", default_value="1000.0"
            ),
            DeclareLaunchArgument(
                "fastlio_acc_cov", default_value="0.1"
            ),
            DeclareLaunchArgument(
                "fastlio_gyr_cov", default_value="0.1"
            ),
            DeclareLaunchArgument(
                "fastlio_b_acc_cov", default_value="0.0001"
            ),
            DeclareLaunchArgument(
                "fastlio_b_gyr_cov", default_value="0.0001"
            ),
            DeclareLaunchArgument(
                "fastlio_time_direction", default_value="clockwise"
            ),
            DeclareLaunchArgument(
                "fastlio_time_source", default_value="sensor_order"
            ),
            DeclareLaunchArgument(
                "fastlio_point_order", default_value="staggered"
            ),
            DeclareLaunchArgument(
                "fastlio_time_sync_en", default_value="false"
            ),
            DeclareLaunchArgument(
                "fastlio_time_offset_lidar_to_imu", default_value="0.0"
            ),
            DeclareLaunchArgument(
                "fastlio_raw_stamp_is_scan_end", default_value="true"
            ),
            DeclareLaunchArgument(
                "visual_video_path",
                default_value="",
                description="Optional anonymous fixed-view MP4 output path",
            ),
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
