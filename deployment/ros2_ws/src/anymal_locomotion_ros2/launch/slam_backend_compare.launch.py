"""Interactive live comparison of LIO-SAM and FAST-LIO2.

The formal model1450 policy is used only as a GT-odometry motion driver so the
simulated quadruped can walk while either SLAM backend observes it.  The
selected backend never feeds this driver.  The first comparison therefore
uses each backend's native odometry topic directly and does not start a
canonical odometry adapter.
"""

from __future__ import annotations

import os
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
from launch.conditions import IfCondition
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


def _backend_actions(context, *_) -> list[object]:
    package_share = Path(
        get_package_share_directory("anymal_locomotion_ros2")
    )
    backend = LaunchConfiguration("slam_backend").perform(context).strip()
    deskew_mode = LaunchConfiguration("deskew_mode").perform(context).strip()
    compare_use_rviz = LaunchConfiguration("compare_use_rviz").perform(
        context
    ).lower()
    point_density = LaunchConfiguration("point_density")
    static_tf_uri = LaunchConfiguration(
        "static_transform_cyclonedds_uri"
    ).perform(context)
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
            "FAST-LIO2 in this comparison always uses its native deskew; "
            "use deskew_mode:=native"
        )

    if backend == "liosam":
        rviz_config = Path(
            get_package_share_directory("lio_sam")
        ) / "config" / "rviz2.rviz"
        rviz_name = "lio_sam_official_rviz"
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

        actions = [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(package_share / "launch" / "lio_sam.launch.py")
                ),
                launch_arguments={
                    "use_rviz": "false",
                    "use_motion_deskew": use_motion_deskew,
                    "feature_cloud_info_topic": feature_cloud_info_topic,
                    "motion_deskew_apply_translation": apply_translation,
                    "motion_deskew_replace_upstream_rotation": (
                        replace_rotation
                    ),
                    "point_density": point_density,
                    "static_transform_cyclonedds_uri": static_tf_uri,
                    "loop_closure_enable": "false",
                }.items(),
            )
        ]
        if compare_use_rviz == "true":
            actions.append(
                Node(
                    package="rviz2",
                    executable="rviz2",
                    name=rviz_name,
                    arguments=["-d", str(rviz_config)],
                    parameters=[{"use_sim_time": True}],
                    output="screen",
                )
            )
        return actions

    try:
        get_package_share_directory("fast_lio")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "FAST-LIO2 is not in the sourced ROS 2 overlay.  Build and source "
            "./scripts/setup_deployment.sh before launching "
            "slam_backend_compare.launch.py with slam_backend:=fastlio2."
        ) from error

    # FAST-LIO2 publishes camera_init -> body.  These static mechanical/frame
    # links make the native TF view visible in the common RViz fixed frame; they
    # are not used by the estimator or by the native odometry comparison.
    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(package_share / "launch" / "fastlio2_native.launch.py")
            ),
            launch_arguments={
                "use_sim_time": "true",
                "config_path": LaunchConfiguration(
                    "fastlio_config_path"
                ),
                "raw_cloud_topic": "/lidar/points_raw",
                "fastlio_cloud_topic": "/fastlio/points",
                "candidate_odom_topic": "/Odometry",
                "enable_visual_outputs": "true",
                "point_density": point_density,
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
            }.items(),
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="fastlio_compare_map_to_camera_init",
            arguments=[
                "--x",
                "0.0",
                "--y",
                "0.0",
                "--z",
                "0.0",
                "--roll",
                "0.0",
                "--pitch",
                "0.0",
                "--yaw",
                "0.0",
                "--frame-id",
                "map",
                "--child-frame-id",
                "camera_init",
            ],
            parameters=[{"use_sim_time": True}],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="fastlio_compare_body_to_base_link",
            arguments=[
                "--x",
                "0.0",
                "--y",
                "0.0",
                "--z",
                "0.0",
                "--roll",
                "0.0",
                "--pitch",
                "0.0",
                "--yaw",
                "0.0",
                "--frame-id",
                "body",
                "--child-frame-id",
                "base_link",
            ],
            parameters=[{"use_sim_time": True}],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="fastlio_compare_base_to_lidar",
            arguments=[
                "--x",
                "0.20",
                "--y",
                "0.0",
                "--z",
                "0.35",
                "--roll",
                "0.0",
                "--pitch",
                "0.0",
                "--yaw",
                "0.0",
                "--frame-id",
                "base_link",
                "--child-frame-id",
                "lidar_link",
            ],
            parameters=[{"use_sim_time": True}],
        ),
    ]
    if compare_use_rviz == "true":
        rviz_config = Path(
            get_package_share_directory("fast_lio")
        ) / "rviz_cfg" / "fastlio.rviz"
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="fastlio2_official_rviz",
                arguments=["-d", str(rviz_config)],
                parameters=[{"use_sim_time": True}],
                output="screen",
            )
        )
    return actions


def _simulation_action(context, *_) -> list[object]:
    project_root = LaunchConfiguration("project_root").perform(context)
    isaaclab_root = LaunchConfiguration("isaaclab_root").perform(context)
    device = LaunchConfiguration("device").perform(context)
    steps = LaunchConfiguration("simulation_steps").perform(context)
    factory_usd_path = LaunchConfiguration("factory_usd_path").perform(context)
    spawn_x = LaunchConfiguration("spawn_x").perform(context)
    spawn_y = LaunchConfiguration("spawn_y").perform(context)
    spawn_yaw = LaunchConfiguration("spawn_yaw").perform(context)
    headless = LaunchConfiguration("headless").perform(context).lower()

    command = [
        os.path.join(isaaclab_root, "isaaclab.sh"),
        "-p",
        os.path.join(project_root, "scripts", "validation", "validate_ros2_bridge.py"),
        "--device",
        device,
        "--steps",
        steps,
        "--real-time",
        "--external-control",
        "--disable-episode-timeout",
        "--enhanced-determinism",
        "--enable-lio-sam",
        "--imu-observation-parity-atol",
        "0.01",
        "--factory-usd-path",
        factory_usd_path,
        "--spawn-x",
        spawn_x,
        "--spawn-y",
        spawn_y,
        "--spawn-yaw",
        spawn_yaw,
        "--locomotion-profile",
        "slam_backend_compare",
    ]
    if headless == "true":
        command.append("--headless")

    simulation = ExecuteProcess(
        cmd=command,
        output="screen",
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
                            reason="SLAM backend comparison simulation exited"
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
    default_isaaclab_root = Path(
        os.environ.get("ISAACLAB_ROOT", Path.home() / "IsaacLab")
    ).expanduser()
    local_cyclonedds_uri = (
        "file://" + str(package_share / "config" / "cyclonedds_local.xml")
    )
    static_transform_cyclonedds_uri = (
        "file://"
        + str(package_share / "config" / "cyclonedds_static_tf.xml")
    )

    project_root = LaunchConfiguration("project_root")
    output_dir = LaunchConfiguration("output_dir")

    policy_node = Node(
        package="anymal_locomotion_ros2",
        executable="policy_node",
        name="anymal_model1450_motion_driver",
        parameters=[
            {
                "use_sim_time": True,
                "backend": "onnx",
                "policy_path": LaunchConfiguration("policy_path"),
                "metadata_path": LaunchConfiguration("metadata_path"),
                # Deliberately use simulator GT for motion generation.  The
                # selected SLAM backend is observation-only in this launch.
                "odometry_topic": "/odom",
                "expected_odometry_child_frame": "base_link",
                "inference_trigger": "synchronized_state",
                "state_timeout_s": 0.25,
            }
        ],
        condition=IfCondition(LaunchConfiguration("use_motion_driver")),
        output="screen",
    )

    bag = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-o",
            PathJoinSubstitution([output_dir, "raw_bag"]),
            "--disable-keyboard-controls",
            "/clock",
            "/odom",
            "/imu/data",
            "/lidar/points_raw",
            "/lio_sam/mapping/odometry",
            "/Odometry",
        ],
        condition=IfCondition(LaunchConfiguration("record_bag")),
        output="screen",
    )

    prepare_output = ExecuteProcess(
        cmd=["mkdir", "-p", output_dir],
        output="log",
    )
    roudi = ExecuteProcess(
        cmd=[FindExecutable(name="iox-roudi"), "--log-level", "warning"],
        output="log",
    )
    teleop = ExecuteProcess(
        cmd=[
            "gnome-terminal",
            "--wait",
            "--title=ANYmal-D SLAM backend comparison teleop",
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
            LaunchConfiguration("open_teleop_terminal")
        ),
        output="screen",
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
            DeclareLaunchArgument("slam_backend", default_value="liosam"),
            DeclareLaunchArgument("deskew_mode", default_value="native"),
            DeclareLaunchArgument(
                "point_density",
                default_value="1.0",
                description="Deterministic fraction of raw scan points retained",
            ),
            DeclareLaunchArgument(
                "fastlio_blind",
                default_value="0.5",
                description="FAST-LIO2 near-range exclusion in metres",
            ),
            DeclareLaunchArgument(
                "fastlio_point_filter_num",
                default_value="2",
                description="FAST-LIO2 native input point stride",
            ),
            DeclareLaunchArgument(
                "fastlio_max_iteration",
                default_value="4",
                description="FAST-LIO2 maximum EKF update iterations",
            ),
            DeclareLaunchArgument(
                "fastlio_filter_size_surf",
                default_value="0.3",
                description="FAST-LIO2 surface voxel size in metres",
            ),
            DeclareLaunchArgument(
                "fastlio_filter_size_map",
                default_value="0.6",
                description="FAST-LIO2 map voxel size in metres",
            ),
            DeclareLaunchArgument(
                "fastlio_cube_side_length",
                default_value="1000.0",
                description="FAST-LIO2 local map cube side length in metres",
            ),
            DeclareLaunchArgument(
                "fastlio_acc_cov",
                default_value="0.1",
                description="FAST-LIO2 accelerometer process covariance",
            ),
            DeclareLaunchArgument(
                "fastlio_gyr_cov",
                default_value="0.1",
                description="FAST-LIO2 gyroscope process covariance",
            ),
            DeclareLaunchArgument(
                "fastlio_b_acc_cov",
                default_value="0.0001",
                description="FAST-LIO2 accelerometer-bias process covariance",
            ),
            DeclareLaunchArgument(
                "fastlio_b_gyr_cov",
                default_value="0.0001",
                description="FAST-LIO2 gyroscope-bias process covariance",
            ),
            DeclareLaunchArgument(
                "fastlio_time_direction",
                default_value="clockwise",
                description="Diagnostic RTX azimuth-to-time direction",
            ),
            DeclareLaunchArgument(
                "fastlio_time_source",
                default_value="sensor_order",
                description="FAST-LIO2 reconstructed Ouster point-time source",
            ),
            DeclareLaunchArgument(
                "fastlio_point_order",
                default_value="staggered",
                description=(
                    "FAST-LIO2 point packing; destaggered is an explicit A/B "
                    "mode"
                ),
            ),
            DeclareLaunchArgument(
                "compare_use_rviz",
                default_value="true",
                description="Open the common RViz view for the selected backend",
            ),
            DeclareLaunchArgument(
                "open_teleop_terminal",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "use_motion_driver",
                default_value="true",
                description=(
                    "Use formal model1450 with GT /odom only to move the "
                    "simulator; selected SLAM never feeds this policy"
                ),
            ),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument(
                "simulation_steps",
                default_value="1000000",
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value=PathJoinSubstitution(
                    [project_root, "logs", "slam_backend_compare", "latest"]
                ),
            ),
            DeclareLaunchArgument(
                "record_bag",
                default_value="false",
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
            DeclareLaunchArgument("spawn_x", default_value="0.0"),
            DeclareLaunchArgument("spawn_y", default_value="-18.0"),
            DeclareLaunchArgument("spawn_yaw", default_value="0.0"),
            DeclareLaunchArgument(
                "fastlio_config_path",
                default_value=PathJoinSubstitution(
                    [
                        package_share,
                        "config",
                        "fastlio2_anymal_ouster32.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "static_transform_cyclonedds_uri",
                default_value=static_transform_cyclonedds_uri,
            ),
            SetEnvironmentVariable(
                "ROS_DOMAIN_ID",
                LaunchConfiguration("ros_domain_id"),
            ),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION",
                LaunchConfiguration("rmw_implementation"),
            ),
            SetEnvironmentVariable("CYCLONEDDS_URI", local_cyclonedds_uri),
            SetEnvironmentVariable("TERM", "xterm-256color"),
            SetEnvironmentVariable("PYTHONPATH", project_python_path),
            prepare_output,
            roudi,
            TimerAction(
                period=1.0,
                actions=[
                    policy_node,
                    OpaqueFunction(function=_backend_actions),
                    teleop,
                    bag,
                    TimerAction(
                        period=2.0,
                        actions=[OpaqueFunction(function=_simulation_action)],
                    ),
                ],
            ),
        ]
    )
