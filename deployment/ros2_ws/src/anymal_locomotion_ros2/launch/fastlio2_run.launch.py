"""Direct interactive FAST-LIO2 launch using the candidate's own launch/RViz.

The upstream ``mapping_ouster64.launch.py`` starts FAST-LIO2 and its native
RViz configuration.  This project wrapper only supplies the simulated raw
LiDAR conversion, the simulator, and the GT-driven locomotion policy used to
move the robot.  It does not start benchmark or odometry adapter nodes.
"""

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
from launch_ros.parameter_descriptions import ParameterValue


def _find_project_root(package_share: Path) -> Path:
    for start in (Path(__file__).resolve(), package_share.resolve()):
        for candidate in (start, *start.parents):
            if (
                (candidate / "pyproject.toml").is_file()
                and (candidate / "deployment" / "ros2_ws").is_dir()
            ):
                return candidate
    raise RuntimeError("Cannot locate the anymal_locomotion repository")


def _simulation_action(context, *_) -> list[object]:
    project_root = LaunchConfiguration("project_root").perform(context)
    isaaclab_root = LaunchConfiguration("isaaclab_root").perform(context)
    command = [
        os.path.join(isaaclab_root, "isaaclab.sh"),
        "-p",
        os.path.join(
            project_root,
            "scripts",
            "validation",
            "validate_ros2_bridge.py",
        ),
        "--device",
        LaunchConfiguration("device").perform(context),
        "--steps",
        LaunchConfiguration("simulation_steps").perform(context),
        "--real-time",
        "--external-control",
        "--disable-episode-timeout",
        "--enhanced-determinism",
        "--enable-lio-sam",
        "--imu-observation-parity-atol",
        "0.01",
        "--factory-usd-path",
        LaunchConfiguration("factory_usd_path").perform(context),
        "--spawn-x",
        LaunchConfiguration("spawn_x").perform(context),
        "--spawn-y",
        LaunchConfiguration("spawn_y").perform(context),
        "--spawn-yaw",
        LaunchConfiguration("spawn_yaw").perform(context),
        "--locomotion-profile",
        "slam_backend_compare",
    ]
    if LaunchConfiguration("headless").perform(context).lower() == "true":
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
                        event=Shutdown(reason="FAST-LIO2 simulation exited")
                    )
                ],
            )
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    package_share = Path(
        get_package_share_directory("anymal_locomotion_ros2")
    )
    fastlio_share = Path(get_package_share_directory("fast_lio"))
    project_root_default = _find_project_root(package_share)
    isaaclab_root_default = Path(
        os.environ.get("ISAACLAB_ROOT", Path.home() / "IsaacLab")
    ).expanduser()
    cyclonedds_uri = "file://" + str(
        package_share / "config" / "cyclonedds_local.xml"
    )

    point_adapter = Node(
        package="anymal_locomotion_ros2",
        executable="fastlio_point_adapter",
        name="anymal_fastlio_point_adapter",
        parameters=[
            {
                "use_sim_time": True,
                "input_topic": "/lidar/points_raw",
                "output_topic": "/fastlio/points",
                "frame_id": "lidar_link",
                "scan_rate_hz": 10.0,
                "raw_stamp_is_scan_end": True,
                "time_source": LaunchConfiguration("time_source"),
                "point_order": LaunchConfiguration("point_order"),
                "point_density": ParameterValue(
                    LaunchConfiguration("point_density"),
                    value_type=float,
                ),
            }
        ],
        output="screen",
    )

    upstream_mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(fastlio_share / "launch" / "mapping_ouster64.launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
            "config_path": LaunchConfiguration("config_path"),
            "rviz": LaunchConfiguration("use_rviz"),
            "rviz_cfg": str(fastlio_share / "rviz_cfg" / "fastlio.rviz"),
            "feature_extract_enable": "false",
            "point_filter_num": LaunchConfiguration("point_filter_num"),
            "max_iteration": LaunchConfiguration("max_iteration"),
            "filter_size_surf": LaunchConfiguration("filter_size_surf"),
            "filter_size_map": LaunchConfiguration("filter_size_map"),
            "cube_side_length": LaunchConfiguration("cube_side_length"),
            "runtime_pos_log_enable": "false",
        }.items(),
    )

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
                "odometry_topic": "/odom",
                "expected_odometry_child_frame": "base_link",
                "inference_trigger": "synchronized_state",
                "state_timeout_s": 0.25,
            }
        ],
        output="screen",
    )

    roudi = ExecuteProcess(
        cmd=[FindExecutable(name="iox-roudi"), "--log-level", "warning"],
        output="log",
    )
    teleop = ExecuteProcess(
        cmd=[
            "gnome-terminal",
            "--wait",
            "--title=ANYmal-D FAST-LIO2 teleop",
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
        condition=IfCondition(LaunchConfiguration("open_teleop_terminal")),
        output="screen",
    )

    project_root = LaunchConfiguration("project_root")
    project_python_path = [
        PathJoinSubstitution(
            [project_root, "deployment", "python_vendor"]
        ),
        ":",
        PathJoinSubstitution(
            [project_root, "source", "anymal_locomotion"]
        ),
        ":",
        EnvironmentVariable("PYTHONPATH", default_value=""),
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "project_root", default_value=str(project_root_default)
            ),
            DeclareLaunchArgument(
                "isaaclab_root", default_value=str(isaaclab_root_default)
            ),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value=EnvironmentVariable(
                    "ROS_DOMAIN_ID", default_value="1"
                ),
            ),
            DeclareLaunchArgument("point_density", default_value="1.0"),
            DeclareLaunchArgument(
                "time_source",
                default_value="sensor_order",
                description="Official reconstructed Ouster column time",
            ),
            DeclareLaunchArgument(
                "point_order",
                default_value="staggered",
                description=(
                    "FAST-LIO2 input packing; staggered keeps the scan-end "
                    "point at the end of the cloud"
                ),
            ),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "point_filter_num",
                default_value="2",
                description="Native input point stride",
            ),
            DeclareLaunchArgument(
                "max_iteration",
                default_value="4",
                description="Maximum iterated EKF update iterations",
            ),
            DeclareLaunchArgument(
                "filter_size_surf",
                default_value="0.3",
                description="Surface feature voxel size in metres",
            ),
            DeclareLaunchArgument(
                "filter_size_map",
                default_value="0.6",
                description="Map voxel size in metres",
            ),
            DeclareLaunchArgument(
                "cube_side_length",
                default_value="1000.0",
                description="Local map cube side length in metres",
            ),
            DeclareLaunchArgument(
                "open_teleop_terminal", default_value="true"
            ),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument(
                "simulation_steps", default_value="1000000"
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value=PathJoinSubstitution(
                    [
                        package_share,
                        "config",
                        "fastlio2_anymal_ouster32_live.yaml",
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
            DeclareLaunchArgument("spawn_x", default_value="0.0"),
            DeclareLaunchArgument("spawn_y", default_value="-18.0"),
            DeclareLaunchArgument("spawn_yaw", default_value="0.0"),
            SetEnvironmentVariable(
                "ROS_DOMAIN_ID", LaunchConfiguration("ros_domain_id")
            ),
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"
            ),
            SetEnvironmentVariable("CYCLONEDDS_URI", cyclonedds_uri),
            SetEnvironmentVariable("TERM", "xterm-256color"),
            SetEnvironmentVariable("PYTHONPATH", project_python_path),
            roudi,
            TimerAction(
                period=1.0,
                actions=[
                    policy_node,
                    point_adapter,
                    upstream_mapping,
                    teleop,
                    TimerAction(
                        period=2.0,
                        actions=[OpaqueFunction(function=_simulation_action)],
                    ),
                ],
            ),
        ]
    )
