"""Start the complete Factory locomotion and LIO-SAM stack."""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node


def _find_project_root(package_share: Path) -> Path:
    """Resolve the repository root from either a source or colcon install path."""
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
        "Cannot locate the anymal_locomotion repository from the installed "
        f"package share: {package_share}"
    )


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("anymal_locomotion_ros2"))
    detected_project_root = _find_project_root(package_share)
    local_cyclonedds_uri = (
        "file://" + str(package_share / "config" / "cyclonedds_local.xml")
    )
    static_transform_cyclonedds_uri = (
        "file://" + str(package_share / "config" / "cyclonedds_static_tf.xml")
    )
    default_isaaclab_root = Path(
        os.environ.get("ISAACLAB_ROOT", Path.home() / "IsaacLab")
    ).expanduser()

    project_root = LaunchConfiguration("project_root")
    isaaclab_root = LaunchConfiguration("isaaclab_root")
    device = LaunchConfiguration("device")
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    policy_path = LaunchConfiguration("policy_path")
    metadata_path = LaunchConfiguration("metadata_path")
    factory_usd_path = LaunchConfiguration("factory_usd_path")
    simulation_steps = LaunchConfiguration("simulation_steps")
    use_rviz = LaunchConfiguration("use_rviz")
    open_teleop_terminal = LaunchConfiguration("open_teleop_terminal")
    enable_locomotion_diagnostics = LaunchConfiguration(
        "enable_locomotion_diagnostics"
    )
    enable_slam_confidence = LaunchConfiguration("enable_slam_confidence")
    controlled_episode_reset_step = LaunchConfiguration(
        "controlled_episode_reset_step"
    )
    locomotion_diagnostics_dir = LaunchConfiguration(
        "locomotion_diagnostics_dir"
    )

    project_python_path = [
        PathJoinSubstitution([project_root, "deployment", "python_vendor"]),
        ":",
        PathJoinSubstitution([project_root, "source", "anymal_locomotion"]),
        ":",
        EnvironmentVariable("PYTHONPATH", default_value=""),
    ]

    policy_parameters = {
        "use_sim_time": True,
        "backend": "onnx",
        "policy_path": policy_path,
        "metadata_path": metadata_path,
    }
    policy_node = Node(
        package="anymal_locomotion_ros2",
        executable="policy_node",
        name="anymal_policy",
        parameters=[policy_parameters],
        condition=UnlessCondition(enable_locomotion_diagnostics),
        output="screen",
    )
    diagnostic_policy_node = Node(
        package="anymal_locomotion_ros2",
        executable="policy_node",
        name="anymal_policy",
        parameters=[
            {
                **policy_parameters,
                "diagnostics_path": PathJoinSubstitution(
                    [locomotion_diagnostics_dir, "policy_diagnostics.json"]
                ),
            }
        ],
        condition=IfCondition(enable_locomotion_diagnostics),
        output="screen",
    )

    lio_sam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(package_share / "launch" / "lio_sam.launch.py")
        ),
        launch_arguments={
            "use_rviz": use_rviz,
            "enable_confidence": enable_slam_confidence,
            "use_motion_deskew": "false",
            "feature_cloud_info_topic": "/lio_sam/deskew/cloud_info",
            "motion_deskew_apply_translation": "true",
            "motion_deskew_replace_upstream_rotation": "true",
            "static_transform_cyclonedds_uri": (
                static_transform_cyclonedds_uri
            ),
        }.items(),
    )

    simulation_command = [
        PathJoinSubstitution([isaaclab_root, "isaaclab.sh"]),
        "-p",
        PathJoinSubstitution(
            [project_root, "scripts", "validation", "validate_ros2_bridge.py"]
        ),
        "--device",
        device,
        "--steps",
        simulation_steps,
        "--real-time",
        "--external-control",
        "--disable-episode-timeout",
        "--enhanced-determinism",
        "--enable-lio-sam",
        "--imu-observation-parity-atol",
        "0.01",
        "--controlled-episode-reset-step",
        controlled_episode_reset_step,
        "--factory-usd-path",
        factory_usd_path,
    ]
    simulation = ExecuteProcess(
        cmd=simulation_command,
        condition=UnlessCondition(enable_locomotion_diagnostics),
        output="screen",
    )
    diagnostic_simulation = ExecuteProcess(
        cmd=[
            *simulation_command,
            "--locomotion-diagnostics-output",
            PathJoinSubstitution(
                [locomotion_diagnostics_dir, "locomotion_diagnostics.json"]
            ),
            "--locomotion-profile",
            "formal_bringup",
        ],
        condition=IfCondition(enable_locomotion_diagnostics),
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

    teleop = ExecuteProcess(
        cmd=[
            "gnome-terminal",
            "--wait",
            "--title=ANYmal-D teleop_twist_keyboard",
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
        condition=IfCondition(open_teleop_terminal),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "project_root",
                default_value=str(detected_project_root),
                description="Auto-detected ANYmal locomotion repository root",
            ),
            DeclareLaunchArgument(
                "isaaclab_root",
                default_value=str(default_isaaclab_root),
                description=(
                    "Read-only Isaac Lab root; defaults to ISAACLAB_ROOT or "
                    "~/IsaacLab"
                ),
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cuda:0",
                description="Isaac Lab simulation device",
            ),
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value=EnvironmentVariable(
                    "ROS_DOMAIN_ID",
                    default_value="1",
                ),
                description=(
                    "ROS 2 DDS domain shared by the complete stack; inherits the "
                    "launching shell and falls back to 1"
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
                description="Exported locomotion ONNX policy",
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
                description="Policy metadata contract",
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
                description="Project-local Factory USD terrain",
            ),
            DeclareLaunchArgument(
                "simulation_steps",
                default_value="1000000",
                description=(
                    "Number of Isaac Sim policy steps; use a finite value for "
                    "clean runtime regressions"
                ),
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Open RViz2 with the LIO-SAM view",
            ),
            DeclareLaunchArgument(
                "open_teleop_terminal",
                default_value="true",
                description="Open the official teleop_twist_keyboard in GNOME Terminal",
            ),
            DeclareLaunchArgument(
                "enable_locomotion_diagnostics",
                default_value="false",
                description=(
                    "Record project-local locomotion and policy traces; disabled "
                    "by default"
                ),
            ),
            DeclareLaunchArgument(
                "enable_slam_confidence",
                default_value="false",
                description=(
                    "Enable fail-closed LIO-SAM confidence instrumentation; "
                    "policy does not consume it"
                ),
            ),
            DeclareLaunchArgument(
                "controlled_episode_reset_step",
                default_value="-1",
                description=(
                    "Run one explicit simulator episode reset before this "
                    "0-based policy step; -1 disables the reset regression"
                ),
            ),
            DeclareLaunchArgument(
                "locomotion_diagnostics_dir",
                default_value=PathJoinSubstitution(
                    [
                        project_root,
                        "logs",
                        "formal_bringup",
                        "latest",
                    ]
                ),
                description=(
                    "Project-local output directory used only when locomotion "
                    "diagnostics are enabled"
                ),
            ),
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION",
                "rmw_cyclonedds_cpp",
            ),
            SetEnvironmentVariable("CYCLONEDDS_URI", local_cyclonedds_uri),
            SetEnvironmentVariable("TERM", "xterm-256color"),
            SetEnvironmentVariable("PYTHONPATH", project_python_path),
            roudi,
            # CycloneDDS creates Iceoryx clients during participant startup.
            # Start the full stack only after RouDi is ready so multi-megabyte
            # LiDAR scans never silently fall back to fragmented UDP.
            TimerAction(
                period=1.0,
                actions=[
                    policy_node,
                    diagnostic_policy_node,
                    lio_sam,
                    teleop,
                    TimerAction(
                        period=2.0,
                        actions=[simulation, diagnostic_simulation],
                    ),
                ],
            ),
        ]
    )
