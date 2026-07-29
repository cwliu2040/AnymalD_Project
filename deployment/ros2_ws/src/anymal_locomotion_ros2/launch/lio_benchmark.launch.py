"""Run one isolated, deterministic Factory LIO-SAM benchmark."""

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


def _shutdown_if_benchmark_failed(event, _context):
    if event.returncode == 0:
        return []
    return [
        EmitEvent(
            event=Shutdown(
                reason=(
                    "LIO benchmark driver failed with "
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
    detected_project_root = _find_project_root(package_share)
    local_cyclonedds_uri = (
        "file://"
        + str(package_share / "config" / "cyclonedds_local.xml")
    )
    default_isaaclab_root = Path(
        os.environ.get("ISAACLAB_ROOT", Path.home() / "IsaacLab")
    ).expanduser()

    project_root = LaunchConfiguration("project_root")
    isaaclab_root = LaunchConfiguration("isaaclab_root")
    device = LaunchConfiguration("device")
    ros_domain_id = LaunchConfiguration("ros_domain_id")
    profile = LaunchConfiguration("profile")
    output_dir = LaunchConfiguration("output_dir")
    policy_path = LaunchConfiguration("policy_path")
    metadata_path = LaunchConfiguration("metadata_path")
    factory_usd_path = LaunchConfiguration("factory_usd_path")
    record_bag = LaunchConfiguration("record_bag")
    motion_deskew_apply_translation = LaunchConfiguration(
        "motion_deskew_apply_translation"
    )
    motion_deskew_replace_upstream_rotation = LaunchConfiguration(
        "motion_deskew_replace_upstream_rotation"
    )
    loop_closure_enable = LaunchConfiguration("loop_closure_enable")
    loop_closure_expectation = LaunchConfiguration(
        "loop_closure_expectation"
    )
    loop_search_radius = LaunchConfiguration("loop_search_radius")
    loop_search_time_diff = LaunchConfiguration("loop_search_time_diff")
    loop_search_keyframes = LaunchConfiguration("loop_search_keyframes")
    loop_fitness_score = LaunchConfiguration("loop_fitness_score")

    project_python_path = [
        PathJoinSubstitution([project_root, "deployment", "python_vendor"]),
        ":",
        PathJoinSubstitution([project_root, "source", "anymal_locomotion"]),
        ":",
        EnvironmentVariable("PYTHONPATH", default_value=""),
    ]

    policy_node = Node(
        package="anymal_locomotion_ros2",
        executable="policy_node",
        name="anymal_policy",
        parameters=[
            {
                "use_sim_time": True,
                "backend": "onnx",
                "policy_path": policy_path,
                "metadata_path": metadata_path,
                "diagnostics_path": PathJoinSubstitution(
                    [output_dir, "policy_diagnostics.json"]
                ),
            }
        ],
        output="screen",
    )
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
            "motion_deskew_apply_translation": (
                motion_deskew_apply_translation
            ),
            "motion_deskew_replace_upstream_rotation": (
                motion_deskew_replace_upstream_rotation
            ),
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
    benchmark_node = Node(
        package="anymal_locomotion_ros2",
        executable="lio_benchmark",
        name="anymal_lio_benchmark",
        parameters=[
            {
                "use_sim_time": True,
                "profile": profile,
                "project_root": project_root,
                "output_path": PathJoinSubstitution(
                    [output_dir, "metrics.json"]
                ),
                "loop_closure_expectation": loop_closure_expectation,
            }
        ],
        output="screen",
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
            "4000",
            "--real-time",
            "--external-control",
            "--disable-episode-timeout",
            "--enhanced-determinism",
            "--enable-lio-sam",
            "--imu-observation-parity-atol",
            "0.01",
            "--factory-usd-path",
            factory_usd_path,
            "--locomotion-diagnostics-output",
            PathJoinSubstitution(
                [output_dir, "locomotion_diagnostics.json"]
            ),
            "--locomotion-profile",
            profile,
            "--benchmark-completion-file",
            PathJoinSubstitution([output_dir, "metrics.json"]),
        ],
        output="log",
        # Kit normally needs more than launch's five-second default to flush
        # extensions and close the PhysX scene after SIGINT.
        sigterm_timeout="30",
        sigkill_timeout="10",
    )
    bag = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-o",
            PathJoinSubstitution([output_dir, "bag"]),
            "/clock",
            "/odom",
            "/imu/data",
            "/lidar/points_raw",
        ],
        condition=IfCondition(record_bag),
        output="screen",
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
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "project_root",
                default_value=str(detected_project_root),
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
            DeclareLaunchArgument("profile", default_value="stationary"),
            DeclareLaunchArgument(
                "output_dir",
                default_value=PathJoinSubstitution(
                    [
                        project_root,
                        "outputs",
                        "lio_sam_benchmarks",
                        "latest",
                    ]
                ),
            ),
            DeclareLaunchArgument("record_bag", default_value="false"),
            DeclareLaunchArgument(
                "loop_closure_enable",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "loop_closure_expectation",
                default_value="disabled",
                description="disabled, required, or forbidden",
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
            DeclareLaunchArgument(
                "motion_deskew_apply_translation",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "motion_deskew_replace_upstream_rotation",
                default_value="true",
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
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION",
                "rmw_cyclonedds_cpp",
            ),
            # A complete benchmark starts more than CycloneDDS Humble's
            # default ten local participants. Keep the larger discovery range
            # scoped to this isolated launch instead of changing user or
            # deployment-wide middleware settings.
            SetEnvironmentVariable(
                "CYCLONEDDS_URI",
                local_cyclonedds_uri,
            ),
            SetEnvironmentVariable("TERM", "xterm-256color"),
            SetEnvironmentVariable("PYTHONPATH", project_python_path),
            prepare_output,
            roudi,
            # CycloneDDS creates Iceoryx clients during participant startup.
            # Give RouDi a deterministic head start so the benchmark never
            # silently falls back to fragmented UDP for the large point clouds.
            TimerAction(
                period=1.0,
                actions=[
                    bag,
                    policy_node,
                    lio_sam,
                    benchmark_node,
                    TimerAction(period=2.0, actions=[simulation]),
                ],
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=benchmark_node,
                    on_exit=_shutdown_if_benchmark_failed,
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=simulation,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(
                                reason="LIO simulation completed",
                            )
                        )
                    ],
                )
            ),
        ]
    )
