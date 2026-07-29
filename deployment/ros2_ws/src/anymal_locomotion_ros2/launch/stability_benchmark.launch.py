"""Run one isolated deterministic Factory locomotion stability benchmark."""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import (
    EnvironmentVariable,
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
                    "stability benchmark driver failed with "
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
    policy_parity_path = LaunchConfiguration("policy_parity_path")
    metadata_path = LaunchConfiguration("metadata_path")
    factory_usd_path = LaunchConfiguration("factory_usd_path")
    factory_friction = LaunchConfiguration("factory_friction")
    enhanced_determinism = LaunchConfiguration("enhanced_determinism")

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
    benchmark_node = Node(
        package="anymal_locomotion_ros2",
        executable="stability_benchmark",
        name="anymal_stability_benchmark",
        parameters=[
            {
                "use_sim_time": True,
                "profile": profile,
                "project_root": project_root,
                "output_path": PathJoinSubstitution(
                    [output_dir, "driver.json"]
                ),
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
            "--validate-observation-parity",
            "--policy-parity-artifact",
            policy_parity_path,
            "--disable-episode-timeout",
            "--enhanced-determinism",
            "--enhanced-determinism-value",
            enhanced_determinism,
            "--imu-observation-parity-atol",
            "0.01",
            "--factory-usd-path",
            factory_usd_path,
            "--factory-friction",
            factory_friction,
            "--locomotion-diagnostics-output",
            PathJoinSubstitution([output_dir, "locomotion_diagnostics.json"]),
            "--locomotion-profile",
            profile,
            "--benchmark-completion-file",
            PathJoinSubstitution([output_dir, "driver.json"]),
        ],
        output="log",
        # Kit normally needs more than launch's five-second default to flush
        # extensions and close the PhysX scene after SIGINT.
        sigterm_timeout="30",
        sigkill_timeout="10",
    )
    prepare_output = ExecuteProcess(
        cmd=["mkdir", "-p", output_dir],
        output="screen",
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
                        "logs",
                        "stability_benchmarks",
                        "latest",
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
                "policy_parity_path",
                default_value=PathJoinSubstitution(
                    [
                        project_root,
                        "exported",
                        "anymal_d_locomotion_v1",
                        "recovery_v0.4.0",
                        "policy.pt",
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
            DeclareLaunchArgument(
                "enhanced_determinism",
                default_value="true",
            ),
            SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION",
                "rmw_cyclonedds_cpp",
            ),
            SetEnvironmentVariable("TERM", "xterm-256color"),
            SetEnvironmentVariable("PYTHONPATH", project_python_path),
            prepare_output,
            policy_node,
            benchmark_node,
            TimerAction(period=2.0, actions=[simulation]),
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
                                reason="stability simulation completed",
                            )
                        )
                    ],
                )
            ),
        ]
    )
