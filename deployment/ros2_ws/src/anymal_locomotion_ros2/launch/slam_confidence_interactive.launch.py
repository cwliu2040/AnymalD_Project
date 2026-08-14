"""Interactive confidence-aware PPO session for either native SLAM backend."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


_CALIBRATION_IDS = {
    "fastlio2": "native-v1-1e6cf8347be1",
    "liosam": "native-v1-edc098b0bd98",
}


def _interactive_session(context, *_) -> list[object]:
    package_share = Path(
        get_package_share_directory("anymal_locomotion_ros2")
    )
    backend = LaunchConfiguration("slam_backend").perform(context).strip()
    if backend not in _CALIBRATION_IDS:
        raise RuntimeError(
            "slam_backend must be 'fastlio2' or 'liosam', "
            f"received {backend!r}"
        )

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(
                    package_share
                    / "launch"
                    / "fastlio2_locomotion_benchmark.launch.py"
                )
            ),
            launch_arguments={
                "slam_backend": backend,
                "expected_confidence_backend": backend,
                "expected_calibration_id": _CALIBRATION_IDS[backend],
                "enable_confidence": "true",
                "enable_velocity_estimator": "true",
                "velocity_estimator_metadata_path": LaunchConfiguration(
                    "velocity_estimator_metadata_path"
                ),
                "policy_odometry_topic": "/locomotion/estimated_odom",
                "policy_inference_trigger": "estimator_joint_state",
                "policy_path": LaunchConfiguration("policy_path"),
                "metadata_path": LaunchConfiguration("metadata_path"),
                "interactive": "true",
                "headless": LaunchConfiguration("headless"),
                "use_rviz": LaunchConfiguration("use_rviz"),
                "open_teleop_terminal": LaunchConfiguration(
                    "open_teleop_terminal"
                ),
                "profile": "interactive_manual",
                "simulation_steps": LaunchConfiguration("simulation_steps"),
                "output_dir": LaunchConfiguration("output_dir"),
            }.items(),
        )
    ]


def generate_launch_description() -> LaunchDescription:
    package_share = Path(
        get_package_share_directory("anymal_locomotion_ros2")
    )
    project_root = package_share
    for candidate in (package_share, *package_share.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "deployment" / "ros2_ws").is_dir()
        ):
            project_root = candidate
            break

    candidate_export = (
        project_root
        / "exported"
        / "anymal_d_locomotion_slam_confidence_phase_separated_gait_v1"
        / "2026-08-13_18-18-26_nonnegative_smoothing_recovery_safe_ppo_gait"
    )
    estimator_export = (
        project_root
        / "exported"
        / "proprioceptive_velocity_estimator"
        / "v1"
        / "clean_lateral_transition_candidate_15"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "slam_backend",
                default_value="fastlio2",
                choices=["fastlio2", "liosam"],
            ),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "open_teleop_terminal", default_value="true"
            ),
            DeclareLaunchArgument(
                "simulation_steps", default_value="1000000"
            ),
            DeclareLaunchArgument(
                "policy_path",
                default_value=str(candidate_export / "policy.onnx"),
            ),
            DeclareLaunchArgument(
                "metadata_path",
                default_value=str(candidate_export / "policy_metadata.yaml"),
            ),
            DeclareLaunchArgument(
                "velocity_estimator_metadata_path",
                default_value=str(
                    estimator_export / "velocity_estimator_metadata.json"
                ),
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value=PathJoinSubstitution(
                    [
                        str(project_root),
                        "logs",
                        "slam_confidence_interactive",
                        LaunchConfiguration("slam_backend"),
                    ]
                ),
            ),
            OpaqueFunction(function=_interactive_session),
        ]
    )
