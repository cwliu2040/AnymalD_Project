"""Launch FAST-LIO2 with the native candidate odometry contract.

This launch intentionally does not start ``fastlio_odom_adapter``.  The
native ``/Odometry`` message and ``camera_init -> body`` TF are the inputs used
by the first backend-only comparison.  The project adapter belongs to a later
locomotion integration launch.
"""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = Path(
        get_package_share_directory("anymal_locomotion_ros2")
    )
    default_config = package_share / "config" / "fastlio2_anymal_ouster32.yaml"

    use_sim_time = LaunchConfiguration("use_sim_time")
    config_path = LaunchConfiguration("config_path")
    raw_cloud_topic = LaunchConfiguration("raw_cloud_topic")
    fastlio_cloud_topic = LaunchConfiguration("fastlio_cloud_topic")
    candidate_odom_topic = LaunchConfiguration("candidate_odom_topic")
    enable_visual_outputs = LaunchConfiguration("enable_visual_outputs")
    enable_effect_diagnostics = LaunchConfiguration(
        "enable_effect_diagnostics"
    )
    point_density = LaunchConfiguration("point_density")

    point_adapter = Node(
        package="anymal_locomotion_ros2",
        executable="fastlio_point_adapter",
        name="anymal_fastlio_point_adapter",
        parameters=[
            {
                "use_sim_time": ParameterValue(
                    use_sim_time,
                    value_type=bool,
                ),
                "input_topic": raw_cloud_topic,
                "output_topic": fastlio_cloud_topic,
                "frame_id": "lidar_link",
                "scan_rate_hz": 10.0,
                "raw_stamp_is_scan_end": True,
                "point_density": ParameterValue(
                    point_density,
                    value_type=float,
                ),
            }
        ],
        output="screen",
    )

    # The candidate config carries the sensor-specific values, including the
    # selected native point_filter_num=2.  Do not add algorithm tuning here.
    fastlio = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fastlio_mapping",
        parameters=[
            config_path,
            {
                "use_sim_time": ParameterValue(
                    use_sim_time,
                    value_type=bool,
                ),
                # These are output-only visualization switches.  The native
                # benchmark passes false; the interactive launch passes true.
                "publish.path_en": ParameterValue(
                    enable_visual_outputs,
                    value_type=bool,
                ),
                "publish.scan_publish_en": ParameterValue(
                    enable_visual_outputs,
                    value_type=bool,
                ),
                "publish.scan_bodyframe_pub_en": ParameterValue(
                    enable_visual_outputs,
                    value_type=bool,
                ),
                "publish.effect_en": ParameterValue(
                    enable_effect_diagnostics,
                    value_type=bool,
                ),
            },
        ],
        # Keep the candidate's native TF visible for the backend-only view.
        # No project canonical TF or odometry message is published here.
        remappings=[
            ("/Odometry", candidate_odom_topic),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "config_path",
                default_value=str(default_config),
                description="Project-owned sensor config for FAST-LIO2",
            ),
            DeclareLaunchArgument(
                "raw_cloud_topic",
                default_value="/lidar/points_raw",
            ),
            DeclareLaunchArgument(
                "fastlio_cloud_topic",
                default_value="/fastlio/points",
            ),
            DeclareLaunchArgument(
                "candidate_odom_topic",
                default_value="/Odometry",
            ),
            DeclareLaunchArgument(
                "point_density",
                default_value="1.0",
                description="Deterministic fraction of raw scan points retained",
            ),
            DeclareLaunchArgument(
                "enable_visual_outputs",
                default_value="false",
                description=(
                    "Enable candidate path and registered-cloud publishers; "
                    "does not change estimator inputs"
                ),
            ),
            DeclareLaunchArgument(
                "enable_effect_diagnostics",
                default_value="false",
                description="Publish native effective points for diagnostics",
            ),
            point_adapter,
            fastlio,
        ]
    )
