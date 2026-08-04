"""Run the external FAST-LIO2 candidate on the project sensor contract."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("anymal_locomotion_ros2"))
    default_config = package_share / "config" / "fastlio2_anymal_ouster32.yaml"

    use_sim_time = LaunchConfiguration("use_sim_time")
    config_path = LaunchConfiguration("config_path")
    raw_cloud_topic = LaunchConfiguration("raw_cloud_topic")
    fastlio_cloud_topic = LaunchConfiguration("fastlio_cloud_topic")
    candidate_odom_topic = LaunchConfiguration("candidate_odom_topic")
    slam_odom_topic = LaunchConfiguration("slam_odom_topic")
    blind = LaunchConfiguration("blind")
    point_filter_num = LaunchConfiguration("point_filter_num")
    filter_size_surf = LaunchConfiguration("filter_size_surf")
    filter_size_map = LaunchConfiguration("filter_size_map")
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
                "preprocess.blind": ParameterValue(
                    blind,
                    value_type=float,
                ),
                "feature_extract_enable": False,
                "point_filter_num": ParameterValue(
                    point_filter_num,
                    value_type=int,
                ),
                "max_iteration": 4,
                "filter_size_surf": ParameterValue(
                    filter_size_surf,
                    value_type=float,
                ),
                "filter_size_map": ParameterValue(
                    filter_size_map,
                    value_type=float,
                ),
                "cube_side_length": 200.0,
                "runtime_pos_log_enable": False,
            },
        ],
        # Keep the candidate's camera_init/body TF out of the project's
        # canonical map/base_link TF tree until the backend passes its gate.
        remappings=[
            ("/tf", "/fastlio/tf"),
            ("/tf_static", "/fastlio/tf_static"),
        ],
        output="screen",
    )
    odom_adapter = Node(
        package="anymal_locomotion_ros2",
        executable="fastlio_odom_adapter",
        name="anymal_fastlio_odom_adapter",
        parameters=[
            {
                "use_sim_time": ParameterValue(
                    use_sim_time,
                    value_type=bool,
                ),
                "source_topic": candidate_odom_topic,
                "output_topic": slam_odom_topic,
                "source_frame_id": "camera_init",
                "source_child_frame_id": "body",
                "output_frame_id": "map",
                "output_child_frame_id": "base_link",
                "derive_twist_from_pose": True,
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value=str(default_config),
                description="FAST-LIO2 candidate parameter file",
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
                "slam_odom_topic",
                default_value="/slam/odom",
                description=(
                    "Canonical experimental backend odometry; never /odom"
                ),
            ),
            DeclareLaunchArgument(
                "blind",
                default_value="0.5",
                description="FAST-LIO2 near-range exclusion in metres",
            ),
            DeclareLaunchArgument(
                "point_filter_num",
                default_value="2",
                description="FAST-LIO2 input point stride",
            ),
            DeclareLaunchArgument(
                "filter_size_surf",
                default_value="0.5",
            ),
            DeclareLaunchArgument(
                "filter_size_map",
                default_value="0.5",
            ),
            DeclareLaunchArgument(
                "point_density",
                default_value="1.0",
                description="Deterministic fraction of raw scan points retained",
            ),
            point_adapter,
            fastlio,
            odom_adapter,
        ]
    )
