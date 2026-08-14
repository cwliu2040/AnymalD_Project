"""Run the external FAST-LIO2 candidate on the project sensor contract."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
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
    effective_points_topic = LaunchConfiguration("effective_points_topic")
    enable_confidence = LaunchConfiguration("enable_confidence")
    enable_visual_outputs = LaunchConfiguration("enable_visual_outputs")
    confidence_config_path = LaunchConfiguration("confidence_config_path")
    confidence_artifact_path = LaunchConfiguration(
        "confidence_artifact_path"
    )
    blind = LaunchConfiguration("blind")
    point_filter_num = LaunchConfiguration("point_filter_num")
    max_iteration = LaunchConfiguration("max_iteration")
    filter_size_surf = LaunchConfiguration("filter_size_surf")
    filter_size_map = LaunchConfiguration("filter_size_map")
    cube_side_length = LaunchConfiguration("cube_side_length")
    point_density = LaunchConfiguration("point_density")
    point_density_profile = LaunchConfiguration("point_density_profile")
    point_density_min = LaunchConfiguration("point_density_min")
    time_source = LaunchConfiguration("time_source")
    point_order = LaunchConfiguration("point_order")
    time_sync_en = LaunchConfiguration("time_sync_en")
    time_offset_lidar_to_imu = LaunchConfiguration(
        "time_offset_lidar_to_imu"
    )

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
                "time_source": time_source,
                "point_order": point_order,
                "point_density": ParameterValue(
                    point_density,
                    value_type=float,
                ),
                "point_density_profile": point_density_profile,
                "point_density_min": ParameterValue(
                    point_density_min,
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
                "max_iteration": ParameterValue(
                    max_iteration,
                    value_type=int,
                ),
                "filter_size_surf": ParameterValue(
                    filter_size_surf,
                    value_type=float,
                ),
                "filter_size_map": ParameterValue(
                    filter_size_map,
                    value_type=float,
                ),
                "common.time_sync_en": ParameterValue(
                    time_sync_en,
                    value_type=bool,
                ),
                "common.time_offset_lidar_to_imu": ParameterValue(
                    time_offset_lidar_to_imu,
                    value_type=float,
                ),
                "cube_side_length": ParameterValue(
                    cube_side_length,
                    value_type=float,
                ),
                "runtime_pos_log_enable": False,
                "publish.effect_en": ParameterValue(
                    enable_confidence,
                    value_type=bool,
                ),
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
            },
        ],
        # Keep the candidate's camera_init/body TF out of the project's
        # canonical map/base_link TF tree until the backend passes its gate.
        remappings=[
            ("/Odometry", candidate_odom_topic),
            ("/cloud_effected", effective_points_topic),
            ("/tf", "/fastlio/tf"),
            ("/tf_static", "/fastlio/tf_static"),
        ],
        output="screen",
    )
    confidence = Node(
        package="anymal_locomotion_ros2",
        executable="fastlio_confidence_extractor",
        name="anymal_fastlio_confidence_extractor",
        parameters=[
            confidence_config_path,
            {
                "use_sim_time": ParameterValue(
                    use_sim_time,
                    value_type=bool,
                ),
                "native_odometry_topic": candidate_odom_topic,
                "canonical_odometry_topic": slam_odom_topic,
                "effective_points_topic": effective_points_topic,
                "lidar_input_topic": fastlio_cloud_topic,
                "calibration_artifact_path": confidence_artifact_path,
            },
        ],
        condition=IfCondition(enable_confidence),
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
                "effective_points_topic",
                default_value="/cloud_effected",
            ),
            DeclareLaunchArgument(
                "enable_confidence",
                default_value="false",
                description=(
                    "Enable fail-closed FAST-LIO2 confidence instrumentation "
                    "and its required effective-point publisher"
                ),
            ),
            DeclareLaunchArgument(
                "enable_visual_outputs",
                default_value="false",
                description=(
                    "Enable FAST-LIO2 registered-cloud and path publishers "
                    "for interactive RViz inspection"
                ),
            ),
            DeclareLaunchArgument(
                "confidence_config_path",
                default_value=str(
                    package_share / "config" / "slam_confidence_fastlio2.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "confidence_artifact_path",
                default_value=str(
                    package_share
                    / "config"
                    / "slam_confidence_fastlio2_native_v2.json"
                ),
                description="Validated native FAST-LIO2 confidence artifact",
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
                "max_iteration",
                default_value="4",
                description="Maximum iterated EKF update iterations",
            ),
            DeclareLaunchArgument(
                "filter_size_surf",
                default_value="0.3",
            ),
            DeclareLaunchArgument(
                "filter_size_map",
                default_value="0.6",
            ),
            DeclareLaunchArgument(
                "cube_side_length",
                default_value="1000.0",
                description="Local map cube side length in metres",
            ),
            DeclareLaunchArgument(
                "point_density",
                default_value="1.0",
                description="Deterministic fraction of raw scan points retained",
            ),
            DeclareLaunchArgument(
                "point_density_profile",
                default_value="constant",
            ),
            DeclareLaunchArgument("point_density_min", default_value="0.01"),
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
            DeclareLaunchArgument(
                "time_sync_en",
                default_value="false",
                description=(
                    "Use FAST-LIO2 native online LiDAR--IMU time sync"
                ),
            ),
            DeclareLaunchArgument(
                "time_offset_lidar_to_imu",
                default_value="0.0",
                description="Native LiDAR-to-IMU timestamp offset in seconds",
            ),
            point_adapter,
            fastlio,
            odom_adapter,
            confidence,
        ]
    )
