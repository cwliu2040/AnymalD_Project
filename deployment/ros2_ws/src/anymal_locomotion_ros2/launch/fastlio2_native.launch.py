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
    fastlio_blind = LaunchConfiguration("fastlio_blind")
    point_filter_num = LaunchConfiguration("point_filter_num")
    max_iteration = LaunchConfiguration("max_iteration")
    filter_size_surf = LaunchConfiguration("filter_size_surf")
    filter_size_map = LaunchConfiguration("filter_size_map")
    cube_side_length = LaunchConfiguration("cube_side_length")
    acc_cov = LaunchConfiguration("acc_cov")
    gyr_cov = LaunchConfiguration("gyr_cov")
    b_acc_cov = LaunchConfiguration("b_acc_cov")
    b_gyr_cov = LaunchConfiguration("b_gyr_cov")
    time_direction = LaunchConfiguration("time_direction")
    raw_stamp_is_scan_end = LaunchConfiguration("raw_stamp_is_scan_end")
    time_source = LaunchConfiguration("time_source")
    point_order = LaunchConfiguration("point_order")

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
                "raw_stamp_is_scan_end": ParameterValue(
                    raw_stamp_is_scan_end,
                    value_type=bool,
                ),
                "time_direction": time_direction,
                "time_source": time_source,
                "point_order": point_order,
                "point_density": ParameterValue(
                    point_density,
                    value_type=float,
                ),
            }
        ],
        output="screen",
    )

    # The config carries the sensor contract.  These explicit overrides are
    # calibration knobs only; they do not add a project deskew stage.
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
                    fastlio_blind,
                    value_type=float,
                ),
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
                "cube_side_length": ParameterValue(
                    cube_side_length,
                    value_type=float,
                ),
                "mapping.acc_cov": ParameterValue(
                    acc_cov,
                    value_type=float,
                ),
                "mapping.gyr_cov": ParameterValue(
                    gyr_cov,
                    value_type=float,
                ),
                "mapping.b_acc_cov": ParameterValue(
                    b_acc_cov,
                    value_type=float,
                ),
                "mapping.b_gyr_cov": ParameterValue(
                    b_gyr_cov,
                    value_type=float,
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
                "fastlio_blind",
                default_value="0.5",
                description="Near-range exclusion in metres",
            ),
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
                default_value="0.5",
                description="Surface feature voxel size in metres",
            ),
            DeclareLaunchArgument(
                "filter_size_map",
                default_value="0.5",
                description="Map voxel size in metres",
            ),
            DeclareLaunchArgument(
                "cube_side_length",
                default_value="200.0",
                description="Local map cube side length in metres",
            ),
            DeclareLaunchArgument(
                "acc_cov",
                default_value="0.1",
                description="FAST-LIO2 accelerometer process covariance",
            ),
            DeclareLaunchArgument(
                "gyr_cov",
                default_value="0.1",
                description="FAST-LIO2 gyroscope process covariance",
            ),
            DeclareLaunchArgument(
                "b_acc_cov",
                default_value="0.0001",
                description="FAST-LIO2 accelerometer-bias process covariance",
            ),
            DeclareLaunchArgument(
                "b_gyr_cov",
                default_value="0.0001",
                description="FAST-LIO2 gyroscope-bias process covariance",
            ),
            DeclareLaunchArgument(
                "time_direction",
                default_value="clockwise",
                description=(
                    "RTX azimuth-to-time direction; clockwise is the native "
                    "OS1 contract"
                ),
            ),
            DeclareLaunchArgument(
                "raw_stamp_is_scan_end",
                default_value="true",
                description=(
                    "Treat RTX scan-buffer header as scan end and publish "
                    "the corresponding scan-start header"
                ),
            ),
            DeclareLaunchArgument(
                "time_source",
                default_value="sensor_order",
                description=(
                    "Point time source; sensor_order is the official Ouster "
                    "column-time contract, sensor_order_fire_time and "
                    "azimuth are diagnostic only"
                ),
            ),
            DeclareLaunchArgument(
                "point_order",
                default_value="staggered",
                description=(
                    "Ouster PointCloud2 packing for FAST-LIO2: staggered "
                    "preserves the raw capture-column end point; "
                    "destaggered follows the official driver default and "
                    "column is a raw-order A/B mode"
                ),
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
