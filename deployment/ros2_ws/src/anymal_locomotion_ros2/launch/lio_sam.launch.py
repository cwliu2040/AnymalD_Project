"""Project-owned LIO-SAM bringup for the Isaac Sim ANYmal-D sensor contract."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("anymal_locomotion_ros2"))
    default_params = package_share / "config" / "lio_sam_params.yaml"
    params_file = LaunchConfiguration("params_file")
    use_rviz = LaunchConfiguration("use_rviz")

    lio_nodes = [
        Node(
            package="lio_sam",
            executable="lio_sam_imuPreintegration",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="lio_sam",
            executable="lio_sam_imageProjection",
            name="lio_sam_imageProjection",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="lio_sam",
            executable="lio_sam_featureExtraction",
            name="lio_sam_featureExtraction",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="lio_sam",
            executable="lio_sam_mapOptimization",
            name="lio_sam_mapOptimization",
            parameters=[params_file],
            # Upstream also broadcasts odom->lidar_link. The high-rate IMU
            # preintegration transform plus base_link->lidar_link is the single
            # authoritative project TF chain, so isolate the duplicate.
            remappings=[("/tf", "/lio_sam/map_optimization_tf")],
            output="screen",
        ),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=str(default_params),
                description="LIO-SAM parameter file",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="false",
                description="Start RViz2 with the upstream LIO-SAM view",
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="anymal_lidar_static_tf",
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
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="anymal_map_to_odom_static_tf",
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
                    "odom",
                ],
                parameters=[{"use_sim_time": True}],
            ),
            Node(
                package="anymal_locomotion_ros2",
                executable="lidar_point_adapter",
                name="anymal_lidar_point_adapter",
                parameters=[
                    {
                        "use_sim_time": True,
                        "input_topic": "/lidar/points_raw",
                        "output_topic": "/lio_sam/points",
                        "frame_id": "lidar_link",
                        "scan_rate_hz": 10.0,
                        "raw_stamp_is_scan_end": True,
                    }
                ],
                output="screen",
            ),
            *lio_nodes,
            Node(
                package="rviz2",
                executable="rviz2",
                name="lio_sam_rviz",
                arguments=[
                    "-d",
                    str(
                        Path(get_package_share_directory("lio_sam"))
                        / "config"
                        / "rviz2.rviz"
                    ),
                ],
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(use_rviz),
                output="screen",
            ),
        ]
    )
