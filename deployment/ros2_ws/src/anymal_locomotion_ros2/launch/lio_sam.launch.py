"""Project-owned LIO-SAM bringup for the Isaac Sim ANYmal-D sensor contract."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("anymal_locomotion_ros2"))
    default_params = package_share / "config" / "lio_sam_params.yaml"
    params_file = LaunchConfiguration("params_file")
    use_rviz = LaunchConfiguration("use_rviz")
    enable_confidence = LaunchConfiguration("enable_confidence")
    confidence_config_path = LaunchConfiguration("confidence_config_path")
    confidence_artifact_path = LaunchConfiguration(
        "confidence_artifact_path"
    )
    use_motion_deskew = LaunchConfiguration("use_motion_deskew")
    motion_deskew_apply_translation = LaunchConfiguration(
        "motion_deskew_apply_translation"
    )
    motion_deskew_replace_upstream_rotation = LaunchConfiguration(
        "motion_deskew_replace_upstream_rotation"
    )
    feature_cloud_info_topic = LaunchConfiguration(
        "feature_cloud_info_topic"
    )
    point_density = LaunchConfiguration("point_density")
    point_density_profile = LaunchConfiguration("point_density_profile")
    point_density_min = LaunchConfiguration("point_density_min")
    point_density_healthy_s = LaunchConfiguration("point_density_healthy_s")
    point_density_ramp_down_s = LaunchConfiguration("point_density_ramp_down_s")
    point_density_hold_s = LaunchConfiguration("point_density_hold_s")
    point_density_ramp_up_s = LaunchConfiguration("point_density_ramp_up_s")
    static_transform_cyclonedds_uri = LaunchConfiguration(
        "static_transform_cyclonedds_uri"
    )
    loop_closure_enable = LaunchConfiguration("loop_closure_enable")
    loop_closure_frequency = LaunchConfiguration("loop_closure_frequency")
    loop_search_radius = LaunchConfiguration("loop_search_radius")
    loop_search_time_diff = LaunchConfiguration("loop_search_time_diff")
    loop_search_keyframes = LaunchConfiguration("loop_search_keyframes")
    loop_fitness_score = LaunchConfiguration("loop_fitness_score")

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
            remappings=[
                (
                    "lio_sam/deskew/cloud_info",
                    feature_cloud_info_topic,
                )
            ],
            output="screen",
        ),
    ]
    map_optimization = Node(
        package="lio_sam",
        executable="lio_sam_mapOptimization",
        name="lio_sam_mapOptimization",
        parameters=[
            params_file,
            {
                "loopClosureEnableFlag": ParameterValue(
                    loop_closure_enable,
                    value_type=bool,
                ),
                "loopClosureFrequency": ParameterValue(
                    loop_closure_frequency,
                    value_type=float,
                ),
                "historyKeyframeSearchRadius": ParameterValue(
                    loop_search_radius,
                    value_type=float,
                ),
                "historyKeyframeSearchTimeDiff": ParameterValue(
                    loop_search_time_diff,
                    value_type=float,
                ),
                "historyKeyframeSearchNum": ParameterValue(
                    loop_search_keyframes,
                    value_type=int,
                ),
                "historyKeyframeFitnessScore": ParameterValue(
                    loop_fitness_score,
                    value_type=float,
                ),
            },
        ],
        # Upstream also broadcasts odom->lidar_link. The high-rate IMU
        # preintegration transform plus base_link->lidar_link is the single
        # authoritative project TF chain, so isolate the duplicate.
        remappings=[("/tf", "/lio_sam/map_optimization_tf")],
        output="screen",
    )
    odom_adapter = Node(
        package="anymal_locomotion_ros2",
        executable="liosam_odom_adapter",
        name="anymal_liosam_odom_adapter",
        parameters=[
            {
                "use_sim_time": True,
                "source_topic": "/lio_sam/mapping/odometry",
                "output_topic": "/slam/odom",
                "source_frame_id": "odom",
                "source_child_frame_id": "odom_mapping",
                "output_frame_id": "map",
                "output_child_frame_id": "base_link",
                "sensor_translation_in_body_xyz": [0.20, 0.0, 0.35],
                "derive_twist_from_pose": True,
                "policy_source_topic": "/lio_sam/odometry/imu",
                "policy_output_topic": "/slam/policy_odom",
                "policy_source_frame_id": "odom",
                "policy_source_child_frame_id": "odom_imu",
            }
        ],
        condition=IfCondition(enable_confidence),
        output="screen",
    )
    confidence = Node(
        package="anymal_locomotion_ros2",
        executable="liosam_confidence_extractor",
        name="anymal_liosam_confidence_extractor",
        parameters=[
            confidence_config_path,
            {
                "use_sim_time": True,
                "motion_deskew_required": ParameterValue(
                    use_motion_deskew,
                    value_type=bool,
                ),
                "calibration_artifact_path": confidence_artifact_path,
            },
        ],
        condition=IfCondition(enable_confidence),
        output="screen",
    )

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
            DeclareLaunchArgument(
                "enable_confidence",
                default_value="false",
                description=(
                    "Enable fail-closed LIO-SAM confidence instrumentation "
                    "and canonical /slam/odom adapter"
                ),
            ),
            DeclareLaunchArgument(
                "confidence_config_path",
                default_value=str(
                    package_share / "config" / "slam_confidence_liosam.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "confidence_artifact_path",
                default_value=str(
                    package_share
                    / "config"
                    / "slam_confidence_liosam_native_v3.json"
                ),
                description=(
                    "Validated native LIO-SAM confidence artifact; override "
                    "with an empty path for fail-closed capture/replay"
                ),
            ),
            DeclareLaunchArgument(
                "static_transform_cyclonedds_uri",
                default_value=EnvironmentVariable(
                    "CYCLONEDDS_URI",
                    default_value="",
                ),
                description=(
                    "Optional CycloneDDS URI used only by static TF publishers"
                ),
            ),
            DeclareLaunchArgument(
                "loop_closure_enable",
                default_value="false",
                description="Enable upstream ICP loop-factor generation",
            ),
            DeclareLaunchArgument(
                "loop_closure_frequency",
                default_value="1.0",
            ),
            DeclareLaunchArgument(
                "loop_search_radius",
                default_value="15.0",
            ),
            DeclareLaunchArgument(
                "loop_search_time_diff",
                default_value="30.0",
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
                "use_motion_deskew",
                default_value="false",
                description=(
                    "Insert project-owned IMU/odometry motion deskew before "
                    "feature extraction"
                ),
            ),
            DeclareLaunchArgument(
                "feature_cloud_info_topic",
                default_value="/lio_sam/deskew/cloud_info",
                description="CloudInfo input used by LIO-SAM feature extraction",
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
            DeclareLaunchArgument("point_density_healthy_s", default_value="3.0"),
            DeclareLaunchArgument("point_density_ramp_down_s", default_value="1.5"),
            DeclareLaunchArgument("point_density_hold_s", default_value="4.5"),
            DeclareLaunchArgument("point_density_ramp_up_s", default_value="3.0"),
            DeclareLaunchArgument(
                "motion_deskew_apply_translation",
                default_value="true",
                description=(
                    "Apply project-owned translational motion compensation"
                ),
            ),
            DeclareLaunchArgument(
                "motion_deskew_replace_upstream_rotation",
                default_value="false",
                description=(
                    "Reconstruct rotation from raw IMU using quaternion "
                    "integration instead of upstream Euler deskew"
                ),
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
                additional_env={
                    "CYCLONEDDS_URI": static_transform_cyclonedds_uri
                },
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
                additional_env={
                    "CYCLONEDDS_URI": static_transform_cyclonedds_uri
                },
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
                        "point_density": ParameterValue(
                            point_density,
                            value_type=float,
                        ),
                        "point_density_profile": point_density_profile,
                        "point_density_min": ParameterValue(
                            point_density_min,
                            value_type=float,
                        ),
                        "point_density_healthy_s": ParameterValue(
                            point_density_healthy_s, value_type=float
                        ),
                        "point_density_ramp_down_s": ParameterValue(
                            point_density_ramp_down_s, value_type=float
                        ),
                        "point_density_hold_s": ParameterValue(
                            point_density_hold_s, value_type=float
                        ),
                        "point_density_ramp_up_s": ParameterValue(
                            point_density_ramp_up_s, value_type=float
                        ),
                    }
                ],
                output="screen",
            ),
            Node(
                package="anymal_locomotion_ros2",
                executable="motion_deskew",
                name="anymal_lio_motion_deskew",
                parameters=[
                    {
                        "use_sim_time": True,
                        "apply_translation": motion_deskew_apply_translation,
                        "replace_upstream_rotation": (
                            motion_deskew_replace_upstream_rotation
                        ),
                    }
                ],
                condition=IfCondition(use_motion_deskew),
                output="screen",
            ),
            *lio_nodes,
            map_optimization,
            odom_adapter,
            confidence,
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
