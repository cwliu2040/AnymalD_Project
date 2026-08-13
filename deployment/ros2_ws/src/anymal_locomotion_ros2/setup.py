from glob import glob

from setuptools import find_packages, setup

package_name = "anymal_locomotion_ros2"
launch_files = sorted(glob("launch/*.launch.py"))

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/config",
            glob("config/*.yaml")
            + glob("config/*.xml")
            + glob("config/*.rviz")
            + glob("config/*.json"),
        ),
        (f"share/{package_name}/launch", launch_files),
    ],
    install_requires=["numpy", "PyYAML", "setuptools"],
    zip_safe=True,
    maintainer="CW Liu",
    maintainer_email="133053216+cwliu2040@users.noreply.github.com",
    description="External ROS 2 runtime for the ANYmal-D locomotion policy.",
    url="https://github.com/cwliu2040/AnymalD_Project",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "keyboard_teleop = anymal_locomotion_ros2.keyboard_teleop:main",
            "lio_benchmark = anymal_locomotion_ros2.lio_benchmark_node:main",
            "lio_replay_evaluator = anymal_locomotion_ros2.lio_replay_evaluator_node:main",
            "stability_benchmark = anymal_locomotion_ros2.stability_benchmark_node:main",
            "lidar_point_adapter = anymal_locomotion_ros2.lidar_point_adapter:main",
            "fastlio_point_adapter = anymal_locomotion_ros2.fastlio_point_adapter:main",
            "fastlio_odom_adapter = anymal_locomotion_ros2.fastlio_odom_adapter:main",
            "fastlio_confidence_extractor = anymal_locomotion_ros2.fastlio_confidence_node:main",
            "fastlio_confidence_fault_validation = anymal_locomotion_ros2.fastlio_confidence_fault_validation:main",
            "slam_confidence_dds_fault_validation = anymal_locomotion_ros2.slam_confidence_dds_fault_validation:main",
            "liosam_odom_adapter = anymal_locomotion_ros2.liosam_odom_adapter:main",
            "liosam_confidence_extractor = anymal_locomotion_ros2.liosam_confidence_node:main",
            "policy_node = anymal_locomotion_ros2.policy_node:main",
            "proprioceptive_velocity_estimator = anymal_locomotion_ros2.proprioceptive_velocity_estimator_node:main",
            "hardware_state_estimator_adapter = anymal_locomotion_ros2.hardware_state_estimator_adapter:main",
            "motion_deskew = anymal_locomotion_ros2.motion_deskew_node:main",
            "yaw_visualizer = anymal_locomotion_ros2.yaw_visualizer_node:main",
        ],
    },
)
