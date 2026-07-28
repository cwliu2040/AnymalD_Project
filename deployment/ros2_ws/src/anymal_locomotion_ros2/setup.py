from glob import glob

from setuptools import find_packages, setup

package_name = "anymal_locomotion_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/config",
            glob("config/*.yaml") + glob("config/*.xml"),
        ),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["numpy", "PyYAML", "setuptools"],
    zip_safe=True,
    maintainer="CW Liu",
    maintainer_email="133053216+cwliu2040@users.noreply.github.com",
    description="External ROS 2 runtime for the ANYmal-D locomotion policy.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "keyboard_teleop = anymal_locomotion_ros2.keyboard_teleop:main",
            "lio_benchmark = anymal_locomotion_ros2.lio_benchmark_node:main",
            "stability_benchmark = anymal_locomotion_ros2.stability_benchmark_node:main",
            "lidar_point_adapter = anymal_locomotion_ros2.lidar_point_adapter:main",
            "policy_node = anymal_locomotion_ros2.policy_node:main",
            "motion_deskew = anymal_locomotion_ros2.motion_deskew_node:main",
        ],
    },
)
