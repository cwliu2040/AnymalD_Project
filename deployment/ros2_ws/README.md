# ROS 2 Deployment Workspace

Reserved for the future external ROS 2 policy node, hardware/simulation
adapters, bringup, and interface tests.

The node will run outside Isaac Sim. It may use `rclpy` or `rclcpp`; the
Isaac Lab training extension may not.

Planned inputs:

- `/cmd_vel` (`geometry_msgs/msg/Twist`)
- IMU
- joint states, remapped by name
- odometry/state estimate

The low-level output interface is intentionally unspecified until the physical
ANYmal-D controller/SDK and safety requirements are confirmed.

Do not copy the legacy workspace or the old in-simulator `rclpy` prototype into
this directory.
