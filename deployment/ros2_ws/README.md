# ROS 2 Deployment Workspace

此目錄保留給未來外部 ROS 2 policy node、hardware/simulation adapter、
bringup 與 interface test。

ROS 2 node 必須在 Isaac Sim process 外執行，可使用 `rclpy` 或 `rclcpp`；
Isaac Lab training extension 則不可使用。

預計輸入：

- `/cmd_vel`（`geometry_msgs/msg/Twist`）
- IMU
- 依 joint name remap 的 joint states
- odometry / state estimate

Low-level output interface 必須等實體 ANYmal-D controller/SDK 與 safety
requirements 確認後才能決定。

不可將 legacy workspace 或舊版在 simulator 內執行的 `rclpy` prototype
複製到此目錄。
