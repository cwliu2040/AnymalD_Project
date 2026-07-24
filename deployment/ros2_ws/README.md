# ROS 2 Policy Runtime v0.1

## 這一版解決什麼

訓練時，Isaac Lab 直接把機器人狀態交給 policy。部署時，policy node 是
Isaac Sim 外部的獨立程式，因此狀態必須透過 ROS 2 Bridge 傳出，policy
算完後再把 joint target 傳回去。

```text
Isaac Sim
  ├─ /joint_states ─┐
  ├─ /imu ──────────┼─> 外部 policy node ─> /joint_command ─> Isaac Sim
  └─ /odom ─────────┘           ↑
                            /cmd_vel
```

`anymal_locomotion_ros2` 已實作第一版外部 policy node 與 ROS-independent
核心邏輯。它不會被 Isaac Lab training package import，也不會在 Isaac Sim
Python 中執行 `rclpy`。

## 為什麼需要 `/odom`

IMU 量到角速度、方向與加速度，但加速度積分成速度會快速累積 bias 和 drift，
所以本版不從 IMU 猜 base linear velocity：

- `/odom.twist.twist.linear`：body-frame base linear velocity
- `/imu.angular_velocity`：body-frame base angular velocity
- `/imu.orientation`：計算 body-frame projected gravity，`frame_id=base_link`
- `/joint_states`：依 joint name 重排 position／velocity
- `/cmd_vel`：body-frame `[vx, vy, wz]`

Isaac Sim Action Graph 的 ROS 2 Publish Odometry node 必須使用
`publishRawVelocities=false`，把 world velocity 投影到 robot frame，並將
`child_frame_id` 設為 `base_link`。實體機則必須由 robot state estimator
提供同一語意的 `/odom`。

## Policy observation

每個 0.02 秒 tick 組成：

| Offset | 維度 | 來源 |
|---:|---:|---|
| 0 | 3 | `/odom` base linear velocity |
| 3 | 3 | `/imu` angular velocity |
| 6 | 3 | `/imu` orientation 算出的 projected gravity |
| 9 | 3 | clamp 後的 `/cmd_vel` |
| 12 | 12 | joint position 減 default position |
| 24 | 12 | joint velocity |
| 36 | 12 | previous raw policy action |

Policy 產生 12 維 raw action，再轉成：

`joint_target = default_joint_position + 0.5 * raw_action`

輸出的 `/joint_command` 是 `sensor_msgs/msg/JointState`，包含 canonical joint
names 與 position targets。模擬端的 ROS2 Subscribe Joint State 必須把
`jointNames` 與 `positionCommand` 都接到 Isaac Articulation Controller。

## 建置

```bash
cd /home/ros/anymal_locomotion/deployment/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select anymal_locomotion_ros2
source install/setup.bash
```

## 推論 runtime

目前 ROS 2 系統 Python 已有 `rclpy` 與 NumPy，但沒有 PyTorch；Isaac Sim
Python 有 PyTorch，但不可拿來執行 ROS 2 node。這是兩個不同的 runtime。

在外部 ROS 2 Python 環境安裝相容的 PyTorch 後，才可啟動：

```bash
ros2 run anymal_locomotion_ros2 policy_node --ros-args \
  -p use_sim_time:=true \
  -p policy_path:=/home/ros/anymal_locomotion/exported/anymal_d_locomotion_v1/high_speed_v0.2.0/policy.pt \
  -p metadata_path:=/home/ros/anymal_locomotion/exported/anymal_d_locomotion_v1/high_speed_v0.2.0/policy_metadata.yaml
```

未安裝 PyTorch時，node 會明確停止並說明缺少 dependency，不會退回 Isaac Sim
Python 或把 `rclpy` 塞進 training process。

## 第一版安全行為

- joint names 缺少、重複或多出時不發布 command。
- `/odom`、`/imu` 或 `/joint_states` 超過 0.1 秒未更新時不發布 command。
- `/cmd_vel` 超過 0.5 秒未更新時自動使用零速度。
- command 會 clamp 到 policy 的訓練範圍。
- observation 或 policy output 出現 NaN／Inf 時不發布 command。
- raw action 絕對值超過 10 時由 simulation guard 拒絕。

這些只是模擬端 guard，不是實體 ANYmal-D 的 safety controller。

## 尚未完成

- 尚未在 Isaac Sim stage 建立實際 Action Graph，因為 robot prim path、IMU
  prim 與 simulation host 尚未固定。
- 尚未安裝外部 PyTorch 或選定 ONNX Runtime。
- 第一版使用 latest-sample，不做多 topic 精確時間同步。
- `/joint_command` 只用於 Isaac Sim；實體 ANYmal-D low-level interface
  尚未確認。
- 尚未完成 ROS 2 closed-loop walking test。

Action Graph 的必要節點與接線見
[Action Graph v0.1 contract](../../action_graph/README.md)。
