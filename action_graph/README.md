# Isaac Sim ROS 2 Action Graph v0.1 Contract

這份文件固定第一版 ROS topic 與 Action Graph 接線。實際 graph asset 尚未
建立，因為目前尚未選定 simulation host、ANYmal-D articulation prim path
與 IMU prim path。

## State publishers

每個 state publisher 都由 simulation time 與 playback tick 驅動：

| Topic | ROS message | Isaac Sim node | 要求 |
|---|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | ROS2 Publish Clock | 使用 simulation time |
| `/joint_states` | `sensor_msgs/JointState` | ROS2 Publish Joint State | target prim 指向 ANYmal-D articulation |
| `/odom` | `nav_msgs/Odometry` | ROS2 Publish Odometry | `chassisFrameId=base_link`、`publishRawVelocities=false` |
| `/imu` | `sensor_msgs/Imu` | ROS2 Publish Imu | `frameId=base_link`，資料必須與 base 對齊 |

ROS2 Publish Clock 的 timestamp input 與其他 publishers 必須使用同一個
Isaac Read Simulation Time source。外部 node 設定 `use_sim_time=true`。

`publishRawVelocities=false` 很重要：ROS2 Publish Odometry node 會把輸入的
world velocity 投影到 robot frame，policy 才能取得訓練時使用的 body-frame
base linear velocity。

若 IMU prim 不是與 `base_link` 對齊，必須先做 frame transform；不可直接把
有 mounting rotation 的 quaternion 當作 base orientation。

## Joint command subscriber

需要以下 nodes：

- `isaacsim.ros2.bridge.ROS2SubscribeJointState`
- `isaacsim.core.nodes.IsaacArticulationController`

設定：

- Subscribe topic：`/joint_command`
- Articulation Controller target prim：ANYmal-D articulation root

必要連線：

```text
SubscribeJointState.outputs:execOut
  -> ArticulationController.inputs:execIn

SubscribeJointState.outputs:jointNames
  -> ArticulationController.inputs:jointNames

SubscribeJointState.outputs:positionCommand
  -> ArticulationController.inputs:positionCommand
```

一定要連接 `jointNames`，不能只假設 message array 與 USD joint index 順序
相同。第一版只發布 position targets，不連 velocity／effort command。

## 架構邊界

- Isaac Sim／Isaac Lab Python 不 import `rclpy`。
- ROS 2 policy node 在 `deployment/ros2_ws` 外部 process 執行。
- 不使用 UDP。
- `/joint_command` 是 simulation adapter interface，不代表實體機介面。
