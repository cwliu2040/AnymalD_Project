# Isaac Sim ROS 2 Action Graph v0.2

第一版 graph 由
`anymal_locomotion.simulation.ros2_bridge.create_ros2_policy_bridge()`
在 Play task 啟動後建立，不修改 Isaac Lab 或官方 ANYmal-D USD。

目前實測的單一環境路徑：

- environment：`/World/envs/env_0`
- ANYmal-D articulation root：`/World/envs/env_0/Robot/base`
- graph：`/ROS2PolicyBridge`

程式會在建立 graph 前重新尋找並驗證 articulation root，不把這個完整路徑
當作其他場景也必然相同。

## State publishers

每個 policy-rate state publisher 都由 simulation time 與 host 發出的
`OnImpulseEvent` 驅動：

| Topic | ROS message | Isaac Sim node | 要求 |
|---|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | ROS2 Publish Clock | 使用 simulation time |
| `/joint_states` | `sensor_msgs/JointState` | Generic ROS2 Publisher | canonical name、position、velocity |
| `/odom` | `nav_msgs/Odometry` | ROS2 Publish Odometry | `chassisFrameId=base_link`、`publishRawVelocities=true` |
| `/tf` | `tf2_msgs/TFMessage` | ROS2 Publish Raw Transform Tree | 動態 `odom → base_link`，與 `/odom.pose` 共用 pose 與 timestamp |
| `/imu/data` | `sensor_msgs/Imu` | Isaac Read IMU + ROS2 Publish Imu | `frameId=base_link`、200 Hz physics sample |
| `/lidar/points_raw` | `sensor_msgs/PointCloud2` | ROS2 Bridge RTX PointCloud Buffer writer | `frameId=lidar_link`、OS1 32-channel、10 Hz full scan |

ROS2 Publish Clock 的 timestamp input 與其他 publishers 必須使用同一個
Isaac Read Simulation Time source。外部 node 設定 `use_sim_time=true`。

Isaac Compute Odometry 的 local linear velocity 直接送入 odometry publisher；
world angular velocity 則先以每幀更新的 base quaternion 轉成 body frame。
這兩個值已經是 policy 需要的 frame，因此使用 `publishRawVelocities=true`
避免 publisher 再做一次座標轉換。

Isaac Sim 5.1 專用的 ROS2 Publish Joint State node 在 GPU PhysX 場景會用
CPU tensor 讀取 articulation，造成 device mismatch。Graph 改用 ROS 2 Bridge
的 Generic Publisher 發布同一種 `sensor_msgs/msg/JointState`。Host 每個
50 Hz policy tick 只把 12 個 joint position 與 12 個 joint velocity 從 GPU
複製到 CPU，依 canonical joint-name order 寫入 Action Graph，再由 Bridge
做 DDS 序列化。模擬、actuator 與 policy tensor 仍留在 GPU；沒有 `rclpy`
或 UDP。

ROS 2 deployment host 會在 physics 啟動前，將真正的 Isaac Sim IMU sensor
prim 建立於：

`/World/envs/env_0/Robot/base/imu_sensor`

它的 parent 是 articulation root `/World/envs/env_0/Robot/base`，mounting
translation 為 `(0, 0, 0)`、mounting quaternion `(w, x, y, z)` 為
`(1, 0, 0, 0)`，因此 sensor frame 與 `base_link` 對齊。sensor period 與
physics dt 都是 `0.005 s`。

高頻 graph 使用 `OnPhysicsStep`，每個 physics step 依序讀取真正 IMU 的
angular velocity、linear acceleration 與 sensor time，再由
`ROS2PublishImu` 發布。Isaac Sim 5.1 在 GPU articulation 上的 IMU orientation
輸出會固定為 identity，因此 orientation 由同一個 200 Hz physics event 的
`IsaacComputeOdometry` 取得；這不是把 50 Hz base state 重播成 200 Hz。
由於該 orientation 相對 reset pose，deployment host 將初始 odom yaw 固定為
零，再把 IMU 的 world angular velocity 轉成 `base_link` frame。

目前 filter width 都是 1，尚未加入 noise 或 bias model。

IMU 不負責提供 base linear velocity；policy 的 base linear velocity 來自
`/odom.twist.twist.linear`。目前 sensor 採 identity mounting；未來若加入
mounting rotation，必須先轉成 `base_link` frame。

## RTX LiDAR 與 TF ownership

加上 `--enable-lio-sam` 時，deployment host 會在 articulation root 下建立
專案擁有的固定 `lidar_link` Xform，translation 為
`(0.20, 0, 0.35) m`、rotation 為 identity，並掛上 Isaac Sim RTX LiDAR。
內建 profile 固定為 `OS1_REV6_32ch10hz1024res`：32 channels、10 Hz、
每圈 1024 個水平 sample。

RTX render product 使用 Isaac Sim 5.1 官方 standalone example 相同的
`RtxLidarROS2PublishPointCloudBuffer` writer 發布 `/lidar/points_raw`，
保持模擬器原始 PointCloud2 格式。
Validation host 會在 `--enable-lio-sam` 時自動啟用 headless off-screen
camera rendering；否則 render product 雖存在，RTX sensor 不會產生 frame。
Deployment host 以 repository 內的
`assets/maps/factory/Factory_Layout.usd` terrain 取代 Flat plane，LIO-SAM
直接使用 Factory 牆面、設備與其他不對稱幾何進行 scan matching。Training
task 與 Flat v1 policy observation 契約不受 deployment 場景替換影響。
外部 `lidar_point_adapter` 再依固定 OS1 beam elevation 與 scan azimuth
產生 LIO-SAM Ouster contract 所需的 `ring` 與每點相對時間 `t`，輸出至
`/lio_sam/points`。這個轉換位於 ROS 2 deployment process，不把 `rclpy`
或 SLAM 邏輯放進 Isaac Sim。

LIO-SAM 模式下的正式 TF ownership 為：

```text
map --static--> odom --LIO-SAM IMU preintegration--> base_link
                                               |
                                               +--static--> lidar_link
```

為避免同一條 TF 有兩個 publisher，simulation ground-truth
`odom → base_link` 在此模式停用。Upstream map-optimization node 額外送出的
重複 TF 被 remap 到隔離 topic；`/odom` topic 仍保留給 locomotion policy
作為模擬 ground-truth velocity，不代表它擁有 TF。

## Joint command subscriber

ROS command 一定先進入以下 node：

- `isaacsim.ros2.bridge.ROS2SubscribeJointState`

設定：

- Subscribe topic：`/joint_command`

官方 ANYmal-D task 使用 Isaac Lab LSTM actuator model。若直接接
`IsaacArticulationController`，會繞過訓練時的 actuator model，造成控制結果
不一致。因此目前已驗證的 Play host 使用：

```text
ROS2SubscribeJointState
  -> 依 joint name 重排 position targets
  -> 還原 raw policy action
  -> ManagerBasedEnv action manager
  -> ANYmal-D LSTM actuator
```

這個 adapter 只讀 OmniGraph outputs，不 import `rclpy`。它嚴格驗證 12 個
joint names，再以 metadata 的 default positions 與 action scale 0.5 還原
raw action。不能只假設 message array 與 USD joint index 順序相同。
Adapter 也會檢查 message timestamp；連續 5 個 policy steps（0.1 秒）沒有
新 command 時，會退回零 raw action，不會永久重播舊訊息。

Graph builder 仍可選擇建立 `IsaacArticulationController`，供使用原生
position drives 的簡單 articulation 測試；官方 ANYmal-D Play closed loop
不使用這條直接控制路徑。

## 啟動與驗證

Graph 使用目前環境的 `ROS_DOMAIN_ID`，未設定時使用 0。外部 ROS 2 process
必須使用相同 domain。

```bash
# Terminal 1：先建置並啟動外部 policy node
cd <repository-root>
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash
export PYTHONPATH="$(pwd)/deployment/python_vendor:${PYTHONPATH}"
ros2 run anymal_locomotion_ros2 policy_node --ros-args -p use_sim_time:=true
```

```bash
# Terminal 2：在 policy node 等待時啟動 simulation host
cd <repository-root>
TERM=xterm-256color PYTHONPATH=source/anymal_locomotion \
  "${ISAACLAB_ROOT:-${HOME}/IsaacLab}/isaaclab.sh" -p \
  scripts/validation/validate_ros2_bridge.py \
  --headless --device cuda:0 --steps 800 --external-control \
  --validate-observation-parity
```

純 headless Isaac Lab 的 `env.step()` 只推進物理，不會可靠地觸發 playback
graph。Validation host 因此在每個 50 Hz environment step 後觸發一次
Action Graph impulse；graph 會同步讀寫 ROS 2，但不額外更新整個 Kit rendering
frame。GUI 與 headless 使用相同的 policy-rate trigger。

外部接收已驗證：

- `/clock`：simulation time
- `/joint_states`：12 個具名關節的 position/velocity/effort
- `/odom`：`odom` 到 `base_link` 的 pose 與 body-frame velocity
- `/tf`：與 `/odom.pose` 一致的動態 `odom` 到 `base_link` transform
- `/imu/data`：`base_link` orientation、angular velocity、linear acceleration
- `/lidar/points_raw`：僅在 `--enable-lio-sam` 模式發布的 RTX full scan

GPU 即時量測的 `/imu/data` DDS 接收率為 `196.046 Hz`（1000-sample
window），高於 180 Hz 驗收目標。250-step closed loop 使用 100 筆 parity
sample 驗證 latest-sample policy path；base angular velocity 最大誤差
`8.03e-4`、projected gravity 最大誤差 `2.11e-4`。
另一次靜止取樣的 linear acceleration 為
`(0.00543, 0.00108, 9.80877) m/s²`。動態 `/tf` 實收率約 `49.2 Hz`，
`tf2_echo odom base_link` 可正常解析。

RTX off-screen rendering 模式的 60 秒 LIO-SAM smoke test 也已通過：
`/lio_sam/points` 的 `ring=0..31`、每點 `t` 約涵蓋一圈 0.1 秒；
Image Projection 的 IMU／odometry coverage 都有效，mapping odometry 與
`map → base_link` TF 可解析。該模式 wall-time point-cloud 接收率約
`6.8–8.1 Hz`，是整體 RTF 小於 1 的結果；sensor profile 與 simulation
timestamp 仍是 10 Hz。

外部 ONNX policy closed loop 已在 RTX 5080 的 `cuda:0` host 驗證；GPU 模式
可持續收到具名 joint commands、0 termination、0 timeout。100 個 controlled
steps 的 48 維 observation parity 最大誤差小於 `1.0e-5`。900-step 即時測試
的 RTF 為 0.975、loop wall rate 為 48.74 Hz，外部
`ros2 topic hz /joint_states` 穩態量測約 49.5 Hz。

## 架構邊界

- Isaac Sim／Isaac Lab Python 不 import `rclpy`。
- ROS 2 policy node 在 `deployment/ros2_ws` 外部 process 執行。
- 不使用自行設計的 UDP 通道；所有資料經 ROS 2 Bridge。
- `/joint_command` 是 simulation adapter interface，不代表實體機介面。
