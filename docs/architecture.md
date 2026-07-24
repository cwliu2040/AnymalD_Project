# ANYmal-D Locomotion 系統架構

ROS 2 模擬部署的決策理由、驗收標準與目前未通過項目，另見
[ROS 2 模擬部署對齊紀錄](ros2_deployment_decisions.md)。

## 範圍

Flat Locomotion v1 是 50 Hz 的 proprioceptive policy。輸入包含 base linear
velocity、base angular velocity、projected gravity、直接 velocity command、
joint position/velocity 與 previous action。

LiDAR、RGB-D、terrain perception、SLAM 與 navigation 不屬於 Flat v1
policy observation。

## 系統生命週期

```text
Isaac Lab v2.3.2 訓練
  （Manager-Based、RSL-RL PPO、不使用 rclpy）
              |
              v
專案內的 checkpoint 與 resolved configs
              |
              v
Policy 匯出（TorchScript / ONNX + policy_metadata.yaml）
              |
              v
外部 ROS 2 policy node
  /cmd_vel + estimated state -> 48-D observation -> 12-D action
              |
              +--------------- 模擬分支 ----------------+
              |                                         |
              v                                         |
ROS 2 messages <-> ROS 2 Bridge / Action Graph <-> Isaac Sim
              |
              +--------------- 實體分支 ----------------+
              |
              v
通過審查的 hardware adapter / safety controller -> 實體 ANYmal-D
```

實體機分支不使用 Isaac Sim Action Graph。它透過 robot-specific、經安全審查
的 hardware adapter 重用同一份版本化 policy contract。實體 ANYmal-D 的
low-level control interface 尚未確認。

## 訓練層

訓練層由 `source/anymal_locomotion` 與 `scripts/rsl_rl` 管理。

- 依賴 Isaac Lab v2.3.2 與 RSL-RL，不複製其 source tree。
- 不 import `rclpy`。
- 第一個 smoke/regression baseline 使用官方 ANYmal-D USD。
- Command 採用 body-frame `[vx, vy, wz]`。
- 所有 run 都寫入本 repository。
- Policy 匯出時必須同時記錄已 commit 的 Git revision 與 metadata。

## Policy Contract

唯一資料來源是 `configs/policy_contract.yaml`。

- Observation：固定七個 term，共 48 維。
- Action：12 個 joint-position action。
- Action 轉換：
  `target_position = default_position + 0.5 * policy_action`。
- Policy period：0.02 秒。
- High-Speed command limits：x 為 -2.0～3.0 m/s、y 為 ±1.5 m/s、
  yaw 為 ±2.0 rad/s。
- Actor/critic observation normalization：關閉。

Action、joint position observation 與 joint velocity observation 使用相同的
canonical joint-name order。Runtime 與 ROS array 必須依 joint name remap。

## 模擬部署邊界

第一版外部 ROS 2 policy node 已建立於
`deployment/ros2_ws/src/anymal_locomotion_ros2`，它會：

1. 接收 `/cmd_vel`（`geometry_msgs/msg/Twist`）。
2. 接收具有 timestamp 的 IMU、joint state 與 state-estimation data。
3. 依照匯出的 metadata 組合 48 維 observation。
4. 在 Isaac Sim 外執行 inference。
5. 對模擬器發布 `sensor_msgs/msg/JointState` `/joint_command`。

Isaac Sim 使用內建 ROS 2 Bridge / Action Graph nodes 傳輸訊息。Isaac Sim 與
Isaac Lab Python module 不可 import `rclpy`，也不可直接修改 command manager
的 private tensor。

專案內的 graph builder 已在 Play task 驗證：

- `/joint_states`、`/odom`、`/imu`、`/clock` 可由外部 ROS 2 process 接收；
- `/joint_command` 使用 JointState joint names 進行 deterministic remap，
  不依賴 USD 內部關節陣列順序；
- ROS2 Context 明確使用目前的 `ROS_DOMAIN_ID`；
- headless 與 GUI host 都在 50 Hz 物理／policy step 後觸發 Action Graph
  impulse，不額外更新整個 Kit rendering frame。

官方 ANYmal-D task 使用 LSTM actuator model，所以 Play host 不把 command
直接接到 USD Articulation Controller。它從 OmniGraph subscriber 取出具名
position targets，還原成 raw policy action，再交給 Isaac Lab action manager。
這能保留訓練時相同的 actuator dynamics，且 Isaac Sim Python 仍不 import
`rclpy`。

Bridge host 使用 GPU PhysX。Isaac Sim 5.1 專用的 ROS2 Publish Joint State
node 在 GPU 場景有 tensor device mismatch，因此 `/joint_states` 改由
ROS 2 Bridge Generic Publisher 發布。每個 50 Hz tick 只將 12 個 joint
position 與 12 個 joint velocity 依 canonical order 從 GPU 複製到 CPU，
供 DDS 序列化；物理、actuator 與 policy tensor 不因此改成 CPU。

ROS 端重建的 48 維 observation 已逐項和 Isaac Lab observation 比對：
base linear/angular velocity、projected gravity、command、joint
position/velocity 與 previous action 均通過 `1e-4` tolerance。固定
`[0.5, 0, 0]` 的 10 秒 closed-loop 測試沒有 termination，但目前 policy
仍有低速側漂；這項限制會保留在驗收結果中，不歸因為 ROS frame 錯接。
RTX 5080 即時整合測試達到 RTF 0.975、48.74 Hz loop rate，
`/joint_states` 穩態約 49.5 Hz。

`/joint_command` 只作為 Isaac Sim adapter interface。實體 ANYmal-D 的
low-level command message type 仍須等 controller/SDK 與 safety requirements
確認後才能決定。

訓練 command range 不等於實體機允許範圍。Hardware adapter 必須另行實作
經安全審查的 clamp、rate limit 與 emergency stop。

## State Estimation

官方 task 可直接使用 simulator ground-truth base linear velocity；IMU 無法
單獨提供無 drift 的 base linear velocity。第一版模擬 deployment 明確使用
body-frame `/odom.twist.twist.linear`，並使用 IMU angular velocity 與
orientation。

官方 Play 場景目前沒有獨立 IMU prim，所以第一版 `/imu` 由 base simulator
state 產生。這是通訊與 observation 整合版本，不是 sensor-noise 模型。實體機
仍需要明確定義 estimator、frame 與 timestamp contract：

- base velocity estimator 與 body/world frame；
- IMU orientation 與 angular-velocity convention；
- odometry source、update rate 與 covariance；
- synchronization 與 stale-data timeout；
- safety clamp、rate limit 與 emergency stop。

## 感知與導航

```text
LiDAR / RGB-D
      |
      v
ROS 2 Bridge / sensor drivers
      |
      v
SLAM / terrain perception / Nav2
      |
      v
/cmd_vel
      |
      v
外部 locomotion policy node
```

LiDAR 與 camera tensor 不進入 Flat v1 observation。Rough locomotion 與
perceptive locomotion 將使用獨立 task 與 policy version。

## 未來 Custom USD

目前以官方 Isaac Sim 5.1 ANYmal-D USD 作為 reference。Custom USD 必須使用
專案擁有的 `ArticulationCfg`，並驗證：

- joint name、axis、sign、limit 與 default position；
- base、foot、IMU、LiDAR 與 camera frame；
- inertial 與 collision property；
- contact body name；
- actuator model compatibility。

系統不會猜測 custom joint name。任何 mapping 變更都必須更新版本化 contract
與相關測試。
