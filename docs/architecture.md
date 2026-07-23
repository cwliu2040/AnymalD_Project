# ANYmal-D Locomotion 系統架構

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
- Command limits：x/y 為 ±1.0 m/s，yaw 為 ±1.0 rad/s。
- Actor/critic observation normalization：關閉。

Action、joint position observation 與 joint velocity observation 使用相同的
canonical joint-name order。Runtime 與 ROS array 必須依 joint name remap。

## 模擬部署邊界

未來外部 ROS 2 policy node 將：

1. 接收 `/cmd_vel`（`geometry_msgs/msg/Twist`）。
2. 接收具有 timestamp 的 IMU、joint state 與 state-estimation data。
3. 依照匯出的 metadata 組合 48 維 observation。
4. 在 Isaac Sim 外執行 inference。
5. 發布經確認的 low-level command interface。

Isaac Sim 使用內建 ROS 2 Bridge / Action Graph nodes 傳輸訊息。Isaac Sim 與
Isaac Lab Python module 不可 import `rclpy`，也不可直接修改 command manager
的 private tensor。

Low-level command message type 必須等實體 ANYmal-D control interface 確認後
才能決定。

## State Estimation

官方 task 可直接使用 simulator ground-truth base linear velocity；實體機則
需要明確定義 estimator、frame 與 timestamp contract。開始 ROS 2 policy
實作前必須確認：

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
