# 實體 ANYmal-D 整合邊界

更新日期：2026-08-03

## 目前狀態

模擬端的 Recovery v0.4.0、reset、map-quality 與 loop-closure gate 已完成；
實體機整合尚未驗收。Repository 目前沒有 ANYmal-D robot driver、controller
SDK、實測 IMU/LiDAR extrinsic 或 safety controller，因此不能把模擬用的
`/joint_command` 直接當成實體 low-level wire protocol，也不能把模擬 mount
數值當成實體標定值。

舊工作區或 Downloads 中的 UDP bridge 不屬於本專案的可接受最終架構；實體
分支必須經由 robot-specific hardware adapter 與安全控制器，不能把 UDP
重新接回正式路徑。

## 不變的 policy-side contract

正式 policy 仍為：

`exported/anymal_d_locomotion_v1/recovery_v0.4.0/policy.onnx`

hardware adapter 應在 policy node 之外，將實體 driver 的資料轉成目前已驗證
的 ROS contract：

| ROS interface | 必要語意 |
|---|---|
| `/joint_states` | `sensor_msgs/msg/JointState`；12 個 canonical joint names、position 與 velocity 依 `configs/policy_contract.yaml` 對齊 |
| `/imu/data` | `sensor_msgs/msg/Imu`；angular velocity 與 orientation 已轉到 `base_link`，timestamp 單調遞增 |
| `/odom` | `nav_msgs/msg/Odometry`；`child_frame_id=base_link`，linear velocity 是 body-frame policy velocity |
| `/cmd_vel` | `geometry_msgs/msg/Twist`；body-frame `[vx, vy, wz]`，經 policy contract limits clamp |

目前 `policy_node` 對 IMU 要求 `header.frame_id=base_link`，對 odometry 要求
`child_frame_id=base_link`。若實體 driver 發布的是 `imu_link`、`base` 或其他
frame，應在 hardware/state-estimation adapter 轉換後再交給 policy；不能只改
字串繞過 frame mismatch。`/joint_command` 的 `sensor_msgs/msg/JointState`
仍只代表模擬 adapter input，不是實體機的最終 command message。

## Sensor extrinsic 必須先完成的資料

需由實體機與 calibration owner 提供版本化資料，放在 repository 的
project-local calibration artifact；在資料到齊前不修改正式 LIO-SAM 數值：

1. robot body reference 的定義與 frame name（`base_link` 的原點、軸向與
   REP-103 convention）。
2. IMU frame、parent frame、`T_base_imu` 的 translation/quaternion，以及
   IMU raw angular velocity、linear acceleration、orientation 的 frame 語意。
3. LiDAR frame、parent frame、`T_base_lidar` 的 translation/quaternion，
   LiDAR 型號、ring order、per-point timestamp 單位與 scan-start timestamp
   規則。
4. 實體 calibration 的方法、日期、工具／reference、估計 covariance 與
   residual；不能只提供一組未標註來源的 xyz 數字。

模擬目前使用 identity IMU mount 與 `base_link → lidar_link=(0.20, 0, 0.35)`
只是 bridge regression fixture。LIO-SAM 的 `extrinsicTrans` 是 upstream
graph 使用的 lidar/IMU 轉換，實機不能直接複製此 fixture 或把
`T_base_lidar` 原向量直接填入；必須依實測 frame convention 與 upstream
implementation 推導，並以靜態 TF、IMU axis test、scan-to-map test 共同驗收。

## Low-level adapter 必須先確認的資料

在實作 hardware adapter 前，必須取得：

- robot driver／SDK 套件名稱、版本、ROS 2 distribution 與生命週期啟動方式；
- state topic 與 command topic 的 message type、QoS、timestamp、update rate、
  frame semantics 與 covariance；
- 12 個 policy joint 到硬體 joint 的 name、axis sign、position zero、gear／
  transmission convention、position/velocity/torque limits；
- command 是 position target、joint trajectory、torque 或其他 controller
  reference，以及 controller 接受的 command rate、插值與 timeout 行為；
- hardware watchdog、通信斷線動作、emergency stop、enable/disable sequence、
  thermal/current/velocity limits 與 operator approval。

Adapter 內部至少要分成三層：

```text
robot driver / state estimator
        -> hardware adapter (frame + joint mapping + timestamp validation)
        -> policy-side ROS contract
        -> safety-reviewed command adapter / controller
```

Project side 現已提供 `hardware_state_estimator_adapter` 的 frame-safe骨架；它只接受
明確設定的 `nav_msgs/Odometry` vendor topic，且必須宣告 input twist 是 body 或 odom
frame。這不代表 ANYbotics topic/schema 已確認，也不得在未取得 SDK 文件時把骨架視為
實機整合完成。若原廠沒有可用 body velocity，才使用另行 qualification 的
proprioceptive estimator；兩者輸出共同的 `/locomotion/estimated_odom` policy-side
contract。

policy node 的 simulation guards（stale data、command timeout、NaN、action
limit）不能取代實體 safety controller。未完成 hardware contract 前，不能
把 `joint_target = default_position + 0.5 * raw_action` 直接送到馬達，也不能
用 simulation policy range 宣告實體安全速度。

## 實機 gate 順序

完成上述資料後，依序執行：

1. 不接 policy 的 read-only sensor bringup：驗證 frame、timestamp、頻率、
   joint mapping 與 calibration residual。
2. 固定姿態與手動小幅 motion：驗證 IMU gravity、angular velocity sign、
   odometry body velocity 與 LiDAR static TF；所有結果保留 raw bag。
3. 低風險、受 safety controller 限制的 policy observation parity；先不發布
   locomotion command。
4. 在硬體 owner 核准的 enclosure／吊掛或等價安全條件下，驗證 50 Hz
   command path、watchdog、reset 與 emergency stop。
5. 最後才進行受限速度的 sim-to-real locomotion gate；結果與模擬
   `no_instability`、LIO 12/12 分開記錄，不能互相替代。

在 driver、extrinsic 與 safety requirements 尚未提供前，這個 gate 停在
資料準備階段；Recovery v0.4.0 維持正式 policy，Recovery v0.5 candidates
不參與實機整合。
