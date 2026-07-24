# ROS 2 模擬部署對齊紀錄

更新日期：2026-07-24

這份文件保存 Grill Me 訪談後確認的架構決策、驗收方式與已知限制。它是跨
對話的共同依據；若後續實作方向改變，應同步更新本文件，不能只依賴聊天
context。

## 不可破壞的邊界

- 不修改 `/home/ros/IsaacLab`，除非使用者明確要求。
- 不修改舊 workspace：
  `/home/ros/Documents/anymal_project/anymal_ws`。
- Isaac Lab training 與 ROS 2 deployment 分開。
- Isaac Sim／Isaac Lab Python 不 import `rclpy`。
- 不使用自製 UDP 傳送 IMU、LiDAR、`cmd_vel` 或 policy state。
- Isaac Sim 與 ROS 2 之間使用 ROS 2 Bridge／Action Graph。
- Joint array 一律依 joint name 做 deterministic mapping，不依賴 USD
  內部陣列順序。
- Git commit 使用繁體中文；每一次 commit 或 push 前都必須重新整理內容並
  取得使用者明確同意。

## 目前共同理解

系統不是「只加感測器」就完成。模擬閉迴路包含三個不同責任：

```text
Isaac Sim / Isaac Lab
  ├─ GPU physics 與 ANYmal-D actuator
  ├─ Action Graph 發布 robot state
  └─ Action Graph 接收 /cmd_vel 與 /joint_command
                  |
                  v
外部 ROS 2 policy node
  robot state + /cmd_vel
          -> 48-D observation
          -> model_1298 匯出的 policy
          -> 12-D joint targets
```

RSL-RL policy 不會自行從 ROS 2 或真實感測器取得完整身體狀態。訓練時，
Isaac Lab 直接提供 observation；部署時，外部 policy node 必須從 ROS 2
messages 重建完全相同的 48 維 observation。

IMU 無法可靠地單獨產生 base linear velocity。模擬階段暫時由 `/odom`
提供 body-frame base linear velocity；IMU 提供 orientation、angular
velocity 與 linear acceleration。實體機階段再由正式 state estimator
提供相同語意的 velocity。

GPU-to-CPU copy 不代表把模擬改成 CPU。Physics、actuator 與 policy tensor
留在 GPU；只有 ROS 2／DDS 必須序列化的小量資料，例如 12 個 joint
position 與 12 個 joint velocity，會在發布邊界複製到 CPU。

## 已確認的工作順序

### 第一階段：可控制的 ROS 2 policy closed loop

- `/cmd_vel` 使用 body-frame `[vx, vy, wz]`。
- 鍵盤採按住式控制：
  - `W/S`：前進／後退。
  - `Q/E`：向左／向右側移。
  - `A/D`：向左／向右旋轉。
  - `Space`：立即停止。
  - `+/-`：調整速度倍率。
- 放開控制鍵後 0.1 秒內回到零命令。
- 預設速度為 `vx=0.5 m/s`、`vy=0.3 m/s`、`wz=0.5 rad/s`。
- 綠色目標箭頭必須顯示收到的外部 body-frame command，不能繼續顯示
  Isaac Lab 的隨機 command。
- ROS 2 組成的 48 維 observation 必須逐項對上 Isaac Lab observation。
- 長時間人工操作使用 `--disable-episode-timeout`，避免 Play task 每 20 秒
  因正常 episode time limit 自動重置。

目前結果：

- 48 維 observation parity 通過；GPU 測試最大誤差小於 `1e-5`。
- GPU ROS 2 closed loop 通過，沒有 joint tensor device mismatch。
- RTX 5080 即時測試：
  - RTF：`0.975`。
  - control loop：`48.74 Hz`。
  - `/joint_states` 穩態：約 `49.5 Hz`。
  - 850/850 controlled steps 收到 joint command。
  - 0 termination、0 timeout。

固定 `[0.5, 0, 0]` 的 10 秒直線驗收結果：

- 前進：`4.88 m`，通過至少 `4.0 m` 的條件。
- heading change：`6.04°`，通過不超過 `10°` 的條件。
- lateral displacement：`0.61 m`，未通過不超過 `0.30 m` 的條件。

原生 checkpoint 固定速度 evaluation 也量到同方向的小幅
`vy=0.020 m/s`、`wz=0.0186 rad/s` bias。因此目前側漂是 policy 的低速
tracking 限制，不是 ROS command 軸向接反。不能透過修改驗收門檻假裝它已
消失。

### 第二階段：GPU 與 ROS 2 Bridge 效能

Isaac Sim 5.1 專用的 ROS2 Publish Joint State node 在 GPU PhysX 場景會
發生 CPU/GPU tensor device mismatch。已決定改用 ROS 2 Bridge Generic
Publisher 發布 `sensor_msgs/msg/JointState`，並以 Action Graph impulse
在 50 Hz policy step 觸發。

這仍是真正的 ROS 2 Bridge／DDS 路徑，不是 `rclpy`、UDP 或假 topic。

### 第三階段：真正的 200 Hz 模擬 IMU

下一步才開始實作，尚未完成：

- 使用 Isaac Sim 的真正 IMU sensor，不從低頻 base state 假造高頻訊息。
- Sensor update period 對齊 200 Hz physics，即 `0.005 s`。
- 透過 ROS 2 Bridge／Action Graph 發布，不使用 UDP。
- 不把 20 Hz 或 50 Hz 資料重複發布成假 200 Hz。
- Policy 維持 50 Hz，每次使用最新的一筆 200 Hz IMU sample。
- Policy 與未來 SLAM 共用同一個 IMU topic，目標名稱為 `/imu/data`；
  不另外製造 `/policy/imu`。
- IMU 必須有明確的 sensor prim、parent prim、frame、mounting transform、
  timestamp 與 update rate。
- 若無法直接掛在 articulation root 的 `base`，應在 base 下建立固定的
  child Xform／rigid sensor frame，再將 IMU sensor 掛在該 frame；不能改用
  自製 UDP sensor 繞過問題。

IMU 驗收至少包含：

- Sensor prim 確實位於 ANYmal-D base hierarchy 下，並記錄完整 prim path。
- 靜止時 orientation、gravity direction 與 angular velocity 合理。
- 已知旋轉或傾斜動作下，IMU 軸向、正負號與 `base_link` frame 一致。
- Timestamp 使用 simulation time，資料單調遞增。
- 200 Hz simulation rate；即時運行時 wall-clock 接收率目標至少 180 Hz。
- Policy 使用的 IMU sample 與 SLAM topic 來自同一個 sensor data source。

## 暫時不做

- 不加入 LiDAR、camera、LIO-SAM、LVI-SAM 或完整 SLAM。
- 不把 LiDAR／camera observation 放進 Flat v1 policy。
- 不直接處理實體 ANYmal-D low-level interface。
- 不盲目繼續訓練 model_1298。
- 不為了讓數字好看而偽造 sensor frequency。

## Git checkpoint

目前應先完成第一階段／GPU bridge 的 commit，再開始真正 IMU sensor。
任何 commit 或 push 仍須依 `AGENTS.md` 重新取得明確同意。本文件本身不代表
已授權 commit 或 push。
