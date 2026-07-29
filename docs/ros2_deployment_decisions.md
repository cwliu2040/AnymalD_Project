# ROS 2 模擬部署對齊紀錄

更新日期：2026-07-27

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
- 鍵盤控制使用已驗證的官方 `teleop_twist_keyboard`：
  - `i/,`：前進／後退。
  - `j/l`：向左／向右旋轉。
  - `Shift+j`／`Shift+l`：向左／向右側移。
  - `k`：停止。
  - `q/z`、`w/x`、`e/c`：調整速度。
- 官方 teleop 預設線速度為 `0.5 m/s`、角速度為 `0.5 rad/s`；holonomic
  側移也使用 `0.5 m/s`。
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

已完成：

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

目前實作與結果：

- Sensor prim：
  `/World/envs/env_0/Robot/base/imu_sensor`。
- Parent prim：
  `/World/envs/env_0/Robot/base`。
- Frame：`base_link`。
- Mounting translation：`(0, 0, 0)`。
- Mounting quaternion `(w, x, y, z)`：`(1, 0, 0, 0)`。
- Sensor period 與 physics dt 都是 `0.005 s`。
- Topic：唯一使用 `/imu/data`；policy node 預設直接訂閱該 topic。
- 動態 `/tf` 使用 `/odom` 的同一筆 pose 與 simulation timestamp，發布
  `odom → base_link`；identity-mount IMU 的 `frame_id` 直接為 `base_link`。
- Graph 由 `OnPhysicsStep` 驅動，sensor time 單調遞增；不是由 50 Hz policy
  impulse 重播資料。
- RTX 5080 即時 DDS 實收率為 `196.046 Hz`（1000-sample window），通過
  至少 180 Hz 的目標。
- 靜止訊息實測 linear acceleration 為
  `(0.00543, 0.00108, 9.80877) m/s²`，z 軸包含正常重力加速度。
- `/tf` 實收率約 `49.2 Hz`，且 `tf2_echo odom base_link` 可解析。
- 250-step external closed loop、100 筆 parity sample：
  - base angular velocity 最大誤差：`8.03e-4`。
  - projected gravity 最大誤差：`2.11e-4`。
  - 0 termination、0 timeout。

Isaac Sim 5.1 的已知限制：

- GPU articulation 上 `IsaacReadIMU` 的 orientation 固定為 identity，但
  angular velocity、linear acceleration 與 sensor time 正常。
- 因此 orientation 改由同一個 200 Hz physics event 的
  `IsaacComputeOdometry` 取得；不是低頻 base state 重播。
- `IsaacReadIMU` 的 angular velocity 是 world frame，graph 以同一筆
  orientation 轉為 `base_link`。
- `IsaacComputeOdometry` orientation 相對 reset pose；bridge 會先合成
  configured initial base orientation，再把 world-frame angular velocity
  轉到 `base_link`。非零 spawn yaw `+90°` 的實測 angular velocity parity
  最大誤差為 `3.17e-4`。
- Episode reset 後 PhysX IMU 可能保留一個 physics tick 的 pre-reset
  sample；validator 將這一 tick 明列為 reset transient metric，正常 contract
  從下一 tick 立即恢復檢查，不把 transient 混成行走期間 warning。
- 真實 sensor 取樣相對 policy state 有一個 physics-step 等級的差異，因此
  IMU angular velocity 與 projected gravity 的 parity tolerance 使用
  `2e-3`；啟用 RTX off-screen rendering 的 LIO-SAM 驗證使用 `1e-2`
  （實測 angular velocity 最大差異 `0.008107`）。其他 observation term
  仍維持 `1e-4`。

### 第四階段：RTX LiDAR 與 LIO-SAM

架構沿用前三階段，不把 SLAM 塞進 locomotion policy 或 Isaac Sim process：

```text
Isaac Sim RTX LiDAR
  -> ROS 2 Bridge /lidar/points_raw
  -> 外部 PointCloud2 adapter
  -> /lio_sam/points
  -> upstream Image Projection
  -> project-owned motion deskew
  -> upstream Feature Extraction / Map Optimization

真正的 200 Hz /imu/data --------------------^
200 Hz incremental odometry ----------------^
```

已確認：

- 使用 Isaac Sim 內建 `OS1_REV6_32ch10hz1024res` RTX profile：
  32 channels、10 Hz、1024 個水平 sample。
- LiDAR frame 為 `lidar_link`，相對 `base_link` 的 mounting translation
  為 `(0.20, 0, 0.35) m`，rotation 為 identity。
- RTX raw PointCloud2 由外部 adapter 補上 LIO-SAM Ouster contract 所需的
  `ring` 與每點相對時間 `t`；點依 `t` 排序，header 使用 scan-start
  simulation time。
- Official LIO-SAM ROS 2 branch 以 `lio_sam.repos` 固定 exact commit，不
  複製或修改 upstream source；專案擁有參數與 launch。
- LIO-SAM 與專案 ROS package 已在乾淨的 ROS Humble environment 完成
  colcon build，所有 adapter／LIO nodes 可啟動。
- 本機大型 PointCloud2 使用 CycloneDDS／Iceoryx shared memory，避免 UDP
  fragmentation 丟 scan；`/tf_static` publisher 則保留 UDP transient
  history。Bringup 會自動啟動 RouDi，不要求使用者設定 RMW。
- 不修改 upstream Image Projection；專案在其後加入 motion deskew，以 raw
  IMU quaternion integration 處理旋轉，並以 200 Hz incremental odometry
  interpolation 處理 point translation。
- LiDAR 不加入 Flat v1 的 48 維 observation；它屬於獨立感知層。
- Deployment host 使用 repository 內的
  `assets/maps/factory/Factory_Layout.usd` 及完整相依 OpenUSD 資產，
  取代原本的 Flat plane；預設出生位置為 `(0, -18, 0)`。Training task
  與 policy 契約仍維持官方 Flat baseline。
- `bringup.launch.py` 預設繼承啟動 shell 的 `ROS_DOMAIN_ID`，未設定時
  回退到 `1`，並將相同 domain 傳給 policy、LIO-SAM、adapter、RViz2、
  Factory simulation 與官方 keyboard teleop；正常操作不需逐項傳入
  launch argument，也不必為此修改 `.bashrc`。
- Factory root layer 對所有 collision 強制繼承 `1.0/1.0 + multiply`
  physics material；配合訓練事件套在 robot rigid bodies 的 `0.8/0.6`
  材質，有效接觸摩擦對齊 Flat training 的 `0.8/0.6`。
- 不在 teleop 與 policy 之間加入固定速度 filter；`/cmd_vel` 仍只受
  High-Speed policy 訓練範圍限制。後續研究方向是比較多種 3D SLAM 的
  信心指標，將信心度納入 PPO observation／控制，使 policy 依特徵可觀測性
  自主減速或提速。

TF 採單一 publisher ownership：

```text
map --static--> odom --LIO-SAM IMU preintegration--> base_link
                                               |
                                               +--static--> lidar_link
```

啟用 `--enable-lio-sam` 時，模擬器停發 ground-truth
`odom → base_link` TF。Upstream map-optimization 的重複 TF remap 到隔離
topic；`/odom` topic 仍保留給 locomotion policy 使用。

Runtime 驗收結果：

- 重開機後 NVIDIA userspace、kernel module 與已載入 driver 均為
  `580.173.02`，Isaac Sim 以 Vulkan 正常使用 RTX 5080。
- Headless RTX sensor 必須啟用 off-screen rendering，並使用官方
  `RtxLidarROS2PublishPointCloudBuffer` writer；只建立 render product
  不會產生 point cloud frame。
- raw 與 adapter point cloud 持續輸出；off-screen 負載下 wall-time
  接收率約 `6.8–8.1 Hz`，而 sensor profile 與 simulation timestamp
  維持 10 Hz。整體 RTF 小於 1 時，不把 wall rate 偽稱為 10 Hz。
- `/lio_sam/points` 的 `ring` 完整涵蓋 `0..31`，每點 `t` 約涵蓋
  `0..99,999,992 ns`，header 已轉成 scan-start simulation time。
- Image Projection 輸出 `imu_available=1`、`odom_available=1`；
  deskew、feature、mapping odometry 與
  `map → odom → base_link → lidar_link` TF 均可正常輸出。
- Factory 場景取代原本為無限平面 smoke test 建立的七個臨時 visual
  landmark。Factory 摩擦材質已對齊訓練；高速直行裂圖另外以完整 scan
  motion deskew 與無丟包點雲路徑修正，不以固定速度上限遮蔽。
- `forward_3_0` deterministic profile 在實速 3.05–3.13 m/s 連續三次
  通過：translation ATE 1.37–1.55 cm、最大 translation jump
  3.5–4.2 cm、vertical wall separation 6.9–10.0 cm，且 ready 後
  scan／IMU／odometry coverage 零缺失。
- 零殘留啟動的 60 秒測試收到 85 筆 mapping correction，全部嚴格遞增；
  四個 LIO-SAM 核心 process 在測試結束後仍存活。重置 simulation time
  前必須一起重啟 LIO-SAM，避免殘留同名 publisher 將兩套 correction
  送入 GTSAM。
- Factory 資產可攜性檢查共有 135 個使用中的 layer，指向舊 workspace
  或其他外部絕對路徑的 layer 數均為 0，並保留 2,082 個 collision prim。
- Factory 的 5-step GPU smoke test 已通過；單一 bringup 的短時間 runtime
  也確認 `/joint_command`、`/lio_sam/points` 與 mapping odometry 有輸出，
  結束後無殘留 process。先前 85 筆 correction 的 60 秒結果屬於舊的臨時
  landmark 場景；Factory 移動精度改由版本化 deterministic benchmark
  獨立驗收。

2026-07-29 完成 Recovery v0.4.0 qualification：

- model 1450 在 RTX/LIO 負載下完成 23 秒 `0.5 m/s + 0.25 rad/s` 圓周，
  停止後 10 秒可恢復站姿；最後 5 秒平面位移 `0.00010 m`、yaw 變化
  `0.00038 rad`，最大 roll/pitch 分別 `0.195/0.106 rad`。
- 正式 turning matrix 36/36 通過：純 yaw `±0.5/±1/±2 rad/s`，以及左右
  `0.5/1.5/3.0 m/s` 曲線，每個 profile 三次。
- true loop-closure qualification 12/12 通過。兩種閉合路徑三次皆在
  enabled replay 產生 constraint、disabled replay 為零；兩種開放路徑
  enabled/disabled 都沒有誤閉環。每個 source bag 都以 fresh LIO graph
  各 replay 兩次。

仍未完成的是實體 ANYmal-D sensor extrinsic 與實體 low-level interface。
模擬 host 的 200 Hz IMU `angular_velocity` warning 已拆成並修正非零初始
yaw frame contract，episode-reset 的單 tick transient 另行量測；沒有改動
upstream LIO-SAM。

## 暫時不做

- 不加入 camera、LVI-SAM、Nav2 或完整 navigation stack。
- 不把 LiDAR／camera observation 放進 Flat v1 policy。
- 不直接處理實體 ANYmal-D low-level interface。
- 不盲目繼續訓練 model_1298。
- 不為了讓數字好看而偽造 sensor frequency。

## Git checkpoint

第一階段、GPU bridge 與第三階段 IMU 都已有可重現驗證結果。
