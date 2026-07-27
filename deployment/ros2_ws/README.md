# ROS 2 Policy Runtime v0.2

## 這一版解決什麼

訓練時，Isaac Lab 直接把機器人狀態交給 policy。部署時，policy node 是
Isaac Sim 外部的獨立程式，因此狀態必須透過 ROS 2 Bridge 傳出，policy
算完後再把 joint target 傳回去。

```text
Isaac Sim
  ├─ /joint_states ─┐
  ├─ /imu/data ─────┼─> 外部 policy node ─> /joint_command ─> Isaac Sim
  ├─ /odom ─────────┘           ↑
  ├─ /tf: odom → base_link   /cmd_vel
  └─ /lidar/points_raw ─> adapter ─> LIO-SAM（獨立感知層）
```

`anymal_locomotion_ros2` 已實作第一版外部 policy node 與 ROS-independent
核心邏輯。它不會被 Isaac Lab training package import，也不會在 Isaac Sim
Python 中執行 `rclpy`。

## 為什麼需要 `/odom`

IMU 量到角速度、方向與加速度，但加速度積分成速度會快速累積 bias 和 drift，
所以本版不從 IMU 猜 base linear velocity：

- `/odom.twist.twist.linear`：body-frame base linear velocity
- `/imu/data.angular_velocity`：body-frame base angular velocity
- `/imu/data.orientation`：計算 body-frame projected gravity，`frame_id=base_link`
- `/joint_states`：依 joint name 重排 position／velocity
- `/cmd_vel`：body-frame `[vx, vy, wz]`

Isaac Compute Odometry 提供的 local linear velocity 直接接到
ROS 2 Publish Odometry，並使用 `publishRawVelocities=true` 避免 publisher
再次轉換。World angular velocity則先依每一幀的 base orientation 轉成
body frame。`child_frame_id` 固定為 `base_link`。實體機必須由 robot state
estimator 提供同一語意的 `/odom`。

同一筆 odometry pose 與 simulation timestamp 也會發布成動態
`/tf`：`odom → base_link`。IMU identity-mount 在 `base_link`，因此目前
不另造一個內容相同的 `imu_link`。

ROS 2 deployment host 現在會在 physics 啟動前建立真正的 Isaac Sim IMU
prim，並在每個 0.005 秒 physics step 發布 `/imu/data`。Policy timer 仍為
0.02 秒，每次使用 callback 收到的最新 IMU sample。Policy 與未來 SLAM
共用這一個 topic；目前尚未加入 noise 或 bias model。

此版本不發布舊 `/imu`。若 `ros2 topic info /imu -v` 只看到舊 policy
subscription，代表 ROS 2 workspace 尚未重建或 process 尚未重啟；請以
`/imu/data` 的 `_ROS2PolicyBridge_PublishImu` publisher 為準。

## Policy observation

每個 0.02 秒 tick 組成：

| Offset | 維度 | 來源 |
|---:|---:|---|
| 0 | 3 | `/odom` base linear velocity |
| 3 | 3 | `/imu/data` angular velocity |
| 6 | 3 | `/imu/data` orientation 算出的 projected gravity |
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

新電腦的正式安裝入口在根目錄 README。完成 Isaac Sim、Isaac Lab、ROS 2
與 Git LFS 安裝後，由 repository root 執行：

```bash
./scripts/setup_deployment.sh
```

它會下載 LFS、匯入固定版本的 LIO-SAM、安裝 ONNX／ROS dependencies，並
執行 colcon build。既有 checkout 可用
`./scripts/setup_deployment.sh --check` 做唯讀完整性檢查。

## ONNX 是什麼

可以把三個檔案這樣理解：

- `model_1298.pt`：完整訓練存檔，可續訓，也包含 optimizer 等訓練資料。
- `policy.onnx`：只留下「48 個數字輸入、12 個 action 輸出」的推論模型。
- `policy_metadata.yaml`：關節順序、預設角度、scale、command limits 等接線
  說明。

ONNX 不會重新訓練，也不會改變權重；它只是讓外部 ROS 2 Python 不必安裝
完整 PyTorch。這個專案的 checkpoint／TorchScript／ONNX parity 已通過。

## 推論 runtime

ROS 2 系統 Python 已有 `rclpy` 與 NumPy。預設使用 repository 版本化的
`policy.onnx` 與 ONNX
reference evaluator；實測單次推論低於 0.1 ms，低於 50 Hz 的 20 ms 預算。
依賴安裝在 repository 內、colcon workspace 外：

```bash
cd <repository-root>
python3 -m pip install \
  --target deployment/python_vendor \
  -r deployment/ros2_ws/requirements-inference.txt
```

建置後可啟動：

```bash
cd <repository-root>
export ANYMAL_PROJECT_ROOT="$(pwd)"
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash
export PYTHONPATH="${ANYMAL_PROJECT_ROOT}/deployment/python_vendor:${PYTHONPATH}"

ros2 run anymal_locomotion_ros2 policy_node --ros-args \
  -p use_sim_time:=true \
  -p backend:=onnx \
  -p policy_path:="${ANYMAL_PROJECT_ROOT}/exported/anymal_d_locomotion_v1/high_speed_v0.2.0/policy.onnx" \
  -p metadata_path:="${ANYMAL_PROJECT_ROOT}/exported/anymal_d_locomotion_v1/high_speed_v0.2.0/policy_metadata.yaml"
```

也保留 `backend:=torchscript`，但必須在外部 ROS 2 Python 另裝 PyTorch，並
把 `policy_path` 改成 `policy.pt`。兩種方式都不會把 `rclpy` 塞進 Isaac Sim。

## 單一 Bringup Launch

根目錄 [README quick start](../../README.md#單一-launchfactory-建圖與鍵盤控制)
是日常操作的正式入口。完成建置後，一個 launch 同時啟動外部 ONNX policy、
LIO-SAM、PointCloud2 adapter、RViz2、Factory GPU simulation，以及官方
`teleop_twist_keyboard`：

```bash
cd <repository-root>
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash

ros2 launch anymal_locomotion_ros2 bringup.launch.py
```

所有 launch argument 都有正式預設值，正常使用不必傳入參數：

- `project_root`：由 package source／colcon install path 自動解析
- `isaaclab_root`：優先使用 `ISAACLAB_ROOT`，否則為 `~/IsaacLab`
- `device=cuda:0`
- `ros_domain_id`：繼承啟動 shell 的 `ROS_DOMAIN_ID`，未設定時為 `1`
- `factory_usd_path=<project_root>/assets/maps/factory/Factory_Layout.usd`
- High-Speed v0.2.0 ONNX policy 與 metadata
- `use_rviz=true`
- `open_teleop_terminal=true`

Factory USD 取代 deployment host 原本的 Flat plane，預設 spawn pose 是
`(0, -18, 0)`。啟用 LIO-SAM 時直接使用 Factory 幾何進行 scan matching，
不再建立 smoke-test 專用的七個 visual landmark。這只改 deployment
場景；training task 與 policy observation 仍是 Flat v1 契約。
Factory collision 的地面 physics material 明確設為
`static=1.0`、`dynamic=1.0`、combine=`multiply`；與訓練時 ANYmal
rigid-body 的 `0.8/0.6` 材質相乘後，有效接觸摩擦維持 `0.8/0.6`。

Launch 會自動開一個 GNOME Terminal。保持該視窗焦點並使用官方按鍵：

- `i/,`：前進／後退
- `j/l`：向左／向右旋轉
- `Shift+j`／`Shift+l`：向左／向右側移
- `k`：停止
- `q/z`：同時提高／降低線速度與角速度
- `w/x`：提高／降低線速度
- `e/c`：提高／降低角速度

Teleop 直接發布 `/cmd_vel`；deployment 不加入固定速度 filter，command
只 clamp 到 High-Speed policy 的訓練範圍。已知高速會提高 10 Hz LiDAR
scan matching 的裂圖風險，後續將比較多種 3D SLAM 的信心指標，並把信心度
回饋給 PPO，讓 locomotion policy 學習依環境可觀測性調整速度。

另開終端可檢查 topic 與實際接收頻率；必須使用與啟動 launch 的 shell
相同的 `ROS_DOMAIN_ID`。若啟動時未另外設定，預設為 `1`：

```bash
source /opt/ros/humble/setup.bash
source <repository-root>/deployment/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=1
ros2 topic list
ros2 topic hz /joint_states
ros2 topic hz /odom
ros2 topic hz /tf
ros2 topic hz /imu/data
ros2 topic hz /joint_command
```

安裝腳本不修改 `.bashrc`。Launch 無法替啟動它的 parent shell source
workspace，但會把選定的 domain 傳給其所有子程序。

目前 GPU 即時驗證結果：`/imu/data` 靜止時
`linear_acceleration.z=9.80877 m/s²`、接收率約 `191.5 Hz`；`/tf` 為
`odom → base_link`、接收率約 `49.2 Hz`，且可由
`tf2_echo odom base_link` 查詢。

綠色箭頭是收到的 body-frame `/cmd_vel` 目標；藍色箭頭是機器人的實際速度。

重新啟動 simulation time 前，先在主 terminal 用 `Ctrl-C` 關閉整個
bringup，並確認 teleop terminal 已關閉。若 executable、路徑、參數、按鍵
或啟動 process 改變，必須同步更新本節與根目錄 README。

## RTX LiDAR 與 LIO-SAM

LIO-SAM 使用官方 `TixiaoShan/LIO-SAM` 的 ROS 2 branch，版本固定在
`deployment/ros2_ws/lio_sam.repos` 記錄的 commit。Upstream source checkout
位於 `src/lio_sam`，由 `.gitignore` 排除；專案只版本化 manifest、參數、
launch 與資料格式 adapter，避免複製第三方 source。

資料流固定為：

```text
Isaac Sim RTX LiDAR（OS1 32ch、10 Hz、1024 horizontal samples）
  -> /lidar/points_raw
  -> lidar_point_adapter（補 ring 與每點相對時間 t）
  -> /lio_sam/points
  -> LIO-SAM

/imu/data（200 Hz）------------------------------------^
```

啟用 LIO-SAM 時，deployment host 會在 Isaac Sim 啟動前開啟 RTX Motion BVH
與 Hydra engine masking。Isaac Sim 5.1 預設關閉 Motion BVH，但移動中的
rotating LiDAR 需要它才能正確產生 intra-scan motion effect。這會增加 VRAM
與 rendering 負擔；`getSimulationTimeMonotonicAtTime` 是否消失仍須另外以
LiDAR header、`/clock` 與移動建圖驗證，不能只把 warning 隱藏。

LiDAR mount 是 `base_link → lidar_link =
(x=0.20, y=0, z=0.35, R=identity)`。LIO-SAM 的正式 TF tree 為
`map → odom → base_link → lidar_link`。啟用 LIO-SAM 時，模擬器不再發布
ground-truth `odom → base_link`；該 transform 由 LIO-SAM IMU
preintegration 單獨負責。`/odom` topic 仍可供 locomotion policy 使用。

目前 login shell 可能繼承其他 ROS workspace。為確保不誤用舊 workspace 的
LIO-SAM，請用乾淨環境匯入並建置：

```bash
cd <repository-root>/deployment/ros2_ws
vcs import --input lio_sam.repos --skip-existing src

unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to anymal_locomotion_ros2
source install/setup.bash
```

若只需單獨除錯 adapter 與 LIO-SAM，仍可使用：

```bash
cd <repository-root>/deployment/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch anymal_locomotion_ros2 lio_sam.launch.py
```

若只需單獨除錯包含 Factory 與 RTX LiDAR 的模擬 host：

```bash
cd <repository-root>
TERM=xterm-256color PYTHONPATH=source/anymal_locomotion \
  "${ISAACLAB_ROOT:-${HOME}/IsaacLab}/isaaclab.sh" -p \
  scripts/validation/validate_ros2_bridge.py \
  --device cuda:0 --steps 1000000 --real-time --external-control \
  --disable-episode-timeout --enable-lio-sam \
  --imu-observation-parity-atol 0.01
```

驗收時至少檢查：

```bash
ros2 topic hz /lidar/points_raw
ros2 topic echo /lio_sam/points --once
ros2 topic hz /imu/data
ros2 topic echo /lio_sam/mapping/odometry --once
ros2 run tf2_ros tf2_echo map base_link
ros2 run tf2_ros tf2_echo base_link lidar_link
```

2026-07-27 重開機後，NVIDIA userspace、kernel module 與已載入 driver
皆為 `580.173.02`。RTX 5080 的 60 秒端到端驗證已通過：

- raw 與 adapter point cloud 都持續輸出；在 off-screen RTX rendering
  負載下，wall-time 接收率約 `6.8–8.1 Hz`。Sensor profile 與 simulation
  timestamp 仍是 10 Hz；此差異來自整體 RTF 小於 1，不把 wall rate 偽稱
  為 10 Hz。
- adapter 輸出 `ring=0..31`，每點 `t` 約涵蓋
  `0..99,999,992 ns`，header 為 scan-start simulation time。
- Image Projection 實測 `imu_available=1`、`odom_available=1`，並持續
  發布 deskew cloud；mapping odometry 與
  `map → odom → base_link → lidar_link` TF 可解析。
- 零殘留啟動的 60 秒測試收到 85 筆 mapping correction，時戳全部嚴格
  遞增；四個 LIO-SAM 核心 process 在測試結束後仍存活。

上述 60 秒結果是在舊的七個臨時 landmark 場景完成。改用 Factory 後已完成
新的短時間整組 bringup smoke test：Factory、policy、RTX LiDAR、adapter 與
四個 LIO-SAM core 均由 `bringup.launch.py` 啟動；`/joint_command` 與
`/lio_sam/points` 都是單一 publisher／subscriber，並已收到 Factory 場景的
mapping odometry。移動建圖精度與裂圖仍需用鍵盤實際走動後驗收，不能沿用
舊 landmark 的 60 秒結果宣稱通過。

每次重跑 simulation time 前應一起重啟整組 LIO-SAM launch。若殘留舊的
adapter／feature／mapping process，同名 topic 會出現多個 publisher，
IMU preintegration 可能因收到兩套相同時戳的 correction 而使 GTSAM
因子圖失效。可先用 `ros2 topic info` 確認核心 topic 的 publisher count
皆為 1。

## 第一版安全行為

- joint names 缺少、重複或多出時不發布 command。
- `/odom`、`/imu/data` 或 `/joint_states` 超過 0.1 秒未更新時不發布 command。
- `/cmd_vel` 超過 0.5 秒未更新時自動使用零速度。
- command 會 clamp 到 policy 的訓練範圍。
- observation 或 policy output 出現 NaN／Inf 時不發布 command。
- raw action 絕對值超過 10 時由 simulation guard 拒絕。
- simulation adapter 連續 0.1 秒沒有新 `/joint_command` 時退回零 raw action。

這些只是模擬端 guard，不是實體 ANYmal-D 的 safety controller。

## 尚未完成

- 第一版使用 latest-sample，不做多 topic 精確時間同步。
- `/joint_command` 只用於 Isaac Sim；實體 ANYmal-D low-level interface
  尚未確認。
- 尚未加入獨立 IMU noise/bias model。
- LIO-SAM 的靜止端到端 smoke test 已通過；移動建圖精度、回環與實體
  ANYmal-D sensor extrinsic 仍未驗收。
- 固定 `[0.5, 0, 0]` 的 10 秒測試可前進 4.88 m、偏航 6.04°，但橫向偏移
  0.61 m，未達原訂 0.30 m；原生 checkpoint 評估也有同方向的小幅
  `vy/wz` bias，因此這是目前 policy 的低速 tracking 限制，不是 ROS 軸向錯接。

專案內的 Action Graph builder、已驗證 topic 與 headless host 注意事項見
[Action Graph v0.2 contract](../../action_graph/README.md)。
