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

它會下載 LFS、匯入固定版本的 LIO-SAM、FAST-LIO2 與 Livox dependencies、
套用 project-owned FAST-LIO2 compatibility patch、安裝 ONNX／ROS
dependencies，並執行 colcon build。既有 checkout 可用
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
  -p policy_path:="${ANYMAL_PROJECT_ROOT}/exported/anymal_d_locomotion_v1/recovery_v0.4.0/policy.onnx" \
  -p metadata_path:="${ANYMAL_PROJECT_ROOT}/exported/anymal_d_locomotion_v1/recovery_v0.4.0/policy_metadata.yaml"
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

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-未設定}"
ros2 launch anymal_locomotion_ros2 bringup.launch.py
```

若要重現長時間行走後停止仍持續搖晃的問題，可只在該次正式 bringup
開啟 project-owned locomotion diagnostics（預設關閉）：

```bash
ros2 launch anymal_locomotion_ros2 bringup.launch.py \
  enable_locomotion_diagnostics:=true
```

診斷會寫入
`logs/formal_bringup/latest/locomotion_diagnostics.json`，並在同一目錄留下
`locomotion_diagnostics.jsonl` 增量 trace。JSONL 每 25 個 policy step flush
一次，包含實際 command、base roll／pitch、四腳 contact force、stance
tangential speed 與 simulation 實際套用的 policy raw action；停止時才產生
完整的 JSON summary。若 bringup 被強制中止，仍可先用 JSONL 保留已 flush
的資料做事後分析。需要保留多次測試時，請用
`locomotion_diagnostics_dir:=<repository 內的新目錄>` 避免覆寫；repository
外的路徑會由 simulation host 拒絕。

所有 launch argument 都有正式預設值，正常使用不必傳入參數：

- `project_root`：由 package source／colcon install path 自動解析
- `isaaclab_root`：優先使用 `ISAACLAB_ROOT`，否則為 `~/IsaacLab`
- `device=cuda:0`
- `ros_domain_id`：繼承啟動 shell 的 `ROS_DOMAIN_ID`；未設定時才為 `1`
- `factory_usd_path=<project_root>/assets/maps/factory/Factory_Layout.usd`
- High-Speed v0.2.0 ONNX policy 與 metadata
- `use_rviz=true`
- `open_teleop_terminal=true`

Launch 會先啟動 Iceoryx RouDi，再讓本機 CycloneDDS participant 使用
shared memory 傳輸多 MB RTX LiDAR scans。固定 TF publisher 使用獨立的
UDP-only CycloneDDS 設定，避免 Humble 所附 Iceoryx 單筆 history 限制讓
晚啟動的 RViz 收不到 `/tf_static`。這些設定只作用於 launch 子程序，使用者
不需要 export `RMW_IMPLEMENTATION` 或 `CYCLONEDDS_URI`。

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
只 clamp 到 High-Speed policy 的訓練範圍。10 Hz LiDAR 的 3 m/s 高速直行
裂圖已由 project-owned scan motion deskew 與 shared-memory 點雲路徑處理。
後續仍將比較多種 3D SLAM 的信心指標，並把信心度回饋給 PPO，讓 locomotion
policy 學習依環境可觀測性調整速度。

若 GNOME Terminal 無法自動開啟，使用
`open_teleop_terminal:=false` 啟動 bringup，再從另一個使用相同
`ROS_DOMAIN_ID` 的 ROS terminal 執行：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args \
  -p speed:=0.5 \
  -p turn:=0.5 \
  -r cmd_vel:=/cmd_vel
```

另開終端可檢查 topic 與實際接收頻率；必須使用與啟動 launch 的 shell
相同的 `ROS_DOMAIN_ID`：

```bash
source /opt/ros/humble/setup.bash
source <repository-root>/deployment/ros2_ws/install/setup.bash
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-未設定}"
ros2 topic list
ros2 topic hz /joint_states
ros2 topic hz /odom
ros2 topic hz /tf
ros2 topic hz /imu/data
ros2 topic hz /joint_command
```

ROS 2 官方建議同一網路上的不同電腦群組使用不同 domain，安全選擇範圍是
`0..101`。實驗室應依每位使用者、每台機器人或每組實驗分配 domain，而不是
讓所有人都使用 `1`。固定只操作一組系統的帳號可把自己的分配值寫入
`.bashrc`；同一帳號需要切換多組系統時，則在各 terminal 個別 export。若
所有 process 都在同一台電腦，可另外使用 `ROS_LOCALHOST_ONLY=1`；跨電腦或
連接實體機時則使用 `0` 或 unset。完整設定範例見根目錄 README。

安裝腳本不修改 `.bashrc`。Launch 無法替啟動它的 parent shell source
workspace，但會把選定的 domain 傳給其所有子程序；未設定 domain 時保留
fallback `1` 只供單機向後相容，不建議在共享網路依賴它。

第一次啟動 Factory 與 RTX LiDAR 時會建立 shader cache，wall time 觀察到的
topic rate 可能暫時偏低。cache 穩定後再量測，這不代表 physics 或 policy
頻率契約改變。

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
  -> upstream Image Projection
  -> motion_deskew（raw IMU quaternion rotation + 200 Hz odometry translation）
  -> /lio_sam/deskew/cloud_info_motion_corrected
  -> upstream Feature Extraction / Map Optimization

/imu/data（200 Hz）------------------------------------^
/lio_sam/odometry/imu_incremental（200 Hz）------------^
```

Upstream LIO-SAM ROS 2 branch 不做 point translation deskew，旋轉則使用逐軸
Euler 累加。專案不修改 upstream source，而是在 Image Projection 與 Feature
Extraction 之間重建 range-image points：旋轉由原始 `/imu/data` 做 quaternion
積分，平移由高頻 incremental odometry 插值。sim ground truth 只供 benchmark
評分，絕不回饋 scan correction。

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
  --disable-episode-timeout --enhanced-determinism --enable-lio-sam \
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

上述 60 秒結果是在舊的七個臨時 landmark 場景完成。改用 Factory 後，正式
`bringup.launch.py` 已完成 Factory、policy、RTX LiDAR、adapter、motion
deskew 與四個 LIO-SAM core 的整組 smoke test。另有 deterministic benchmark
在實速 3.05–3.13 m/s 下連續三次通過，translation ATE 為
1.37–1.55 cm、最大 pose jump 為 3.5–4.2 cm、vertical wall separation
為 6.9–10.0 cm，且穩態 scan／IMU coverage 零缺失。完整門檻、限制與重跑
命令見 [高速 LIO-SAM 驗證](../../docs/validation/lio_sam_high_speed.md)。

可在 repository root 執行單次隔離 benchmark：

```bash
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash
ros2 launch anymal_locomotion_ros2 lio_benchmark.launch.py \
  profile:=forward_3_0 \
  output_dir:="$(pwd)/outputs/lio_sam_benchmarks/forward_3_0"
```

`metrics.json` 與選用的 rosbag 都寫入 ignored `outputs/`；精簡驗證結論才放入
Git。Benchmark 預設關閉 loop closure，並以 simulation time 檢查 10 Hz
scan 完整性；wall-time rate 只用來判斷主機負載。

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
- LIO-SAM 的 3 m/s Factory 直行已完成三次定量驗收，修正版 loop-closure
  matrix 為 12/12；目前仍未驗收的是可穩定達成的高速旋轉，以及實體
  ANYmal-D sensor extrinsic 與 low-level interface。
- 固定 `[0.5, 0, 0]` 的 10 秒測試可前進 4.88 m、偏航 6.04°，但橫向偏移
  0.61 m，未達原訂 0.30 m；原生 checkpoint 評估也有同方向的小幅
  `vy/wz` bias，因此這是目前 policy 的低速 tracking 限制，不是 ROS 軸向錯接。

專案內的 Action Graph builder、已驗證 topic 與 headless host 注意事項見
[Action Graph v0.2 contract](../../action_graph/README.md)。
