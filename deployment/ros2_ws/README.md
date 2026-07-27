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
  └─ /tf: odom → base_link   /cmd_vel
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

```bash
cd /home/ros/anymal_locomotion/deployment/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select anymal_locomotion_ros2
source install/setup.bash
```

## ONNX 是什麼

可以把三個檔案這樣理解：

- `model_1298.pt`：完整訓練存檔，可續訓，也包含 optimizer 等訓練資料。
- `policy.onnx`：只留下「48 個數字輸入、12 個 action 輸出」的推論模型。
- `policy_metadata.yaml`：關節順序、預設角度、scale、command limits 等接線
  說明。

ONNX 不會重新訓練，也不會改變權重；它只是讓外部 ROS 2 Python 不必安裝
完整 PyTorch。這個專案的 checkpoint／TorchScript／ONNX parity 已通過。

## 推論 runtime

ROS 2 系統 Python 已有 `rclpy` 與 NumPy。預設使用 `policy.onnx` 與 ONNX
reference evaluator；實測單次推論低於 0.1 ms，低於 50 Hz 的 20 ms 預算。
依賴安裝在 repository 內、colcon workspace 外：

```bash
cd /home/ros/anymal_locomotion
python3 -m pip install \
  --target deployment/python_vendor \
  -r deployment/ros2_ws/requirements-inference.txt
```

建置後可啟動：

```bash
cd /home/ros/anymal_locomotion
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash
export PYTHONPATH=/home/ros/anymal_locomotion/deployment/python_vendor:${PYTHONPATH}

ros2 run anymal_locomotion_ros2 policy_node --ros-args \
  -p use_sim_time:=true \
  -p backend:=onnx \
  -p policy_path:=/home/ros/anymal_locomotion/exported/anymal_d_locomotion_v1/high_speed_v0.2.0/policy.onnx \
  -p metadata_path:=/home/ros/anymal_locomotion/exported/anymal_d_locomotion_v1/high_speed_v0.2.0/policy_metadata.yaml
```

也保留 `backend:=torchscript`，但必須在外部 ROS 2 Python 另裝 PyTorch，並
把 `policy_path` 改成 `policy.pt`。兩種方式都不會把 `rclpy` 塞進 Isaac Sim。

## GPU 模擬與鍵盤控制

ROS 2 workspace 位於：

`/home/ros/anymal_locomotion/deployment/ros2_ws`

先在 Terminal 1 啟動上面的 policy node，再於 Terminal 2 啟動鍵盤控制：

```bash
cd /home/ros/anymal_locomotion
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash
ros2 run anymal_locomotion_ros2 keyboard_teleop
```

鍵盤視窗必須保持焦點。按住按鍵才持續送命令，放開後會在 0.1 秒內停止：

- `W/S`：前進／後退
- `Q/E`：向左／向右側移
- `A/D`：向左／向右旋轉
- `Space`：立即停止
- `+/-`：調整速度倍率
- 數字鍵盤 `KP_Add/KP_Subtract`：同樣調整速度倍率

Terminal 3 啟動 GPU simulation host；不要加 `--headless` 才能看到視窗：

```bash
cd /home/ros/anymal_locomotion
TERM=xterm-256color PYTHONPATH=source/anymal_locomotion \
  /home/ros/IsaacLab/isaaclab.sh -p \
  scripts/validation/validate_ros2_bridge.py \
  --device cuda:0 --steps 1000000 --real-time --external-control \
  --disable-episode-timeout
```

另開終端可檢查 topic 與實際接收頻率：

```bash
source /opt/ros/humble/setup.bash
ros2 topic list
ros2 topic hz /joint_states
ros2 topic hz /odom
ros2 topic hz /tf
ros2 topic hz /imu/data
ros2 topic hz /joint_command
```

目前 GPU 即時驗證結果：`/imu/data` 靜止時
`linear_acceleration.z=9.80877 m/s²`、接收率約 `191.5 Hz`；`/tf` 為
`odom → base_link`、接收率約 `49.2 Hz`，且可由
`tf2_echo odom base_link` 查詢。

若有設定 `ROS_DOMAIN_ID` 或 `RMW_IMPLEMENTATION`，所有終端必須使用相同值。
綠色箭頭是收到的 body-frame `/cmd_vel` 目標；藍色箭頭是機器人的實際速度。

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
- 固定 `[0.5, 0, 0]` 的 10 秒測試可前進 4.88 m、偏航 6.04°，但橫向偏移
  0.61 m，未達原訂 0.30 m；原生 checkpoint 評估也有同方向的小幅
  `vy/wz` bias，因此這是目前 policy 的低速 tracking 限制，不是 ROS 軸向錯接。

專案內的 Action Graph builder、已驗證 topic 與 headless host 注意事項見
[Action Graph v0.2 contract](../../action_graph/README.md)。
