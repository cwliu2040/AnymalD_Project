# ANYmal-D Locomotion

這是一個基於 Isaac Sim 5.1、Isaac Lab v2.3.2 與 RSL-RL PPO 的
ANYmal-D locomotion External Project。目標是建立乾淨、可維護，並能逐步
延伸至實體 ANYmal-D 的 sim-to-real 系統。

Flat Locomotion v1 以官方 ANYmal-D Flat task 為基準，採用
Manager-Based workflow 與 48 維 proprioceptive observation。LiDAR、
RGB-D、SLAM 與 navigation 不放入 v1 policy observation；LIO-SAM 是獨立的
ROS 2 感知層，不會改變既有 policy 契約。

## 目前基準

- 專案根目錄：`/home/ros/anymal_locomotion`
- Isaac Sim：`5.1.0`
- Isaac Lab：`v2.3.2`
- Isaac Lab commit：`37ddf626871758333d6ed89cf64ad702aef127d0`
- RL：Manager-Based、single-agent、RSL-RL PPO
- Train task：`Isaac-Velocity-Flat-Anymal-D-Locomotion-v0`
- Play task：`Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0`

Isaac Lab 必須固定在支援的 tag，不以持續變動的 `origin/main` 作為基準。

## v1 Policy 契約

- Observation：48 維
- Action：12 維 joint-position action
- Policy 頻率：50 Hz
- Physics 頻率：200 Hz
- Action scale：0.5
- Command：body-frame `[vx, vy, wz]`
- Command 範圍：`vx = [-2.0, 3.0]`、`vy = [-1.5, 1.5]`、
  `wz = [-2.0, 2.0]`
- Joint order：依 `configs/policy_contract.yaml` 固定，runtime 與 ROS array
  一律依 joint name remap
- Terrain：flat plane
- Policy 不包含 height scan、LiDAR 或 camera

## 目錄

```text
anymal_locomotion/
├── source/anymal_locomotion/   # Isaac Lab extension，不 import rclpy
├── scripts/rsl_rl/             # train / play
├── scripts/validation/         # runtime 與契約驗證
├── configs/                    # policy 與 artifact 契約
├── deployment/ros2_ws/         # 外部 ROS 2 policy/runtime
├── action_graph/               # ROS 2 Bridge / Action Graph 契約
├── logs/                       # RSL-RL run 與 TensorBoard
├── checkpoints/                # 挑選後保留的 checkpoint
├── exported/                   # TorchScript / ONNX 與 metadata
├── tests/
└── docs/
```

Isaac Lab 與 RSL-RL 都是 dependency，不會複製進本專案。

## 驗證

靜態測試：

```bash
cd /home/ros/anymal_locomotion
PYTHONPATH=source/anymal_locomotion python3 -m pytest -q tests
```

Isaac Sim runtime smoke test：

```bash
cd /home/ros/anymal_locomotion
PYTHONPATH=source/anymal_locomotion \
  /home/ros/IsaacLab/isaaclab.sh -p \
  scripts/validation/validate_project.py --headless
```

ROS 2 Action Graph smoke test：

```bash
TERM=xterm-256color PYTHONPATH=source/anymal_locomotion \
  /home/ros/IsaacLab/isaaclab.sh -p \
  scripts/validation/validate_ros2_bridge.py \
  --headless --device cuda:0 --steps 250 \
  --external-control --validate-observation-parity
```

目前已在 RTX 5080 主機完成驗證：

- 48 observations
- 12 actions
- 50 Hz policy
- 無 height scanner
- deterministic canonical-to-runtime joint mapping
- 完整 runtime smoke test 通過
- 外部 ROS 2 已接收 `/clock`、`/joint_states`、`/odom`、`/tf`、`/imu/data`
- GPU PhysX、ROS 2 Bridge 與 ONNX closed loop 已完成整合驗證
- 專案擁有的 RTX LiDAR、PointCloud2 adapter 與 LIO-SAM 已完成 60 秒
  端到端驗證；deskew、mapping odometry 與 `map → base_link` TF 可正常輸出
- ROS 組成的 48 維 observation 與 Isaac Lab observation 逐項一致
- 官方 ±1.0 baseline 已完成 300 iterations
- High-Speed v0.2.0 已由 baseline checkpoint 接續完成 1,000 iterations
  （最終 timeout 92.55%、XY velocity error 0.376 m/s）

## 三個 Terminal：開啟視窗並用鍵盤控制

這是日常人工操作的正式 quick start。三個 Terminal 都使用相同的
`ROS_DOMAIN_ID=27`，而且先清除其他 ROS workspace 可能留下的環境變數。
第一次執行前，須先依
[ROS 2 deployment README](deployment/ros2_ws/README.md) 完成 workspace
建置與 `deployment/python_vendor` 安裝。

Terminal 1：啟動外部 ONNX locomotion policy：

```bash
cd /home/ros/anymal_locomotion
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash
export PYTHONPATH=/home/ros/anymal_locomotion/deployment/python_vendor:${PYTHONPATH}
export ROS_DOMAIN_ID=27

ros2 run anymal_locomotion_ros2 policy_node --ros-args \
  -p use_sim_time:=true \
  -p backend:=onnx \
  -p policy_path:=/home/ros/anymal_locomotion/exported/anymal_d_locomotion_v1/high_speed_v0.2.0/policy.onnx \
  -p metadata_path:=/home/ros/anymal_locomotion/exported/anymal_d_locomotion_v1/high_speed_v0.2.0/policy_metadata.yaml
```

Terminal 2：啟動鍵盤控制視窗：

```bash
cd /home/ros/anymal_locomotion
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=27

ros2 run anymal_locomotion_ros2 keyboard_teleop
```

Terminal 3：啟動可見的 GPU simulation；不要加入 `--headless`：

```bash
cd /home/ros/anymal_locomotion
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=27

TERM=xterm-256color \
PYTHONPATH=/home/ros/anymal_locomotion/source/anymal_locomotion:${PYTHONPATH} \
  /home/ros/IsaacLab/isaaclab.sh -p \
  scripts/validation/validate_ros2_bridge.py \
  --device cuda:0 --steps 1000000 --real-time --external-control \
  --disable-episode-timeout
```

鍵盤控制視窗必須保持焦點，按住移動鍵才會持續送命令：

- `W/S`：前進／後退
- `Q/E`：向左／向右側移
- `A/D`：向左／向右旋轉
- `Space`：立即停止
- 主鍵盤 `+/-` 或數字鍵盤 `KP_Add/KP_Subtract`：調整速度倍率

模擬視窗中的綠色箭頭是 `/cmd_vel` 目標，藍色箭頭是實際速度。若鍵盤視窗
顯示的倍率有改變但機器人速度不變，先確認三個 Terminal 的
`ROS_DOMAIN_ID` 相同，並檢查 Terminal 1 是否持續發布 `/joint_command`。

### 同時啟用 LIO-SAM

完整 LIO-SAM 會多一個 Terminal。先啟動下列 Terminal 4，再啟動 Terminal 3：

```bash
cd /home/ros/anymal_locomotion/deployment/ros2_ws
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=27

ros2 launch anymal_locomotion_ros2 lio_sam.launch.py
```

並在 Terminal 3 的 simulation 指令最後加上：

```text
--enable-lio-sam --imu-observation-parity-atol 0.01
```

重新啟動 simulation time 前，必須先用 `Ctrl-C` 關閉舊的 LIO-SAM launch，
避免殘留同名 publisher。這份 quick start 與
[deployment/ros2_ws/README.md](deployment/ros2_ws/README.md) 的
「GPU 模擬與鍵盤控制」互相對應；未來若 executable、路徑、參數、按鍵或
Terminal 數量改變，必須在同一批修改中同步更新兩處。

## 訓練

訓練入口：

```bash
cd /home/ros/anymal_locomotion
PYTHONPATH=source/anymal_locomotion \
  /home/ros/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-D-Locomotion-v0 \
  --headless \
  --seed 42
```

High-Speed v0.2.0 預設訓練 1,000 iterations，command range 與官方 Spot
example 相同。訓練範圍不等於已驗證的最高速度，也不等於實體機安全速度；
後續須用固定速度 evaluation 驗證，實體機部署另套經安全審查的限制。

訓練產物會寫入：

`logs/rsl_rl/anymal_d_locomotion_v1/`

每個 run 會保存 resolved environment/agent config、版本與 seed manifest、
TensorBoard events 以及 RSL-RL checkpoint。Checkpoint 是可續訓或匯出
policy 的模型存檔，不會提交到 GitHub。

## Play 與匯出

有 checkpoint 後，可用 play task 載入：

```bash
PYTHONPATH=source/anymal_locomotion \
  /home/ros/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0 \
  --checkpoint <checkpoint-path>
```

固定速度量測使用 `evaluate.py`，例如以 128 個環境測試 3 m/s、20 秒：

```bash
PYTHONPATH=source/anymal_locomotion \
  /home/ros/IsaacLab/isaaclab.sh -p scripts/rsl_rl/evaluate.py \
  --headless --num_envs 128 --steps 1000 --warmup_steps 100 \
  --vx 3.0 --vy 0.0 --wz 0.0 --checkpoint <checkpoint-path>
```

結果寫入 `logs/evaluation/`；目前量測結果見
[High-Speed v0.2.0 Evaluation](docs/high_speed_evaluation.md)。

匯出產物放在 `exported/anymal_d_locomotion_v1/`，包含 TorchScript、ONNX
與 `policy_metadata.yaml`。

目前提升為 deployment candidate 的 High-Speed v0.2.0 checkpoint、artifact
hash 與 checkpoint／JIT／ONNX output parity 結果見
[High-Speed Policy v0.2.0](docs/policy_release_v0.2.0.md)。

## 架構限制

- Training 與 Isaac Sim Python 不 import `rclpy`。
- ROS 2 policy node 必須在 Isaac Sim process 外執行。
- Isaac Sim 通訊使用 ROS 2 Bridge / Action Graph。
- UDP 不作為最終架構。
- 未來 custom USD 必須通過 joint contract 驗證。

詳細內容請見：

- [系統架構](docs/architecture.md)
- [ROS 2 模擬部署對齊紀錄](docs/ros2_deployment_decisions.md)
- [Baseline 分析](docs/baseline_analysis.md)
- [ROS 2 Policy Runtime v0.2](deployment/ros2_ws/README.md)
