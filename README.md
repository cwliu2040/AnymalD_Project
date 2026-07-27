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
├── assets/maps/factory/        # project-local Factory OpenUSD 與相依資產
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

## 單一 Launch：Factory 建圖與鍵盤控制

這是日常人工操作的正式入口。第一次執行前，須先依
[ROS 2 deployment README](deployment/ros2_ws/README.md) 完成 workspace
建置與 `deployment/python_vendor` 安裝。之後只需一個啟動命令：

```bash
cd /home/ros/anymal_locomotion
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash

ros2 launch anymal_locomotion_ros2 bringup.launch.py
```

此 launch 會繼承啟動 shell 的 `ROS_DOMAIN_ID`，若未設定則使用 `1`；
ONNX policy 路徑、Factory map、`cuda:0`、LIO-SAM、RViz2、RTX LiDAR、
simulation parity tolerance 與 episode timeout 設定均已有預設值，不必
在每次啟動時逐項傳入。它會開啟 Isaac Sim、RViz2，並另外開一個 GNOME
Terminal 執行官方 `teleop_twist_keyboard`。

目前 `~/.bashrc` 已 source ROS 2 Humble 與本專案建置後的 workspace，
因此新開的互動式 Terminal 可直接執行上述 `ros2 launch`，不必修改
`.bashrc` 或再次手動 source。上方完整 source 流程保留作為乾淨環境與
問題排查用；非互動式 shell 或其他尚未設定的電腦仍須先 source。

Factory 地圖及其相依 OpenUSD 資產位於
`assets/maps/factory/`。ROS 2 deployment host 以 Factory USD terrain
取代原本的 Flat plane，ANYmal 預設出生於 `(x=0, y=-18, yaw=0)`；訓練
task 與 48 維 policy 契約仍維持官方 Flat baseline。

保持 teleop terminal 焦點，按住按鍵控制：

- `i/,`：前進／後退
- `j/l`：向左／向右旋轉
- `Shift+j`／`Shift+l`：向左／向右側移
- `k`：停止
- `q/z`：同時提高／降低線速度與角速度
- `w/x`：提高／降低線速度
- `e/c`：提高／降低角速度

模擬視窗中的綠色箭頭是 `/cmd_vel` 目標，藍色箭頭是實際速度。若沒有
動作，確認 teleop terminal 保持焦點，並檢查 `/cmd_vel` 與
`/joint_command` 是否持續發布。

重新啟動 simulation time 前，必須先在主 launch terminal 按 `Ctrl-C`
關閉整組 process，並確認自動開啟的 teleop terminal 也已關閉，避免舊
LIO-SAM publisher 把兩套 correction 送入 GTSAM。這份 quick start 與
[deployment/ros2_ws/README.md](deployment/ros2_ws/README.md) 的
「單一 Bringup Launch」互相對應；未來若 executable、路徑、參數、按鍵或
啟動 process 改變，必須在同一批修改中同步更新兩處。

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
