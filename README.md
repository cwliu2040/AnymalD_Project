# ANYmal-D Locomotion

這是一個基於 Isaac Sim 5.1、Isaac Lab v2.3.2 與 RSL-RL PPO 的
ANYmal-D locomotion External Project。目標是建立乾淨、可維護，並能逐步
延伸至實體 ANYmal-D 的 sim-to-real 系統。

Flat Locomotion v1 以官方 ANYmal-D Flat task 為基準，採用
Manager-Based workflow 與 48 維 proprioceptive observation。LiDAR、
RGB-D、SLAM 與 navigation 不放入 v1 policy observation。

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
├── deployment/ros2_ws/         # 未來外部 ROS 2 policy/runtime
├── action_graph/               # 未來 ROS 2 Bridge / Action Graph
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

目前已在 RTX 5080 主機完成驗證：

- 48 observations
- 12 actions
- 50 Hz policy
- 無 height scanner
- deterministic canonical-to-runtime joint mapping
- 完整 runtime smoke test 通過
- 官方 ±1.0 baseline 已完成 300 iterations
- High-Speed v0.2.0 已由 baseline checkpoint 接續完成 1,000 iterations
  （最終 timeout 92.55%、XY velocity error 0.376 m/s）

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

## 架構限制

- Training 與 Isaac Sim Python 不 import `rclpy`。
- ROS 2 policy node 必須在 Isaac Sim process 外執行。
- Isaac Sim 通訊使用 ROS 2 Bridge / Action Graph。
- UDP 不作為最終架構。
- 未來 custom USD 必須通過 joint contract 驗證。

詳細內容請見：

- [系統架構](docs/architecture.md)
- [Baseline 分析](docs/baseline_analysis.md)
