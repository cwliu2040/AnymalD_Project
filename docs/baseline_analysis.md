# ANYmal-D Locomotion 基準分析

分析日期：2026-07-23

> 實作更新（2026-07-23）：本文件第 3 節記錄的是實作前的
> workspace 狀態。External Project baseline 現已建立。已確認的版本與架構
> 決策整理在第 12 節，並優先於較早的待確認項目。

## 1. 結論摘要

- 實際專案根目錄是 `/home/ros/anymal_locomotion`。
  `/home/ros/Documents/anymal_locomotion` 目前不存在。
- 分析當時，新專案尚未建立 Isaac Lab External Project scaffold，且
  `.git/` 是空目錄，因此此專案當時不是有效的 Git repository。
- 歷史分析使用的 Isaac Lab checkout 是
  `v2.2.1-232-gcbf51abb5e`。本報告涉及的官方
  locomotion task 與 robot asset 檔案沒有本地修改；checkout 的 dirty
  狀態來自其他未追蹤內容。
- 官方 ANYmal-D Flat 是 Manager-Based velocity locomotion 共用設定的
  薄覆寫：4096 environments、50 Hz policy、48-dimensional
  proprioceptive observation、12-dimensional joint-position action、
  action scale 0.5、RSL-RL PPO 300 iterations。
- 官方 Spot Flat 不是 ANYmal-D baseline 的直接替代品。它雖然也使用
  48-dimensional proprioceptive observation 與 50 Hz policy，但有更大的
  velocity range、action scale 0.2、不同的 actuator dynamics、一套 Spot
  專用 gait rewards，以及實際包含 random-rough patches 的 terrain
  generator。
- 找到一個舊 ANYmal-D run：
  `2026-01-25_17-29-14`。其 resolved YAML 與目前官方 ANYmal-D Flat 的
  關鍵設定一致。Checkpoint 確認 actor/critic input 都是 48，actor output
  是 12。
- 新版 v1 應先重現官方 ANYmal-D Flat，再做少量且可追蹤的變更：
  建立自己的 External Project task registration、顯式 joint schema、
  policy I/O contract 與 project-local artifact paths。第一版 policy 不加入
  LiDAR、camera 或 height scan。

## 2. 分析範圍與保護措施

本次只在新專案新增這份報告。以下位置全程唯讀：

- `/home/ros/IsaacLab`
- `/home/ros/IsaacLab/logs/rsl_rl/anymal_d_flat`
- `/home/ros/Documents/anymal_project/anymal_ws`

沒有修改 Isaac Lab core、舊 workspace、舊 log/checkpoint，也沒有啟動
training。

## 3. 分析當時的 External Project 結構

實際盤點結果：

```text
/home/ros/anymal_locomotion/
├── .git/                     # empty; not a valid Git repository
└── docs/
    └── baseline_analysis.md
```

因此目前還沒有下列典型 External Project 元件：

- `source/<extension_name>/`
- Python package、`setup.py` 或 `pyproject.toml`
- `config/extension.toml`
- task registration
- environment、robot asset、agent configuration
- `scripts/rsl_rl/train.py` 與 `play.py`
- tests、project-local logs、checkpoints、exports
- 獨立的 ROS 2 deployment workspace/package

## 4. 找到的官方檔案

Isaac Lab source snapshot：

- Git description：`v2.2.1-232-gcbf51abb5e`
- Commit：`cbf51abb5e98`
- 舊 run 的 asset URL 指向 Isaac Sim 5.1 content。

### 4.1 ANYmal-D Flat

Task registration：

- `/home/ros/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/__init__.py`
- Gym ID：`Isaac-Velocity-Flat-Anymal-D-v0`

Environment inheritance chain：

1. `/home/ros/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
   - 共用 scene、commands、actions、observations、events、rewards、
     terminations、curriculum、simulation timing。
2. `/home/ros/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/rough_env_cfg.py`
   - 將 robot 換成 `ANYMAL_D_CFG`。
3. `/home/ros/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py`
   - 改為 plane、移除 height scanner 與 terrain curriculum，並覆寫三個
     reward weights。

Robot 與 actuator：

- `/home/ros/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/anymal.py`

RSL-RL PPO：

- `/home/ros/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py`

### 4.2 Spot Flat

Task registration：

- `/home/ros/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/__init__.py`
- Gym ID：`Isaac-Velocity-Flat-Spot-v0`

Environment、observations、commands、rewards、terrain：

- `/home/ros/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py`

Spot-specific reward implementation：

- `/home/ros/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py`

Spot reset event implementation：

- `/home/ros/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/events.py`

Robot 與 actuator：

- `/home/ros/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/spot.py`

RSL-RL PPO：

- `/home/ros/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/agents/rsl_rl_ppo_cfg.py`

### 4.3 舊 ANYmal-D run

Run directory：

`/home/ros/IsaacLab/logs/rsl_rl/anymal_d_flat/2026-01-25_17-29-14`

找到：

- `params/env.yaml`
- `params/agent.yaml`
- `git/IsaacLab.diff`
- `events.out.tfevents.1769333365.123.26255.0`
- `model_0.pt`、每 50 iterations 的 checkpoints、`model_299.pt`
- `exported/policy.pt`
- `exported/policy.onnx`

`IsaacLab.diff` 記錄當時工作樹乾淨，但 run 沒有保存 commit hash，所以不能
證明當時 source 與目前 checkout 在 byte level 完全相同。Resolved YAML
中的本報告關鍵欄位則與目前官方 ANYmal-D Flat 一致。

## 5. 三方設定比較

### 5.1 核心比較

| 項目 | 官方 ANYmal-D Flat | 官方 Spot Flat | 舊 ANYmal-D run |
|---|---|---|---|
| Workflow | Manager-Based | Manager-Based | Manager-Based |
| 演算法 | RSL-RL PPO | RSL-RL PPO | RSL-RL PPO |
| 環境數量 | 4096 | 4096（繼承） | 4096 |
| Episode 長度 | 20 s | 20 s | 20 s |
| Physics / policy 頻率 | 200 Hz / 50 Hz（`dt=0.005`、decimation 4） | 500 Hz / 50 Hz（`dt=0.002`、decimation 10） | 200 Hz / 50 Hz |
| Observation | 48 維 proprioception | 48 維 proprioception | 48 維，checkpoint 已確認 |
| Action | 12 維 joint-position target | 12 維 joint-position target | 12 維，checkpoint 已確認 |
| Action scale | 0.5 | 0.2 | 0.5 |
| Policy 是否包含 height scan | 否 | 否 | 否 |
| Terrain | 平面 | Cobblestone generator：flat + random rough | 平面 |
| Terrain curriculum term | 關閉 | 保留繼承設定 | 關閉 |
| 訓練 observation corruption | 開啟 | 關閉 | 開啟 |
| 最大 iterations | 300 | 20,000 | 300，完成至 299 |

### 5.2 速度命令

| Command 設定 | 官方 ANYmal-D Flat | 官方 Spot Flat | 舊 ANYmal-D run |
|---|---:|---:|---:|
| `lin_vel_x` | `[-1.0, 1.0]` m/s | `[-2.0, 3.0]` m/s | `[-1.0, 1.0]` m/s |
| `lin_vel_y` | `[-1.0, 1.0]` m/s | `[-1.5, 1.5]` m/s | `[-1.0, 1.0]` m/s |
| `ang_vel_z` | `[-1.0, 1.0]` rad/s | `[-2.0, 2.0]` rad/s | `[-1.0, 1.0]` rad/s |
| Heading 範圍 | `[-pi, pi]` | 無 | `[-pi, pi]` |
| Heading command | 開啟 | 關閉；直接 yaw-rate command | 開啟 |
| Heading environments | 100% | 0% | 100% |
| Standing environments | 2% | 10% | 2% |
| 重新取樣間隔 | 10 s | 10 s | 10 s |

ANYmal-D 的 policy observation 最後仍收到三維
`[vx_command, vy_command, wz_command]`；heading mode 會由 command term 根據
heading error 產生 `wz_command`。要讓 `/cmd_vel.angular.z` 成為直接 yaw-rate
命令，v1 應明確決定是否將 `heading_command=False`，而不是在 inference
途中寫入 command manager 的內部 tensor。

### 5.3 Observations

兩個官方 task 的 observation categories 相同，且都是 48 維：

| 順序 | Term | 維度 |
|---:|---|---:|
| 1 | base linear velocity | 3 |
| 2 | base angular velocity | 3 |
| 3 | projected gravity | 3 |
| 4 | velocity command | 3 |
| 5 | relative joint position | 12 |
| 6 | relative joint velocity | 12 |
| 7 | previous action | 12 |
|  | 總計 | 48 |

差異在 noise：

| Term | ANYmal-D noise | Spot 設定的 noise |
|---|---:|---:|
| base linear velocity | ±0.1 | ±0.1 |
| base angular velocity | ±0.2 | ±0.1 |
| projected gravity | ±0.05 | ±0.05 |
| joint position | ±0.01 | ±0.05 |
| joint velocity | ±1.5 | ±0.5 |

Spot 設定了 noise term，但 `enable_corruption=False`，所以 training 時不套用
這些 corruption。ANYmal-D training 則會套用。

舊 ANYmal-D checkpoint 的實際 tensor shapes：

```text
actor first layer:  (128, 48)
critic first layer: (128, 48)
actor output:       (12, 128)
policy std:         (12,)
```

這確認舊 policy 是 48 observations / 12 actions，且沒有 observation
normalization。

### 5.4 Rewards

Official ANYmal-D Flat 與舊 run 相同：

| Reward term | 權重 |
|---|---:|
| `track_lin_vel_xy_exp` | +1.0 |
| `track_ang_vel_z_exp` | +0.5 |
| `lin_vel_z_l2` | -2.0 |
| `ang_vel_xy_l2` | -0.05 |
| `dof_torques_l2` | -2.5e-5 |
| `dof_acc_l2` | -2.5e-7 |
| `action_rate_l2` | -0.01 |
| `feet_air_time` | +0.5 |
| `undesired_contacts` | -1.0 |
| `flat_orientation_l2` | -5.0 |
| `dof_pos_limits` | 0.0 |

其中 linear/yaw tracking kernel 的 `std=0.5`，feet air-time threshold 是
0.5 s，undesired contact 對象為 `.*THIGH`。

Spot 使用不同的 robot-specific reward formulation：

| Reward term | 權重 |
|---|---:|
| air time | +5.0 |
| base angular velocity tracking | +5.0 |
| base linear velocity tracking | +5.0 |
| foot clearance | +0.5 |
| diagonal gait synchronization | +10.0 |
| action smoothness | -1.0 |
| air-time variance | -1.0 |
| base motion | -2.0 |
| base orientation | -3.0 |
| foot slip | -0.5 |
| hip joint acceleration | -1.0e-4 |
| joint position deviation | -0.7 |
| joint torque | -5.0e-4 |
| hip joint velocity | -1.0e-2 |

Spot 的 reward functions 使用不同的 norm、kernel、gating 與 body/joint naming；
weight 的數值不能直接與 ANYmal-D weight 比大小。尤其 gait、foot clearance、
remotized knee dynamics 都是 Spot task 的整體設計，不能只複製幾個 weight
到 ANYmal-D。

### 5.5 Actions 與 actuators

#### ANYmal-D

- Action：`default_joint_position + 0.5 * policy_action`
- Joint selector：`[".*"]`
- Official config 沒有顯式列出 12 個 joint names，resolved YAML 顯示
  `preserve_order: false`。目前 mapping 因此依賴 articulation/USD resolved
  joint order，不足以作為未來 ROS 2 與實體機的明確介面契約。
- Actuator：`ActuatorNetLSTMCfg`
- Saturation effort：120 Nm
- Effort limit：80 Nm
- Velocity limit：7.5 rad/s
- Network：`anydrive_3_lstm_jit.pt`
- 官方註記指出：沒有公開的 ANYmal-D actuator network，因此使用 ANYmal-C
  的 network；這可能影響 sim-to-real。

#### Spot

- Action：`default_joint_position + 0.2 * policy_action`
- Hip：delayed PD actuator，effort limit 45 Nm、stiffness 60、damping 1.5。
- Knee：remotized PD actuator，以實驗 lookup table 處理 torque limit。
- Hip/knee 都有 0–4 physics steps delay；在 2 ms physics step 下是 0–8 ms。

Spot 的 action scale 與 actuator model 是一起設計的，不應用 Spot 的 0.2
scale 單獨替換 ANYmal-D 的 0.5。

### 5.6 PPO

| 參數 | 官方 ANYmal-D Flat | 官方 Spot Flat | 舊 ANYmal-D run |
|---|---:|---:|---:|
| 每個環境的 steps | 24 | 24 | 24 |
| 最大 iterations | 300 | 20,000 | 300 |
| actor/critic MLP | 128, 128, 128 | 512, 256, 128 | 128, 128, 128 |
| 初始 noise std | 1.0 | 1.0 | 1.0 |
| learning rate | 1.0e-3 | 1.0e-3 | 設定為 1.0e-3 |
| schedule | adaptive | adaptive | adaptive |
| desired KL | 0.01 | 0.01 | 0.01 |
| 最大 gradient norm | 1.0 | 1.0 | 1.0 |
| epochs / mini-batches | 5 / 4 | 5 / 4 | 5 / 4 |
| gamma / lambda | 0.99 / 0.95 | 0.99 / 0.95 | 0.99 / 0.95 |
| PPO clip | 0.2 | 0.2 | 0.2 |
| value loss coefficient | 1.0 | 0.5 | 1.0 |
| entropy coefficient | 0.005 | 0.0025 | 0.005 |
| observation normalization | 關閉 | 關閉 | 關閉 |

舊 run 的 TensorBoard 沒有記錄 actual KL scalar，只能確認
`desired_kl=0.01`。同樣地，gradient norm 沒有 scalar；只能確認
`max_grad_norm=1.0`。

adaptive schedule 使舊 run 的實際 learning rate 不固定：

- first logged：`6.6667e-4`
- min：`2.6012e-4`
- max：`7.59375e-3`
- final / checkpoint optimizer：`3.90184e-4`

因此 `learning_rate=1e-3` 是起始設定，不是整段 run 的固定值。

### 5.7 Terrain 與訓練設定

Official ANYmal-D Flat：

- Plane terrain。
- Height scanner 被移除。
- Policy height observation 被移除。
- Terrain-level curriculum 被移除。
- 仍保留 base mass、COM、reset state、material 與 push events。
- Material startup range 實際是固定 static 0.8 / dynamic 0.6。
- Base mass randomization：±5 kg。
- Base COM randomization：x/y ±0.05 m、z ±0.01 m。
- Push interval：10–15 s，x/y velocity ±0.5 m/s。

Official Spot Flat：

- 名稱雖為 Flat，實際使用 `COBBLESTONE_ROAD_CFG` terrain generator。
- Generator 是 9 rows × 21 columns、8 m × 8 m tiles，包含 flat 與
  random-rough sub-terrains。
- Random rough noise range 0.02–0.05 m。
- Height scanner 被移除。
- Base mass randomization：±2.5 kg。
- Friction randomization：static 0.3–1.0、dynamic 0.3–0.8。
- Reset joint position/velocity、base velocity ranges比 ANYmal-D 更寬。
- Inherited `terrain_levels` curriculum term 沒有在 train config 中被設為
  `None`；但是 generator 的 `curriculum` 沒有在此檔顯式設定。實作前若要
  參考此機制，應以 resolved config/runtime behavior 再確認，不能只由 class
  inheritance 推定。

## 6. 舊 ANYmal-D 訓練實際結果

舊 run 完整執行 300 TensorBoard steps，checkpoint iteration 為 299。

重要 final metrics：

| 指標 | 最終值 | 最佳值／範圍 |
|---|---:|---:|
| mean reward | 21.5068 | 最大 21.7607 |
| mean episode length | 991.7 steps | 最大 1000 |
| XY velocity error | 0.2115 | — |
| yaw velocity error | 0.2064 | — |
| timeout fraction | 0.9677 | 最大 0.9734 |
| base-contact termination | 0.0323 | — |
| policy mean noise std | 0.2866 | 起始 0.9908 |
| value loss | 0.00301 | 最小 0.00228 |

在 50 Hz policy rate 與 20 s episode 下，1000 steps 是完整 episode。
因此最後約 96.8% termination 為 timeout、mean length 991.7，顯示 baseline
已學會大多數 episode 不跌倒。這是值得保留的 regression reference，但還
不是 sim-to-real readiness 證明：

- 測試仍在 official USD 與 ANYmal-C actuator network approximation 上。
- 沒有 hardware estimator、latency、joint mapping contract 或 real control
  interface validation。
- TensorBoard 沒有 evaluation-only run，也沒有不同 seeds 的統計。
- 沒有 actual KL、gradient norm、torque/velocity saturation ratio 等
  sim-to-real 重要 diagnostics。

## 7. 對新版 ANYmal-D locomotion v1 的建議

### 7.1 Baseline policy

建議 v1 保持：

- Manager-Based workflow。
- RSL-RL PPO。
- 48-dimensional proprioceptive observation。
- 不加入 LiDAR、RGB-D、camera 或 height scan。
- 50 Hz policy interface。
- Official ANYmal-D reward set 與 PPO hyperparameters 作為第一個可重現
  baseline。
- Official asset 作為第一個 smoke/regression target；custom USD 用獨立
  robot config 切換，不修改 Isaac Lab asset。

不建議第一版採用：

- Spot 的完整 reward set。
- Spot action scale。
- Spot actuators。
- Rough/perceptive terrain。
- 在 Isaac Sim Python 中使用 `rclpy`。
- 直接存取 command manager private fields。

### 7.2 建議的專案邊界

```text
anymal_locomotion/
├── source/anymal_locomotion/        # Isaac Lab extension; no rclpy
│   ├── config/extension.toml
│   ├── setup.py / pyproject.toml
│   └── anymal_locomotion/
│       ├── assets/                  # local custom ANYmal-D ArticulationCfg
│       └── tasks/manager_based/locomotion/velocity/
│           └── config/anymal_d/
│               ├── flat_env_cfg.py
│               └── agents/rsl_rl_ppo_cfg.py
├── scripts/
│   ├── rsl_rl/                      # train/play/export entry points
│   └── validation/                  # config and policy-contract checks
├── deployment/
│   └── ros2_ws/src/                 # external ROS 2 nodes/packages
├── action_graph/                    # graph assets/configuration notes
├── configs/                         # versioned deployment/policy schemas
├── logs/                            # project-local only
├── checkpoints/                     # project-local only
├── exported/                        # policy + metadata
├── tests/
└── docs/
```

Training extension 與 ROS 2 deployment package 必須是兩個 dependency
boundaries：

- `source/anymal_locomotion` 不依賴、不 import `rclpy`。
- Isaac Sim 使用 ROS 2 Bridge / Action Graph 交換 `/cmd_vel`、IMU、
  joint states、odometry/state estimate 與 actuator command。
- ROS 2 policy node 在 Isaac Sim process 外執行；未來同一 policy I/O
  contract 可接 simulation bridge 或實體 robot driver。
- LiDAR/RGB-D 只進 perception、SLAM、Nav2 graph，不進 locomotion v1
  policy observation。

### 7.3 Deterministic policy contract

這是 v1 最重要的新增設計，不應等到 ROS 2 deployment 才補：

1. 顯式定義 12 個 canonical joint names 與 policy indices。
2. Environment startup 時檢查 USD joint set 完全一致，不允許 missing、
   duplicate 或 unexpected joints。
3. Action config 使用顯式 joint list，並確認 `preserve_order` 行為。
4. ROS `JointState` 永遠依 name remap，不依 message array 原始順序。
5. Export policy 時一起輸出 versioned metadata，例如：
   - joint order
   - observation order與 dimensions
   - frame conventions
   - default joint positions
   - action scale/offset
   - policy period
   - command limits
   - normalization
   - checkpoint/config hash
6. Inference 前對 metadata、robot model 與 topic interface 做 fail-fast
   validation。

Official task 的 `joint_names=[".*"]` 與 `preserve_order=false` 可以用來重現
baseline，但不應成為最終 deployment contract。

### 7.4 State-estimation contract

Official observation 的 `base_lin_vel` 在 simulation 中可直接由 simulator
state 取得；實體機不能依賴 ground truth。v1 在開始 sim-to-real 前必須定義：

- `base_lin_vel` 的 estimator 與 body/world frame。
- IMU angular velocity、orientation與 gravity projection conventions。
- joint position/velocity signs、units、timestamps。
- observation sampling、synchronization 與 stale-data timeout。
- odometry/state-estimation source與 covariance handling。
- command timeout、estop、saturation、rate limit。

ROS 2 bridge simulation 應刻意使用與實體 deployment 相同的 message-level
contract，避免 policy node 對 simulator ground truth 形成隱藏依賴。

## 8. 建議實作階段

### 階段 0 — 決策與版本固定

- 確認實際 project root。
- 確認 Isaac Lab commit/tag、Isaac Sim 5.1、RSL-RL version 的支援組合。
- 確認 custom USD 與實體 ANYmal-D control interface。
- 決定 v1 command mode 與 velocity limits。

### 階段 1 — 最小 External Project scaffold

- 使用目前 Isaac Lab external template 建立專案 package。
- 建立自己的 Gym task ID，避免覆蓋官方 ID。
- 以 subclass/import 重用官方 ANYmal-D Flat；只覆寫 project-owned
  experiment/log paths與已確認的 v1 差異。
- 加入 config-resolution tests：
  48 observations、12 actions、no height scan、50 Hz、預期 rewards/PPO。
- 加入 source check，禁止 training package import `rclpy`。

### 階段 2 — 重現 baseline

- 先用 official USD 做 zero/random action smoke test。
- 以固定 seed 重現 300-iteration baseline。
- 加入 evaluation-only script 與 regression thresholds。
- 所有 logs/checkpoints/exported policies 寫入本專案。
- 與舊 run 的 reward、episode length、tracking error、termination 比較。

此階段需在使用者確認後才開始 training。

### 階段 3 — Custom ANYmal-D USD

- 新增 local `ArticulationCfg`，不改 `isaaclab_assets`。
- 驗證 joint/link names、axes/signs、default pose、limits、inertials、
  collision、foot/contact body names與sensor frames。
- 用 deterministic mapping test 比較 official/custom USD。
- 評估 ANYmal-C actuator network approximation 是否仍可接受；若沒有
  實機識別資料，先保留為已知限制。

### 階段 4 — Policy 匯出與 contract 驗證

- Export TorchScript/ONNX 及 metadata。
- 建立 Python-independent golden observation/action vectors。
- 測試 checkpoint、TorchScript、ONNX 在相同 input 下的 output tolerance。
- 測試 command clamp、timeout、NaN/Inf 與 stale state fail-safe。

### 階段 5 — ROS 2 模擬部署

- 在 Isaac Sim 建立 ROS 2 Bridge / Action Graph。
- 外部 ROS 2 policy node 接收 IMU、joint states、state estimate 與
  `/cmd_vel`，輸出已確認的 low-level command。
- 禁止在 Isaac Sim/Isaac Lab process import `rclpy`。
- 先做 topic rate、timestamp、frame、latency與joint remapping tests。
- UDP prototype 不進 final architecture。

### 階段 6 — 感知與導航

- Action Graph/Bridge 發佈 LiDAR、RGB-D、TF、odometry。
- ROS 2 外部執行 SLAM/Nav2。
- Nav2 只透過 `/cmd_vel` 驅動 locomotion command。
- 感知資料不加入 v1 locomotion observation。

### 階段 7 — Rough/perceptive locomotion 與 sim-to-real

- Flat v1 穩定後另開 Rough task。
- Perceptive locomotion 作為獨立 policy/version，不改變 Flat v1 contract。
- 加入 latency、sensor noise/bias、dynamics、payload、friction、actuator
  domain randomization 與 hardware-in-the-loop validation。
- 經過限速、吊掛/安全場地、estop 與低風險 gait validation 後才上實體機。

## 9. 下一步建議修改內容

在使用者確認後，建議下一個 implementation turn 只做：

1. 建立標準 External Project scaffold 與有效 Git repository。
2. 建立 project-owned task registration 與 minimal ANYmal-D Flat subclasses。
3. 新增 canonical policy I/O schema；若 custom USD 尚未提供，先將 joint
   order validation 設計完成，但不猜測實體 joint contract。
4. 設定 project-local `logs/`、`checkpoints/`、`exported/` 與 ignore rules。
5. 新增 config smoke tests 和 dependency-boundary test。
6. 建立 README/commands，但不啟動 training。

暫時不要修改：

- Isaac Lab core/source/assets。
- 官方 reward functions。
- Spot code。
- 舊 ANYmal-D logs/checkpoints。
- 舊 ROS 2 workspace。
- Rough/perceptive policy。
- ROS 2/Action Graph runtime implementation。

## 10. 需要使用者確認

1. **Project root**  
   是否接受實際路徑 `/home/ros/anymal_locomotion`？若一定要使用
   `/home/ros/Documents/anymal_locomotion`，需先決定建立/搬移方式；目前該
   路徑不存在。

2. **Version pin**  
   是否以目前 `cbf51abb5e98` checkout 作為基準，或要改用指定的 Isaac Lab
   release/tag？目前 source description 比 `v2.2.1` 多 232 commits。

3. **Command semantics**  
   v1 是否改成 `heading_command=False`，直接訓練
   `[vx, vy, wz]` 以對齊 `/cmd_vel`？建議是。

4. **Velocity limits**  
   是否先保留官方 ±1.0 範圍，或使用更保守且非對稱的 deployment limits？
   舊 prototype 曾使用 `vx=±0.8`、`vy=±0.4`、`wz=±0.8`，但那不是舊
   training range。

5. **Custom USD**  
   USD 的最終路徑、canonical joint/link names、base/foot/IMU/camera/LiDAR
   frames 是否已確定？

6. **Physical control interface**  
   實體 ANYmal-D 最終接受 joint position、velocity、torque，還是 vendor
   controller command？控制頻率、joint limits、SDK/ROS 2 interface 與 safety
   supervisor 必須在 deployment design 前確認。

7. **Joint order authority**  
   canonical 12-joint order應以 custom USD、實體 driver，或既有 ANYbotics
   interface中的哪一個為準？不能只沿用 policy checkpoint 的隱含 order。

8. **State estimator**  
   實體機可用的 IMU/odometry estimator、topic、frame與update rate為何？
   這會決定如何提供 policy 所需的 base linear velocity。

9. **Baseline reproduction**  
   scaffold 完成後，是否先只做 config/simulation smoke tests，待再次確認才
   啟動 300-iteration reproduction training？建議如此。

## 11. 已知舊版 prototype 風險

`/home/ros/IsaacLab/scripts/anymal_d_test/` 與
`/home/ros/IsaacLab/bringup/` 是目前 checkout 中的未追蹤內容，不是上述官方
baseline 的一部分。

其中 `play_flat_ros2_cmd_vel.py` 在 Isaac Sim process 內直接 import
`rclpy`，並直接操作 command manager tensor/private term storage。這與新
專案規則衝突，不應移植為 final architecture。本次沒有修改或刪除它。

## 12. 已確認的 v1 實作決策

以下決策已於 External Project baseline 階段確認，取代第 10 節中相對應的
未決項目：

- Project root：`/home/ros/anymal_locomotion`
- Isaac Sim：`5.1.0`
- Isaac Lab baseline：`v2.3.2`
- Isaac Lab tag commit：
  `37ddf626871758333d6ed89cf64ad702aef127d0`
- Historical working/rollback reference：
  `cbf51abb5e98d1b3d497c8c73dc989e9f3628b89`
- 不追蹤 `origin/main` 作為開發 baseline。
- Command semantics：direct body velocity `[vx, vy, wz]`，
  `heading_command=False`。
- v0.1.0 baseline training ranges：`vx/vy/wz = [-1.0, 1.0]`，已完成
  300 iterations 重現。
- v0.2.0 High-Speed ranges：`vx=[-2.0, 3.0]`、`vy=[-1.5, 1.5]`、
  `wz=[-2.0, 2.0]`；預設訓練預算提高為 1,000 iterations。
- 第一個 implementation target：官方 Isaac Sim 5.1 ANYmal-D USD。
- Custom USD：尚未整合，也不猜測其 joint names。
- Physical ANYmal-D low-level control interface：尚未確認。
- Flat v1 observation 維持 48-D proprioception；LiDAR、RGB-D、camera、
  height scan 不進入 policy。
- Canonical joint order已由官方 USD 建立並版本化於
  `configs/policy_contract.yaml`。
- Training extension 與外部 ROS 2 deployment 維持 dependency boundary；
  Isaac Sim communication最終採 ROS 2 Bridge / Action Graph。
