# Proprioceptive body-velocity estimator v1

狀態：契約、模擬資料 collector、訓練／export、離線 gate、ROS runtime 與 vendor
adapter 已實作。2026-08-13 的最佳 candidate 04 通過平均、RMSE、p95、stopped
bias 與 finite checks，但最大單點誤差 `2.2541 m/s` 超過 frozen `1.5 m/s`，因此
整體 gate 失敗，不得接入正式 model1450 或 model19 live matrix。

## 責任邊界

本 estimator 只解決 locomotion policy 所需的 50 Hz、`base_link` frame
`[vx, vy, vz]`。它不估計地圖位置，也不改變 SLAM confidence：

```text
LIO-SAM / FAST-LIO2 -> global pose + /slam_confidence
IMU + joints + foot contacts -> /locomotion/estimated_odom.twist.linear
                                      |
                                      v
                              48-D / 51-D policy
```

固定輸入為最近 20 筆、每筆 37 維：IMU angular velocity、linear
acceleration、projected gravity、12 維 relative joint position、12 維 joint
velocity、依 `LF/LH/RF/RH` 固定順序的四腳 contact probability。模型輸入不含：

- simulator ground-truth velocity；
- `/odom` 或其他 ground-truth topic；
- SLAM pose、SLAM twist、map 或 confidence。

模擬 ground truth 只可寫進 dataset 的 `labels`，用於 supervised training 與
offline evaluator。此限制由 dataset role、metadata 與 37-D input contract 明確記錄。

## 模擬資料與 gate

collector 從正式 Recovery v0.4.0 model1450 rollout 取樣。每個 capture 應包含不同
seed 與固定 command profiles；至少要有 stopped、forward、reverse、left/right
lateral、left/right yaw 與混合 command。正式資料量與 profiles 尚未 freeze。
第一個導航用 accuracy gate 明確停用 `push_robot` 與
`base_external_force_torque`；人工推擠保留為後續非阻擋 robustness gate。

範例：

```bash
PYTHONPATH=source/anymal_locomotion /home/ros/IsaacLab/isaaclab.sh -p \
  scripts/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Anymal-D-Locomotion-v0 \
  --checkpoint logs/rsl_rl/anymal_d_locomotion_v1/2026-07-29_00-12-45_recovery_v040_seed42_4096env/model_1450.pt \
  --num_envs 512 --seed 42 --evaluation_steps 1000 \
  --velocity_estimator_fixed_command 1.5 0.0 0.0 \
  --velocity_estimator_dataset_output logs/velocity_estimator/captures/forward_seed42.npz \
  --headless --device cuda:0
```

訓練時按 simulator environment 切分，不能把同一 environment 的相鄰時間窗分到
train 與 holdout：

```bash
/home/ros/IsaacLab/_isaac_sim/python.sh scripts/train_proprioceptive_velocity_estimator.py \
  --dataset logs/velocity_estimator/captures/*.npz \
  --output-dir exported/proprioceptive_velocity_estimator/v1/candidate_01
```

離線 gate 定義在 `configs/proprioceptive_velocity_estimator_gate.yaml`。候選必須通過
axis MAE/RMSE、vector p95/max、stopped bias 與 finite-output checks；之後還需要：

1. TorchScript／ONNX parity；
2. 將 estimator prediction 替代 simulator GT velocity 的 model1450 closed-loop gate；
3. 相同替代下的 model19 confidence behavior gate；
4. FAST 與 LIO 兩 backend live locomotion matrix。

上述 gate 未全數通過前，正式 policy 和正式 odometry source保持不變。

### 2026-08-13 candidates

正式 capture 使用 model1450：隨機 command `256 env × 600 step`（seed 42）與
stopped `128 env × 600 step`（seed 43），共 230,400 rows。按 environment split
後 holdout 有 44,680 windows。

最佳 candidate 04 是兩層、hidden 128 的 windowed GRU：XYZ MAE
`0.0377/0.0379/0.0218 m/s`、XYZ RMSE `0.0696/0.0614/0.0328 m/s`、vector p95
`0.1586 m/s`、stopped planar/vertical bias `0.0339/0.0221 m/s`，但最大 vector
error `2.2541 m/s`。失敗樣本集中於外力 push／四腳離地 transition，而非跨 episode
window 或 timestamp regression。對 worst 5% 額外加權的 candidate 05 反而退化，
不再繼續盲目調 loss。機讀結果見
`docs/validation/proprioceptive_velocity_estimator_candidate04_gate.json`。

## ROS runtime 與實體 ANYmal

learned runtime 訂閱 `/imu/data`、`/joint_states`、`/foot_contacts`，只有完整歷史且
frame、timestamp、receipt age、數值與輸出 guard 全部合法時才發布
`/locomotion/estimated_odom`。任何 fault 都清空歷史並停止發布，讓 policy 的 state
watchdog fail closed，不回退到 SLAM pose-delta velocity。

若實體 ANYmal SDK 已提供經驗證的 state-estimator odometry，使用獨立
`hardware_state_estimator_adapter`，依文件明確指定原始 linear velocity 是 body 或
odom frame，再輸出同一 policy-side topic。Repository 不猜測 ANYbotics 私有 topic 或
message schema；在取得 driver 文件前，adapter 不設定預設 input topic。
