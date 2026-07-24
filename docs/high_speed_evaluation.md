# High-Speed v0.2.0 固定速度測試

## 測試目的

確認 command range 不只存在於設定檔，policy 也能在 flat simulation
實際追蹤命令。測試載入 `model_1298.pt`，不修改訓練設定，也不直接寫入
command manager 的私有 tensor。

`scripts/rsl_rl/evaluate.py` 會在建立環境前，把官方
`UniformVelocityCommandCfg` 的各軸 range 設成單一固定值。每個正式案例使用：

- 128 個平行環境
- seed 42
- 1,000 policy steps，即 20 秒
- 前 100 steps 作為 warm-up，不納入追蹤誤差
- observation corruption、push 與 external force 關閉
- 官方 ANYmal-D Flat reward、action scale 與 actuator 維持不變
- startup mass、center-of-mass 與 reset randomization 維持啟用

`survival` 表示 20 秒內沒有因 base contact 提前終止的環境比例。MAE 是
warm-up 後、尚未失敗環境的 body-frame velocity mean absolute error。

## 結果

| Command `[vx, vy, wz]` | 平均實際速度 | MAE | Survival |
|---|---|---|---:|
| `[3.0, 0.0, 0.0]` | `[2.969, 0.003, 0.022]` | `[0.097, 0.060, 0.071]` | 100% |
| `[-2.0, 0.0, 0.0]` | `[-1.967, 0.000, 0.011]` | `[0.071, 0.048, 0.053]` | 100% |
| `[0.0, 1.5, 0.0]` | `[0.002, 1.504, -0.010]` | `[0.044, 0.034, 0.084]` | 100% |
| `[0.0, -1.5, 0.0]` | `[-0.008, -1.467, 0.027]` | `[0.057, 0.053, 0.047]` | 100% |
| `[0.0, 0.0, 2.0]` | `[0.002, -0.003, 2.019]` | `[0.018, 0.017, 0.044]` | 100% |
| `[0.0, 0.0, -2.0]` | `[0.007, -0.007, -1.990]` | `[0.029, 0.033, 0.053]` | 100% |
| `[3.0, 1.5, 2.0]` | `[2.801, 1.274, 1.207]` | `[0.203, 0.227, 0.798]` | 99.22% |
| `[-2.0, -1.5, -2.0]` | `[-1.706, -1.529, -1.716]` | `[0.294, 0.069, 0.287]` | 99.22% |

線速度的單位為 m/s，yaw rate 的單位為 rad/s。

## 判讀

- Policy 已在 flat simulation 達到 Spot example 的各軸 command 邊界。
- 純前進 3 m/s 並非只設定 command limit；128 個環境都完整存活，平均速度
  為 2.969 m/s。
- 三軸同時最大仍大致穩定，但會優先保留線速度，yaw tracking 明顯下降。
- Command 各軸範圍不應解讀成實體機可以同時安全達到所有角落。

目前不追加完全相同設定的 PPO iterations。最後約 300 iterations 的
TensorBoard 指標已接近平臺，而單軸目標已達成；直接續訓的預期效益偏低。
若未來需求明確要求同時達到最大平移與轉向，應另外設計 command curriculum
或 command distribution，再建立獨立 fine-tuning run。

## 尚未涵蓋

- 多個 evaluation seeds
- observation corruption、push 與外力
- rough terrain
- custom ANYmal-D USD
- actuator delay、sensor latency 與 state-estimation error
- 實體 ANYmal-D 的 torque、thermal、速度與安全限制

因此這份結果只證明目前 policy 在官方 flat simulation 的固定命令表現，
不構成 sim-to-real 安全驗證。
