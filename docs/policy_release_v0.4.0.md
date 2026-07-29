# Recovery Policy v0.4.0

## 發布狀態

`model_1450.pt` 已提升為正式 project-local deployment policy，取代
High-Speed v0.2.0 的 model 1298。最終 training checkpoint model 1497
雖有改善，但仍超過 body-instability gate，因此沒有採用。

這次處理的是長時間行走後收到 zero command 仍持續搖晃、無法回復水平的
policy robustness 問題，不以提高摩擦或硬切 nominal joint pose 掩蓋。

## 正式產物

- Checkpoint：
  `checkpoints/anymal_d_locomotion_v1/recovery_v0.4.0/model_1450.pt`
- 匯出目錄：
  `exported/anymal_d_locomotion_v1/recovery_v0.4.0/`

| 產物 | SHA256 |
|---|---|
| `model_1450.pt` | `3bbaff6247fdb59808af42a59e2ec9477c047531a3e2aa6cd0df6208a796feee` |
| `policy.pt` | `95f20f7618adc9be808d9b3d424d61116abf52012d88b3e92c9c0ff6e55cfe5b` |
| `policy.onnx` | `721a918533cd00e605cf6edfcd8bf8bba9cbd56f26e22c2eda26604f03ee54c0` |

## 訓練變更

Recovery task 以正式 Flat locomotion rewards 為基礎，將 command
resampling time 改為 5 秒、standing environment 比例由 0.02 提高至
0.2。正式 run 使用 seed 42、4096 environments、200 iterations；未加入
robust model 的 feet-slide reward。

## Qualification

| Gate | 結果 |
|---|---:|
| RTX/LIO 長時序圓周後 zero-command recovery | Pass |
| 純平移 3 m/s regression | 1/1 |
| turning matrix | 36/36 |
| true loop-closure matrix | 12/12 |
| checkpoint／TorchScript／ONNX parity | Pass |

長時序 profile 為 23 秒 `vx=0.5 m/s, wz=0.25 rad/s`，接著 zero command
10 秒。model 1298 發生 `foot_slip_first`，最大 roll `2.584 rad` 並
hard-fail；model 1450 為 `no_instability`，最大 roll/pitch
`0.195/0.106 rad`、最低高度 `0.539 m`，最後 5 秒平面位移
`0.00010 m`、yaw 變化 `0.00038 rad`。

Turning matrix 涵蓋純 yaw `±0.5/±1/±2 rad/s`，以及左右
`0.5/1.5/3.0 m/s` 曲線，每個 profile 三次。Loop-closure matrix 涵蓋
兩種真閉環與兩種開放路徑，每種三次；每個 bag 都以 closure disabled 與
enabled 的 fresh LIO graph replay。

Parity 使用 256 組 seed 42 float32 observation。checkpoint 與
TorchScript 最大誤差為 0；checkpoint 與 ONNX 最大絕對誤差
`2.861e-6`，低於 `1e-5` tolerance。

## 已知限制

- Release 當時的模擬 host 200 Hz IMU `angular_velocity` warning，後續確認
  包含非零初始 yaw 未合成與 episode-reset 單 tick transient；deployment
  bridge／validator 已修正，policy artifact 本身未變更。
- 實體 sensor extrinsic 與 low-level interface 尚未完成。
- 此結果是模擬 qualification，不是實體 ANYmal-D 安全速度宣告。
