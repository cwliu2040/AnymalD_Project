# Native-deskew yaw-rate stress validation

本 gate 只建立固定 Factory 起點的 rotation degradation dataset 與量測管線；
不定義 `/slam_confidence`、不修改 PPO，也不宣稱完成 FAST-LIO2 full-map
deployment qualification。

## 固定實驗 contract

- Branch：`exp/slam-fastlio2`。
- 正式 policy：Recovery v0.4.0 model1450；capture 時只使用 simulator GT
  `/odom` 驅動 locomotion，SLAM 不回饋控制。
- Backend：LIO-SAM native deskew 與 FAST-LIO2 native deskew；本矩陣不包含
  project-based deskew reference arm。
- 運動：`vx=vy=0`，左右原地旋轉，commanded yaw rate 為
  `0.25/0.5/1.0/1.5/2.0 rad/s`。
- Timeline：5 s warmup、2 s ramp-up、8 s hold、2 s ramp-down、10 s
  recovery，共 27 s。
- Loop closure 關閉。
- 點雲退化軸正式稱為 `uniform_point_density`，使用 deterministic evenly
  spaced thinning；它不代表遮擋、ring loss、雨霧或 scan dropout。
- FAST-LIO2 `map_en=false`：不影響 native estimator/iKD-Tree/odometry，但
  本 profile 不具 full-map export qualification。

兩個 backend 都需要 simulator raw cloud 到 Ouster message contract 的格式轉換：
補 `ring`、per-point `t` 與 Ouster fields。兩路的點、排序與 timestamp 必須相同；
FAST-LIO2 fork 只因編譯期欄位 contract 使用 `ambient` 名稱及 reliable output
QoS。Topic remap、parameter 選擇與 `map_en=false` 不稱為 adapter。

## 實驗規模

Pilot 使用 seed 42、10 source bags、full density、兩 backend，共 20 replays。
Pilot gate 通過後，formal 使用 seeds 42/43/44；同一 seed index 跨全部 motion
cells 配對。30 source bags × 4 densities × 2 backends，共 240 replays。

Replay schedule 由固定 seed counterbalance：同一 source bag 的兩 backend 成對，
但跨 cells 交換先後；density/cell 順序固定打散。每次 replay 都啟動全新 backend
process，不保留 map、bias、keyframes 或 iKD-tree state。

## 操作入口

只產生並檢查 pilot plan，不啟動 simulator：

```bash
python3 scripts/validation/run_slam_yaw_stress.py --stage plan
```

完成 code review 且工作樹位於乾淨 baseline 後，才可 capture：

```bash
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash
python3 scripts/validation/run_slam_yaw_stress.py --stage capture
```

Pilot capture 驗證後才執行 replay：

```bash
python3 scripts/validation/run_slam_yaw_stress.py --stage replay
```

需要同時產生固定視角盲測影片時加上 `--render-visuals`。Renderer 固定使用
`camera_init` frame 的俯視範圍、720×720、10 fps 與 30 s point decay；畫面不顯示
backend 或 cell 名稱。

`--allow-dirty-smoke` 只允許非 qualification 的本機 smoke。不得用它產生正式
pilot/formal 結論。Formal 需另傳 `--formal`，不應在 pilot review 前執行。

## Capture 與結果 gate

Source bag 保存 backend-neutral `/clock`、GT `/odom`、raw IMU、raw LiDAR、
`/cmd_vel`、joint state 與 TF。Reset/ack topics 也列入 recorder；沒有 episode
reset 的正常 capture 可以是零訊息，seed/spawn/reset contract 另由 provenance
與 simulator diagnostics 保存。

Evaluator 以第一筆 command 的 simulation time 定位 profile 起點，分別輸出
warmup/ramp-up/hold/ramp-down/recovery metrics。Yaw 同時報 wrapped orientation
error 與 unwrapped cumulative rotation，避免多圈 cycle slip 被 `[-pi, pi]`
包角隱藏。Tracking gaps 報 `0.10/0.25/0.50 s` 以及 `3×` steady median period。

Native support diagnostics 保留 backend-specific 語意：LIO-SAM 直接讀
`CloudInfo.cloud_corner + cloud_surface`；FAST-LIO2 以預設關閉的 downstream
`publish.effect_en` 發布原生 point-to-plane measurement selected points。這個
publisher 在 state update 後才序列化診斷點雲，不改 feature selection、EKF 或
map update。兩種 count 不可直接視為相同比例 confidence。

同一份 `yaw_stress_left_0_5` smoke bag 已完成 FAST-LIO2 diagnostics off/on
A/B：全程 trajectory 與五個 phase evaluation JSON 逐值相同；開啟時取得 273
筆 effective-point samples，關閉時為 0。這驗證目前觀測 publisher 未改變該
replay 的 estimator 結果；physical-clock overhead 仍需在 throughput gate 另測。

盲測 manifest 可由下列入口建立：

```bash
python3 scripts/validation/prepare_yaw_stress_blind_review.py \
  <experiment-root> --require-videos
```

`blind_manifest.json` 只含 `yaw_review_NNNN`、固定視角 contract 與預期影片路徑；
backend/cell 對照只放在分開的 `reveal_manifest.json`。`--require-videos` 會檢查
影片可開啟、720×720、10 fps、至少涵蓋 26 s，並抽查首／中／末幀不是空白畫面。
同一份單-cell smoke 已為兩 backend 各產生 273 幀、27.3 s 的 MP4 並通過此
contract；影片尚未完成人工標註，因此不構成裂圖盲測結論。

Yaw-stress mode 不以既有 ATE/yaw/jump 閾值決定品質 pass/fail。初始化未在
ramp 前完成、timestamp 非單調、缺資料、backend crash 或 evaluator 缺失仍是
明確 failure。Locomotion 失穩的 bag 保留但需標為 `motion_driver_invalid`，只可
分析首次 body instability 前的區段。

第一版只做 bag/run-level paired descriptive statistics，不把時間序列 sample
視為獨立樣本，不做 `n=3` 顯著性優越宣稱。固定單一起點的結果只能定位
candidate degradation interval，不能直接決定 production backend 或 confidence
calibration。
