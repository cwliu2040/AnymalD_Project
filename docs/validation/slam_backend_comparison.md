# LIO-SAM 與 FAST-LIO2 比較基線

本文件只適用於 `benchmark/slam-liosam-fastlio2` branch。正式
`main` 的 Recovery v0.4.0 與既有 project motion deskew 設定不在本文件
中切換。

## 第一階段比較原則

LIO-SAM 與 FAST-LIO2 必須使用同一批 raw LiDAR、同一份 raw IMU、相同
時間戳、ring/per-point time、sensor extrinsic 與 replay clock。Simulator
ground truth `/odom` 只能由 evaluator 使用，不能作為 SLAM backend 的輸入。

第一個公平比較使用兩個 backend 的原生 deskew：

- LIO-SAM：關閉 project-owned `motion_deskew`，保留 LIO-SAM upstream
  原生 deskew。
- FAST-LIO2：關閉 project-owned deskew，使用 FAST-LIO2 自己的
  undistortion/deskew。

LIO-SAM 的 native online benchmark 參數：

```text
use_motion_deskew:=false
feature_cloud_info_topic:=/lio_sam/deskew/cloud_info
motion_deskew_apply_translation:=false
motion_deskew_replace_upstream_rotation:=false
motion_deskew_required:=false
```

`lio_replay_benchmark.launch.py` 使用相同的前四個 deskew 參數；它不需要
`motion_deskew_required`，因為 replay evaluator 不以 project deskew readiness
作為啟動條件。

`motion_deskew_apply_translation` 與
`motion_deskew_replace_upstream_rotation` 在 node 關閉時不會生效，但
native profile 仍明確設為 `false`，避免實驗 metadata 產生歧義。

## 必須保留的 reference arm

目前通過的 LIO-SAM project pipeline 仍需保留為第二組 reference：

```text
use_motion_deskew:=true
feature_cloud_info_topic:=/lio_sam/deskew/cloud_info_motion_corrected
motion_deskew_apply_translation:=true
motion_deskew_replace_upstream_rotation:=true
motion_deskew_required:=true
```

這組結果代表目前 project deployment stack，不是與 FAST-LIO2 native
deskew 的純 backend 對比。FAST-LIO2 第一階段不直接接 project deskew
輸出，避免發生重複 undistortion；若之後需要共用前端，必須另定義
deskewed-cloud input contract。

## 測試順序

1. 用相同 raw replay 執行 LIO-SAM native profile。
2. 用相同 raw replay 執行 FAST-LIO2 native profile。
3. 執行 LIO-SAM project-deskew reference profile。
4. 比較 ATE、RPE、yaw RMSE、pose jump、scan/IMU coverage、latency 與
   tracking validity。
5. 在不加入 confidence 的情況下，讓目前 v0.4.0 policy 接收各 backend
   的 odometry，另測 locomotion stability；此結果不能用 simulator
   ground-truth `/odom` 取代。
6. backend gate 通過後，才在獨立
   `feature/ppo-slam-confidence` branch 加入 confidence-conditioned PPO。

## 共用 SLAM confidence contract

confidence 的固定語意是：

> 在目前時間戳與可觀測 backend diagnostics 下，接下來短時間內的
> odometry 是否仍適合 locomotion policy 使用的機率，範圍為 `[0, 1]`。

它不是 ATE，也不是某個 backend 的 raw ICP fitness、feature count 或
residual。ATE 只能在 offline calibration 使用，不能直接當成 online
confidence。

每個 backend adapter 都必須提供相同語意的四個欄位：

| 欄位 | 語意 |
| --- | --- |
| `odom` | `nav_msgs/Odometry`；`child_frame_id=base_link`，twist 為 body frame |
| `slam_confidence` | `float32`，已校正且限制在 `[0, 1]`，越大越可靠 |
| `slam_tracking_valid` | 硬性 tracking 狀態；失效時不可只用 confidence 偽裝成正常 |
| `confidence_age_s` | 從 confidence source timestamp 到目前時間的秒數；stale 時必須增大 |

confidence 的 timestamp 必須與 adapter 的 odometry source timestamp 可對齊；
不能只用 ROS callback 到達時間掩蓋 backend latency。初期可以用既有 ROS
standard messages 分開發布，但 adapter 必須在 diagnostics 中記錄同一個
source timestamp、publish timestamp 與 age。

## Calibration 方法

LIO-SAM 與 FAST-LIO2 各自使用自己的 raw diagnostics，但最後都校正到同一個
offline label：

```text
usable(t) =
  tracking_valid(t .. t+H)
  AND translation_error(t .. t+H) <= E_trans
  AND yaw_error(t .. t+H) <= E_yaw
  AND no odometry jump in (t .. t+H)
```

`H`、`E_trans`、`E_yaw` 先由 replay sampling rate 與 locomotion reaction
time 決定，不直接沿用某個 SLAM 的原始閾值。用 ground truth 只產生 offline
label，再對每個 backend 的 diagnostics 訓練或擬合 calibration map；online
不能讀 ground truth。

第一版至少要記錄：有效特徵／點的支持度、backend residual 或 innovation、
退化／失敗旗標、估計 covariance（若 backend 提供）、output age、丟訊息率與
局部 odometry jump。各 backend 的 raw 欄位可以不同，但 calibration 後的
輸出必須遵守同一個 `[0,1]` 語意。

Calibration gate 應檢查 reliability diagram、Brier score、ECE、低 confidence
對實際 unusable odometry 的召回率，以及正常場景的 false-low-confidence
比例；不能只看 ATE 平均值。

## 目前 branch 狀態

- branch：`benchmark/slam-liosam-fastlio2`
- FAST-LIO2 adapter：尚未存在，待建立共用 contract 後實作。
- 現有 root `docs/project_knowledge.md` dirty 必須與本 branch 的程式變更
  分開處理。
- nested upstream LIO-SAM 的既存 dirty 不得修改、還原或提交。

## FAST-LIO2 ROS 2 Humble smoke test（尚未通過 benchmark gate）

本 branch 暫時測試的 candidate 是
`Taeyoung96/FAST_LIO_ROS2`，commit
`373aa886402b6307db2995ca12b3f4596ef4f633`。它不是 hku-mars 官方 ROS2
release，而是將官方 FAST-LIO/FAST-LIO2 code 改成 ROS2 `rclcpp`、ament、
ROS2 launch 與 `livox_ros_driver2` 的 fork。

隔離測試結果：

- ROS 2 Humble build 成功；需要額外的官方 Livox SDK2 shared library。
- 預設 ROS2 YAML 的 `extrinsic_T: [0, 0, 0.28]` 混用 integer/double，會讓
  ROS2 parameter parser 在 node 初始化前失敗；只在 `/tmp` smoke copy 修正。
- node 可以初始化，也能在 project bag 收到 IMU 與 point cloud，並發布
  `/Odometry`。
- project adapter 的 best-effort QoS 與 candidate 的 reliable subscription
  不相容；測試用 `/tmp` relay 暫時隔離這個問題。
- candidate Ouster input 要求 `ambient` field，但 project cloud 使用
  `noise` field；測試出現 `Failed to find match for field 'ambient'` 與多次
  `No Effective Points`。
- `/Odometry` 的 frame/topic 仍是 `camera_init`、`body`、`/Odometry`，尚未
  符合本專案的 `/odom`、`base_link` 共用 contract。

因此目前結論是「ROS2 port 可 build、可啟動、可收到資料」，但尚未是可接受
的 FAST-LIO2 backend，也尚未執行 ATE。必須先處理 QoS、point-field mapping、
frame/output adapter 與 effective-point gate，再和 LIO-SAM 做公平比較。
