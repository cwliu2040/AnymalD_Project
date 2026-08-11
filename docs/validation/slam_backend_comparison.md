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

上述 reference arm 屬於一般 backend comparison 的既有 deployment 參考；
`slam_yaw_stress` gate 依使用者決策只比較兩個 backend 的 native deskew，不把
LIO-SAM project-deskew 加入該 240-run 主矩陣。詳細 contract 見
`docs/validation/slam_yaw_stress.md`。

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

共同 confidence 的 normative schema／state machine／backend signal inventory 與
validation gates 由 `docs/slam_confidence_contract.md` 定義。固定語意是：

```text
C(t) = P(U(t, t + 0.50 s) = true |
         t 當下與過去可部署取得的 backend diagnostics)
```

`t` 固定是20 Hz `evaluation_stamp`；`source_stamp`只指向該tick可用的latest
canonical pose。Offline horizon也從evaluation time起算，避免較慢backend的
pipeline age造成label leakage。

`U` 表示該視窗內 pose 持續符合共同的 freshness、finite、translation、yaw、
jump 與 tracking-failure label。它不是當下 pose 完全正確的機率、ATE、
covariance，也不是某個 backend 的 raw ICP fitness、feature count 或 residual。
GT `/odom` 只能在 offline label/evaluator 使用，不能進 online extractor、
deployment confidence 或 PPO observation。

共同輸出不是四個彼此獨立、可能錯配的 topic，而是單一 atomic message：

| 介面 | 語意 |
| --- | --- |
| `/slam/odom` | canonical `nav_msgs/Odometry`；不可覆蓋 GT `/odom` |
| `/slam_confidence` | `SlamConfidence` atomic snapshot：score、hard validity、state、source/evaluation stamp、age 與 reason mask |
| `/slam_tracking_valid` | 只便利 mirror message 內的 bool；consumer 仍須有 receipt watchdog／DDS deadline |

`source_stamp` 必須精確等於被評估的 `/slam/odom.header.stamp`；新的 IMU 或
LiDAR callback 不得單獨推進它。`slam_tracking_valid` 是 pose 現在能否使用的
hard gate，優先於連續 score；即使上一筆 score 是 `0.95`，source stale 時也
必須 `LOST/invalid`。ROS logical age、consumer steady-clock liveness 與 physical
callback latency分開量測，不能用 callback arrival time 掩蓋 backend latency。

## Calibration 方法

LIO-SAM 與 FAST-LIO2 各自使用目前真正可觀測的 diagnostics，再校正到上述
同一個 `U`。逐 timestamp label只做 initial-SE(2) alignment；ATE與yaw RMSE
仍是整包 aggregate，不能拿aggregate pass替整包每個 frame貼 healthy。

資料以 capture group切分，final holdout前凍結 signal transform、threshold、
hysteresis 與 calibration ID。Gate同時檢查 gradual degradation的event recall、
lead time、frame AUROC、healthy false-low time，以及 probability calibration的
Brier score、ECE與reliability diagram；bootstrap以bag/episode為單位，不能把
20 Hz frames當成獨立樣本。Outage、NaN、reset、stale-high等 abrupt failure由
hard-validity fault-injection gate檢查，不要求保留的 score也必須先下降。

## 目前 branch 狀態

- benchmark base：`benchmark/slam-liosam-fastlio2`，commit
  `fee8c9f 建立 SLAM backend 比較基線`
- 實驗 branch：`exp/slam-fastlio2`
- FAST-LIO2 adapter：已建立輸入與輸出 adapter，並已通過第一個 live
  locomotion runtime gate；尚未完成正式 backend qualification。
- FAST-LIO2 project launch：`fastlio2_benchmark.launch.py`；replay launch：
  `fastlio2_replay_benchmark.launch.py`。
- FAST-LIO2 locomotion launch：`fastlio2_locomotion_benchmark.launch.py`；policy
  讀取 `/slam/odom`，simulation 的 ground-truth `/odom` 只供 stability
  diagnostics 與 readiness 使用。
- FAST-LIO2 confidence第一階段：`fastlio_confidence_extractor`已實作exact-stamp
  source assembly與共同state/reason，但尚無calibration artifact，固定
  `UNCALIBRATED/invalid`。三個FAST project wrappers皆以
  `enable_confidence:=false`預設關閉；true只開instrumentation，policy仍不consume。
- LIO-SAM confidence第一階段：`liosam_odom_adapter`與
  `liosam_confidence_extractor`已實作mapping/canonical/incremental/CloudInfo的
  exact-stamp assembly；project deskew arm另要求motion status，native arm不要求。
  同樣固定`UNCALIBRATED/invalid`並由`enable_confidence:=false` opt-in控制；
  `smoke_out_and_back`兩arm已收到source-valid snapshot且trajectory gate通過。
- FAST-LIO2 project cloud topic：`/fastlio/points`，使用與 LIO-SAM 相同的
  raw `/lidar/points_raw`，但輸出欄位名稱為 `ambient` 並使用 reliable QoS。
- FAST-LIO2 project odometry topic：`/slam/odom`，由
  `fastlio_odom_adapter` 將候選的 `/Odometry`、`camera_init/body` 轉為
  `map/base_link`，並從 pose delta 推導 body-frame twist；不會覆蓋 simulator
  ground-truth `/odom`。
- 現有 root `docs/project_knowledge.md` dirty 必須與本 branch 的程式變更
  分開處理。
- nested upstream LIO-SAM 的既存 dirty 不得修改、還原或提交。

FAST-LIO2 config：
`deployment/ros2_ws/src/anymal_locomotion_ros2/config/fastlio2_anymal_ouster32.yaml`。
其中 extrinsic 是 LiDAR 在 IMU/base frame 的位置 `[0.20, 0.0, 0.35]`，不是
把 project motion-deskew 的輸出接進 FAST-LIO2。

## FAST-LIO2 ROS 2 Humble smoke/replay test

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

project adapter 建立後，以下結果來自同一份
`smoke_out_and_back` bag（185 raw scans、29.87 s；實際約 6.19 Hz），
LIO-SAM 使用 native deskew：

| Profile | Middleware | Mapping odom | Translation ATE RMSE | Yaw RMSE | 結果 |
| --- | --- | ---: | ---: | ---: | --- |
| LIO-SAM native | Cyclone DDS | 182 | 0.0374 m | 0.084 deg | passed |
| FAST-LIO2 stride 4, blind 0.5 | Cyclone DDS | 182 | 0.3018 m | 1.169 deg | failed |
| FAST-LIO2 stride 4, blind 2.0 | Cyclone DDS | 182 | 0.3310 m | 1.663 deg | failed |
| FAST-LIO2 stride 1, blind 0.5 | Fast DDS | 182 | 0.1133 m | 0.180 deg | failed ATE |
| FAST-LIO2 stride 2, blind 0.5 | Fast DDS | 182 | 0.0944 m | 0.225 deg | passed |
| FAST-LIO2 stride 2, blind 0.5 | Cyclone DDS | 182 | 0.0944 m | 0.225 deg | passed |

stride 2 是目前此單一 replay 的實驗預設；它尚未代表 FAST-LIO2 已完成正式
backend qualification。仍需在其他 locomotion/退化場景重跑、確認 effective-point
coverage、latency/queue drop 與 physical-clock throughput，再做 locomotion
stability 與 confidence calibration。stride 1 在 Cyclone/iceoryx 下曾因
4.25 MB shared-memory chunk 不足而 crash；這是 middleware/吞吐 gate，不能
用 partial trajectory 當精度結論。

## FAST-LIO2 live locomotion smoke

在 `exp/slam-fastlio2` 使用正式 Recovery v0.4.0 `model1450` policy，執行
stationary 與 `forward_0_5` live smoke：

| Metric | Stationary | Forward 0.5 m/s |
| --- | ---: | ---: |
| Simulation steps | 1000 | 1200 |
| Stability profile | passed | passed |
| Termination / truncation | 0 / 0 | 0 / 0 |
| Controlled joint-command freshness | 406 / 406 (1.0) | 656 / 656 (1.0) |
| Joint-command wait timeout | 0 | 0 |
| Policy diagnostic records | 534 | 784 |
| Event-order classification | `no_instability` | `no_instability` |
| FAST-LIO2 policy odometry source | `/slam/odom` | `/slam/odom` |

`forward_0_5` 的 GT-only stability diagnostics 另外得到：target actual
velocity `0.4552 m/s`、target MAE `0.0789 m/s`、base roll max `0.0479 rad`、
base pitch max `0.0402 rad`；這些是 locomotion assessment，不是 SLAM ATE。
這次 controlled loop 的 real-time factor 約 `0.741`，因此 physical-clock
throughput 仍是未完成的 backend gate。

這次測試確認 FAST-LIO2 odometry 已能進入 policy，且 policy 能持續輸出
joint command；並非只把 `/odom` ground truth 接回 policy。因為 FAST-LIO2
輸出約 10 Hz，而 locomotion policy 為 50 Hz，這個專用 launch 使用 timer
trigger 取最新 SLAM state，並以 0.25 s receipt-age timeout 防止 stale state
繼續推動 policy。原本要求 joint、IMU、SLAM odom 在 10 ms 內對齊的
`synchronized_state` 不適合這個低頻 backend。

forward smoke 的 IMU bridge parity 在 `imu_observation_parity_atol=0.01` 下
通過，最大 angular-velocity error 為 `0.002587`；這個 threshold 只屬於
LiDAR-enabled live simulator 的 bridge validation，不是 SLAM confidence 或
locomotion stability threshold。

這次的 stationary stability assessment 沒有把 simulator ground truth 當成
policy input；GT 僅供 stability diagnostics 與驗證使用。尚未加入 confidence，
也尚未測試 forward locomotion、點雲退化與 SLAM confidence calibration。
