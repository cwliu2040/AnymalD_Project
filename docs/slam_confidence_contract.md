# Backend-neutral SLAM confidence contract

狀態：`SlamConfidence` v1 wire schema、production freshness/state semantics與
native-only calibration pipeline已凍結。FAST-LIO2 的24-group與LIO-SAM 的40-group
strict group-split calibration皆已通過最後未看過的holdout，兩者都已安裝可發布
非零分數的fingerprinted artifact。FAST 的13-case synthetic hard-fault matrix與
兩backend的實際replay source assembly、artifact ID／score／timestamp驗證也已通過；
兩backend在CycloneDDS隔離graph上的12-case topic/publisher fault qualification，及
FAST cube=1000的左右`wz=2.0`與雙向lateral live qualification亦已通過。
共同介面已達後續PPO observation的資料前置條件；本階段仍不修改SLAM演算法、
不把confidence接入正式Recovery v0.4.0 policy，也不開始PPO。

## 範圍與權威來源

共同 ROS 2 型別是
`anymal_locomotion_interfaces/msg/SlamConfidence`；可機讀的 provisional 參數在
`configs/slam_confidence_contract.yaml`，兩 backend 的 runtime 起點安裝於
`anymal_locomotion_ros2/config/slam_confidence_{fastlio2,liosam}.yaml`。message
定義 wire layout，本文定義欄位與狀態語意，YAML 有 static parity test。任一處不一致
時不得讓 tracking 保持 valid，必須先修正 contract。

本 contract 只處理 SLAM pose 是否可用及其短視窗品質，不改變正式 Recovery
v0.4.0 policy，也不把 LiDAR/SLAM 塞入既有 48-D observation。正式 policy 仍是：

`exported/anymal_d_locomotion_v1/recovery_v0.4.0/policy.onnx`

## ROS 2 topics 與 QoS

| Topic | Type | 角色 |
| --- | --- | --- |
| `/slam/odom` | `nav_msgs/msg/Odometry` | backend adapter 產生的 canonical pose；不可覆蓋 GT `/odom` |
| `/slam_confidence` | `anymal_locomotion_interfaces/msg/SlamConfidence` | 唯一權威、atomic 的 score／validity／age／reason snapshot |
| `/slam_tracking_valid` | `std_msgs/msg/Bool` | `slam_tracking_valid` 的便利 mirror，不是另一套判定 |

`/slam_confidence` 與 mirror 使用 reliable、volatile、keep-last depth 1，依 ROS
logical time 以 20 Hz 發布。當 backend 停止但 ROS clock 仍前進時，extractor
持續發布：`source_stamp` 與最後 score 固定，`confidence_age` 增加，並在
freshness gate 到期後立即 invalid。

`/slam_tracking_valid` 單獨不是安全介面。Extractor process 死亡或 `/clock`
停止時不會再有新訊息；subscriber 若只保留最後一筆 `true` 會 fail open。
任何 consumer 都必須有 deployment-specific 的 steady-clock receipt watchdog
或 DDS deadline。YAML 的 `0.15 s` 只是 physical deployment 起點，不進 ROS
timestamp 計算，也不作 replay calibration feature。

## `slam_confidence` 的共同語意

`slam_confidence(t)` 在安裝合格 calibration artifact 後，才是已校正的連續
短視窗可用性信心：

```text
C(t) = P(U(t, t + H) = true | t 當下與過去可部署取得的 diagnostics)
H = 0.50 s
```

這裡的 `t` 是 `evaluation_stamp`，不是較舊的 backend `source_stamp`。`U` 的
offline 定義見後文。這不是當下 pose「完全正確」的機率、不是 ATE、
不是 covariance，也不是任一 backend 的 raw feature count／residual。LIO-SAM
與 FAST-LIO2 各自有 signal transform 與 calibration，但使用相同 `U`、`H`、
threshold 及 holdout gate，因此校正後才有 backend-neutral 語意。

每個新完整 source bundle 在其後第一個 20 Hz ROS-time evaluation tick，以當時的
source age與最新可部署signals計算一次score；同一source stamp在後續tick採
sample-and-hold，age／freshness與state仍逐tick更新。這與calibration transform只讓
每個source stamp更新一次temporal state的語意一致，也避免DDS callback排程改變
age feature。分數先量化成wire `float32`再進threshold/hysteresis。Hard bundle會
推進causal transform但不覆寫最後score；沒有source時score為`0`。新bundle若無法
在evaluation tick產生score，則以`SIGNAL_MISSING` fail closed。

`slam_tracking_valid` 是現在能否使用 pose 的 hard gate，優先於 score。它不等於
`slam_confidence > 某個 threshold`：source/odom/sensor stale、時間戳倒退、
NaN、可明確觀測的 backend reset、未校正等條件都會繞過 score dwell，立即 invalid。

## Message 欄位與 invariants

| 欄位 | v1 語意 |
| --- | --- |
| `schema_version` | 必須等於 `SCHEMA_VERSION=1`；未知版本 fail closed |
| `backend_id` | bounded algorithm-family ID，目前只允許 `fastlio2`／`liosam`；精確 provenance 不塞在這裡 |
| `calibration_id` | bounded、可稽核 calibration artifact fingerprint；缺失、不相容或 `uncalibrated` 必須 invalid |
| `source_stamp_valid` | 明確表示 source 是否存在；ROS time `0` 是合法值，不能當 sentinel |
| `source_stamp` | 被 confidence 評估的 canonical `/slam/odom.header.stamp`，必須逐位相等 |
| `evaluation_stamp` | extractor 建立 snapshot 時的 ROS logical time |
| `confidence_age` | 有 source 時為 `max(0, evaluation_stamp-source_stamp)` 的發布當下快照 |
| `slam_confidence` | finite、clamp 到 `[0,1]` 的 calibrated `C(t)`；source stale 時保留最後值 |
| `slam_tracking_valid` | current pose usability hard gate |
| `tracking_state` | `INITIALIZING`／`TRACKING`／`DEGRADED`／`LOST` |
| `degradation_reasons` | `REASON_*` bitwise OR；原因可同時存在，不設假 priority |

必要 invariants：

- `slam_tracking_valid == (state == TRACKING or state == DEGRADED)`。
- `INITIALIZING`、`DEGRADED`、`LOST` 的 reason mask 必須非零。
- `REASON_NONE` 只能是整個 mask 的值 `0`，不能與其他 bit 並存。
- 未知 state、未知 reason bit、未知 schema、空白/未知 calibration 都由 consumer
  本地判為 invalid；不可沿用上一筆 valid。
- `source_stamp_valid=false` 時，`source_stamp` 與 `confidence_age` 內容不可拿來
  算 freshness；publisher 將兩者清成 zero，score 設為 `0`。
- 新 IMU 或 LiDAR callback 不得單獨推進 `source_stamp`。只有完成 diagnostics
  對齊、且存在同 stamp canonical odom 的 confidence sample 才能推進。
- FAST-LIO2 的完整 source 必須以 integer nanoseconds exact-join native odom、
  canonical odom 與 `/cloud_effected`。三個 callback 可任意順序抵達；從第一個
  required input callback 的 ROS logical arrival time起算 `0.05 s` grace，不能
  從較舊的 sensor/source stamp 起算。Grace 內不因暫時缺另一個 topic 而
  invalid，逾時才設`SIGNAL_MISSING`。最多保留 64 個 pending stamps。
- 若較新的完整 bundle 已 commit，仍不完整的較舊 stamp 會先產生一次 latched
  `SIGNAL_MISSING` hard event再evict；它不能無聲消失，也不能永久阻止新source
  進入recovery。遲到的舊component依monotonic規則視為timestamp regression。
- Pending cache超過64筆時同樣先產生一次`SIGNAL_MISSING`再evict最舊stamp，
  不能靜默丟資料。
- 首次完整 source 前，任何fault都保持`INITIALIZING/invalid`並保留實際reason；
  曾有 source 後的 immediate hard condition 才進
  `LOST/invalid`。No-source startup 不設 `RECOVERY_PENDING`；已有 source、hard
  已清除但 readiness/hysteresis 尚未回到 `TRACKING` 時才可設定。
- no-source、clock reset、backend/calibration instance switch 清 source/score；
  stale、malformed 或 rejected input 保留 last-good source/score但 hard invalid。
  任何accepted complete bundle即使帶hard reason仍會推進source stamp並hold最後
  score；zero effective support就是目前FAST的例子。Root-cause bits（包含hard
  reason或low-confidence LOST）latch 到完整 recovery 成功。

`schema_version` 只能版本化同一 wire layout 的語意；ROS type 增刪欄位不是
向後相容。v1 一旦凍結便不再改 layout；未來不相容變更使用新 message/type 與
明確 topic migration，不假設舊 subscriber 能靠 message 內版本自動相容。

### 初始化與 stale-high 範例

尚未有完整 source：

```text
source_stamp_valid = false
slam_confidence = 0.0
slam_tracking_valid = false
tracking_state = INITIALIZING
reasons = INITIALIZING | SIGNAL_MISSING
```

目前 FAST-LIO2 instrumentation-only runtime 會再加 `UNCALIBRATED`。首次完整
bundle 抵達後仍因 `UNCALIBRATED` 轉成 `LOST/invalid`；這是刻意的 fail-closed，
不是 estimator tracking failure。

最後一次量測很健康，但已 stale：

```text
source_stamp_valid = true
slam_confidence = 0.95       # 保留最後一次 calibrated prediction
slam_tracking_valid = false
tracking_state = LOST
reasons includes SOURCE_STALE and/or ODOMETRY_STALE
```

這正是 score、validity 與 age 必須分開的原因。Safety/PPO integration 永遠先
gate validity，再使用 score；不能把 stale 0.95 當成現在仍健康。

## 時間模型與 replay determinism

三種時間不可混用：

1. **ROS logical time**：`source_stamp`、`evaluation_stamp`、age、freshness 與
   hysteresis。sim/replay 使用 `/clock`，實機使用 system ROS time。
2. **consumer steady time**：process/message liveness watchdog。它不寫進 message，
   不當成 calibration feature。
3. **physical latency**：callback-to-publication、CPU/queue throughput；以額外
   steady-clock telemetry 評估，不從兩個 ROS stamp 推測。

`source_stamp` 若比 evaluation time 超前超過 provisional `5 ms`，或同一 ROS
clock epoch 內倒退，立即 `LOST + TIMESTAMP_INVALID`。完全相同 stamp／內容的
duplicate 是 idempotent，不推進 dwell；相同 stamp 但內容衝突視為 invalid。

ROS clock 向後跳代表新 epoch：清除 source cache、score、所有 dwell，進
`INITIALIZING + CLOCK_RESET`。單一 backend source 倒退而 ROS clock 未重置，則
進 `LOST + TIMESTAMP_INVALID`，不清成一個看似正常的新 epoch。大幅向前跳由
freshness gate 自然失效。

Replay 的 state update 使用固定 20 Hz ROS-time grid；同 stamp tie-break、
duplicate 與 out-of-order 規則固定。相同 bag 重跑或改 `ros2 bag play --rate`
後，evaluation-stamp score/state transition sequence與各自source stamp必須一致。
Live steady-clock
watchdog 不混入此 deterministic sequence。

目前純core已驗證logical elapsed-time determinism，但runtime timer的grid origin、
missed-tick catch-up及同一logical time的input/timer tie-break仍須在下一個replay
gate前固定並做不同rate實測；instrumentation-only node尚未宣稱通過這項gate。

## State machine 與 hysteresis

本輪可機讀起點：

| Parameter | Provisional value |
| --- | ---: |
| `degrade_below` / dwell | `0.45` / `0.10 s` |
| `invalidate_below` / dwell | `0.25` / `0.20 s` |
| `recover_at_or_above` / dwell | `0.55` / `0.50 s` |

所有 dwell 以 ROS logical elapsed time 累積，不以 callback 次數累積。條件中斷、
clock epoch reset、backend/calibration switch 都清除對應 accumulator。

| Current state | Condition | Next state / validity |
| --- | --- | --- |
| `INITIALIZING` | hard conditions clear，score `>=0.55` 持續 `0.50 s` | `TRACKING / true` |
| `INITIALIZING` | 尚未完成上列 dwell | 保持 invalid，`INITIALIZING`，加 `RECOVERY_PENDING` |
| `TRACKING` | score `<0.45` 持續 `0.10 s` | `DEGRADED / true` |
| `TRACKING` | score `<0.25`；degrade dwell 尚未到 | 仍 `TRACKING / true`，設 `LOW_CONFIDENCE`，繼續兩個 dwell |
| `TRACKING` 或 `DEGRADED` | score `<0.25` 持續 `0.20 s` | `LOST / false` |
| `DEGRADED` | score `>=0.55` 持續 `0.50 s` | `TRACKING / true` |
| `DEGRADED` | score 位於中間 hysteresis band | 保持 `DEGRADED / true`，設 `RECOVERY_PENDING` |
| `LOST` | hard conditions clear，score `>=0.55` 持續 `0.50 s` | `TRACKING / true` |
| 已有完整 source的任意 state | immediate hard condition | 當次 evaluation 進 `LOST / false`；clock reset 例外進 `INITIALIZING / false` |
| 尚未有第一筆完整 source | 任一 pre-source fault | 保持 `INITIALIZING / false`，保留實際 reasons |

若 `backend_id` 或 `calibration_id` runtime 改變，視為新 instance：清 cache/dwell、
回 `INITIALIZING`。Hard或low-confidence root-cause bit latch 到 recovery 完成；
實際 condition清除但還在 dwell 時加 `RECOVERY_PENDING`。成功回 `TRACKING`
後才清 latch。

## Degradation reasons

| 類別 | Reasons | 判定 |
| --- | --- | --- |
| lifecycle | `INITIALIZING`, `LOW_CONFIDENCE`, `RECOVERY_PENDING` | 由共同 state machine 設定 |
| immediate hard | source/odom/LiDAR/IMU stale、可明確觀測的 estimator reset、timestamp/numeric invalid、backend error、clock reset、uncalibrated | bypass score dwell，立即 invalid |
| calibrated soft | degenerate geometry、high residual、not converged | 納入 backend-specific score；單一 bit 不直接跨 backend 比較 |
| conditional | insufficient support、signal missing | zero support 或缺 required signal 是 hard；低非零 support 或缺 optional signal 是 soft |

`SOURCE_STALE` 表示「最後一個完整、與 canonical odom 對齊的 confidence
diagnostic bundle」過期；`ODOMETRY_STALE` 表示最新 canonical pose 過期。Odom
可繼續但 diagnostics 停止，反之亦然，所以兩個 bit 不合併。

`ESTIMATOR_RESET` 只可在 backend 有 deterministic epoch/reset signal 時設置。
目前 FAST-LIO2 現成 topics 沒有這個 signal，不能把 pose jump 或 process猜測
硬標成 exact reset；第一版會以實際可觀測的 stale、timestamp 或 backend error
fail closed。未來若加入 project-owned process/epoch health，再另外驗證 exact
reset reason。

## Backend-specific signal audit

下面區分三種 availability：

- **現成 topic**：第一版 project-owned extractor 可直接訂閱。
- **wrapper-derived**：可從現成 topic 的 stamp/內容可靠衍生。
- **需 instrumentation**：只存在 backend internal；本階段不可假裝已可用，
  也不以 `/rosout` 文字 parsing 充當 safety signal。

### FAST-LIO2

| Signal | Availability / source | 第一版角色與限制 |
| --- | --- | --- |
| native/canonical odom freshness | `/Odometry`；`fastlio_odom_adapter` 轉 `/slam/odom` | stamp/finite/monotonic/stale 是 hard；fresh odom 也可能只是 IMU prediction |
| LiDAR / IMU freshness | estimator實際輸入 `/fastlio/points`、`/imu/data` | wrapper-derived stamp age/gap hard gate；raw `/lidar/points_raw` 只在 adapter 上游，不能用它掩蓋 adapted input stale |
| effective support | patched `/cloud_effected`，width × height；stamp 與 odom 同為 `lidar_end_time` | count `0` 代表 final measurement update invalid，hard；低非零 count 只作 FAST-local calibration |
| support denominator | visual output 關閉時不一定有；`/cloud_registered` 語意又受 dense mode 影響 | diagnostics/calibration，不能列為第一版 required signal |
| ESKF pose covariance | `/Odometry.pose.covariance` 有值，但目前 publisher 有兩個缺陷 | lagged calibration feature only，不能當 hard uncertainty |
| residual、final update-valid、iteration/convergence、Jacobian conditioning | `res_mean_last`、`dyn_share.valid/converge` 等 internal | 需 downstream diagnostics instrumentation；project sidecar 無法直接讀 globals |
| exact initialization / internal queue | internal flags/buffers | 第一版以 finite monotonic odom + 同 stamp positive effective support + recovery dwell 定義 readiness |

FAST-LIO2 covariance 不是「未填」，但也不能直接使用：candidate 在
`laserMapping.cpp` 先 publish odom，才把當前 `kf.get_P()` 寫回重用 message，
所以 message `n` 帶的是 `P[n-1]`，第一筆全零；matrix block order又是
`[rotation, position]`，不是 ROS `[position, rotation]`。Wrapper 可用
`[3,4,5,0,1,2]` 同時 permute rows/columns 修正順序，但不能消除一掃 lag。
若 buffer 到下一掃再配回會增加約 `0.1 s` age 且遺失最後一筆；same-scan P
需要 publisher instrumentation。本輪不修改 estimator/publisher。

`/cloud_effected` 的 source flow 已確認：zero effective points 時 FAST-LIO2 仍
可發布 fresh prediction-only odom與 width-0 cloud；更早的 empty/too-few scan
則兩者都不發布。因此第一版不能只看 odom heartbeat，必須把 zero support 與
missing/stale 分開測試。

### LIO-SAM

| Signal | Availability / source | 第一版角色與限制 |
| --- | --- | --- |
| extracted feature proxy | `/lio_sam/feature/cloud_info` 的 corner/surface cloud count | 10 Hz continuous calibration；不是 voxel 後 accepted correspondence |
| deskew availability | 同一 `CloudInfo.imu_available/odom_available` | readiness/soft feature；完整 IMU coverage 不足時 scan 可能直接消失，仍須 freshness |
| mapping odom freshness | `/lio_sam/mapping/odometry`；`liosam_odom_adapter`轉成 `/slam/odom` | stamp/finite/monotonic/stale hard gate；fresh pose可能只是 initial guess |
| scan-to-map degeneracy | `/lio_sam/mapping/odometry_incremental.pose.covariance[0]` | 明確是 0/1 flag，不是 covariance；只作 soft/hysteresis |
| high-rate predictor freshness | `/lio_sam/odometry/imu_incremental`／`/lio_sam/odometry/imu` | diagnostics；200 Hz predictor不能掩蓋 stale mapping correction |
| project deskew status | `/lio_sam/deskew/motion`，僅 project-deskew arm | 該 arm 要求 deskew 時可作 hard readiness；native comparison不使用 |
| accepted constraints/residual、LM iteration/eigenvalues | map optimization internal | 需 instrumentation；目前 protected upstream boundary 下不可依賴 |
| graph covariance | internal、只在 keyframe更新，沒有寫入 odom | diagnostics candidate，不是 current scan quality |
| IMU failure reset reason/counter | internal warning + `resetParams()` | 精確 reason 需 instrumentation；每 100 key的例行 graph reset不是 failure |

LIO-SAM `isDegenerate` 只在 LM 第一次迭代更新；feature 不足或 selected
correspondence `<50` 提前退出時可能沿用舊值。Global/high-rate odom covariance
預設全零，不能解讀成零 uncertainty。既有 yaw pilot 的 feature count很接近，
yaw RMSE仍跨明顯範圍，也證明單一 extracted count不是 confidence。

LIO-SAM odom使用 scan-start stamp，ImageProjection 又刻意 buffer scans；現有
replay source age可約 `0.34 s`。FAST-LIO2 使用 scan-end stamp且既有 receipt age
約 `0.04 s`。因此不能把 FAST 的 `0.30 s` threshold 靜默套給 LIO。YAML 暫列
FAST `0.30 s`、LIO `0.50 s` 作 instrumentation/unit-test 起點；production
threshold 必須同時滿足 healthy p99、pose consumer reaction budget與 live gate，
並隨 calibration ID 凍結。

### 第一版 extractor 允許的 signal set

第一版只使用現成 topic與 wrapper-derived signals：

- 共同 freshness、finite、monotonic與clock/source lifecycle；
- FAST-LIO2 同 stamp effective count（zero hard，低非零作 calibration）；
- LIO-SAM extracted corner/surface proxy、availability 與 degeneracy flag；
- backend-specific initialization readiness。

未經 causal motion/velocity compensation 的 local pose delta 不作第一版 hard gate，
以免把正常的 3 m/s 或 `wz=2.0` 每掃位移誤判成 jump。Residual、conditioning、
optimizer exit reason與精確 reset先標記 unavailable。
Loop-closure fitness是稀疏、延遲的候選檢查，不代表 current local tracking；CPU、
DDS latency與queue gap是 transport health，不可偽裝成幾何 confidence。

## Extractor 與 calibration 邊界

- Extractor 是 `deployment/ros2_ws` 內的外部 project-owned ROS 2 node；Isaac
  Sim／Isaac Lab Python 不 import `rclpy`。
- Runtime extractor、launch與 calibration inference 不得 subscribe simulator
  GT `/odom`。GT只存在 offline label generator/evaluator。
- 本階段不修改 protected LIO-SAM，也不改 FAST-LIO2 estimator。未來若要
  internal signals，先另行審查只讀 diagnostics instrumentation、on/off parity與
  maintenance成本。
- `calibration_id` artifact 至少 fingerprint：algorithm family/source commit、
  build/patch、estimator config、adapter與timestamp contract、sensor model/
  extrinsic、sim或physical domain、signal transforms、`H`/label、dataset split、
  score thresholds與hysteresis。
- Artifact provenance schema v2把backend revision/source manifest、estimator
  config、split manifest、sensor/domain/extrinsic、timestamp/input adapter contract
  與score timing直接封入fingerprinted JSON；runtime loader逐項比對，不只檢查
  backend名稱。Calibration ID維持既有model/report identity，provenance envelope
  fingerprint可在不重訓權重下獨立更新。
- RTX reconstructed Ouster calibration 不得沿用到實體 Ouster官方 driver；
  sensor/domain改變就必須換 calibration ID並重過 gate。

目前兩個extractor都負責operational instrumentation、exact-stamp assembly與共同
state/reason publication，並各自載入通過holdout的native artifact。Artifact缺失，
或fingerprint/schema/backend/native-deskew/runtime provenance不相容時，node會拒絕
啟動或以空artifact override維持`uncalibrated` fail closed。LIO 的完整 source
exact-join native mapping odom、
canonical body odom、incremental odom與 feature `CloudInfo`；project motion
deskew arm再要求同stamp deskew status，native arm不要求。Launch都以
`enable_confidence:=false`預設關閉；FAST設為true時才額外開
`publish.effect_en`。正式 policy 不訂閱此輸出。

## Offline target 與 event label

GT `/odom` 只由 offline evaluator使用。對 deterministic 20 Hz
`evaluation_stamp` grid 的每個 tick：

1. 保存該 evaluation tick 當時已抵達、runtime 可用的 latest complete source
   features；不可用 bag 中稍後抵達的資料回填。
2. 以 source stamp 對齊 estimate pose與GT，依現有 evaluator相同規則插值 GT，
   套用該 backend sensor offset。
3. 只用第一個有效 matched pair 做 initial SE(2) alignment；禁止用整段 trajectory
   最佳化 alignment 掩蓋 drift。
4. 輸出逐 timestamp 的 `translation_error_m(t)`、`yaw_error_deg(t)` 與相鄰
   estimate-vs-GT step residual。ATE/yaw RMSE仍是整包 aggregate summary，
   不是 event onset 名稱。

`U(t,t+H)` 的 horizon 從 `evaluation_stamp=t` 開始，不從較舊的 source stamp
開始。否則 LIO-SAM 約 `0.34 s` pipeline delay 會讓一部分「未來」在 evaluation
前已經發生，造成 label leakage，並破壞與 FAST-LIO2 的可比性。Source stamp只
負責 pose alignment 與 provenance。

在 `H=0.50 s` 內符合下列全部條件才令 `U=true`：

- finite、monotonic pose持續存在，odom source停止推進不超過 `0.30 s`；固定
  pipeline latency只進`confidence_age` feature，不等同odom outage；
- translation error不連續超過 `0.10 m` 達 `0.20 s`；
- yaw error不連續超過 `1.0 deg` 達 `0.20 s`；
- 沒有單步 translation residual `>0.20 m` 或 yaw residual `>2.0 deg`；
- 沒有已知 backend reset/tracking failure。

Pose-error event以違規視窗起點作 onset（確認需要看滿 `0.20 s`）；jump、NaN、
timestamp與outage依 hard condition onset。相距小於 `1.0 s` 的同類失敗合併，
只有 label-negative持續 `1.0 s` 才 re-arm。既有 replay pass gate的 translation
ATE是 `max(0.10 m, path length 1%)`；它與固定 `0.10 m` 逐時刻 confidence
label用途不同，不能因整包 pass就把每個 frame都標 healthy。

資料以capture group（bag + seed + motion profile）分成model train、probability
calibration、threshold validation與final holdout；五組pilot使用2/1/1/1。同一
trajectory的frame不得跨split；threshold與calibration ID在final holdout前凍結。

## Validation gates

每個 backend各自通過後，才做 identical-bag paired comparison；一個 backend
通過不會替另一個完成 calibration。

1. **Schema / build**：interface與project ROS packages build成功；schema、bounded
   IDs、unknown enum/version、no-source、clock reset、duplicate/conflicting stamp、
   stale-high與完整 state transition都有 deterministic tests。
2. **No leakage**：static dependency、launch graph與runtime topic audit證明
   extractor沒有 `/odom` subscription、沒有 simulator-only import；GT只在
   offline label process。
3. **Signal coverage**：每個 healthy validation bag在初始化後，至少99%的
   source odom stamps有matched raw-signal record；missing和zero分開，不能補成
   healthy。FAST必測 `fresh odom + empty /cloud_effected`。
4. **Replay determinism**：相同bag至少重跑三次，並用不同 replay rate；按
   evaluation stamp比較score/state/reason transition一致，source stamp也須逐筆
   相同。Clock reset、同stamp
   tie-break與dwell使用logical time。
5. **Healthy availability**：只以 final-holdout failure-negative、也就是
   `U=true` healthy windows為分母，
   `slam_tracking_valid>=99%`；任何 unexplained invalid episode `>0.30 s` 個別
   review。整包aggregate pass不能取代此gate。
6. **Hard-fault detection**：LiDAR/IMU/odom freeze、diagnostic missing、future/
   regressing/conflicting duplicate stamp、NaN/Inf、zero support、clock backward
   與publisher death全數在一個20Hz tick或對應freshness期限內invalid，並驗證
   可觀測 reason與recovery sequence。Backend restart只有在另有明確 epoch/reset
   signal時才要求 exact `ESTIMATOR_RESET`；目前 FAST topics 不假裝能辨識。
   Extractor publisher death由consumer steady-clock watchdog測試，不期待已死亡的
   publisher再發reason。
7. **Gradual predictive calibration**：只對幾何／support逐步退化事件評分；至少
   20個independent events、5個capture groups。`C<0.45` 在onset前命中率至少
   80%，median lead time至少`0.20 s`，frame AUROC至少`0.80`，healthy
   false-low time不超過5%。Abrupt outage/NaN/reset由hard gate評估，不因
   stale-high score而判 predictive miss。
8. **Probability calibration**：final holdout報告Brier score、ECE、reliability
   diagram與score-bin failure rate；bootstrap以bag/episode cluster抽樣並報95%
   CI，不把20Hz frame假設成獨立樣本。這些先作報告，acceptance bound在
   calibration pilot後、final holdout前凍結。
9. **Instrumentation parity / overhead**：任何新增 diagnostics做off/on A/B；
   trajectory/source stamps不得改變，logical output 20Hz容許±5%。Estimator
   parity與physical throughput分開；physical p99 callback-to-publication低於
   `50 ms`，以steady telemetry量測。
10. **Live qualification**：cube=1000 baseline完成左右多圈`wz=2.0`與lateral
    live run，無backend crash、unexplained invalid或freshness loss。Simulator
    驗logical 20Hz並以RTF正規化wall rate；physical 20Hz gate只適用RTF≈1或
    實機system-time deployment。

Gate輸出必須逐 backend列event count、capture groups、threshold/calibration ID、
false positive、lead time、AUROC/calibration與所有failures，不只報平均ATE。

### ROS 2/DDS hard-fault與FAST live gate（2026-08-11）

安裝型`slam_confidence_dds_fault_validation`在兩個隔離topic namespace直接建立
rclpy publisher、extractor與subscriber，實際使用`rmw_cyclonedds_cpp`。FAST與LIO
各測IMU/LiDAR freeze、odometry publisher death、diagnostic publisher death、
zero-support payload及extractor publisher death，共12/12通過。前五類皆在對應
freshness/join期限加一個20 Hz排程容差內fail closed；可用score為1.0時仍令tracking
invalid。Extractor死亡後consumer的0.15 s steady receipt watchdog將adapter中的
confidence與valid強制為0。可恢復case均回到TRACKING；FAST zero-support因凍結artifact
的causal support memory約需8.5 s恢復，屬預期模型歷史而非DDS失效。

FAST正式live gate使用native deskew、cube=1000、正式Recovery v0.4.0 model1450，
policy feedback為`/slam/odom`，GT `/odom`沒有進runtime confidence或policy observation。
左右`wz=2.0`各27 s與左右`vy=1.5`各14 s共四個profile皆完成：共1,648筆confidence
snapshot，進入TRACKING後0次invalid、0 freshness loss、0 unknown reason/ID/timestamp/
UNCALIBRATED violation；四次locomotion均0 termination、0 truncation、0 non-finite且
分類為`no_instability`。高yaw IMU parity將projected gravity容差維持0.01，僅angular
velocity依量測上限0.0269354 rad/s分離為0.03 rad/s；root-body與Action Graph odometry
angular velocity彼此一致，差異來自獨立取樣的physics IMU時序。

一次較早的`yaw_left_2_0_imu_debug`在19.5 s termination，confidence隨support下降由
TRACKING轉DEGRADED/LOST；此失敗診斷保留、不算入正式pass run。機讀摘要為
`docs/validation/slam_confidence_dds_and_fast_live_gate_summary.json`，詳細raw reports
位於ignored `logs/slam_confidence/`。

## 先前 pilot 停止點（已由下節取代）

本輪已凍結 v1 wire schema與共同 state/時間語意，並完成兩 backend 的
instrumentation-only extractor、exact-stamp assembler、deterministic unit/static
tests，以及 evaluation-anchored offline label core。FAST另有13-case pure-core
synthetic hard-fault matrix，涵蓋source/odom、LiDAR、IMU freeze，effect
missing/zero/malformed，future/regressing/conflicting stamp、numeric invalid、frame
contract、clock reset與missing-scan recovery；所有hard fault都保留stale-high score
但令tracking invalid，clock reset則依contract清source/score。

LIO-SAM已在既有 `smoke_out_and_back` bag以project與native deskew兩種arm收到
363筆source-valid confidence snapshot，stamp無倒退且trajectory gate通過；project
arm instrumentation on/off同為182筆mapping，ATE差約0.17 mm、yaw RMSE差約
0.00037 deg，未見軌跡退化。這仍不是ROS DDS/topic-level fault injection，也不
代表confidence已校準。

後續已固定正式capture為native deskew，並以`forward_3_0`、`lateral_1_5`、
`combined`、`curve_3_0_left_0_5`、`warehouse_final_turn`五個source bag，各跑
`1.00/0.50/0.10` deterministic uniform density；同一source bag的三個variant
保持在相同capture-group split。每個backend共15個derived captures、4,392個有
完整0.5 s horizon的20 Hz rows，其中4,272個source-valid rows的feature coverage
為100%。GT只在evaluator完成後產生label，extractor runtime沒有`/odom`。

Pilot採backend-local standardized logistic transform加isotonic calibration。
FAST的probability-calibration capture只有單一label class，isotonic無法合法fit，
所以不碰final-holdout score metrics；holdout本身有819 frames、failure prevalence
38.22%與3個gradual events。LIO-SAM final holdout有819 frames、failure prevalence
28.45%、1個event；AUROC 0.9632、Brier 0.2502、ECE 0.3498、recall 100%、median
lead 0.50 s，但healthy false-low 75.26%。兩邊都少於20個independent events；
FAST另缺calibration split class coverage，LIO另失敗false-low。因此沒有產生或
安裝runtime artifact，score維持0。

以上是2026-08-10的歷史pilot結論；後續狀態以下節為準。

## Native gradual-v2 calibration 狀態（2026-08-11）

正式capture profile固定native deskew，使用未修改source bags，依source scan time
逐步做ring/FOV/density loss：3.0 s healthy、1.5 s ramp-down、4.5 s最低0.1%
support、3.0 s recovery。Calibration capture明確不載入runtime artifact；GT `/odom`
只在replay完成後建立evaluation-stamp anchored、H=0.5 s labels。

Estimator不是PPO。每個backend各自使用standardized logistic、獨立isotonic probability
calibration與一個以causal 0.2 s support projection／motion exposure為基礎的可解釋
safety envelope。FAST raw effective points與LIO corner/surface counts從不直接互比。
Artifact包含backend、feature schema/transform、threshold、input SHA、implementation
SHA、split manifest與artifact fingerprint；runtime loader逐項驗證，不符合即拒絕啟動。

FAST第一個holdout false-low失敗後已retire，另用四個完全未看過的source bags建立新
final holdout；`configs/slam_confidence_fastlio2_split.yaml`固定24個互斥groups。
新holdout通過：AUROC 0.973621、Brier 0.036620、ECE 0.078300、event recall 100%、
median lead 0.20 s、healthy false-low 4.938%，且support hard-gate前置recall 100%、
median lead 0.50 s。FAST已產生fingerprinted native-v2 artifact。

LIO先後誠實retire四輪final：run01的recall 75%／lead 0.10 s、run02的recall 25%、
run03雖recall 100%但healthy false-low 8.0%、fresh4的recall 50%。這些資料後續只進
development，未重複冒充fresh holdout。Frozen causal-v3模型使用corner/surface
support ratio、trend、0.2 s projection、memory、motion exposure與public
degeneracy/availability；bootstrap odometry gap不套degeneracy soft cap。最後四個
內容SHA全新的yaw source groups只評估一次：2,132 frames、5 gradual events、
AUROC 0.986497、Brier 0.021720、ECE 0.034500、`C<0.45` recall 100%、median lead
0.20 s、healthy false-low 3.145%，所有gate通過。LIO已產生native-v3 artifact。

凍結權重的final holdout另以capture instance為cluster、固定seed 0做2,000次
bootstrap。FAST AUROC/Brier/ECE的95% CI分別為
`[0.971146,0.977826]`／`[0.022120,0.046795]`／`[0.039236,0.108795]`；LIO為
`[0.964587,0.999549]`／`[0.011747,0.031692]`／`[0.033189,0.037379]`。
可提交摘要位於`docs/validation/slam_confidence_final_holdout_summary.json`；這次
只重評既有artifact，沒有refit或重訓。

已安裝artifact ID為FAST `native-v1-1e6cf8347be1`、LIO
`native-v1-edc098b0bd98`。兩個artifact-enabled native replay均確認exact ID、非零且
多值float32 score、tracking state/reason切換、零timestamp violation、零
`UNCALIBRATED`及runtime extractor不訂閱GT。刻意降到0.1% support的replay trajectory
gate失敗是預期fault outcome；confidence node沒有回饋或改寫任何SLAM input/output。

未來PPO observation介面固定為`slam_confidence`、`slam_tracking_valid`與
`clip(confidence_age/0.50 s, 0, 1)`；consumer另須使用steady receipt watchdog
`<=0.15 s`。Receipt過期時，即使最後一筆confidence很高也輸出confidence 0、valid 0。
這套轉換已在`slam_confidence_observation_core.py`測試，但尚未接入policy。
