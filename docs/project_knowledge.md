# Current project knowledge

更新日期：2026-08-17

這份文件保存跨對話補充知識，讓 Work locally 模式的新對話在直接閱讀
repository 的程式、設定與其他 `docs/` 時，也能知道目前正式產物、近期
診斷、驗證狀態與未完成事項。它不取代 repository 或其他文件；詳細設計與
歷史仍以 architecture、release、validation 文件及實際程式碼為準。

## Repository state

- 專案根目錄：`/home/ros/anymal_locomotion`
- 目前 branch：`exp/slam-fastlio2`。
- 使用者於2026-08-17再次確認：`main`保存已完成、供國科會計畫使用的既有LIO-SAM成果，
  model48＋estimator15的正式化與論文工作只在`exp/slam-fastlio2`進行；不得為此合併或修改
  `main`。近期目標是第一級simulation deployment release及可發表的完整數據鏈，不宣稱實體
  ANYmal-D qualification。
- `main` 與 `origin/main` 仍停在 `98b43dd`（`驗證：完成 reset、地圖品質與
  閉環驗證`）；benchmark base branch
  `benchmark/slam-liosam-fastlio2` 與目前實驗 branch 都以
  `fee8c9f 建立 SLAM backend 比較基線` 為共同基礎。
- `exp/slam-fastlio2` 的本機與遠端baseline均為
  `9b015c7 建立 SLAM confidence 發表與正式化基礎`；其前一個commit為
  `7a5675a 修正 LIO-SAM 信心誤判並完成互動驗證`，再前一個commit為
  `8392743 加入雙 SLAM backend 互動驗證入口`，再前一個parent為
  `0d38247 完成 confidence-aware PPO 與狀態估計候選`，再前一個commit為
  `fbee219 更新 SLAM confidence 對話交接狀態`，均已push。使用者確認
  `exp/slam-fastlio2`作為完整SLAM confidence開發線，因此51-D observation/training、
  deployment consumer、第二輪bounded safe-command、proprioceptive estimator與gait-value
  實驗均保存在此branch。本機`feature/ppo-slam-confidence`目前亦指向`4c063a4`，但未建立
  同名遠端branch。其先前
  baseline `cb9e3e2 校準 FAST-LIO2 並建立 confidence 前置基線`與parent
  `7b9994d 修正虛擬雷射輸入與 FAST-LIO2 校準流程` 固定虛擬 Ouster input
  contract。更早的
  `649fcda 實作：加入原地旋轉 SLAM 壓力測試` 包含 yaw-stress、renderer、
  effective-support diagnostics 與 pilot tooling。兩backend真實DDS fault qualification、
  FAST live confidence monitor、IMU parity gate分離、四個cube=1000 live profiles及文件，
  均已提交在`0e9a571`。51-D observation/training/deployment consumer、bootstrap/export/
  evaluator tooling、proprioceptive estimator、gait-value gate、測試與驗證報告均已提交在
  `4c063a4`。Confidence
  milestone已在`b8b8c89`提交並push；第一次confidence-conditioned PPO已失敗並retire。
  第二輪 bounded safe-command `model_19` 已通過五個模擬confidence-state profiles與
  export parity，但兩backend live screening在LIO-SAM `forward_1_5`失敗，因此不可promotion，
  尚未接入或取代正式policy。
  C-v4至C-v10 confidence-conditioned PPO實作、estimator-closed-loop訓練接線、測試、
  驗證文件與interactive false-stop修正均已提交並push；C-v10模擬候選已通過固定behavior及
  gait-value gate、export parity與FAST/LIO完整五方向live matrix，但尚未取代正式policy。
  Publication data contract、B/D artifacts與release skeleton已在`9b015c7`建立；目前仍未執行
  disjoint pilot或publication formal runs，也未授權promotion。
- 目前尚未提交的project-owned變更是第二批publication execution實作：A/B/C/D live matrix、
  雙backend replay、offline usability／false-stop、trajectory、map consistency、run-level
  statistics、cluster bootstrap、sample-size pilot，以及estimator15 foot-contact replay gate；
  同批亦包含對應config、tests、validation evidence、五方向recovery regression evidence與
  byte-identical curated model48 checkpoint。完成驗證後需另取得使用者明確同意才可commit／push。
  root `build/`、`install/`、`log/`
  與`lidar_type`是未追蹤runtime產物，不納入提交，其他 `logs/`／`outputs/` 實驗
  產物也不提交。
- 更早的核心修正仍位於歷史 commit，包括：
  `e34f023 修正：改用增量診斷並穩定視窗效能`、
  `a76981c 修正：更正 IMU 座標並加入狀態回放診斷`，以及
  `c53f7d1 修正：改善高速 LIO-SAM 去畸變與點雲傳輸`。
- 根目錄 `AGENTS.md` 是 `.gitignore` 中的本機工作規則，不可 stage 或
  push。本文件是可由 repository 共享的跨對話補充知識。
- Nested upstream LIO-SAM 位於
  `deployment/ros2_ws/src/lio_sam`，目前是 detached HEAD。
- Nested LIO-SAM 的 `config/rviz2.rviz` 有對話開始前就存在的 dirty；
  不可還原、stage 或提交；目前檔案SHA為
  `0c3a25d41df63d9fa711af98845f01d99856f27b3dc8ea05fc11fcddee82adc1`。
- Nested FAST-LIO2 checkout 的 `FAST_LIO/src/laserMapping.cpp` 是既有 downstream
  visualization-only map publisher patch，`FAST_LIO/Log/*.txt` 是 runtime
  輸出；不可把 nested checkout 的 dirty stage 到本輪 root commit。

## Non-negotiable boundaries

- 不修改 `/home/ros/IsaacLab`，除非使用者明確要求；可唯讀檢查。
- 不修改 `/home/ros/Documents/anymal_project/anymal_ws`。
- 不修改 upstream LIO-SAM。
- Isaac Lab training 與 ROS 2 deployment 分離。
- Isaac Sim／Isaac Lab Python 不 import `rclpy`。
- 不以限速、提高摩擦或硬切 nominal joint pose 掩蓋 policy 問題。
- 每次 commit／push 前重新整理內容並取得使用者明確同意。
- Commit message 使用繁體中文。

## Future SLAM comparison and confidence-conditioned PPO plan

- 使用者目前先比較 LIO-SAM 與 FAST-LIO2，並把標準化的 SLAM
  tracking confidence 輸入 locomotion PPO，使 policy 在特徵不足或 tracking
  退化時學會降低速度、減少機身晃動。FAST-LIO2 的第一階段 adapter／replay
  baseline 已在 `exp/slam-fastlio2` 建立；confidence contract、兩backend
  extractor、strict group-split calibration與fresh holdout gate均已完成並安裝
  artifacts；51-D PPO observation與ROS deployment consumer已接通。第一次training已失敗；
  第二輪bounded safe-command候選已通過模擬confidence-state gate；兩backend live screening
  的FAST arm通過、LIO arm失敗。後續以獨立proprioceptive velocity estimator關閉LIO
  policy-state問題，並完成C-v10 structured-gait PPO候選、export parity及兩backend最難
  右移live cell，人工LIO操作與兩backend完整五方向live matrix也已完成。使用者目前要求先
  完善工程實作，再依凍結publication protocol補齊replay、readiness gates與
  safety-efficiency trade-off量化；不繼續外部限速器或盲目reward sweep。
- 不為每個「SLAM 方法 × PPO 版本」建立永久 branch。先在共用介面 branch
  定義 backend selector、共同輸出與 benchmark，再合併穩定的中性基礎回
  `main`。每個侵入性較大的 SLAM 實作可暫時使用獨立實驗 branch，例如
  `exp/slam-<method>`；只保留比較結果的失敗候選不必整條合併。
- 所有 SLAM backend 應透過相同 contract 輸出 canonical odom（正式 locomotion
  topic 仍由 bringup 決定，實驗 backend 使用 `/slam/odom`，不可覆蓋 GT
  `/odom`）、標準化的
  `/slam_confidence`、`/slam_tracking_valid` 與 confidence age／timestamp
  狀態，讓 PPO branch 不依賴 LIO-SAM 或其他特定方法。預期以 launch/config
  selector 在同一個 repository 內切換 backend，而不是靠切 branch 才能比較。
- PPO confidence training保存在使用者指定的完整confidence開發線
  `exp/slam-fastlio2`，先用可重播／可控制的 simulated
  confidence 驗證 observation 與行為，再與各 SLAM backend 做相同資料集的
  matrix comparison。不同 SLAM 的 raw ICP fitness／residual 不可直接當成
  可比較的 confidence；需要先定義 `[0, 1]` semantics、validity、age 與
  calibration。低 confidence 的 speed cap／stop safety fallback 仍應有
  deterministic supervisor，不可只依賴 PPO 自己學會保護。
- 最終只把通過 gate、需要長期維護的 SLAM adapter、confidence contract
  與 PPO training/config 合併回 `main`；正式 Recovery v0.4.0 baseline 在
  比較期間保持可重現，不切換 Recovery v0.5。

### Backend-neutral confidence contract and backend instrumentation (2026-08-10)

- `docs/slam_confidence_contract.md`、`configs/slam_confidence_contract.yaml` 與
  `anymal_locomotion_interfaces/msg/SlamConfidence.msg` 已形成 schema v1
  contract；v1 wire layout已凍結，calibration與production thresholds未凍結。
  唯一權威 topic 是 atomic `/slam_confidence`，Bool
  `/slam_tracking_valid` 只作便利 mirror且consumer必須另有receipt watchdog。
  目前輸出不能標成production confidence。
- 共同 score 定義為 `P(U(t,t+0.5s) | deployment diagnostics through t)`；
  `source_stamp` 精確等於被評估的 canonical `/slam/odom.header.stamp`，另有
  `source_stamp_valid` 避免把合法ROS time zero當sentinel。`slam_tracking_valid`
  是優先於score的hard gate；stale `0.95`仍必須`LOST/invalid`並帶reason。
- ROS logical age/hysteresis、consumer steady-clock liveness與physical callback
  latency分開。LIO-SAM使用scan-start stamp且有pipeline buffer，現有replay age
  可約`0.34 s`；FAST-LIO2使用scan-end stamp，不能共用未校準的單一freshness
  threshold。YAML中的FAST `0.30 s`、LIO `0.50 s`只是測試起點並屬calibration
  provenance，不是production門檻。
- FAST-LIO2第一版可觀測signal是native odom/input freshness與同stamp
  `/cloud_effected` support；fresh odom加zero effective points代表prediction-only，
  必須hard invalid。FAST odom covariance雖有填值，但message n帶`P[n-1]`且matrix
  block order是`[rotation,position]`而非ROS順序，只能作lagged calibration feature。
- LIO-SAM第一版可觀測signal是mapping odom/input/CloudInfo freshness、extracted
  corner/surface proxy、availability與incremental odom covariance[0]借放的
  degeneracy bit。Feature不足仍可能發布initial-guess pose，degeneracy也可能沿用
  舊值，兩者不可單獨hard gate。兩backend的exact residual、accepted constraints、
  convergence、conditioning與reset原因目前都需source instrumentation；本階段
  不解析`/rosout`、不修改protected upstream。
- `C(t)` 的預測與offline `U(t,t+H)` 都以20 Hz `evaluation_stamp=t`為anchor，
  source stamp只作latest canonical pose alignment。這避免LIO-SAM約0.34 s的
  pipeline delay讓label視窗部分落在evaluation之前而造成leakage。每個新source
  bundle在其後第一個logical tick以當時age計分，同stamp後續tick sample-and-hold；
  message age、freshness與state仍逐tick更新，score先量化為wire float32。
- Offline `U`只用GT產生逐timestamp initial-SE(2)-aligned label；資料按capture
  group切分。Gate分開檢查hard-fault detection、healthy availability、gradual
  event recall/lead time/AUROC、Brier/ECE/reliability、replay determinism與
  instrumentation parity/overhead。
- 已加入ROS-independent `slam_confidence_core.py`：integer-ns time、hard/soft
  reason severity、float32 thresholds、stale-high、parallel degrade/invalidate
  dwell、recovery latch、duplicate/conflict/regression/future/clock-reset與
  backend/calibration instance reset都有deterministic tests。兩個timer tick間的
  zero-support、numeric或future-stamp hard event也會至少被發布一次，不能被下一
  筆healthy source洗掉。
- 已加入FAST `fastlio_confidence_core.py`／`fastlio_confidence_node.py`。完整source
  必須用exact integer stamp join native `/Odometry`、canonical `/slam/odom`與
  `/cloud_effected`；cross-topic arrival order不影響結果。Join grace從第一個
  required callback的ROS logical arrival起算50 ms，而非從較舊sensor stamp起算；
  pending bundles上限64。`/fastlio/points`與`/imu/data`只更新input freshness，
  不推進source。
- 兩 backend node在沒有合法artifact時仍固定`calibration_id=uncalibrated`、score 0
  並帶`UNCALIBRATED`而fail closed；目前兩者已有通過holdout的native artifact，
  loader會驗證backend/schema/transform/guard/fingerprint。Wrappers仍以
  `enable_confidence:=false`預設關閉；正式model1450 policy不consume confidence。
- 已加入LIO `liosam_odom_adapter.py`、`liosam_confidence_core.py`與
  `liosam_confidence_node.py`。完整source exact-join native mapping odom、canonical
  body odom、incremental odom及feature `CloudInfo`；project motion deskew arm再要求
  同stamp status，native arm不要求。Corner `<=10`或surface `<=100`只代表可明確
  確認的insufficient support並hard invalid；positive count、degeneracy與
  `odom_available=false`只作backend-local soft feature。Offline
  `slam_confidence_label_core.py`也已實作evaluation-anchored horizon labels，GT仍只
  存在offline。
- FAST instrumentation的第一個真實replay smoke已用既有
  `smoke_out_and_back` bag完成。Confidence-on時實際收到source-valid message：
  source `11.509902085 s`、evaluation `11.669999739 s`、age `0.160097654 s`、
  score 0、valid false、state LOST、reason只有`UNCALIBRATED`，表示exact join
  在該健康區間沒有誤報missing/stale。On/off evaluator都為182 mapping samples、
  ATE `0.05350850477000207 m`、yaw RMSE `0.19210894896032746 deg`；兩份JSON
  SHA-256同為`efbac866e183de08f95bc98ec8d717c1359a81b88c17b4c6fa03636089e02add`，
  byte-for-byte相同。Artifacts在`logs/slam_confidence/`，不提交。這只完成單bag
  instrumentation parity smoke，不是不同rate determinism、fault injection或
  calibration gate。
- LIO instrumentation已用同一bag驗證project/native deskew兩條路徑：兩次
  confidence-on皆收到371筆snapshot、363筆source-valid，backend/calibration為
  `liosam/uncalibrated`、valid 0，source/evaluation stamp violation為0。Project
  arm的五個required streams各182筆且exact-match 182；native arm的四個required
  streams各182筆。Project arm on/off mapping都182筆，ATE為`0.0368127/0.0366414 m`、
  yaw RMSE為`0.0984778/0.0981068 deg`，未見instrumentation退化。第一次replay也
  找到並修正project motion deskew共用header造成feature frame_id被status覆寫的
  alias bug；修正使用deep-copy，沒有修改nested LIO-SAM。Artifacts在
  `logs/slam_confidence/`，不提交。
- Confidence正式capture現在固定native deskew；`lio_sam.launch.py`、bringup、
  comparison與replay calibration預設均為native，project deskew只保留顯式相容
  arm且不能進calibration。Evaluator新增native-only labelled sidecar輸出，GT
  `/odom`只在offline label階段使用，runtime extractor仍不訂閱GT。
- 2026-08-10/11 已完成native-only gradual-v2 calibration matrix。正式fault profile
  在每個未修改source bag內依scan timestamp做`3.0 s healthy -> 1.5 s ramp-down ->
  4.5 s low-support hold -> 3.0 s recovery`，最低保留0.1%且保留sensor order，因而
  同時逐步降低density、rings與FOV；沒有project deskew、沒有frame-level split，GT
  `/odom`只在replay結束後產生0.5 s offline label。Runtime feature改成backend-local
  support、causal peak ratio/trend/0.2 s projection、support-motion exposure、odom motion
  與LIO public degeneracy/availability，沒有future sample或GT。
- FAST-LIO2第一個20-group holdout因`motion_forward_0_5`的healthy low-support control
  暴露25.73% false-low而被誠實retire；沒有以該holdout宣稱pass。加入四個完全未看過
  的model1450 run01 source bags作新final holdout後，24個independent gradual events
  全部有雙class coverage。新holdout為1,198 frames、failure prevalence 72.95%、
  AUROC/Brier/ECE `0.97362/0.03662/0.07830`、event recall 100%、median lead
  `0.199999996 s`（1 us timestamp tolerance對應0.20 s gate）、healthy false-low
  4.938%；support hard-gate advance recall 100%、median lead 0.50 s。全部gate通過，
  artifact為`config/slam_confidence_fastlio2_native_v2.json`，目前calibration ID
  `native-v1-1e6cf8347be1`。
- LIO-SAM共有四輪retired final evidence：run01 recall 75%/lead 0.10 s；run02 recall
  25%；run03 recall 100%但false-low 8%；fresh4 recall 50%。它們只在retire後進
  development。Frozen causal-v3 transform/guard通過development gate後，另capture
  四個內容SHA全新yaw groups作fresh5 final；2,132 frames、5 events、prevalence
  85.084%、AUROC/Brier/ECE `0.986497/0.021720/0.034500`、recall 100%、lead
  `0.199999996 s`、false-low 3.145%、support advance recall 100%/lead 0.30 s，全部
  gate通過。artifact為`config/slam_confidence_liosam_native_v3.json`，ID
  `native-v1-edc098b0bd98`。
- FAST與LIO artifact皆已用實際localhost native gradual-v2 replay驗證：FAST 332筆、
  6個distinct float32 score、72筆valid；LIO 552筆、11個distinct score、243筆valid；
  兩者ID正確、沒有`UNCALIBRATED`或timestamp violation、runtime extractor不訂閱GT。
  FAST off/on的166 scans、162 mapping samples與trajectory metrics逐欄完全相同。
  LIO off/on輸入scan、mapping sample與GT path完全相同；估計trajectory受upstream
  multi-thread scheduling影響不具bitwise determinism，但artifact-on沒有quality
  regression，confidence graph也沒有回饋到SLAM。
- 2026-08-11 review修正runtime/offline計分時序：新source只在其後第一個20 Hz
  evaluation tick計分，同stamp後續sample-and-hold；hard bundle更新causal transform
  但不覆寫wire score。兩份既有estimator權重沒有重訓。Artifact provenance升為
  schema v2並由loader驗證backend revision、config/split hash、sensor/domain/
  extrinsic、timestamp/input adapter與score timing；calibration ID不變。凍結artifact
  的2,000次capture-cluster bootstrap 95% CI保存在
  `docs/validation/slam_confidence_final_holdout_summary.json`。
- PPO observation介面已固定並測試：`slam_confidence`、
  `slam_tracking_valid`、`clip(confidence_age/0.50,0,1)`，再加consumer-local 0.15 s
  steady receipt watchdog；receipt失效時confidence/valid均歸零。兩backend已達
  PPO-ready資料介面條件；51-D training branch與候選runtime consumer已完成，但正式
  Recovery v0.4.0仍為48-D且不訂閱confidence。

### Yaw-stress implementation contract (in progress)

- 使用者已確認 yaw-stress 直接留在 `exp/slam-fastlio2`；目前不 merge 回
  `main`，也不提前建立 PPO branch。
- 只比較 LIO-SAM／FAST-LIO2 native deskew；不加入 project-based deskew arm。
- 固定 Factory pose，左右原地旋轉，yaw-rate grid 為
  `0.25/0.5/1.0/1.5/2.0 rad/s`；timeline 為 5 s warmup、2 s ramp-up、8 s
  hold、2 s ramp-down、10 s recovery。
- Pilot 為 10 bags／20 full-density replays；正式為三個 paired seeds、30 bags、
  四個 uniform point densities、兩 backend，共 240 replays。Pilot review 前不跑
  formal matrix。
- `uniform_point_density` 只代表 deterministic evenly spaced thinning；不可泛稱
  一般 LiDAR degradation。固定單一起點資料不可直接決定 production backend 或
  confidence calibration。
- FAST-LIO2 `map_en=false` 不影響 estimator，但 full-map export 尚未
  qualification，需另設 gate。
- 實作與 smoke 可在 dirty tree 進行；pilot/formal 必須來自乾淨、可識別的 Git
  baseline。任何 commit/push 仍須當次重新取得使用者明確同意。
- 2026-08-04 非 qualification 單-cell smoke 已驗證
  `yaw_stress_left_0_5`／seed 42：27 s capture contract 通過、model1450 motion
  driver `passed`、276 raw scans、bag 約 106 MB。相同 bag 的 LIO-SAM 與
  FAST-LIO2 native full-density replay 都產生完整 yaw-stress report 並通過
  infrastructure/tracking-ready gate。這不是 10-cell pilot 結論。
- 第一次 sandbox capture 因 Iceoryx 無權建立 `/tmp/roudi` Unix socket 失敗，
  沒有產生 bag；改在正常 host 權限後通過。失敗 artifact 位於 ignored
  `outputs/slam_yaw_stress/capture_smoke_20260804`，不可混入正式資料。
- Native effective-support instrumentation 已加入並以同一份 smoke bag 驗證：
  LIO-SAM 記錄 extracted corner+surface feature count；FAST-LIO2 透過可重建、
  預設關閉的 downstream `publish.effect_en` 記錄 point-to-plane selected
  effective points。兩者只作 backend-specific calibration feature，不當成共同
  confidence 比例。FAST-LIO2 rebuild 與兩 backend replay regression 通過。
- FAST-LIO2 `publish.effect_en` off/on A/B 的全程 trajectory 與五 phase
  evaluation JSON 逐值相同；on 取得 273 筆 samples，off 為 0。這只證明該
  deterministic replay 的 estimator parity，publisher 的 physical-clock overhead
  仍屬 throughput gate。
- Blind review 已有 deterministic blind/reveal manifest generator 與固定視角
  MP4 renderer。相同單-cell smoke 的兩 backend 各產生 720×720、10 fps、273
  幀（27.3 s）影片；首／中／末幀非空白檢查通過。Blind manifest 不含 backend
  對照，reveal manifest 分開保存。這次 single-cell renderer smoke 本身不構成
  10-cell pilot 或裂圖盲測結論；後續完整 pilot 結果另記於下一項。
- 2026-08-04 已從 clean commit `649fcda` 完成 seed 42 yaw-stress pilot capture
  與 full-density replay：10/10 source bags 通過 source/motion gate，每包 276
  raw scans；20/20 backend replay reports 通過。20 支匿名影片皆為 720×720、
  10 fps、273 幀（27.3 s），且通過可開啟、時長與非空白抽幀 contract。人工
  blind annotation 於揭盲前完成：20/20 均無可見裂圖或形變，severity 0、無
  onset、recovery 不適用。這表示本 pilot 未重現使用者原先看到的快速旋轉裂圖。
- 揭盲後的軌跡數值並非完全等價：LIO-SAM／FAST-LIO2 全程 translation ATE
  RMSE 中位數約為 0.0234／0.1214 m，full yaw RMSE 中位數約為
  0.4009／0.4309 deg；hold translation ATE 中位數約為 0.0290／0.0466 m。
  固定俯視累積點雲影片適合人工辨識裂圖，但對整體 translation drift 不敏感，
  不可因影片看起來相同就宣稱兩 backend 軌跡等價。
- 2026-08-06 使用者以 live `/cmd_vel` 持續 `wz=2.0` 旋轉約一至數圈後觀察到
  FAST-LIO2 明顯裂圖；先前 replay pilot 沒有重現裂圖。後續用同一個 live
  profile 做 point-order A/B，已重現為虛擬 LiDAR input packing／FAST-LIO2
  scan-end inference 的問題候選；仍不能把這個結果解讀成 estimator 已完成
  calibration。
- 2026-08-06 已將 native／replay／live calibration launch 的 estimator
  overrides 接通，但尚未改正式 YAML。既有 `smoke_out_and_back` 的小型 sweep
  顯示目前候選 `stride=2, iteration=4, voxel=0.5` 為
  `0.094370 m / 0.224615°`；`iteration=3` 為 `0.136902 m / 0.309133°`，
  `stride=1` 為 `0.113285 m / 0.180294°`，兩者都未過目前 ATE gate。
  `filter_size_surf=filter_size_map=0.3` 得到 `0.060280 m / 0.097934°`，
  `0.8/0.8` 則失敗（ATE `0.509859 m`）。
- `0.3/0.3` 只視為待 live 驗證的 candidate：在既有 `yaw_stress_left_2.0`
  holdout replay 為 `0.073535 m / 0.881365°`，右轉 `2.0` holdout 為
  `0.065025 m / 0.833303°`，都通過 infrastructure／tracking gate，但這些
  是 replay 數字，不能宣稱已消除使用者 live 裂圖，也不能直接寫回正式 config。
- `slam_backend_compare.launch.py` 現在不再使用共用 comparison RViz：selector
  會載入 backend 官方設定。LIO-SAM 使用 `lio_sam/config/rviz2.rviz`、fixed
  frame `map`；FAST-LIO2 使用 `fast_lio/rviz_cfg/fastlio.rviz`、fixed frame
  `camera_init`。兩者不再同時顯示 registered cloud；官方視窗 geometry 與
  view 保持原樣。
- FAST-LIO2 native launch 現在提供 project-owned calibration overrides：
  `fastlio_blind`、`fastlio_point_filter_num`、`fastlio_max_iteration`、
  `fastlio_filter_size_surf`、`fastlio_filter_size_map` 與
  `fastlio_cube_side_length`；目前預設為 `0.5/2/4/0.3/0.6/1000`。這些只覆寫
  estimator parameter，不加入 project-based deskew；`scan_line`、timestamp
  unit、scan rate、ring／per-point time 與 LiDAR-IMU 外參仍固定為 sensor
  contract。`point_filter_num=2` 目前只是 calibration 起點，不是完成 holdout
  後的正式 baseline。
- 2026-08-06 已完成 broad FAST-LIO2 native replay calibration，涵蓋平滑
  `vx=0.5/3.0`、`vy=1.5`、`vx=1.5,wz=1.0`、`vx=3.0,wz=0.5`、左右
  `wz=2.0`，以及瞬間切入 `vx=2.3,wz=2.0` 的急加速／急轉案例。`warehouse_final_turn`
  source 的最高線速度 2.372 m/s、yaw rate 2.176 rad/s，含 29 個 high-yaw
  scans；所有 FAST replay 仍為 native deskew，沒有 project-based deskew。
- 固定 `surf=0.3` 的 map sweep 顯示真實取捨：`map=0.3` 對左右 yaw 與急切換
  較好但側移／混合／曲線失敗；`map=0.6` 對前進、側移、混合、曲線、loop
  通過但左右 `wz=2.0` 失敗；`map=0.4` 通過側移／混合／曲線與右轉，但左轉
  yaw 1.202°、急切換 ATE 0.1068 m 仍失敗；`map=0.58` 左右 yaw 通過但側移
  失敗。stride、iteration、blind、IMU covariance 的額外 sweep 也沒有消除
  此方向／運動型態取捨；這是後來確認使用錯誤 `cube=200` 時的歷史結果，
  不再代表目前 `cube=1000` baseline 的 gate 狀態。
- 同一批 bag 的 LIO-SAM native control 全部通過（前進 ATE 0.0596 m、側移
  0.0430 m、混合 0.0509 m、曲線 0.0576 m、左右 yaw 0.0278／0.0298 m），
  將問題定位在 FAST-LIO2 estimator／live transport，而非 evaluator 或 source
  motion contract。FAST per-point timestamp 的 `counterclockwise` A/B 在混合
  yaw 5.847°、左右 yaw 約 54°／52°，確認 `clockwise` 預設正確；反向介面只
  作診斷用途。
- 因 broad replay 尚未得到通用 candidate，`fastlio2_anymal_ouster32.yaml`
  與 `fastlio2_anymal_ouster32_live.yaml` 仍維持原始 `surf/map=0.5/0.5`，
  沒有把 `.3/.3`、`.3/.4` 或 `.3/.6` 靜默寫入正式 config。`sensor_order`
  加 `staggered` 已先成為 FAST-LIO2 live／replay entry-point 的實驗預設，
  但它是 input contract 修正，不是 estimator parameter qualification。
- 2026-08-06 進一步核對虛擬 RTX LiDAR 的 raw bag：每個水平欄位以 ring
  `0..31` 連續排列，欄位之間以 ring wrap 分隔；單包約 29.3k 點、1024 個
  欄位，header timestamp 是完整 scan end。官方
  [Ouster ROS point-cloud composition](https://github.com/ouster-lidar/ouster-ros/blob/master/src/point_cloud_compose.h)
  對 native cloud 使用 ring-major（ring outer、column inner）排列與 column
  timestamp，因此 adapter 的正式 `sensor_order` 現在先以 wrap 重建欄位時間，
  再輸出同一個 ring-major layout；`fireTimeNs` 與 azimuth 仍只是 diagnostics。
  RTX sensor 也明確設定為 `NONCOMPENSATED`，FAST-LIO2／LIO-SAM 各自執行原生
  deskew，沒有加入 project-based deskew。
- 這個排列修正已重建 ROS package 並通過 adapter/dependency tests。ring-major
  FAST-LIO2 native replay（`point_filter_num=2`、`max_iteration=4`、
  `surf=0.4/map=0.3`）在 `forward_3_0` ATE `0.062100 m`、
  `warehouse_final_turn` ATE `0.096243 m`、`combined` ATE `0.072520 m`，三者
  都過 replay gate；但這只是 calibration candidate，不能寫入正式 YAML。
  `destaggered` 保留為官方排列 A/B，FAST-LIO2 entry point 目前使用
  `staggered`，讓最後一個輸入點保有接近 scan-end 的 capture time。
  對照顯示 `.5/.3` 雖讓快速轉 ATE `0.093132 m`，前進卻為 `0.142680 m`；所以
  不能用單一旋轉包選 estimator 參數。
- ring-major LIO-SAM native control 同一批 `warehouse_final_turn`／
  `forward_3_0` 分別為 ATE `0.046419 m`／`0.060128 m`，均通過；因此目前
  adapter contract 沒有破壞 LIO-SAM reference。以 per-emitter `fireTimeNs`
  取代 column time 的 FAST A/B 在快速轉 ATE `0.100346 m`，剛好越過 gate，
  支持正式 contract 保留 column timestamp。
- point-order A/B 的 replay 與 live 結果已補齊：`sensor_order+staggered` 的
  `forward_3_0`（surf/map `.3/.6`）ATE `0.061231 m`、yaw RMSE `0.141860°`；
  同一 profile 的 `destaggered` lateral replay ATE `0.519142 m`，而
  `staggered` 為 `0.124354 m`，所以 destaggered 不是目前 FAST-LIO2 的安全
  default。live `yaw_stress_left_2_0` 的 staggered run 有 1,336 筆 policy
  odometry、0 termination、0 truncation、classification `no_instability`；
  完全相同 profile 改成 destaggered 後出現重複 policy reset、FAST-LIO2
  `No Effective Points!`、10 次 termination，最後 joint-command freshness
  timeout。這是目前最強的虛擬 LiDAR contract 證據；該 staggered run 另有
  simulation IMU parity `0.021747 > 0.01` 的獨立 validation failure，不能誤記
  成 FAST-LIO2 crash。
- 2026-08-07 已在 commit `7b9994d` 固定上述虛擬 LiDAR contract，並以同一
  `sensor_order + staggered`、native deskew 做新的 estimator sweep；測試產物在
  `logs/fastlio2_tuning/calibration_20260807`。`surf=.3/map=.5` 的
  `forward_3_0` ATE `0.055355 m` 通過，但 `combined` ATE `0.132723 m` 失敗，
  證明不能用單一平移包選參數。
- 在 `surf=.3` 的 combined sweep 中，`map=.52` ATE `0.168713 m`、`.58`
  ATE `0.285471 m` 失敗；`.55` ATE `0.086879 m` 與 `.60` ATE `0.086487 m`
  通過，顯示 voxel leaf 對場景幾何有離散效應，不是單調的「越大越穩」。
  `.3/.55` 的 curve 通過（ATE `0.086864 m`），但 lateral/warehouse 分別為
  `0.149239/0.161322 m`，仍失敗。
- 固定 `map=.6` 的 surface sweep 沒有消除 holdout 取捨：
  `surf=.35` 讓 lateral ATE `0.119846 m`，但 warehouse ATE `0.138215 m`；
  `point_filter_num=1` 與 `max_iteration=5` 也沒有改善兩個 holdout。補測
  `map=.4/.45` 時 warehouse 最佳約 `0.102514 m`，但 lateral 仍為
  `0.143658 m`。這只證明固定 `cube=200` 的該輪沒有可鎖定 candidate；後續
  cube audit 已取代這個暫時結論。
- `surf=.3/map=.6`、stride 2、iteration 4 的 live
  `yaw_stress_left_2_0`（IMU bridge parity atol `0.03` 僅用於隔離 bridge gate）
  完成 27.1 s、1,334 筆 policy odometry、0 termination/truncation，classification
  `no_instability`；這是 cube audit 前的 live-safe calibration evidence，不等於
  新的 `cube=1000` 預設已完成 live qualification。
- 2026-08-07 將 FAST-LIO2 原生 `common.time_sync_en` 與
  `common.time_offset_lidar_to_imu` 暴露到 project-owned native/replay/live
  launch；預設仍為 `false/0.0`，不改 sensor contract，也不加入 project
  deskew。固定 `.3/.6`、`sensor_order+staggered` 的 offset holdout 中，
  `lateral_1_5` 在 `-10/-5/+5/+10 ms` 分別為 `0.1544/0.1484/0.1299/0.1285 m`，
  `warehouse_final_turn` 為 `0.1030/0.1068/0.1470/0.1907 m`；沒有通用 offset，
  且 offset `-10 ms` 仍未將 warehouse 穩定壓到 `0.1 m` gate。
- 重新驗證舊的 azimuth time 假設後，`time_source=azimuth` 在 forward/lateral/
  combined/curve 分別得到 `0.0893/0.0946/0.0897/0.0922 m`，但
  `warehouse_final_turn` 惡化至 `0.1803 m`；因此 azimuth 只保留 diagnostic，
  正式實驗仍用 `sensor_order`。在 `sensor_order` 下對 `map=.46-.50`、
  `surf=.25-.35` 做 holdout fine sweep；在兩個難場景局部最佳為
  `surf=.3,map=.49` 的 lateral `0.1027 m`／warehouse `0.0798 m`，以及
  `surf=.25,map=.49` 的 lateral `0.0874 m`／warehouse `0.1034 m`，但完整驗證
  `surf=.3,map=.49` 的 forward/combined/curve 又分別失敗於
  `0.1469/0.1924/0.1914 m`。`.33-.345` 也沒有找到兩者同時通過的組合，
  `.345` warehouse replay 還觸發 180 s timeout。這是固定 `cube=200` 時的
  voxel 結論；後續 `cube=1000` 已使五個 translation ATE gate 全數通過。
- `surf=.3/map=.6` diagnostics 沒有發現 transport 掉包：raw/adapted scan gap
  median 約 `0.1 s`、odometry receipt age median 約 `40 ms`。FAST selected
  effective point 數 lateral 為 p10 `3602`、min `1434`，warehouse 為 p10
  `1831`、min `889`；後者退化較明顯，支持把 effective support 納入未來
  confidence，但這些 backend-specific counts 尚不能直接當成跨 backend 比例。

- 2026-08-07 完整 parameter audit 發現先前漏測官方 Ouster launch 的
  `cube_side_length=1000`；專案原本 `cube=200`、`det_range=100` 會讓 estimator
  從初始化開始就反覆進入 local-map 搬移條件。固定 corrected contract、native
  deskew、stride 2、iteration 4、`surf=.3/map=.6`，只改 cube 為 1000 後，
  forward/lateral/combined/curve/warehouse ATE 分別為
  `0.065137/0.099775/0.041051/0.023490/0.045948 m`，五包全數通過；project-owned
  FAST launch 預設已統一為 `surf=.3/map=.6/cube=1000`；2026-08-11後續正式
  live左右長時間急轉與雙向lateral qualification皆已通過。
- 2026-08-10 使用者以 live backend comparison 入口選擇FAST-LIO2，人工觀察
  目前cube=1000預設點雲，回報「沒什麼裂圖、調整看起來不錯」，並同意繼續
  confidence工作。這是user-observed visual smoke，沒有保存本次量化log、左右
  多圈`wz=2.0`或lateral metrics；只能支持繼續instrumentation，不能取代正式
  live qualification gate。
- 官方 Ouster YAML 的 `blind=2.0` 在新 cube baseline 上補測後，lateral/
  warehouse ATE 為 `0.099982/0.045983 m`，和專案 `blind=.5` 的
  `0.099775/0.045948 m` 幾乎相同，因此保留 `.5`。官方 `det_range=200` 在此
  fork 只參與 local-map 搬移、不裁切 scan；對 cube=1000、約 10 m 的 replay
  不會改變 estimator。其餘 scan line 32、stride 2、iteration 4 與 LiDAR–IMU
  extrinsic 都是依虛擬硬體與既有 sweep 有意選定，不是遺漏的官方 default。
- 同批 bag header stamp 實測 IMU `200.000 Hz`、LiDAR `10.000 Hz`，不是 sensor
  頻率不足；lateral/warehouse 每個 raw scan 都可重建完整 1024 column 與
  `0..99.902344 ms` point-time span。`/lidar/points_raw` 實際只有 `x/y/z`，不是
  官方 Ouster driver output；adapter 重建 Ouster-like 欄位與 column time，且為
  配合 candidate 的 scan-end inference 使用 `staggered` packing。因此仍缺真實
  UDP packet、sensor/PTP clock、packet loss、measurement-id 與硬體回波特性，
  不可把目前 topic 描述成「完全等同正式 Ouster driver」。

## Formal policy

正式 deployment policy 仍是 Recovery v0.4.0 model1450，Recovery v0.5
尚未發布：

- Checkpoint：
  `checkpoints/anymal_d_locomotion_v1/recovery_v0.4.0/model_1450.pt`
- ONNX：
  `exported/anymal_d_locomotion_v1/recovery_v0.4.0/policy.onnx`
- Bringup source 與 install launch 預設都指向上述 ONNX。
- Checkpoint／TorchScript／ONNX parity 已通過；ONNX 最大絕對誤差
  `2.861e-6`。
- Release 文件記錄的正式 v0.4 qualification：
  - 低速長圓周後 zero-command recovery 通過。
  - 純平移 3 m/s regression 通過。
  - Turning matrix 36/36。
  - True loop-closure matrix 12/12。

2026-08-03 以目前 corrected motion-deskew pipeline、正式 v0.4.0 ONNX 與
同一個 formal metadata 重新執行 LIO loop-closure matrix，初始結果為 11/12。
調查確認 `loop_open_backward/run_01` 在 `minimum_time_difference=8 s` 時，
因初始 keyframe 與尚未走遠的 current keyframe 距離仍小於 1.5 m，於
`8.509999807 s` 產生可重現的 false constraint。將 project-owned matrix
參數提高至 `minimum_time_difference_s=10.0` 後，保留 1.5 m radius；正式
v0.4.0 replay 結果為 12/12，詳細輸出在
`outputs/lio_sam_loop_closure/qualification_v1_formal_v040_20260803_time10`。
此修正未修改 upstream LIO-SAM。

詳情見 `docs/policy_release_v0.4.0.md`。

## Warehouse fall diagnosis

正式使用者跌倒 trace：

`logs/formal_bringup/warehouse_fall_01/locomotion_diagnostics.json`

已確認：

- Factory friction 為 1.0，地面是同一個 GroundPlane。
- Classification 為 `foot_slip_first`。
- First foot slip：8.10 s。
- First body instability：30.38 s。
- First hard failure：30.60 s。
- Peak raw command：`vx=2.2974865 m/s, vy=0, wz=2.0 rad/s`。
- 29.00 s 開始最後高速左轉；29.46 s 在 `(-0.328, 12.991)` 仍穩定，
  30.38 s 才出現 body instability。
- PaintLine 38–43 沒有 `PhysicsCollisionAPI`；其 world z 約
  `0.00543 m`，不是物理凸起。
- 同位置 `(-0.413, 12.604, yaw=1.065)` 直接施加最後 2.96 s
  `vx=2.3, wz=2.0`，結果為 0 termination、`no_instability`。
- 完整 command history 在 Factory 外 `(-50,-50)` 的 GroundPlane replay
  也是 0 termination、`no_instability`。因此不是 policy 隨時間自行累積
  失穩，也不是 PaintLine 位置或最後 combined command 單獨造成。
- 原始 refinery mesh 有兩個三角形組成
  `x=±15.123、y=±11.775、z=0` 的大地板，與無限 GroundPlane 共面。
  使用者提到的 `x=3` 與 `x=-2` 都會跨過其 `y=11.775` 邊界。
- 舊 Formal trace 在 27–29 s 持續收到 `vx=2.3`，卻停在
  `(-0.4,12.6)`；當時與後腳跨在 refinery 共面邊界的幾何位置相符。
- 關掉 refinery collision 的舊 A/B 曾顯示 robot 不再停在 y=12.6；這足以
  證明共面大地板是需要清理的場景缺陷，但後續修正版仍在相近區域跌倒，
  所以不可再把它記為整個跌倒問題的必要或充分原因。
- 修正資產只移除該兩個共面 faces，保留其餘 refinery collision。
- 目前不應以增加 training iteration 取代閉迴路診斷；下一個必要 gate 是
  重現相同路徑與 gait phase，或以 formal joint/action state 做
  state-transplant replay。
- Simulator episode reset 後 external policy `previous_action` 未同步
  reset，是使用者看到 reset 後「腳打結」的另一個問題。

## Refinery fix follow-up

修正版 Factory 上的使用者 trace：

`logs/formal_bringup/refinery_fix_01`

已確認：

- First body instability 為 28.48 s，約在 world
  `(10.83, 13.54)`；29.12 s termination。當時 effective command 仍是
  `vx≈1.899, wz=0`，不是直接撞擊後立即 reset。
- Policy watchdog 的 effective command 與 simulator raw command 必須
  分開解讀；raw trace 會保留最後收到的命令，看不出 0.5 s watchdog
  已把 policy command 歸零。
- 使用者路徑包含兩次長 `/cmd_vel` refresh gap，分別造成約
  0.50 s 與 1.38 s 的 effective zero；之後是原地轉向與重新高速前進。
- 近似 effective-command replay 在 Factory 外 `(-70,-70)` 也能於
  36.94 s 進入 body instability、37.76 s 翻倒。失穩發生在高速命令被
  watchdog 瞬間切零後約 6 s，證明 v0.4 policy 存在「高速歷史後 abrupt
  stop」的閉迴路弱點；但該 replay 的最後 stop 比使用者實際操作更激烈，
  不能單獨解釋 28.48 s 的 forward fall。
- 實驗性 watchdog deceleration（linear/angular 均為 `2.0`）在同一近似
  replay 下通過：0 termination、`no_instability`、最低 height
  `0.533 m`、最大 roll `0.099 rad`。這目前只是診斷候選，尚未設成正式
  bringup 預設。
- 更接近原始 received-command history、保留兩次 refresh gap 的 replay，
  在 GroundPlane 與修正版 Factory 都通過。Factory run 在 29 s 到
  `(13.15,14.23)`；相對使用者 trace 已因 yaw 累積偏差約 3 m，尚未重現
  完全相同的足端接觸序列。
- 修正版 Factory 的局部直線穿越：
  - `(3,8)` 朝 `+y`、1.5 m/s 穿過 `(3,13)`：`no_instability`，最低
    height `0.547 m`。
  - `(-2,8)` 朝 `+y`、1.5 m/s 穿過 `(-2,13)`：`no_instability`，最低
    height `0.538 m`。
- 因此 PaintLine 38–43、`(3,13)` 或 `(-2,13)` 的位置／collision
  本身不是充分條件。尚待分辨的是高速轉向歷史、精確 gait phase、命令
  refresh discontinuity 與局部接觸的組合。

## t=25 state-transplant result

已從 `refinery_fix_01` 自動建立 t=25.00 s transplant manifest：

`logs/formal_bringup/refinery_fix_01/state_transplant_t25.json`

關鍵結論：

- 瞬時 transplant 可以恢復 base、canonical joints 與 policy
  `previous_action`，pre-rollout 48-D 誤差為 `1.45941973e-4`；但第一個
  physics step 後會因 actuator recurrent state 與 PhysX contact cache
  遺失而分岔。早期的 Factory/GroundPlane `no_instability` 結果
  `factory_baseline_06` / `groundplane_baseline_01` 只能證明瞬時 pose
  不是充分條件，不能當 exact history A/B。
- 單獨補存並恢復 `ActuatorNetLSTM` hidden/cell 仍不足；runtime regression
  的 joint velocity 仍分岔，證明接觸 solver history 也重要。
- 最終改用 formal trace 的 1250 個 `applied_raw_action`，從原始 spawn
  重播 0→25 s；另補兩個 bridge bootstrap actions，並依 formal metadata
  建立、逐步 render RTX LiDAR。這會自然重建 actuator 與 PhysX history。
- Exact action-history replay 第一筆 external policy observation：
  base linear/angular velocity、projected gravity、joint position/velocity、
  previous action全部逐值 0 誤差；command 只有 `2.384e-7` 浮點誤差。
- 依 formal command changes（yaw→forward，最後明確 zero packet）重播的
  Factory baseline：
  `logs/state_transplant_t25/action_replay_factory_received_exact_01`
  - `foot_slip_first`；
  - first foot slip 0.46 s；
  - first body instability 4.32 s；
  - first hard failure 8.12 s；
  - 1 termination，最低 height 0.1638 m。
- Watchdog deceleration 2.0 A/B：
  `logs/state_transplant_t25/action_replay_factory_received_decel2_01`
  - 結果與 baseline 完全相同；
  - formal 在 29.62 s 明確送入 zero packet，effective command 立即歸零，
    watchdog timeout deceleration 不會介入；
  - 因此 deceleration 不是此 failure 的解法，不可設成正式預設。
- 將相同 open-loop action history 平移到 GroundPlane 時，在到達 t=25 前
  已有 5 次 termination，無法形成有效 post-t25 A/B。這表示 t<25 的
  Factory 接觸／閉迴路歷史不可忽略；不能用瞬時 GroundPlane transplant
  排除 history/contact 組合。
- Exact replay 的 body-instability timing 仍比使用者 formal trace晚約
  0.84 s，但前幾個 policy steps 可做到 observation 約 `1e-6`、action 約
  `1e-7`，之後 PhysX 浮點差逐步放大。Command change 與 final zero packet
  已對齊，這個 timing 差不再阻擋 failure classification。
- 新 diagnostics 仍會在每個 flush boundary 保存 public actuator LSTM
  checkpoint，供較短的瞬時診斷使用；exact contact regression 則優先使用
  action-history replay。

## IMU angular-velocity frame root cause

Exact replay 的 episode-reset parity 診斷最後找出真正的 deployment
根因，已不再把本次跌倒分類為尚未證實的 v0.4 recovery weakness：

- `IsaacReadIMU.outputs:angVel` 已是 IMU sensor-local frame；目前 IMU
  identity-mounted 到 `base_link`，所以 raw value 就是 policy contract
  所需的 body angular velocity。
- 舊 Action Graph 又用 base world→body matrix 旋轉一次，再發布到
  `sensor_msgs/Imu.angular_velocity`。純 yaw／小姿態時不易察覺，但 roll／
  pitch rate 變大時會嚴重錯誤。
- Worst-case 實測：
  - raw IMU：`[3.27705, -0.13917, -0.11505]`；
  - Isaac native body：`[3.27671, -0.14173, -0.12155]`；
  - 舊二次旋轉後 bridge：`[-2.63955, -1.95018, 0.03499]`。
- 修正後 `ReadImuSensor.outputs:angVel` 直接接到
  `PublishImu.inputs:angularVelocity`，`read_imu_state()` 也直接讀 raw
  `angVel`；IMU orientation 的 world pose composition 保留不變。
- 使用完全相同的 0→25 s formal actions、Factory contact history 與 RTX
  LiDAR rendering，只從 t=25 起讓 external policy 使用修正後 IMU：
  `logs/state_transplant_t25/action_replay_factory_received_imu_frame_fix_01`
  - 正常 rollout 的 48-D observation parity 全部通過；
  - angular velocity max error `0.0035273 < 0.01`；
  - `no_instability`、0 termination、最低 height `0.5413 m`、最大
    roll `0.1077 rad`。
- 修正後第一筆 external angular observation 與舊 formal manifest 相差
  `0.6591` 是預期的因果介入，不是 replay 失真；舊 manifest 保存的就是
  二次旋轉後錯值。其他第一筆 term 仍為 0，command 誤差
  `2.384e-7`。
- 這個 A/B 證明：保留相同 t<25 policy/action/contact history，只修正 IMU
  frame 即足以避免原 failure。正式 v0.4 policy 尚未更換，watchdog 預設
  也未更動。
- 使用者另從 t=0 執行 corrected bridge 的一般 formal bringup：
  `logs/formal_bringup/imu_frame_fix_manual_01`
  - LIO-SAM 啟用、1975 samples、39.48 s；
  - `no_instability`、0 termination、0 truncation；
  - 最低 height `0.5154 m`；
  - 最大 roll/pitch 分別 `0.0847/0.0955 rad`；
  - 有一次 6.04 s 的 contained foot-slip event，但沒有後續 body
    instability 或 hard failure。

## Implemented diagnostics and reset synchronization

Commit `2a7159a` 已加入：

- Bringup 的 `enable_locomotion_diagnostics` 預設仍為 `false`。
- 啟用時，diagnostics directory 同時產生：
  - `locomotion_diagnostics.json`
  - `policy_diagnostics.json`
- Policy trace 記錄 received command、watchdog 後 effective command、
  command age、watchdog timeout、observation 與 raw action。
- `/simulation/episode_reset` 與 `/simulation/episode_reset_ack` 明確
  handshake。
- Policy reset 時清除 runtime `previous_action`；simulator 等待 ACK 後才
  恢復 external control。
- 沒有使用模糊 episode-reset heuristic。

## Deterministic warehouse profile

`warehouse_mapping_stress` 現在依
`warehouse_fall_01` 的實際 command plateau 重播高速直行、左右最大 yaw、
最後高速左轉與 zero-command settle：

- 已補回舊 profile 遺漏的 2.70 s 初始等待，以及第一次左轉後 0.56 s
  zero-command。
- Profile 在 31.80 s 停止刷新 `/cmd_vel`。
- Policy 的 0.5 s command watchdog 負責產生真正 effective zero command。
- 補齊路徑後，舊 isolation spawn `(-28,-18)` 會走回 Factory 幾何，不再
  collision-free；目前完整空地對照使用 `(-50,-50, yaw=0)`。
- 該位置仍使用 Factory 的相同 GroundPlane 與 friction 1.0，但附近 LIO
  幾何特徵不足；因此 locomotion 結果與 LIO metric 必須分開判讀。

## FloorB finding

- `FloorB_01` 與 `FloorB_02` 的 transform、mesh points、face indices 與
  composed bounding box 完全相同，屬於重疊的 visual geometry。
- 兩個 FloorB prim 都沒有 `PhysicsCollisionAPI`，也沒有
  `physics:collisionEnabled`。
- 真正地面碰撞來自 `/World/GroundPlane/CollisionPlane`。
- FloorB 重複可能造成 z-fighting、RTX/LiDAR 重複表面或渲染負擔，但不會
  形成雙層 PhysX 腳部接觸，不能解釋這次跌倒。
- Factory map 尚未因這項發現而修改。

## Viewport FPS warning storm

使用者在完整 GUI bringup 靜置數分鐘後看到 viewport 反覆 FPS drop／回穩；
關閉並重開 RViz 累積點雲 display 沒有改善。2026-07-30 的同步量測確認：

- Locomotion diagnostics 全程關閉，因此不是大型 diagnostics JSON flush。
- Isaac Sim 5.1 的
  `isaacsim.core.simulation_manager.plugin` 對每個 RTX frame 連續輸出：
  - `No adjacent samples found for interpolation`
  - `getSimulationTimeMonotonicAtTime: no data found`
- 舊基準 240 s 內第一種 warning 新增 7,653 筆（32.02/s），Kit log
  增加 3.64 MiB，Isaac process 寫入增加 3.85 MiB。
- 舊基準 GPU 通常為 23–33%，CPU 約 210–216%，並非持續資源飽和；但
  warning 約 8,353 筆時曾有一秒 GPU utilization 由 29% 降至 4%，下一秒
  回到 28%，與使用者看到的 drop／回穩型態一致。
- 移除 host 額外 `sim.render()` 的 A/B 沒有消除 warning，並把 RTX
  rendering/reference-frame 推進率由約 32 Hz 降到約 21 Hz，因此已完整
  回復，不採用該方向。
- NVIDIA 新版 Isaac Sim release notes 已將這個訊息由 WARNING 降為 INFO；
  目前 5.1 runtime 只針對
  `isaacsim.core.simulation_manager.plugin` 設定 Kit log channel 為
  `error`。其他 Kit、PhysX、RTX 與 ROS warning 仍保留，該來源的 error
  也不會被隱藏。
- 修正後完整 Factory＋RTX LiDAR＋LIO-SAM＋RViz idle 300 s：
  - interpolation warning：0；
  - GPU utilization 平均 32.31%，範圍 13–38%；
  - Isaac CPU 平均 213.38%，範圍 209–215%；
  - Kit log 僅增加 1.602 MiB；
  - Isaac process write 增加 1.879 MiB；
  - VRAM 增加 152 MiB、RSS 增加 70.02 MiB，沒有週期性尖峰。
- 修正後 RTX raw point cloud 仍為約 29k 點、ring `0..31`、simulation
  scan period 0.100 s；以 bringup 相同 CycloneDDS/Iceoryx 設定量到 GUI＋
  RViz 負載下 wall-rate 約 6.2 Hz，mapping odometry 約 3.05 Hz。
- 效能原始資料：
  - `logs/validation/fps_idle_full_2026-07-30.csv`
  - `logs/validation/fps_idle_channel_filter_2026-07-30.csv`

2026-08-03 使用者以 diagnostics-enabled 的完整 bringup（含 LIO-SAM 與 RViz）
實跑後確認沒有再出現原本的週期性 FPS drop／回穩。若後續仍復發，下一個
gate 才是加入真正的 per-frame latency telemetry，而不是再以 GPU utilization
單點推測 FPS。

## Diagnostics runtime I/O

先前使用者以 `enable_locomotion_diagnostics:=true` 進行 formal bringup 時，曾
觀察到 viewport FPS drop／回穩。當時輸出在約 45 s 已達：

- `locomotion_diagnostics.json`：2,250 samples、約 7.66 MiB；
- `policy_diagnostics.json`：2,267 records、約 5.05 MiB。

原 simulator host 每 25 個 policy step 都會重新對「從 t=0 到目前」的完整
sample list 做 summary、pretty JSON encode 與 atomic replace；因此每次 flush
的工作量隨時間增長，且總寫入量呈 O(n²)。這是 diagnostics 開啟時特有的
viewport drop 候選，與前述 Kit interpolation warning storm 是兩個獨立來源。

目前已提交的修正改為：

- runtime 只將新 sample 追加到 `locomotion_diagnostics.jsonl`，每個 flush
  boundary（預設 25 steps）flush 一次；不再週期性重寫完整 JSON；
- bringup 正常停止時，從 JSONL 一次重建既有 schema v1 的
  `locomotion_diagnostics.json`，所以既有 stability evaluator 介面不變；
- 若主機被強制中止，JSONL 仍保留最近一次 flush 的完整 raw samples，可直接
  用於跌倒前後追查。

2026-08-03 的修正版 full bringup 已完成使用者確認：沒有再出現週期性
viewport FPS drop。該次資料位於
`logs/formal_bringup/diagnostics_io_fix_manual_02`，包含 3,700 筆
locomotion JSONL samples（simulation time 74.0 s）與 3,717 筆 policy records，
且 locomotion samples 的 termination／truncation 都是 0。完整 canonical JSON
若以 Ctrl-C 中止可能不會完成寫出；JSONL 是這次可保留的正式 raw trace。

## LIO-SAM motion deskew status

目前 bringup 確實在 upstream Image Projection 與 Feature Extraction 之間啟用
project-owned `motion_deskew`，但它不是單純把同一個旋轉 deskew 重做兩次：

- upstream Image Projection 先產生 `CloudInfo` 與原始 rotational deskew；
- `motion_deskew` 以 raw `/lio_sam/points` 的 ring／per-point time、200 Hz
  `/imu/data` 與 200 Hz incremental odometry 重建每點的完整 SE(3) correction；
- bringup 設定 `motion_deskew_replace_upstream_rotation=true`，所以 custom
  quaternion gyro integration 取代 upstream Euler rotational deskew，同時補上
  upstream 缺少的 translational deskew；
- Feature Extraction 接收 remap 後的
  `/lio_sam/deskew/cloud_info_motion_corrected`。

因此現在 LIO-SAM 穩定的主要原因是 scan 內 100 ms 的機體平移／旋轉畸變被用
正確的時間與 frame contract 補償，feature extraction 與 map optimization 看到
較一致的幾何，而不是把 instability 用額外節點或速度限制掩蓋。三次
`forward_3_0` map-quality benchmark 已通過；初始 loop matrix 暴露的
`loop_open_backward` early false constraint 已由 project-owned 10 s
minimum-time gate 排除，formal replay matrix 現為 12/12。使用者的 full
bringup 結果已確認 pipeline 可穩定持續運作。

## FAST-LIO2 experiment status

目前只在 `exp/slam-fastlio2` 實驗 branch 加入 FAST-LIO2；沒有修改 upstream
LIO-SAM，也沒有切換正式 Recovery policy。候選 source 是
`Taeyoung96/FAST_LIO_ROS2` fork，commit
`373aa886402b6307db2995ca12b3f4596ef4f633`；它不是 hku-mars 官方 ROS 2
release。2026-08-04 已從易消失的 `/tmp` external overlay 移到 project-local
`deployment/ros2_ws/src/fast_lio`，並由 `fastlio2.repos` 與
`scripts/setup_deployment.sh` 固定、重建。第三方 checkout 由 root Git ignore；
dirty `/tmp/fastlio2_ros2_smoke` 只保留作舊 smoke artifact，不再使用。

2026-08-04 實測發現此 candidate 的 ROS 2 port 會每秒累積並發布無界的
`/Laser_map`。在 project 的 CycloneDDS/Iceoryx 設定下，累積訊息達
`4366153` bytes 時超過預設最大 `4194304` bytes chunk，觸發
`rclcpp::exceptions::RCLError`、`SIGABRT`（`exit code -6`）；因此先前看似
`/Odometry` topic 存在但沒有 publisher。project-owned
`docs/validation/fastlio2_map_pub_downstream.patch` 只新增
`publish.map_en` 參數，config 設為 `false`，停用這個 visualization-only
publisher，不改 FAST-LIO2 的 raw input、native deskew、EKF 或 `/Odometry`。
套用 patch 後重新編譯，FAST-LIO2 已超過原本約 84 秒的 abort 點仍持續發布
`/Odometry`；第二終端確認 `Publisher count: 1` 並成功 echo
`camera_init -> body`。手動停止時為正常 `SIGINT`（`exit code -15`），不是
再次 abort。`scripts/setup_deployment.sh` 會在 project-local pinned source
重建時冪等套用這個 downstream patch。

FAST-LIO2 的 `map_en=false` 不是限制最多保存幾個點，而是完全不啟動每秒把
registered scan append 到 `pcl_wait_pub`、再整包發布 `/Laser_map` 的 timer；
內部 iKD-Tree mapping、`/cloud_registered`、`/Odometry` 與 `/path` 不受影響。
upstream FAST-LIO2 RViz 對 `/cloud_registered` 使用 30 s decay。相較之下，
LIO-SAM `/mapping/map_global` 只有 subscriber 時才以 0.2 Hz 建立，並使用
radius、key-pose density 與 1 m voxel leaf 降採樣；但目前 protected nested
upstream `rviz2.rviz` 對 `/lio_sam/mapping/cloud_registered` 的 decay 是
1000 s，長時間仍可能在 RViz client 累積而變慢。若要改善，應新增
project-owned LIO-SAM RViz config，不修改既有 dirty nested config。

project-owned integration：

- `fastlio_point_adapter` 讀 raw `/lidar/points_raw`，轉為 candidate 要求的
  Ouster `x/y/z/intensity/t/reflectivity/ring/ambient/range`，使用 reliable
  output；不使用 project motion deskew。
- `fastlio_odom_adapter` 將 candidate `/Odometry`、`camera_init/body` 轉成
  `/slam/odom`、`map/base_link`，並由 pose delta 推導 body-frame twist；不會
  發布或覆蓋 simulator GT `/odom`。
- config：`config/fastlio2_anymal_ouster32.yaml`，Ouster 32 ring、timestamp
  unit ns、extrinsic T `[0.20, 0.0, 0.35]`、目前 point filter stride 2。
- replay launch：`fastlio2_replay_benchmark.launch.py`；live locomotion
  launch：`fastlio2_locomotion_benchmark.launch.py`。

同一份 `smoke_out_and_back` replay（185 raw scans、29.87 s，實際約 6.19 Hz）
的 native deskew 結果：LIO-SAM ATE/yaw `0.0374 m / 0.084 deg`；FAST-LIO2
stride 2 為 `0.0944 m / 0.225 deg`，目前 replay gate 通過但仍只是單一場景。
stride 4 失敗；stride 1 在 CycloneDDS/Iceoryx 可能因 4.25 MB shared-memory
chunk 不足 crash，不能把 partial trajectory 當精度結果。

目前正式 Recovery v0.4.0 model1450 policy 已接收 FAST-LIO2 `/slam/odom` 做
無 confidence 的 live baseline：

- stationary：1000 simulation steps，0 termination/truncation，joint command
  freshness `406/406=1.0`，event classification `no_instability`。
- `forward_0_5`：1200 simulation steps，0 termination/truncation，freshness
  `656/656=1.0`，event classification `no_instability`；GT-only diagnostics
  的 target actual velocity `0.4552 m/s`、target MAE `0.0789 m/s`、最大
  roll/pitch `0.0479/0.0402 rad`。
- FAST-LIO2 約 10 Hz，policy 50 Hz，因此 live launch 使用 timer trigger
  取最新 odom，`state_timeout_s=0.25`；原本 10 ms 的
  `synchronized_state` 不適合低頻 SLAM odom。
- LiDAR-enabled live simulator 的 IMU parity tolerance 明確設為 `0.01`，
  與既有 LIO-SAM live benchmark 一致；這只是 bridge validation threshold，
  不是 confidence 或 locomotion stability threshold。
- forward run 的 controlled real-time factor 約 `0.741`，physical-clock
  throughput 仍未通過正式 gate。

詳細結果在 `docs/validation/slam_backend_comparison.md`。目前正式runtime／policy
仍未加入`slam_tracking_valid` safety supervisor或PPO confidence observation；
repository中的51-D consumer與兩backend calibrated confidence extractor都屬候選path，
尚未進production或取代正式model1450。

## FAST-LIO2 native comparison pilot implementation (2026-08-03)

- 新增 `slam_backend_compare.launch.py` 作為 live selector：一次只啟動
  `liosam` 或 `fastlio2`；預設 LIO-SAM native deskew。正式 model1450 只讀
  GT `/odom` 產生 joint command，selected SLAM 是 observation-only，因此
  不會把 GT 偷接回 SLAM policy input。FAST-LIO2 live 分支保留 native
  `/Odometry` 與 `camera_init/body`，不啟動 `fastlio_odom_adapter`。
- 2026-08-04 的 LIO live smoke 發現 nested LIO-SAM 的 `use_rviz=false` 會
  覆蓋外層同名 launch configuration，導致 selector 的 RViz 沒有啟動；已將
  外層開關改名為 `compare_use_rviz`。`open_teleop_terminal:=false` 本來就
  是關閉鍵盤，`simulation_steps:=300` 完成後本來就會正常 shutdown。
- candidate、Livox ROS driver 與 Livox SDK2 source 現在分別位於
  `deployment/ros2_ws/src/fast_lio`、`livox_ros_driver2`、`livox_sdk2`；固定版本
  由 `deployment/ros2_ws/fastlio2.repos` 記錄。SDK2 install 位於被忽略的
  `deployment/ros2_ws/vendor/livox_sdk2`；不使用 Docker。
- project-owned CycloneDDS local/static-TF config 已補上 explicit localhost
  peer `127.0.0.1`；同一 `ROS_DOMAIN_ID` 的第二終端現在可 discovery
  `/Odometry`、`/cloud_registered`、`/path` 與 TF。2026-08-04 進一步確認
  direct FAST-LIO2 launch 原本強制 `ROS_LOCALHOST_ONLY=1`，但一般使用者 shell
  為 `0`，且 project profile 完全停用 multicast，造成第二終端 discovery
  隔離。`fastlio2_run.launch.py` 已移除額外 localhost 覆寫，CycloneDDS
  profile 則開放 discovery-only SPDP multicast；大型 user data 仍走 unicast
  或 Iceoryx。第二終端只需 source ROS 2 與 project overlay，並使用相同
  `ROS_DOMAIN_ID`，不再需要手動 export DDS profile。使用者已在正常 host
  terminal 重新啟動 stack，確認另一個只 source ROS 2／project overlay 的
  terminal 可以直接看到 FAST-LIO2 topics。
- `scripts/setup_deployment.sh` 會由 Livox `package_ROS2.xml` 產生被忽略的
  `package.xml`、初始化 FAST-LIO2 submodule、冪等套用 downstream patch，並與
  project workspace 一起建置。candidate 內 nested driver 沒有 ROS 2
  `package.xml`，因此 colcon 只辨識 workspace root 的 driver package，不會
  產生 duplicate package。
- `slam_backend_native_replay.launch.py` 對相同 raw bag 直接評估
  LIO-SAM `/lio_sam/mapping/odometry` 或 FAST-LIO2 `/Odometry`；project
  `/slam/odom` adapter 不在 native comparison path。既有 Factory
  `smoke_out_and_back` bag 的新 evaluator 結果：LIO-SAM native
  `0.037391 m / 0.083753 deg`、FAST-LIO2 native stride 2
  `0.094370 m / 0.224615 deg`，兩者 passed。
- 2026-08-04 新增 `fastlio2_run.launch.py` 作為第一次人工操作的直接入口；
  它 include candidate 原本的 `mapping_ouster64.launch.py` 與
  `rviz_cfg/fastlio.rviz`，不經 `slam_backend_compare` selector，也不啟動
  benchmark/evaluator/odom adapter。project 只補 simulator raw PointCloud2
  adapter、model1450 GT motion driver、teleop 與 local DDS。upstream RViz fixed
  frame 維持 `camera_init`。headless 1200-step smoke 通過；第二終端確認
  `/Odometry` 與 `/cloud_registered` 各有 1 個 publisher，並成功讀到 width
  `4199` 的 registered cloud。
- 使用者已親自用 `fastlio2_run.launch.py` 操作並看到 FAST-LIO2 native RViz
  輸出；主觀觀察是整體比目前 LIO-SAM native 好一點，但快速旋轉時仍會出現
  明顯裂圖。這次目的只是建立 FAST-LIO2 效能直覺，不是正式 quality gate，
  不可把「稍好」寫成量化結論；快速旋轉裂圖應保留為後續 yaw-rate／rotation
  stress scene 與 confidence degradation label 的候選現象。
- adapters 新增 deterministic `point_density`（預設 1.0；pilot 使用
  1.0/0.75/0.5/0.25），不複製或修改 source bag。50% FAST-LIO2 Factory
  smoke 已被 evaluator 正確判為 expected backend failure
  (`ATE=0.248619 m`)，不是 infrastructure failure。
- 新增 project-owned flat feature-poor scene：
  `assets/maps/ground_plane/GroundPlane.usda/.usd`。它只有 100 m square
  collision mesh 與同 contract 的 `FactoryPhysicsMaterial`，不含牆、物件或
  visual features；`validate_ros2_bridge` smoke 通過，resolved collision
  prims `1`、300 steps、joint freshness `1.0`。source bag 位於
  `outputs/slam_backend_pilot/source_groundplane_forward_0_5/bag`（約 15.4 s、
  120 scans、2416 IMU、605 GT odom）；Flat scene 的 LIO/FAST replay 均為
  expected backend failures，不能當成 host/bridge failure。
- 新增 `scripts/validation/run_slam_backend_pilot.py`：兩 scene × 兩 backend
  × 四 densities 的 16 unique cells；100% baseline 與 50% boundary 三次，
  其他一次，共 32 sequential runs。只有 evaluator JSON 明確回報的 backend
  failure 會繼續；timeout、missing JSON 或 pre-evaluator crash 會停止整套。
  這次實際完成 `passed=11`、`expected_backend_failure=21`、
  `infrastructure_failure=0`；輸出在
  `logs/slam_backend_pilot/matrix/summary.json`。Factory native baseline
  三次結果為：LIO-SAM ATE `0.037397..0.037446 m`、yaw
  `0.083725..0.083863 deg`；FAST-LIO2 stride 2 ATE `0.094370 m`、yaw
  `0.224615 deg`。Factory LIO-SAM 在 75/50/25% density 仍通過；FAST-LIO2
  在 75/50/25% 被 evaluator 判為 backend failure。GroundPlane 的兩個
  backend、所有 density 都是預期的 feature-poor failure，不是
  infrastructure failure。操作與 gate 邊界見
  `docs/validation/slam_backend_live_and_replay.md`。

## Experimental Recovery v0.5

Recovery v0.5 training foundation 已加入，但所有候選都仍是實驗產物：

- Warehouse failure sequence replay。
- High-combined、combined-to-straight、reverse-yaw、rapid-zero sampling。
- Turning regression replay。
- 左右對稱 data augmentation。
- 40 s episode 與 recovery sampling。
- 純低 yaw、低速曲線與 3 m/s 曲線的條件式 rewards。
- 尚未訓練的新 targeted 設定另保留 15% environments，逐段重播
  `refinery_fix_01` 的 25 段 effective policy-command history
  （42.10 s；episode 會涵蓋前 40 s），包含兩次 watchdog-effective zero
  gap，以及最後 `wz=0.817349` 原地轉向 → `vx=1.898749` 直行 → 明確
  zero 的 failure tail。
- 新設定在上述 refinery moving envelope 額外施加
  flat-orientation 與 stance-foot-slide rewards；dedicated 分佈為
  warehouse 35%、refinery 15%、turning regression 40%，另外保留 10%
  給既有一般／高組合隨機 transition sampling。
- 這是 command-history 加上既有 5–10 s push disturbance 的 targeted
  training distribution，不是 PhysX contact-cache transplant。訓練仍無法
  直接保存 Factory contact solver history，最終有效性必須由 exact
  action-history replay qualification 判定。
- IMU frame root cause 確認後，不應立刻啟動此 targeted training；先完成
  corrected bridge 的完整正式路徑、reset 與 LIO regression。只有 bridge
  修正後仍存在 policy gate failure 時才續訓新 candidate。
- 2026-08-03 的正式 v0.4 diagnostics-enabled bringup 已在
  `vx≈2.2975 m/s, wz=-2.0 rad/s` 的 command envelope 下
  `no_instability`、0 termination；因此 Recovery v0.5 目前維持候選封存，
  不需要為了這次穩定性結果切換 policy 或立即續訓。

目前最重要的候選結果：

### model2402

Checkpoint：

`logs/rsl_rl/anymal_d_locomotion_v1/2026-07-29_12-38-24_recovery_v050_low_curve_3iter_seed42_4096env/model_2402.pt`

- Warehouse locomotion 3/3 為 `no_instability`、0 termination。
- 完整 turning fail-fast 通過前 30/36。
- `curve_3_0_left_0_5` 失敗：
  `vx MAE=0.510151 > 0.2`。

### model2420

Checkpoint：

`logs/rsl_rl/anymal_d_locomotion_v1/2026-07-29_13-19-15_recovery_v050_high_curve_reward16_10iter_seed42_4096env/model_2420.pt`

- 增加 3 m/s curve replay 與條件式 reward 後，ONNX parity 通過。
- `curve_3_0_left_0_5` 改善為
  `vx MAE=0.331432 > 0.2`，仍未達 gate。
- 因最小 gate 未過，沒有執行此候選的 warehouse 三次、完整 turning 或
  loop-closure regression。

不可把 model2402 或 model2420 設為正式 policy。

## Latest validation state

### First confidence-conditioned PPO (2026-08-11)

- 新task `Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-v0` 的policy
  observation固定為51-D；原48-D順序與數值不變，offset 48..50依序是
  `confidence/tracking_valid/normalized_age`。Training使用ROS-independent的
  deterministic simulated confidence，沒有GT或backend-specific raw feature；ROS
  consumer則只在51-D metadata時訂閱atomic `/slam_confidence`，並保留0.15 s steady
  receipt watchdog與fail-closed `[0,0,1]`。
- 正式Recovery v0.4.0 model1450已用零初始化新輸入columns擴成51-D bootstrap；初始actor
  對所有舊48-D observation與confidence值保持原policy輸出。Bootstrap checkpoint位於
  `logs/rsl_rl/anymal_d_locomotion_slam_confidence_v1/bootstrap_model1450_51d/model_1450.pt`，
  只屬runtime artifact，不取代正式policy。
- 4096-env、seed 42、150 iterations的第一次PPO run已完成，run為
  `logs/rsl_rl/anymal_d_locomotion_slam_confidence_v1/2026-08-11_21-50-19_first_ppo`，
  checkpoint為`model_1599.pt`，共14,745,600 simulation steps。最後20個
  `Train/mean_reward`為`9.5478 +/- 0.3755`，relative slope/iteration為
  `+0.000511`；最後50點slope為`-0.001671 reward/iteration`，可視為第一次score
  已回穩。最後20點episode length為`985.67 +/- 7.14 / 1000`。
- `model_1599` checkpoint/TorchScript/ONNX 256-sample parity通過；TorchScript max
  absolute error為0，ONNX為`5.72e-6`。新三個input column L2 norms分別為
  `0.1463/0.1225/0.2605`，固定舊48-D時timeout相對healthy的action mean L2差為
  `0.1816`，證明policy有使用新輸入，而不是維持零權重。
- 但seed 43、512 env、1000-step confidence-state evaluator顯示第一次候選沒有學會
  invalid時停下：healthy/invalid mean planar speed分別為`1.4660/1.4521 m/s`，invalid
  linear RMSE為`1.6016 m/s`。因此此候選是重要的負結果，不得通過deployment或
  locomotion gate，也不得取代正式Recovery v0.4.0 model1450。下一次訓練需先修正
  reward/curriculum，使invalid stop成為明確且可達的學習目標，再要求分段行為gate。
- 使用者於2026-08-12確認第二輪主線仍從正式model1450 warm-start，不從0重新學走路。
  Actor沿用model1450，51-D新增三個input columns仍以0初始化，確保iteration 0 locomotion
  行為與48-D正式policy相同；critic與optimizer重新初始化，因confidence-conditioned
  reward已改變value target語意，也避免沿用舊optimizer momentum。正式訓練前先在完全
  相同seed、command與confidence schedule評估未訓練bootstrap，作為iteration 0 baseline。
  Aggregate reward不再是主要gate，必須分開記錄healthy tracking、degraded deceleration、
  invalid stop與recovery；可以另做小型from-scratch control，但不得把它當主要候選。

### Second confidence-conditioned PPO round (2026-08-12)

- 已實作真正的actor-only model1450 warm-start：只複製`std`與`actor.*`，actor第一層
  48→51的新三欄精確zero-init；critic由runner依seed重新初始化，Adam state為空，iteration
  為0。256-sample任意confidence輸入的actor parity最大誤差為0，fresh critic不等於source
  critic。Legacy bootstrap保留作第一輪重現，但已明確標為會擴critic並保留optimizer、不可
  作第二輪正式起點。
- 機讀behavior gate固定於`configs/slam_confidence_behavior_gate.yaml`（SHA-256
  `298df1eb772b2f2a077af6a0eacd0e746fd4e8a0a446b740882e60a81cf8eaef`）：seed 43、512 env、
  1000 steps、固定`vx=1.5`與同步10 s confidence cycle，分開healthy、degraded-late、
  invalid-settled與recovery-settled，且hard termination與正常timeout分開。
- iteration-0 baseline已保存；healthy/invalid settled speed為`1.494046/1.494028 m/s`，
  degraded/healthy ratio `0.999964`，hard termination 0、正常20 s timeout 512。它保留
  model1450 locomotion並如預期未過degraded/invalid gate。
- 三個4096-env、seed42、50-iteration（各4,915,200 steps）tranche都從model1450重新開始，
  沒有resume model1599或前一個model49：unrestricted actor、confidence-input-only，以及
  input-only加invalid action-L2。三者invalid settled speed分別為
  `1.501238/1.300765/1.355592 m/s`，degraded/healthy ratio為
  `0.998959/1.003298/1.000356`；全都未學會degraded deceleration或invalid stop。
  後兩者invalid yaw也失敗。三個model49皆retire，不續訓、不export、不進backend matrix。
- 最後input-only checkpoint的舊48欄、downstream actor與std仍bit-exact，新三欄norm為
  `0.6283/0.6205/1.0548`，證明失敗不是誤訓舊actor或完全忽略confidence。詳細路徑、hash、
  gate與下一設計選項見`docs/validation/slam_confidence_ppo_second_round.md`。
- 後續short-credit-path實驗依序排除了free action residual、recovery失敗的nominal teacher、
  無部署上界的unbounded gain，以及右側移48/512 hard termination的exact unbounded target；
  這些checkpoint都retire，不續訓也不進backend matrix。
- 通過的架構固定model1450 actor backbone，以confidence-conditioned per-action gain把原command
  朝zero command做bounded blend；teacher target與runtime gain皆硬限制為`0.8`。gain為0時完整
  actor與正式model1450 bit-exact，critic與optimizer皆fresh。新task是
  `Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-BoundedSafeCommand-v0`。
- bounded iteration-0 checkpoint位於
  `logs/rsl_rl/anymal_d_locomotion_slam_confidence_bounded_safe_command_v1/2026-08-12_14-16-51_bounded_safe_command_bootstrap/bootstrap_model_0.pt`
  （SHA-256 `7d827f76573ce2fa48f79d8a44add24bd33645601b5956c6898ce405a4d8deea`）。
  gain全0、backbone/std exact、optimizer state為0；behavior baseline重現
  healthy/invalid `1.494046/1.494028 m/s`與ratio `0.999964`，如預期只保留正式行為、未過stop gate。
- 第二輪qualified candidate是
  `logs/rsl_rl/anymal_d_locomotion_slam_confidence_bounded_safe_command_v1/2026-08-12_14-20-44_bounded_safe_command_qualification_20iter/model_19.pt`
  （SHA-256 `73d5870de4acc5c79b0d0fa30f15cd70ed5c9539f9b7ec9a4d35d30d6377fb16`）。
  它從正式model1450重新開始，使用4096 env、seed42、20 iterations、1,966,080 simulation
  steps；learned gain為`0.792782..0.796848`，未超過`0.8`。
- seed43、512 env、1000 steps的forward、combined、lateral-left、lateral-right、reverse
  五個profiles全部通過原behavior gate與distinct hard-terminated-env fraction `<=1%` overlay。
  healthy speed為`1.454..1.516 m/s`、degraded/healthy ratio為`0.278..0.324`、invalid settled
  speed為`0.052..0.207 m/s`、recovery為`1.179..1.223 m/s`；forward/combined為0個hard
  terminated env，三個directional holdout各1/512。機讀aggregate位於
  `docs/validation/slam_confidence_bounded_safe_command_model19_qualification.json`
  （SHA-256 `b0474d52a9164da3c8529a7e50e10496e44708eba8dd114351f1de68570f81c2`）。
- 同一checkpoint的256-sample TorchScript/ONNX parity通過，最大絕對誤差為`0`與
  `4.2915e-6`。候選ONNX位於
  `exported/anymal_d_locomotion_slam_confidence_bounded_safe_command_v1/2026-08-12_14-20-44_bounded_safe_command_qualification_20iter/policy.onnx`
  （SHA-256 `511383d9a6b0d4d6c26588667e195f9d66b7e703db991af59fcc8aad4aeb84df`）。
  通過模擬gate只表示可進兩backend live locomotion matrix；正式Recovery v0.4.0 model1450
  仍不變。
- 兩backend fail-fast live screening已執行`forward_1_5`。FAST-LIO2使用正式
  `native-v1-1e6cf8347be1`，0 termination，706/736個policy records讀到tracking-valid，
  confidence identity、policy diagnostics與locomotion stability全部通過。LIO-SAM使用正式
  `native-v1-edc098b0bd98`，identity與51-D consumption通過，但出現1 termination、
  `foot_slip_first`、`vx/vy MAE=1.09995/0.320096 m/s`，所以matrix fail-fast停止，剩餘8格未跑。
- LIO failure在policy於7.73 s第一次讀到invalid confidence前已開始。以GT只作offline evaluator
  的same-stamp比較顯示LIO mapping pose-delta twist平均L2誤差約`1.1066 m/s`。相同LIO path改用
  正式48-D model1450仍有3 terminations；0.5 s causal pose window仍3 terminations；native
  IMU-preintegration velocity control降為0 termination但仍`foot_slip_first`且
  `vx MAE=0.266499 > 0.2`。兩個twist arm都retire並從source移除，沒有改upstream LIO、
  confidence estimator、native deskew或正式artifact。機讀摘要見
  `docs/validation/slam_confidence_bounded_safe_command_model19_backend_screening.json`。
- 2026-08-12 已凍結 `configs/liosam_policy_state_quality.yaml`，並在project-owned adapter
  分離 `/slam/odom` mapping/confidence authority與實驗性 `/slam/policy_odom`。後者讀取
  LIO-SAM既有map-corrected高頻 `/lio_sam/odometry/imu`，把upstream world-frame linear
  velocity轉成`base_link`；不修改upstream、不使用project deskew，GT只新增到offline
  diagnostics/evaluator。正式model1450的3個registered `forward_1_5` repetitions全部通過
  direct policy-state gate：age p95約`0.010 s`，XYZ MAE範圍約
  `0.0558..0.0589 / 0.0674..0.0729 / 0.0208..0.0248 m/s`，0 timestamp regression。
- 上述3次仍全部在停止尾端出現`foot_slip_first`，body instability時間為
  `14.72/14.84/14.92 s`；雖然0 termination，locomotion stability仍為0/3。
  Offline one-step GT velocity replacement顯示active action差異尚在provisional limit，
  但stopped-tail action mean absolute difference為`0.1443..0.1776`、max
  `0.4940..0.8227`，所以action-sensitivity為0/3。這表示mapping pose-delta的大錯已解，
  但native IMU predictor在停止漂移時的殘差仍會實質改變model1450 action，尚不可稱
  policy-grade。0.1 s high-rate pose regression會把map correction微分成尖峰，XYZ MAE
  `1.185/0.570/1.334 m/s`且1 termination，已從source移除。機讀摘要見
  `docs/validation/liosam_policy_state_v1_forward_control.json`。

在 `2a7159a` 提交前：

- Pytest：86 passed、3 skipped。
- Python compileall：通過。
- `git diff --check`：通過。
- `scripts/setup_deployment.sh --check`：通過。
- Git LFS fsck：通過。

本次 IMU frame／state-transplant commit：

- Pytest：98 passed、3 skipped。
- Python compileall：通過。
- `git diff --check`：通過。
- `scripts/setup_deployment.sh --check`：通過。
- Git LFS fsck：通過。
- 修正版 Factory 81 點 drop-grid：81/81 通過，唯一 infinite GroundPlane
  正常。
- `warehouse_refinery_exit` 局部序列：0 termination、`no_instability`。
- 完整空地 `warehouse_mapping_stress`：0 termination、`no_instability`。
- `refinery_fix_received_trace_replay` 在 GroundPlane 與 Factory：
  0 termination、`no_instability`。
- `(3,13)` 與 `(-2,13)` 局部直線穿越：皆為 0 termination、
  `no_instability`。
- 近似 effective trace 的 abrupt-stop 基準會跌倒；watchdog deceleration
  實驗 A/B 通過。
- Exact t=25 action-history replay 已做到第一筆 external 48-D observation
  全 term parity，並重現 `foot_slip_first` 與 termination。
- Exact received-command baseline 與 watchdog deceleration 2.0 A/B 結果
  完全相同；正式 zero packet 會讓 deceleration 無機會介入。
- Policy/reset parity 現在對 episode reset 後的 IMU transient 使用最多
  4 個 50 Hz steps 的有界 grace；只在 IMU/native error 尚超標時略過，
  收斂即提前結束 grace。
- Recovery v0.5 新設定已由實際 Isaac Sim instantiate：
  `RecoveryV05VelocityCommandCfg`、refinery probability `0.15` 與新增
  reward weights `-3.0/-0.1` 均成功解析。
- Corrected IMU frame 的 exact-history runtime observation parity 已通過，
  且原 failure 變為 `no_instability`、0 termination。
- 使用者從 t=0 執行的 corrected formal bringup 亦為 `no_instability`、
  0 termination；log 為
  `logs/formal_bringup/imu_frame_fix_manual_01`。
- 早先的 `imu_frame_fix_manual_01` 沒有 episode reset，因此當時不能把四 tick
  grace 記成已通過；後續 `reset_gate_lio_clean_01` 已在 LIO 負載下完成一次
  reset ACK、IMU transient 與 `previous_action` reset regression。
- Viewport warning-storm 修正後：
  - Pytest：99 passed、3 skipped；
  - Python compileall、`git diff --check`、
    `scripts/setup_deployment.sh --check`、Git LFS fsck 全部通過；
  - 完整 GUI idle 300 s interpolation warning 為 0；
  - RTX LiDAR、adapter 與 mapping odometry 持續輸出。
- 2026-07-30 diagnostics-enabled 短 smoke（完整 LIO、無 RViz／teleop）在
  SIGINT 前留下可解析的 `locomotion_diagnostics.jsonl`：775 samples、約
  1.63 MiB；執行中沒有週期性重寫 canonical JSON。
- 2026-08-03 使用者完成 diagnostics-enabled full bringup（含 RViz）並確認
  沒有再出現週期性 FPS drop；run 為 3,700 locomotion samples、3,717 policy
  records、0 termination／truncation；peak command 約為
  `vx=2.2974865 m/s, wz=-2.0 rad/s`，最低 base height 約 `0.509 m`，
  最大絕對 roll／pitch 約 `0.146/0.130 rad`，classification 為
  `no_instability`。
- 2026-08-03 LIO 負載下的可控 reset gate 已乾淨通過：
  `logs/formal_bringup/reset_gate_lio_clean_01`，
  `simulation_steps=360`、reset step `220`、reset ACK count `1`、
  termination/truncation `0`。Validator 的 IMU reset transient angular
  velocity／projected gravity max error 都是 `0`；policy diagnostics 顯示
  reset 前 `previous_action` max abs 約 `0.81401`，reset 後第一筆 observation
  的 12 維 `previous_action` 全為 `0`。
- 目前修正版 pipeline 的 `forward_3_0` map-quality benchmark 連續三次通過：
  `outputs/lio_sam_benchmarks/reset_gate_forward_3_0_run{1,2,3}`；三次
  translation ATE RMSE 為 `0.013370/0.012135/0.013071 m`，yaw RMSE 為
  `0.031043/0.031649/0.033574 deg`，post-ready motion-deskew unavailable
  與 IMU/odom unavailable 都是 `0`。
- 正式 policy loop matrix 的初始輸出位於
  `outputs/lio_sam_loop_closure/qualification_v1_formal_v040_20260803`；
  policy path 是正式 `exported/.../recovery_v0.4.0/policy.onnx`，結果為
  11/12。唯一失敗的 `loop_open_backward/run_01` 在
  `8.509999807 s` 看到 15 個 loop marker 與 1 個 constraint edge；同一
  sensor bag 的獨立 fresh-graph repeat 重現同一 edge，證實不是隨機
  evaluator false positive。
- 將 matrix `minimum_time_difference_s` 從 `8.0` 改為 `10.0` 後，使用同一
  12 個 capture bags、正式 v0.4.0 policy 與 metadata 完成修正版 replay：
  `outputs/lio_sam_loop_closure/qualification_v1_formal_v040_20260803_time10`
  的 `matrix_summary.json` 為 expected 12、passed 12、failed 0。12 個
  enabled cases 全部符合 required／forbidden expectation；open-backward
  三個 run 均為 0 marker／0 edge。disabled control reports 沿用同一 sensor
  bags 的既有 passed reports，因 loop closure 明確 disabled，不使用搜尋門檻。
- `feature/ppo-slam-confidence` 目前 Python pytest 為 `271 passed, 3 skipped`；confidence
  tests涵蓋共同state machine、兩backend exact-stamp assembler、offline label core、
  contract、launch/config integration與FAST 13-case synthetic fault matrix。3個skip
  仍是Isaac Sim runtime unavailable的既有external-project tests。
  `git diff --check`、Python compile與本里程碑ROS interfaces/runtime package build
  均通過；`--packages-up-to anymal_locomotion_ros2`的完整閉包在目前host會被既有
  `livox_ros_driver2`缺少`/usr/local/lib/liblivox_lidar_sdk_shared.so`擋住，本次沒有
  修改或繞過該外部SDK狀態。
  `scripts/setup_deployment.sh --check` 也通過。
- 安裝型`fastlio_confidence_fault_validation`以與runtime相同的pure core在固定
  20 Hz grid重播13種hard fault，13/13通過。Freeze detection為IMU 50 ms、
  odom/LiDAR 250 ms；effect missing在join grace後100 ms；zero support、numeric、
  timestamp/frame fault同tickinvalid。missing scan可在新完整bundle後經0.50 s
  recovery回TRACKING；除clock reset清source/score外，所有case都證明約0.90的
  stale-high score仍會invalid。Report是runtime artifact
  `logs/slam_confidence/fastlio_fault_validation_installed_20260810.json`，不提交。
  後續實際CycloneDDS qualification已補FAST與LIO各6類共12/12通過，包括topic
  freeze、publisher death、zero-support payload及extractor死亡的0.15 s receipt
  watchdog；機讀摘要為
  `docs/validation/slam_confidence_dds_and_fast_live_gate_summary.json`。
- Generated `SlamConfidence` serialization round-trip為80 bytes，超過32-byte的
  `backend_id`會被ROS generated type拒絕。localhost-only runtime smoke實際收到
  `backend_id=fastlio2`、`calibration_id=uncalibrated`、score 0、valid false、
  `STATE_INITIALIZING`與reason mask 139265（INITIALIZING + SIGNAL_MISSING +
  UNCALIBRATED），確認no-source startup fail closed。
- FAST-LIO2 stationary／forward live smoke 的 output 分別位於：
  `logs/stability_benchmarks/fastlio2/stationary_smoke_timer` 與
  `logs/stability_benchmarks/fastlio2/forward_0_5_smoke_atol01`。第一次
  forward run 只因 default IMU parity `0.002` gate 超過
  `0.0028817` 而停止，沒有 termination 或 policy freshness failure；依照
  既有 LIO-SAM live benchmark 改用 `0.01` 後重跑通過，最後 error
  `0.0025872`。

## Unfinished work

依目前 gate 順序：

1. FAST-LIO2 的虛擬 Ouster input contract、estimator calibration 與五個主要
   replay holdout 已完成；branch baseline 固定為 native deskew、
   `sensor_order+staggered`、stride 2、iteration 4、`surf=.3/map=.6/cube=1000`。
   不加入 project-based deskew。
2. 新 baseline的live左右`wz=2.0`與雙向lateral共四個正式profile已通過；進入
   TRACKING後0 invalid，locomotion皆0 termination/truncation。一次較早左轉debug run
   在19.5 s跌倒且confidence隨support下降進LOST，保留為失敗診斷，不計入pass。
3. Backend-neutral confidence v1 wire schema與共同state/time語意已freeze；兩個
   backend extractor、offline label、strict capture-group calibration/fresh holdout與
   artifact-enabled runtime replay都已通過。正式artifact為FAST
   `slam_confidence_fastlio2_native_v2.json`（`native-v1-1e6cf8347be1`）及LIO
   `slam_confidence_liosam_native_v3.json`（`native-v1-edc098b0bd98`）。Stale
   high-confidence仍必須invalid；不同backend raw count不可直接互比，GT `/odom`
   仍只可作offline label。兩backend隔離CycloneDDS graph的12-case hard-fault
   qualification已全數通過。
4. `feature/ppo-slam-confidence`已完成51-D observation、metadata、symmetry、export與ROS
   runtime parity，也完成actor-only/fresh-critic/fresh-optimizer起點與固定分段behavior gate。
   reward-only、不安全short-credit-path與外部gait-mode候選均已retire；bounded safe-command
   `model_19`已通過五profile模擬behavior qualification與export parity但不具gait value。
   C-v10 phase-separated structured-gait model48現已通過固定512-env behavior、
   gait-value A/B、export parity及兩backend完整五方向live matrix；仍是實驗deployment
   candidate，不是正式policy。
5. `model_19`兩backend minimum screening為FAST通過、LIO失敗。LIO
   policy-state quality contract、evaluator與high-rate native predictor candidate已實作；
   direct age/accuracy/outlier為3/3 pass，但正式model1450 locomotion與stopped-tail action
   sensitivity均為0/3，故整體contract仍fail。下一個架構gate是獨立於SLAM的policy-grade
   proprioceptive body-velocity estimator已由candidate08在模擬中關閉，並已放進PPO rollout
   observation loop。人工LIO操作與相同estimator/confidence artifacts的FAST/LIO五方向
   live matrix現已完成；replay與凍結protocol的safety-efficiency／false-stop／SLAM
   survival量化仍未執行，且使用者目前先要求完善工程實作。完整publication與實體gate
   通過前不得取代model1450。
   實體 ANYmal-D
   sensor extrinsic、
   low-level interface；IMU frame contract
   必須維持 sensor-local `base_link` semantics。目前尚無 driver／SDK、實測
   extrinsic 或 safety controller；準備清單見
   `docs/physical_anymal_d_integration.md`。
6. Recovery v0.5 model2402/model2420 維持實驗候選封存；除非正式 bridge
   gate 再次出現 policy failure，否則不切換 policy、不立即續訓。

GroundPlane open-loop prehistory 在 t=25 前失敗，不能當有效 counterfactual；
後續結論仍必須保留 Factory/contact history dependency。

## Latest validation state and remaining gates

### Model48 deployment candidate v2 qualification (2026-08-13)

- 新增 `slam_confidence_interactive.launch.py` 作為人工操作入口：以
  `slam_backend:=fastlio2|liosam` 選 backend，自動選正式 calibration ID，固定載入
  model48 confidence-aware PPO 與 estimator15，預設開 Isaac Sim GUI、backend RViz
  及官方 keyboard teleop。此模式不啟動 scripted stability driver，避免兩個
  `/cmd_vel` publisher 互搶；原 benchmark defaults 維持 headless scripted behavior。

- C-v10 model48 原始 export 在 FAST-LIO2 `lateral_right_1_5` live cell 可重現
  foot-slip-first fall；512-env fixed holdout 為 316/512 distinct hard terminations。問題不是
  DDS 或 LIO latency，而是 1.0 intent transition、PPO gait delta 與缺 lateral-transition
  estimator data 疊加。
- 保持 confidence 在 51-D PPO observation 內，將 PPO intent coordinate 的結構上限設為
  0.79、gait delta 依 safe scale power 10 淡出，invalid 時完全關閉 gait delta。這是 policy
  action-space bound，不是 ROS node 外部限速器；PPO gait coordinates 在 healthy/degraded/
  recovery 仍參與動作。
- 新 estimator15 位於
  `exported/proprioceptive_velocity_estimator/v1/clean_lateral_transition_candidate_15/`，以
  candidate08 初始化，加入固定左右側移與左右 confidence stop/recover captures；GT 只作
  supervised label/offline evaluator。environment-disjoint holdout 243,932 windows，XYZ MAE
  0.0147/0.0152/0.0109 m/s、p95 0.0762 m/s、max 1.0211 m/s，通過 frozen accuracy gate。
- model48 + estimator15 五方向 simulation behavior matrix全通過：forward 0/512、left
  0/512、right 5/512、reverse 1/512、combined 0/512 hard-terminated env；各 profile 的
  invalid stop與recovery checks亦全過。
- 新 export ONNX SHA-256
  `c8225b05a2b8fd1a35fb919593a68326b0ddac3d955b69191708def5d071f4a5`；checkpoint/
  TorchScript/ONNX固定輸入與200-step previous-action recurrence parity全過。
- ROS live最難右移 cell已在 FAST-LIO2與LIO-SAM各通過一次，policy diagnostics分別
  686/685筆全finite、watchdog timeout 0，兩者 confidence calibration/identity/timestamp
  checks全過。LIO-SAM不再提供 locomotion velocity；兩backend只提供 pose/confidence，
  速度由同一 proprioceptive estimator提供。
- 此結果仍是experimental deployment candidate；正式 recovery v0.4.0 model1450未被
  取代，實體ANYmal-D qualification與包含新 `/foot_contacts` 的bag replay仍未完成。

### Model48 two-backend full live matrix (2026-08-17)

- `run_slam_confidence_locomotion_matrix.py`原本有三個結果可信度問題：phase matrix未指定
  `--output-root`時會寫入model19預設目錄；subset smoke會覆寫完整
  `matrix_summary.json`；passed cell只看`passed=true`，未核對matrix、calibration及artifact
  identity。現已改為由安全的`matrix_id`推導輸出目錄、subset使用selection hash命名summary，
  並以matrix SHA、backend/profile/repetition、calibration、policy/estimator/parity及stability
  config SHA共同決定可否skip舊cell。
- 通用stability gate仍保留原始command tracking門檻。Model48 phase-separated matrix另明確
  選用`configs/stability_diagnostics_confidence_aware.yaml`：termination、truncation、finite
  samples及instability的硬安全門檻完全相同；raw-command tracking仍保存在summary，但因
  confidence-conditioned policy在low confidence時本來就會主動偏離raw command，故不作此
  matrix的pass/fail條件，gate輸出明示`tracking_gate_applied=false`。這不是放寬安全門檻，
  也不宣稱任務效率已驗證。
- 修正前第一次完整執行FAST五項全過，LIO forward只因`vx_mps_mae=0.940452>0.2`被誤判；
  該run為0 termination/truncation、`no_instability`、policy watchdog 0，confidence確實在前進
  時退化並觸發model48降速。新語意先對同一trace重算通過，再重新執行完整matrix。
- 最終matrix位於
  `logs/slam_confidence_locomotion_matrix/phase-separated-model48-intent079-estimator15-v2/`，
  matrix SHA為`351895fbc3faecb633f802813257ed540bde78608131942db3cd8184451336f7`；FAST-LIO2與
  LIO-SAM的forward/left/right/backward/curve共10/10通過。每格driver、硬安全、51-D policy
  consumption與confidence identity/calibration checks均通過，policy watchdog timeout皆0。
  這完成目前simulation live backend implementation matrix，但仍不是publication因果證據、
  實體ANYmal-D qualification或promotion授權。
- 本輪相關core/integration/contract/estimator/stability/matrix測試共76項通過，三個修改的
  Python入口亦通過`py_compile`，`git diff --check`通過。Matrix teardown後FAST-LIO2或
  LIO-SAM部分upstream程序仍會被launch以SIGTERM結束並印出`process has died ... -15`；
  benchmark host、driver、policy與launch總回傳碼皆成功，因此不影響10/10結論，但屬於可再
  整理的project-owned launch shutdown噪音，且不可藉此修改或還原nested upstream。
- 使用者已明確決定publication收數延後。近期工程優先順序是保持正式model1450不變，先完善
  deployment／qualification tooling；仍缺包含新`/foot_contacts`契約的bag replay、實體
  ANYmal-D driver/SDK、extrinsic與safety controller整合。Publication B/D artifacts、正式
  repetitions與因果統計等到實作穩定後再做。

### Branch-only simulation release and publication readiness (2026-08-17)

- 使用者其後將目標明確提升為：仍只在`exp/slam-fastlio2`上，完成第一級正式simulation
  deployment release，並完善到可執行publication A/B/C/D收數與產生數據支持；`main`不動。
  在所有promotion gates通過且使用者再次核准切換前，model1450仍是branch內正式default與
  rollback，formal publication collection也尚未開始。
- 新增`configs/slam_confidence_sim_release_v1.yaml`，固定branch-only／no-main-merge邊界、
  model1450 rollback、共同estimator15與confidence artifacts、A/B/C/D identity及promotion
  gates。C checkpoint現已以相同SHA curate到tracked `checkpoints/`路徑，但不得把artifact
  readiness誤寫成已promotion。
- 新增B/D可執行artifact exporter。B是frozen model1450 backbone在deterministic safe command
  上的51-D control；D是model48 learned intent-only，structured gait delta精確為零。暫定輸出
  位於`exported/anymal_d_locomotion_slam_confidence_publication_v1/`，B/D ONNX SHA分別為
  `9f485d39...ec8b`與`72c9e534...a0d`；module→ONNX最大誤差分別`2.861e-6`與`5.007e-6`，
  TorchScript誤差皆0，ROS external ONNX backend 51-D smoke亦通過。第一階段code／provisional
  artifacts已提交並push為`9b015c7 建立 SLAM confidence 發表與正式化基礎`；其後已從clean
  `9b015c7`重產B/D，ONNX SHA保持不變，metadata SHA更新為B
  `bd22d7a3...1df5`、D `31920173...2e8f`。正式收數前仍須提交這批重產artifact與hash manifest；
  export目錄符合`.gitignore`的`exported/*`，提交時須明確force-add。
- 新增offline mechanism sidecar：由實際policy diagnostics observation與model48 checkpoint
  逐幀重建safe scale、intent blend、raw/applied stride、crouch、stance width、action smoothing、
  structured-delta L2及A/B/C/D actions，並以live ONNX raw action作一致性gate。實際LIO-SAM
  `forward_1_5` 735 records全部finite，checkpoint reconstruction對live ONNX最大誤差
  `7.153e-7`，D structured delta精確為零。
- Mechanism sidecar已接入locomotion matrix runner：cell identity另綁checkpoint/agent config
  SHA，sidecar missing、stale、nonzero return code或action reconstruction超差都會使cell失敗。
  一格excluded LIO-SAM forward integration smoke已通過driver、硬安全、51-D consumption、
  confidence identity與mechanism gate；selection summary為
  `matrix_summary.selection-5a19625513cb.json`。此smoke不算正式publication樣本。

### 2026-08-14 to 2026-08-17 interactive manual observation and publication gate

- LIO-SAM interactive 啟動後已重現一個與定位品質無關的 false stop：開啟官方
  RViz 時，feature `CloudInfo` 可約20 Hz輸出，而受`mappingProcessInterval`節流的
  mapping odometry只對其中部分stamps輸出。舊assembler把feature/incremental-only
  stamps也建立成expected mapping bundle，因而持續輸出score `1.0`但
  `tracking_valid=false`、`SIGNAL_MISSING/RECOVERY_PENDING`，使model48正確地fail
  closed而完全不走；同一pipeline在headless control因收到的topic rates接近而可進
  TRACKING。Project-owned assembler現改由native/canonical mapping odometry anchor
  bundle candidate；auxiliary-only stamps仍可exact join，但不再單獨宣告mapping source
  遺失。`0.20 s` grace、exact-stamp requirements、source freshness與artifact均不變；
  52項core/integration/contract/estimator測試（含anchored missing fault regression）通過。
  重新build後以domain 1執行GUI＋官方RViz interactive smoke，穩定後實測
  `tracking_state=TRACKING`、`tracking_valid=true`、confidence `1.0`、reasons `0`，
  已確認原本持續false stop消失。
- 2026-08-17使用者完成LIO-SAM interactive人工行走，主觀觀察為行走快速、點雲沒有裂圖，
  且明顯快於先前FAST-LIO2人工操作。該launch未覆寫artifact，確定載入model48 51-D ONNX
  （SHA `c8225b05...f4a5`）與estimator15；正式policy仍為model1450，model48＋estimator15
  仍只是experimental candidate。LIO healthy confidence接近1時，model48依架構精確回到
  frozen model1450 healthy path，因此這次結果證明LIO healthy deployment鏈路可用，不能
  單獨證明low-confidence learned gait adaptation。
- 兩次interactive diagnostics不是受控速度比較：FAST記錄4.0 s、active平均平面command
  `0.622 m/s`、實際速度`0.378 m/s`；LIO記錄25.5 s、command `1.914 m/s`、實際速度
  `1.562 m/s`。active confidence平均FAST/LIO約`0.793/0.792`，tracking-valid比例約
  `0.793/0.797`。LIO command本身約為FAST三倍，故不可由主觀速度差推論FAST尚未調好；
  必須使用相同command trace、route、seed與duration重測。
- 使用者以新interactive入口人工操作FAST-LIO2，最高yaw command約`2.5 rad/s`。RViz觀察到
  旋轉後極小、短暫的點雲錯位，繼續行走後視覺上重新重合。這是人工觀察，尚未由bag或
  map-consistency metric量化；目前FAST-LIO2設定沒有loop closure，因此不可歸因為閉環修正，
  較可能是後續scan-to-map registration與registered-cloud更新造成局部重新對齊。
- FAST-LIO2同一次人工操作主觀感受為：feature不足或confidence下降時policy容易停下，整體控制速度
  偏慢。這證明fail-closed鏈路有介入，但是否過度保守尚未量化；不可只憑主觀操作promote或
  否定研究方向。LIO-SAM interactive manual arm其後亦已完成，結果如上所述。
- 論文必要的新gate是安全性與效率的Pareto比較，而非只報告跌倒率。至少比較
  confidence-unaware model1450、model1450＋相同deterministic confidence supervisor、
  confidence-aware model48三組；固定相同backend／route／command／seed，記錄fall/base
  contact、foot slip、tilt/action-rate、command tracking、distance/progress、平均速度、
  confidence-triggered stop比例、false-stop時間、recovery latency與SLAM ATE/RPE。裂圖另需
  定義map-level consistency metric；現有confidence只預測短期tracking usability，不能當
  explicit split-map detector。
- 工程上的closed loop已接通，但publication claim尚未完成。除前三組外，宜加入關閉四個
  learned gait coordinates的intent-only ablation，並同步記錄safe scale、intent blend、
  stride/crouch/stance-width/smoothing coordinates，證明`confidence下降→步態改變→機身晃動
  /foot slip下降→SLAM survival提高→仍保有效率`。現有simulation gait-value A/B是支持證據，
  但不能取代真實backend closed-loop重複實驗與confidence interval。
- 2026-08-17已建立第一版凍結protocol：機讀設定為
  `configs/slam_confidence_publication_protocol.yaml`，說明為
  `docs/validation/slam_confidence_publication_protocol.md`。Live matrix固定A/B/C/D四arms、
  兩backend、四條route、native／controlled-support-loss兩conditions與五個paired blocks；
  replay明確只作matched-input backend／量測可重現性，不能冒充closed-loop gait causality。
  正式收數仍未授權；B/D hash-locked artifacts、mechanism sidecar、live gradual challenge與
  完整excluded smoke block、false-stop offline label與map-consistency evaluator現已完成，
  仍缺sample-size justification與其餘formal readiness gates。
- 2026-08-17 publication執行鏈已補上共同`simulation_seed`、兩backend完全相同的
  `point_density/profile/min`傳遞與raw sensor/command/reset/SLAM/confidence bag錄製。
  `scripts/validation/run_slam_confidence_publication_matrix.py`會hash-lock A/B/C/D與
  estimator15、產生固定320-cell balanced schedule，並把block 43..47直接作為Isaac Lab
  seed；預設資料角色固定為`excluded_smoke`。`--formal`在protocol未授權、artifact SHA不符、
  tracked worktree不乾淨或輸出不在`outputs/slam_confidence_publication_v1/`時
  fail closed；執行時的clean HEAD直接寫入run manifest，避免在被hash的release文件中建立
  不可能的self-referential commit hash。B/C/D mechanism sidecar也已能依實際executed arm逐sample重建action；A為48-D
  confidence-unaware control，不偽造51-D mechanism sidecar。完整excluded smoke block已如下一項
  完成。其後runner已把offline future-usability／false-stop與map-consistency設為每格必要
  measurement-completeness gates；尚未實作run-level publication statistics。
- 2026-08-17 block43 `lateral_right_1_5`完整excluded smoke已完成：兩backend × native／
  gradual × A/B/C/D共16/16 cells通過，raw bags合計120,656 messages；B/C/D action sidecar
  最大重建誤差`1.66893e-6`（`atol=1e-5`），D structured delta exact zero。受控loss在driver
  中明確作為experimental outcome保留，不再誤判成infrastructure failure；identity、timestamp、
  unknown reason與logical publish gap仍維持hard checks。機讀摘要為
  `docs/validation/slam_confidence_publication_excluded_smoke_block43.json`。這只證明harness與
  單一excluded block可用，不是正式effect-size、promotion或完整四route／五block證據。
- 同一block43的16格raw bags已全數通過offline usability evaluator：4,512個labels，
  10,972／10,972個policy records成功配對，1,497個known-label confidence-stop samples中
  有3個false stops。GT只在run結束後形成future-horizon label；runtime policy與confidence
  extractor均未讀GT。這些excluded數字只證明false-stop定義及資料鏈可執行，不可作正式
  noninferiority結論。
- 已新增map-consistency core與每格bag evaluator。它用相同raw scan samples、已凍結的
  base-to-lidar平移`(0.20,0,0.35)`，分別由offline GT與canonical SLAM pose建立
  route-observed Factory surfaces，再以單次SE(2)＋Z alignment計算reference-distance
  p50/p95、off-reference與duplicate-surface fractions。合成no-split與0.25 m split-map四項
  gate全過（`docs/validation/slam_confidence_map_consistency_synthetic.json`）；FAST-LIO2與
  LIO-SAM各一個native arm C excluded bag亦端到端通過measurement-completeness gate。
  該gate刻意不套efficacy threshold，不能把單次smoke值解讀成backend排名或promotion；
  threshold／confidence interval必須由完整paired formal matrix決定。
- Publication run-level資料鏈也已實作：每格將locomotion diagnostics、offline usability／
  trajectory、map consistency與mechanism sidecar聚合為一個`accepted_live_run` record，
  frame samples明確不作independent replicates。分析器對C–D／C–B／B–A輸出continuous
  paired differences、binary risk differences、效率ratio及backend interaction，固定以
  `(paired_block_id, profile)`做10,000次cluster bootstrap；正式claim需320個唯一run
  identities精確匹配frozen schedule且所有run gates通過。既有16格excluded smoke已全部
  backfill並完成descriptive analysis，但因只有block43且role為excluded，分析器固定拒絕
  formal claim／complete support。sample-size justification仍未完成。
- Publication replay runner已實作：每個accepted live raw bag以content SHA fingerprint，
  分別進FAST-LIO2／LIO-SAM native backend各兩次；正式320 live runs會形成1,280 replay
  cells。Replay永久標記不可作closed-loop gait causality。第一個excluded source smoke為
  4/4通過：FAST兩次137 mapping samples且ATE同為`0.143664 m`；LIO兩次138 samples，
  ATE `0.038180/0.044856 m`（差`0.006677 m`），yaw RMSE差`0.004102 deg`。因此counts、
  schema、finite與timestamps屬hard gate，數值差異則完整量化為backend variability，不以
  任意closeness threshold排除；FAST超過舊0.10 m qualification threshold亦保留為outcome。
  機讀證據為`docs/validation/slam_confidence_publication_replay_excluded_smoke.json`，完整
  replay matrix仍待formal live data後執行。
- Sample-size gate已凍結disjoint pilot，不再允許用formal outcomes事後決定樣本數：pilot
  blocks/seeds為`143..147`，formal為`43..47`，四routes×gradual×兩backend×A/B/C/D共
  160格，role固定`excluded_pilot`且永不併入formal efficacy。Precision planner要求完整
  160個唯一records與all gates passed，再以10,000次stratified simulation比較預註冊的
  slip `0.02 m/s`、roll/pitch-rate `0.05 rad/s`、RMST `0.50 s`及log-efficiency ratio
  `0.05` half-width目標，候選formal blocks為5到30，效率下界仍須`>=log(0.90)`。
  Pilot schedule dry-run為160/160；目前尚未執行這160格，因此sample-size justification
  與formal collection authorization仍不可通過。
- model48 checkpoint已由ignored training log以相同SHA `888bc682...4fa0` curate到tracked
  `checkpoints/anymal_d_locomotion_slam_confidence_sim_v1/model_48.pt`，publication protocol、
  release manifest與mechanism sidecar的正式dependency均改用此路徑；export metadata保留的
  原logs source path只作training provenance。這關閉「clean commit沒有model48 checkpoint」
  的artifact缺口。
- 新publication raw-bag contract加入`/foot_contacts`與`/locomotion/estimated_odom`，並新增
  estimator15 offline replay parity：由同一ONNX、policy joint mapping與deterministic
  source-stamp bundle重建IMU/joint/contact 20-step history，逐exact stamp對live estimator output，要求match比例
  `>=0.95`與最大誤差`<=1e-5 m/s`，全程不讀GT。舊block43 bags缺少新增topics，故只能保留
  為舊schema smoke；新的160-cell disjoint pilot每格必須通過此gate，才能關閉common estimator
  foot-contact replay readiness。
- model48＋estimator15的既有五方向Isaac Lab recovery behavior evidence已curate並以SHA鎖定；
  forward／left／right／reverse／combined各為512 env、1000 steps且使用五個不同seed，所有
  behavior checks通過。總計2560個環境中6個曾hard terminate，pooled fraction為`0.00234375`，
  各profile亦低於凍結的1%上限。機讀gate為
  `docs/validation/slam_confidence_model48_estimator15_regression_gate.json`。這關閉simulation
  recovery regression gate，但不取代尚未執行的pilot、正式雙backend matrix或promotion審查。
- 在clean baseline `d1128fe`上執行第一個new-schema excluded-pilot qualification cell
  （FAST-LIO2／lateral-right／gradual／block143／arm C）後，live driver、policy diagnostics、
  `/foot_contacts`與stability資料均產生，但整格正確fail-close。原因有二：publication runner的
  offline child process未明確加入project source `PYTHONPATH`；policy內建estimator雖供inference
  使用，卻未發布bag contract要求的`/locomotion/estimated_odom`。目前修正為offline evaluator
  依Python ABI分離環境（Isaac mechanism不注入ROS vendor；estimator replay才注入deployment
  vendor），並由policy node直接發布實際供inference使用的同一estimate與joint exact stamp。
  第二次excluded smoke證明topic與環境修正有效，但685個exact-stamp outputs中有48個因policy
  subscriber與bag recorder的跨topic DDS callback順序不同而產生最大`0.03837 m/s`誤差；gate
  未放寬並正確拒絕。Runtime與replay現共用source-stamp synchronizer：每個joint stamp等到IMU
  與contact的per-topic watermark後，再以nearest stamp和固定`0.025 s` tolerance組bundle，tie
  固定選較早stamp，timestamp regression會fail-close，因此不依賴跨topic callback順序。
  後續永久排除的FAST-LIO2與LIO-SAM A/B/C/D共8格smoke均完整通過：合計5486個replay
  outputs全部exact-stamp match，最大速度誤差皆為`0.0 m/s`，stability、mechanism（A不要求）、
  offline usability、map measurement與run record亦全過。機讀證據為
  `docs/validation/slam_confidence_estimator15_stamp_sync_qualification.json`。前兩次失敗cell只作
  qualification/debug且永不納入pilot；完整160-cell pilot仍須在修正的clean commit後逐格通過。

### Proprioceptive estimator and gait-value A/B (2026-08-13)

- 普通flat navigation的clean proprioceptive estimator candidate 08已通過accuracy與
  256-sample TorchScript/ONNX parity。Artifact位於
  `exported/proprioceptive_velocity_estimator/v1/clean_candidate_08/`；runtime只使用
  IMU、projected gravity、joint position/velocity與四腳contact的20-step history，GT
  velocity仍只作supervised label與offline evaluator。正式model1450 closed-loop為
  2/256 distinct hard-terminated env（0.78125%），因此可用於下一輪模擬A/B，但仍是
  實驗產物，不取代任何正式ANYmal state estimator。
- 已凍結`configs/slam_confidence_gait_value_gate.yaml`，把「51-D的價值」定義成相對於
  model1450加相同confidence command limiter，在degraded＋invalid階段的action-rate、
  body tilt、roll/pitch rate、vertical speed與stance-foot slip五項平均至少改善5%，任何
  單項不得惡化超過10%；healthy/recovery tracking與gait亦不得顯著退步，distinct hard
  terminated env fraction必須`<=1%`。
- 同seed43、512 env、1000 steps、固定`vx=1.5`、相同confidence schedule、相同candidate
  08 estimator且push/external-force關閉的A/B已完成。A組正式model1450加純command limiter
  雖通過原stop/recovery behavior gate，但71/512環境曾base-contact。B組bounded
  safe-command model19降為5/512，healthy與recovery未退步；然而degraded＋invalid gait
  composite ratio為`1.60595`（越低越好），invalid action-rate/vertical-speed/stance-slip
  分別為baseline的`1.8908/4.4845/1.8418`倍。因此model19未通過新增gait-value gate，
  不可promote。機讀報告是
  `docs/validation/slam_confidence_gait_ab_model19_value_gate.json`。
- 已實作新task
  `Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-Gait-v0`。Actor由凍結model1450
  backbone、0.8-bounded exact safe-command gain與只在low-confidence啟用的0.25-bounded
  gait residual組成；gain及residual output皆exact-zero-init，所以iteration 0對所有
  confidence輸入仍與model1450相同。Critic與optimizer皆fresh。Residual由PPO依明確的
  low-confidence action-rate、vertical motion、roll/pitch motion、tilt與stance-slip rewards
  學習；safe gain另由deterministic teacher更新。此task明確關閉push與external force。
- 新task的64-env真實Isaac Sim instantiate及bootstrap-only成功。Iteration-0 checkpoint為
  `logs/rsl_rl/anymal_d_locomotion_slam_confidence_gait_v1/2026-08-13_12-17-22_gait_iteration0_bootstrap/bootstrap_model_0.pt`。
  同一正式behavior profile加candidate08 estimator的baseline為healthy/invalid speed
  `1.47195/1.47565 m/s`、degraded ratio`1.00184`、0 hard termination；如預期保留
  model1450並尚未學會confidence stop。報告位於
  `docs/validation/slam_confidence_gait_iteration0_baseline.json`。
- 64-env、1-iteration update smoke成功；model_0 safe gain從0更新為`0.0249859`，residual
  final-layer weight norm從0更新為`0.0286741`，證明safe teacher與reward-driven gait
  residual兩條更新路徑皆有實際更新。Smoke不屬候選。
- 第一個safe-gait主run為
  `logs/rsl_rl/anymal_d_locomotion_slam_confidence_gait_v1/2026-08-13_12-19-57_gait_candidate_100iter_seed42_4096env`，
  從正式model1450重新actor-only warm-start，4096 env、seed42、100 iterations、
  9,830,400 simulation steps。model99通過原behavior gate且0/512 hard-terminated env，
  但invalid residual仍造成action-rate、vertical-speed與stance-slip相對純limiter為
  `2.390/6.613/2.675`倍，gait composite ratio `1.97065`，未通過gait-value gate。
  此run與checkpoint均retire，不export、不進backend matrix。
- 依上述失敗做單一因果修正：gait residual改為只在safe-scale介於0與1的deceleration／
  recovery transition啟用，healthy與fully-invalid由架構強制為0；safe-command gain target
  改為1.0，使settled invalid直接使用正式model1450 zero-command actor。新iteration-0
  checkpoint位於
  `logs/rsl_rl/anymal_d_locomotion_slam_confidence_gait_v1/2026-08-13_12-24-12_transition_gait_iteration0_bootstrap/bootstrap_model_0.pt`，
  同profile baseline再次重現healthy/invalid `1.47195/1.47565 m/s`與0 termination。
- transition-gated主run為
  `logs/rsl_rl/anymal_d_locomotion_slam_confidence_gait_v1/2026-08-13_12-25-30_transition_gait_candidate_100iter_seed42_4096env`，
  同樣從model1450重新開始，4096 env、seed42、100 iterations。model99 SHA-256為
  `4dbfc4789644a1ea4eeaedb020975300d4ece49de0809ee44ff8181455cc2c64`，gain為
  `0.994369`，但23/512 distinct hard-terminated env（4.492%）超過1%。中段checkpoint
  sweep只測必要的model40與model60：model40為2/512但gait composite `2.64022`；model60
  為31/512且composite `1.57289`。三者皆fail，證明目前架構在不完整停止與過快切到完整
  zero-command間仍有明確trade-off。停止sweep與訓練；不得export、不得進backend matrix、
  不得取代正式model1450。對應機讀報告位於`docs/validation/slam_confidence_*gait*`。
- 下一個設計不得只調reward或續訓。需要把confidence transition本身做成具有rate limit／
  dwell的policy-side gait-mode state，或使用能直接控制步幅／抬腳高度／stance posture的
  結構化gait parameter head；仍須exact model1450 healthy path、fresh critic/optimizer、
  iteration-0 baseline與同一凍結gait-value gate。正式model1450維持不變。
- 已開始實作下一版policy-side gait-mode governor，但尚未執行Isaac Sim gate或訓練。
  v1使用`TRACK/DECELERATE/HOLD/RECOVER`四狀態、confidence hysteresis、hold/recovery
  dwell，並從confidence開始下降就追蹤continuous safe scale；command-scale降速上限為
  `0.8/s`、恢復上限為`2.0/s`。狀態保存在各simulation environment與ROS
  `PolicyRuntime`，不放入stateful ONNX actor。候選actor只在governor改寫command後執行
  frozen model1450 backbone，沒有12-D residual。Torch與deployment NumPy core的逐tick
  parity、stale-high、chatter、reset及runtime observation接點單元測試已通過；設計與待跑
  gate見`docs/validation/slam_confidence_gait_mode_v1.md`。這仍不是候選promotion證據，
  下一步必須先保存iteration-0 bootstrap並跑相同behavior/gait-value A/B，不得直接長訓練。
- 2026-08-13 gait-mode B architecture gate已完成且失敗。晚門檻、`2.0/s`版本雖然
  invalid settled gait穩定，但degraded tracking未過且9/512 distinct env hard terminate；
  改成從confidence開始下降即追蹤continuous safe scale、降速上限`0.8/s`後，velocity
  checks全過但hard termination惡化到89/512，recovery roll/pitch rate RMS為
  `0.925 rad/s`。因此停止rate sweep；B證明外部限速有作用，但不是可promotion解法。
  evaluator已修正為用distinct terminated env fraction執行1%安全門檻，並新增phase counts。
- 2026-08-13 已實作C structured-gait PPO head。raw command不由confidence governor限速；
  frozen model1450之外只學4-D gait coordinates：stride modulation、crouch、stance width、
  action smoothing，再透過固定canonical 12-joint basis形成動作。healthy severity為零，
  因此即使head訓練後仍精確走model1450。64-env seed43 iteration-0 bootstrap SHA為
  `5f1c0b48...b0df`；單一PPO iteration產物SHA為`d3f1d599...e3e3`，6/6 head tensors
  有更新、8/8 backbone tensors bit-identical。這只證明訓練接線，尚未通過behavior、
  safety或gait-value，不能export或取代正式policy。詳細見
  `docs/validation/slam_confidence_structured_gait_v1.md`。
- 2026-08-13 C-v1固定pilot已完成並失敗：512 env、seed43、25 iterations，final
  `model_24.pt` SHA為`10fa91ba...b1d`。固定behavior gate為0/512 hard termination，
  但invalid-settled仍以`1.500 m/s`前進、100% samples高於`0.5 m/s`，degraded speed
  ratio為`1.013`，表示4-D posture/stride/width/smoothing head維持了平衡卻沒有學會
  confidence-conditioned stop。停止gait-value、export、續訓及事後挑中間checkpoint。
  下一版不可只調reward；應新增一個由PPO共同決定的low-dimensional locomotion-intent
  coordinate，在frozen raw-command與safe-command model1450 action間blend，同時保留
  balance gait coordinates。這不是B的外部固定限速器。
- 2026-08-13 C-v2/C-v3 intent-gait固定pilots亦未過behavior gate。兩者皆為512 env、
  seed43、25 iterations且只評估預先指定final model24；C-v2 gain1為0/512 terminate、
  degraded ratio `0.982`、invalid speed `1.453 m/s`；C-v3 gain20為0/512、`0.962`、
  `1.390 m/s`。相較C-v1方向正確但幅度遠不足，證明intent coordinate可達、PPO也有
  使用，然而現有間接reward在固定budget內推力不足；禁止再做gain/checkpoint sweep或
  單純續訓。下一個合理實驗需要明確的intent auxiliary learning signal，而PPO仍負責
  四個balance coordinates；必須另立設計gate。詳細見
  `docs/validation/slam_confidence_intent_gait_v1.md`。
- 2026-08-13 C-v4將intent auxiliary與四個PPO gait coordinates分離後，使用GT velocity
  評估為0/512 hard termination，但接candidate08 estimator為78/512，確認主要缺口是
  training/evaluation state-estimator closure。現已實作`VelocityEstimatorTrainingWrapper`，
  在rollout內執行完全相同的20-step TorchScript estimator、warmup時action fail-closed為0，
  reset清history，並把artifact SHA/contract寫入run manifest；不在Isaac環境import ROS。
- C-v5 estimator-closed-loop直接訓練反而107/512跌倒；C-v6加入termination與recovery
  action-rate/roll-pitch/orientation costs後降為61/512；C-v7再限制stride與smoothing只能為正，
  model24降至17/512。C-v8 smoothing gain10惡化至63/512，停止gain sweep。C-v7固定續訓25
  iterations的model48（SHA
  `888bc682eda65e2c427eccd445d8b9cb1e787635b5da86e6fa8a85b83fda4fa0`）達0/512並通過
  behavior gate，但degraded stance slip `0.14052 m/s`超過相對model1450的1.10門檻。
  C-v9加degradation slip reward後model62仍0/512但slip惡化至`0.14985 m/s`，已retire。
- model48因果ablation證明crouch不是主因；全域zero stride可把degraded slip降至
  `0.11692 m/s`，卻造成78/512 recovery跌倒；stride 0.5仍為9/512且slip `0.13069 m/s`。
  因此不能用單一全域scale，必須分開degradation與recovery。C-v10在policy內用deployable
  `[confidence, valid, normalized_age]`辨識live degradation與post-outage recovery，僅對前者
  以cubic envelope連續衰減PPO stride coordinate，intent與其他三個gait coordinates仍照PPO
  輸出；不是policy外部限速器，也不使用sim phase或GT。
- C-v10固定seed43、512-env、1000-step behavior gate已全部通過：2/512 distinct hard
  termination（`0.390625%`）、degraded stance slip `0.117835 m/s`；degraded/recovery gait
  coordinates約為`[0.0006,-0.0542,0.0155,0.1805]`與
  `[0.1451,-0.0602,0.0185,0.1999]`。相對model1450＋相同confidence supervisor的
  gait-value gate九項checks全過，degraded＋invalid composite ratio `0.770354`（門檻
  `<=0.95`），最差單項ratio `1.05730`（門檻`<=1.10`）。完整suite為336 passed、3個既有
  Isaac-runtime skips；`git diff --check`通過。機讀報告在
  `docs/validation/slam_confidence_phase_separated_model48_{behavior,gait_value}.json`，詳細因果
  與失敗候選見`docs/validation/slam_confidence_intent_gait_v1.md`。Export parity與
  FAST/LIO完整五方向live matrix已通過；尚未完成正式promotion、commit或push，正式policy
  仍是recovery v0.4.0 model1450。

2026-08-13 已完成第一版 proprioceptive velocity gate。Project-owned v1 契約固定
50 Hz、20-step／0.4 s history、每步 37-D：IMU angular velocity／linear
acceleration／projected gravity、12 joint position／velocity與四腳contact；輸出是
`base_link` frame `[vx,vy,vz]`。GT velocity只可作simulation supervised label與
offline evaluator，模型runtime不含GT、SLAM pose或SLAM twist。已實作資料collector、
environment-disjoint window/split、windowed GRU training與ONNX/TorchScript export、
機讀accuracy gate、fail-closed ROS estimator，以及不猜測ANYbotics私有schema的vendor
Odometry adapter骨架。人工push與external force從普通flat estimator gate關閉後，clean
random seed44與stopped seed45共230,400 rows；candidate08 XYZ MAE為
`0.03356/0.03432/0.02231 m/s`、vector p95 `0.13508 m/s`、max vector error
`0.91619 m/s`，accuracy與TorchScript/ONNX parity通過。正式policy與model19均未切換；
candidate08只作後續模擬qualification的實驗性state estimator。

可控 reset gate、三次 LIO-SAM map-quality benchmark、修正版 12/12 loop-closure
matrix，以及 FAST-LIO2 input contract、parameter audit 與五個主要 translation
ATE holdout 都已完成。Confidence／tracking-valid／age v1 wire schema、兩backend
extractor、native-only calibration/fresh holdout、正式artifact與runtime replay均已
完成；兩backend目前都具備PPO-ready資料介面。FAST-LIO2新baseline的live多圈yaw／
lateral qualification與兩backend ROS topic/DDS hard-fault injection均已完成。
`feature/ppo-slam-confidence`的dimension/name/order/export/runtime parity、第一次PPO、
第二輪actor-only起點與固定behavior gate均已完成；reward-only model49與失敗的residual／
teacher變體均已retire。Bounded safe-command `model_19`保有iteration-0 healthy parity，並已
通過五profile behavior qualification與同checkpoint export parity。兩backend minimum live
screening結果為FAST pass、LIO fail。LIO policy-state contract現已freeze並實作：分離的高頻
native predictor讓direct state quality達3/3，但model1450停止尾端stability與action sensitivity
仍0/3；pose regression arm也已明確失敗並移除。獨立proprioceptive locomotion state
estimator現已接入model48並通過兩backend完整live matrix；仍不得據此promote候選。
Recovery v0.4.0仍是正式policy。

Recovery v0.5 的 `curve_3_0_left_0_5 <= 0.2` 仍是未來 candidate 的必要
gate，但目前不是 bridge fix 發布前置條件，也不可用未通過的 model2420
取代正式 v0.4 policy。

Watchdog deceleration 2.0 已證明對明確 zero packet failure 無效，正式
bringup 預設維持不變。

## Handoff maintenance checklist

每次準備切換到新對話時：

1. 更新本文件的日期、Git baseline 與 dirty worktrees。
2. 同步正式 policy；實驗候選必須分開列出。
3. 更新最新完成的修正、診斷結論和 validation gate。
4. 移除已解決的 unfinished item，加入新發現的阻塞。
5. 寫清楚下一個 log、測試或使用者決策。
6. 再產生可貼入新對話的摘要。
