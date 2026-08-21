# Current project knowledge

更新日期：2026-08-21

這是新對話的短版入口，只保存目前有效狀態、邊界、最新結論與下一步。完整截至
2026-08-21的舊版交接內容逐字保存在`docs/project_history.md`；個別實驗的方法、數字與
失敗原因仍以`docs/validation/`及機讀artifact為準。新對話先讀本文件與repository實作，
只有當目前任務需要追查特定歷史時才讀對應的history／validation文件。

## Repository state

- 根目錄：`/home/ros/anymal_locomotion`；目前branch為`exp/slam-fastlio2`。
- 本機HEAD為包含本文件的`建立動作介入風險試驗與精簡專案交接`commit；精確SHA以
  `git rev-parse HEAD`為準。遠端仍為`bc88e75`，本機ahead 2，沒有push授權。`main`與
  `origin/main`仍為`98b43dd`，不得為目前論文工作修改或合併`main`。
- Formal analysis修正、800-cell診斷、risk predictability audit、action-risk intervention
  pilot的config／runner／analyzer／測試、短版current knowledge與完整history均已納入
  `11f3931`。目前沒有已知project-owned source／config／docs dirty；每次後續commit／push前
  仍必須重新整理並取得使用者明確同意。
- Root `build/`、`install/`、`log/`、`lidar_type`及其他`logs/`／`outputs/`是runtime或
  實驗產物，不納入提交。
- `AGENTS.md`是`.gitignore`中的本機規則，不可stage或push。
- Nested LIO-SAM `deployment/ros2_ws/src/lio_sam/config/rviz2.rviz`有使用者既存dirty；
  SHA-256為`0c3a25d41df63d9fa711af98845f01d99856f27b3dc8ea05fc11fcddee82adc1`，
  不可還原、修改、stage或提交。Nested FAST-LIO2的visualization patch與runtime logs亦不可
  混入root提交。

## Non-negotiable boundaries

- 除非使用者明確要求，不修改`/home/ros/IsaacLab`；絕不修改舊工作區
  `/home/ros/Documents/anymal_project/anymal_ws`。
- Isaac Lab training與ROS 2 deployment分層；Isaac環境不import `rclpy`，最終架構不用UDP。
- 維持deterministic joint-name-to-policy-index mapping；模擬器專屬邏輯與實體deployment分離。
- 不以提高摩擦、硬切nominal joint pose或單純限速掩蓋policy問題。
- 不啟動3,200-cell replay、model48 promotion、default switch或實體ANYmal-D宣稱，除非使用者
  另行明確授權且前置gate通過。

## Formal artifacts and status

- 正式locomotion policy仍是Recovery v0.4.0 model1450：
  `exported/anymal_d_locomotion_v1/recovery_v0.4.0/policy.onnx`；checkpoint為
  `checkpoints/anymal_d_locomotion_v1/recovery_v0.4.0/model_1450.pt`。
- 正式confidence artifacts：FAST-LIO2 `native-v1-1e6cf8347be1`，LIO-SAM
  `native-v1-edc098b0bd98`。兩backend共同輸出canonical `/slam/odom`、atomic
  `/slam_confidence`及tracking-valid/age contract；GT只可作offline label。
- Proprioceptive velocity estimator15是實驗鏈共用artifact，不是正式ANYmal estimator：
  `exported/proprioceptive_velocity_estimator/v1/clean_lateral_transition_candidate_15/`。
- C-v10/model48通過既有simulation behavior、gait-value、export parity及雙backend五方向live
  matrix，但800-cell正式結果否定其額外learned gait價值；不得promotion或續訓挽救。
- B是model1450加deterministic confidence command supervisor，為後續最強simple baseline與
  invalid/stale hard-safety boundary，不取代正式model1450。

## 800-cell formal result

- 25 paired blocks（443..467）、兩backend×四profiles×A/B/C/D，共800/800 accepted live
  cells完成。Formal schedule、100個`(block, profile)`clusters及10,000次bootstrap完整；
  accepted-only結果在ignored
  `outputs/slam_confidence_publication_v1/formal_analysis_v1/`。
- Primary C−D不支持舊C的learned gait價值：slip差`-0.000492 m/s`、body-rate差
  `+0.002432 rad/s`、tracking RMST差`-0.0442 s`，區間皆不支持完整因果鏈；
  `complete_support=false`。
- C−B顯示舊C不如B：slip`+0.01441 m/s`、body-rate`+0.01824 rad/s`，兩者95% CI排除零；
  progress與tracking survival沒有可靠補償。舊C不可promotion，也不可只調reward或續訓。
- B−A證明deterministic supervisor是強基準：slip`-0.25809 m/s`、body-rate
  `-0.11137 rad/s`、tracking RMST`+2.047 s`；代價是completion下降18 percentage points。
- 分層診斷顯示C−D在matched speed沒有恢復價值；舊C主要在3–9 s degradation onset過度積極。
  Nominal healthy 0–3 s早於5 s command warmup，後續protocol必須對齊active-motion healthy
  control與精確phase origin。
- 詳細正式與exploratory解讀：
  `docs/validation/slam_confidence_publication_stratified_diagnostics.md`。

## Existing-800 temporal predictability audit

- Audit只讀accepted JSON artifacts，不讀bag、不跑ROS／Isaac／PPO。A的48-D diagnostics缺少
  SLAM state而排除；B/C/D 600 runs中596有合格causal window，共108,866個10 Hz prediction
  ticks。推論以完整`(block, profile)`cluster bootstrap，不把ticks當獨立實驗。
- 固定比較1 s SLAM-state-only、motion-without-estimator-velocity及with-estimator-velocity
  histories；用equal-run weighted L2 logistic regression做block 5-fold與leave-one-arm-out。
- 原定「0.5 s hazard＋invalid後1.0 s recovery一起做新C」的事前gate為`FAIL`。Motion history
  對hazard有development signal，但recovery不穩：FAST block recovery Brier惡化
  `+0.02456`，95% CI `[+0.00545,+0.04395]`；多個recovery cells的ECE亦惡化超過0.02。
- Estimator15 linear velocity幾乎沒有額外收益。此結果不推翻現有confidence；它只禁止把
  passive motion predictor當成新貢獻，或直接進joint hazard/recovery PPO。
- 詳細報告：`docs/validation/slam_risk_predictability_audit.md`；機讀結果與90 MB feature cache
  位於ignored `outputs/.../formal_analysis_v1/risk_predictability_audit_v1/`。

## Current research decision

- 論文目標是證明locomotion action能改變下一刻localization risk，而不是重新做一個SLAM
  confidence predictor。現有confidence已定義為未來0.5 s usability probability；被動資料中的
  motion-hazard關聯不等於action的causal controllability。
- 下一個唯一允許的研究步驟是小型simulation intervention pilot：在相同state／command／
  terrain／friction下，對bounded 12-D joint residual做事前凍結、平衡交叉配對的zero-mean
  intervention，含exact-zero control；測量action→body motion/slip→下一刻SLAM risk。
- FAST-LIO2與LIO-SAM必須分開報告；matched command是設計條件，matched speed應用於gate／
  分層而不是事後宣稱隨機因果。不得使用GT、future label、backend ID或intervention ID作
  runtime policy input。
- Pilot先凍結protocol、資料schema、futility gate與最小smoke，再決定是否執行。未證明
  intervention能降低risk前，不建立新student、不跑PPO、不做live qualification。
- 若pilot通過，才可研究action-conditioned transition/risk model：凍結model1450、zero-init
  bounded residual、既有B supervisor負責invalid/stale/recovery，並以zero/shuffle/delay
  ablations排除只是降速或讀到phase。未來live採12→24→72 staged gate，不直接投入800 cells。

## Current unfinished order

1. 凍結action-conditioned localization-risk intervention pilot protocol，明確定義intervention、
   matched conditions、SLAM outcome、mechanism endpoints、futility stop與資料角色。
2. Frozen protocol與純函式已通過validation；smooth／zero／antismooth三個experimental
   policies已hash-lock匯出並通過TorchScript/ONNX、zero Arm-B、invalid fallback及residual
   bound parity。6-run runner已接通real ROS backend launch、intervention trace、stability、
   offline 0.5 s usability、map consistency、estimator replay與run-level record；plan mode通過。
   預防性SLAM endpoint只計算「當下tracking仍有效且有requested motion」的ticks，判斷未來
   0.5 s是否變成unusable；已失效ticks不重複計入hazard，clock無法配對則完整性gate失敗。
   自動decision analyzer已凍結完整性、四strata方向、speed、safety與PASS／INCONCLUSIVE／FAIL
   規則；不完整或不合法records直接FAIL。Runner已硬性要求committed release逐stage授權、從raw
   records重算前一stage gate，且遇接線／非有限值／殘差超界／smooth安全失敗立即停止；不能
   直接跳過smoke執行pilot，也不能在pilot非INCONCLUSIVE時執行72-run expansion。
3. Smoke執行要求clean hash-locked worktree；先整理待提交內容並取得使用者另一次明確commit
   同意。經smoke確認接線後才執行24-run pilot；不默認進PPO或擴大stage。
4. 根據pilot決定研究路線：PASS才設計action-conditioned model；FAIL就停止這個新C方向；
   INCONCLUSIVE只允許一次預先限定的小幅補充，不做參數掃描。
5. 所有新結果更新本文件的current state，完整方法與數字寫入對應validation文件；不得把既有
   800當未來新C blind test。

## Detailed-record index

- 完整截至2026-08-21的舊交接內容：`docs/project_history.md`。
- 架構與部署：`docs/architecture.md`、`docs/ros2_deployment_decisions.md`、
  `docs/physical_anymal_d_integration.md`。
- Confidence contract：`docs/slam_confidence_contract.md`、
  `configs/slam_confidence_contract.yaml`、`configs/policy_contract_slam_confidence.yaml`。
- Publication protocol：`configs/slam_confidence_publication_protocol.yaml`、
  `docs/validation/slam_confidence_publication_protocol.md`。
- 舊PPO／gait候選完整歷史：`docs/validation/slam_confidence_ppo_second_round.md`、
  `docs/validation/slam_confidence_intent_gait_v1.md`、
  `docs/validation/slam_confidence_gait_mode_v1.md`。
- Backend驗證：`docs/validation/slam_backend_live_and_replay.md`、
  `docs/validation/slam_backend_comparison.md`。

## Handoff maintenance

每次切換新對話前：

1. 只在本文件修正目前有效事實、Git/dirty、正式artifact、最新gate與下一步；移除過時現況，
   不在此累積實驗流水帳。
2. 新實驗的完整方法、數字與失敗原因寫入主題validation文件；跨主題的重要歷史可追加到
   `docs/project_history.md`，不得覆寫或遺失既有紀錄。
3. 產生可貼入新對話的短摘要。新對話仍以repository與本文件驗證摘要是否過時；只按任務需要
   讀取history與詳細docs。
4. Commit／push仍需新的使用者明確同意；整理文件本身不構成授權。
