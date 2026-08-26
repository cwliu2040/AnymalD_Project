# Current project knowledge

更新日期：2026-08-26

這是新對話的短版入口，只保存目前有效狀態、邊界、最新結論與下一步。完整截至
2026-08-21的舊版交接內容逐字保存在`docs/project_history.md`；個別實驗的方法、數字與
失敗原因仍以`docs/validation/`及機讀artifact為準。新對話先讀本文件與repository實作，
只有當目前任務需要追查特定歷史時才讀對應的history／validation文件。

## Repository state

- 根目錄：`/home/ros/anymal_locomotion`；目前branch為`exp/slam-fastlio2`。
- 最近已提交並推送baseline為`c3db39f 完成 SLAM 動作介入與分量式因果驗證`；精確HEAD以
  `git rev-parse HEAD`為準，`origin/exp/slam-fastlio2`已同步到該commit。`main`與
  `origin/main`仍為`98b43dd`，不得為目前論文工作修改或合併`main`。
- Formal analysis修正、800-cell診斷、risk predictability audit、action-risk intervention
  pilot的config／runner／analyzer／測試、短版current knowledge與完整history均已納入
  baseline。其後兩版offline component risk model、fresh richer component-identification、
  command-preserving residual、touchdown-headroom／phase-estimator／pilot，以及full-policy
  joint-training與non-learning preflight均已整理為後續研究commits，不再是未整理的
  project-owned working-tree dirty；
  另有兩個component model reports、一個rejected v1 phase artifact，以及v2 development
  candidate與fresh-validation frozen artifact位於ignored
  `exported/`，未來commit時須明確判斷是否force-add。
  Live授權已重新關閉。後續在
  完整實驗段落結束後才整理一次commit；每次commit／push前仍必須重新
  整理並取得使用者明確同意。
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

- **2026-08-26 superseding decision:** the new formal target is original-command
  confidence-conditioned full locomotion-policy joint training／fine-tuning. Initialize the full
  actor from model1450, keep the exact unscaled requested XYZ/yaw command, add deployable
  confidence/validity/age plus command/body/joint/previous-action history, and learn the complete
  12-D joint action. Preserve linear/yaw tracking and progress while reducing unnecessary roll/pitch,
  high-frequency yaw error, acceleration/jerk, body vibration and LiDAR scan-time motion, then test
  future SLAM usability.
- Old C/model48 does not show that confidence cannot enter a locomotion policy. It failed because it
  used a confidence-scaled velocity target, making slowdown/stopping an easy PPO solution. The new
  target forbids that semantic: confidence/history may condition gait/body response but may not scale
  the requested command. B remains an invalid/stale safety fallback and risk--progress baseline.
- Freeze three formal comparators before training: J0 is original model1450; J1 uses the same
  vibration-aware training stack without confidence/history; J2 adds confidence/history. `J1-J0`
  tests generic smoother locomotion, while `J2-J1` is the localization-aware contribution. Evaluate
  all with fresh original commands and matched realized speed; report FAST-LIO2 and LIO-SAM separately.
- Full-policy fine-tuning must use model1450 behavior anchoring and prospectively frozen reward,
  curriculum and gates: strong command/yaw tracking and progress, body/LiDAR endpoints,
  STOP/SHUFFLE/POSTURE/LIMITER collapse, cadence, stance width, clearance, joint margin, torque,
  energy, slip, fall and base contact. Future SLAM/GT/internal diagnostics may be simulation-only
  privileged signals or held-out labels; runtime still forbids GT/future/backend/intervention/route
  phase/hardcoded timers.
- Do not hand-design another fixed foot/joint mechanism, retune touchdown 0/25/50% or phase thresholds,
  continue Risk v3, or build another scaling selector. The touchdown frozen pilot remains
  `INCONCLUSIVE`; blocks602..605 must not run. Its audit found that the three formal matched-motion
  failures were curved-route endpoint aliasing, but targeted physical response was still not
  repeatable, so the family is retired. See
  `docs/validation/slam_low_level_touchdown_headroom_v1.md`.
- Before new training, freeze corrected future metrics once: monotone/accumulated curved-route
  progress, paired zero-relative gait noninferiority plus absolute physical limits, and event-aligned
  body/LiDAR windows. Do not change these after seeing J1/J2 outcomes.
- This is currently a research objective, not execution permission. Teacher, history encoder,
  full-policy PPO, live ROS wiring, default switch and physical robot remain unexecuted and
  unauthorized. Blocks581..584 remain untouched; model1450 and B safety contracts remain unchanged.
- Joint-training static implementation now exists in
  `configs/slam_confidence_joint_training_v1.yaml`, `full_policy_bootstrap.py`,
  `joint_training_contract.py`, `policies/joint_training.py`, `algorithms/behavior_anchored_ppo.py`
  and the J1/J2 Isaac task configs. The actual model1450 checkpoint passed exact 48-to-1068 actor
  parity, exact frozen-reference copy and fresh-critic checks; behavior anchoring and the offline
  matched-motion block evaluator also have focused tests. No simulator was launched for these checks.
- J1/J2 now share a 1068-D dense actor: current 48-D input plus 20 oldest-to-newest 51-D history
  frames. J1 neutralizes only confidence/validity/age, while J2 receives those channels. Both retain
  the original command and use the same full random/edge command distribution, source locomotion
  reward stack, non-zero anchor and staged body/LiDAR distortion curriculum. The retired 51-D
  `--adapt-confidence-input-only` mode is explicitly rejected for this full-policy task.
- The offline block evaluator freezes exact command/timestamp identity, 0.05 m/s and 0.05 rad/s
  matched-motion guards, accumulated progress, absolute and paired anti-collapse gates, shared
  event-aligned body/LiDAR windows and held-out SLAM direction. It also prevents a false pass when
  both arms stop. Details: `docs/validation/slam_confidence_joint_training_v1.md`.
- The non-learning Isaac runtime preflight passed for both J1 and J2: each arm ran 8 environments ×
  100 steps with seed1450 and model1450 inference. The 1068-D layout, reset/history flattening,
  exact original command, J1 neutral localization and J2 confidence/validity/age contracts all
  passed; model1450 bootstrap action error was exactly `0.0`; termination/truncation counts were
  zero. Projected new-motion reward p99/max were `0.0473193/0.0622891`, below frozen `1.0/5.0`
  limits. The ignored merged report is
  `outputs/slam_confidence_joint_training_v1/nonlearning_preflight.json`, SHA-256
  `6c3ab914d91d57365ba78397c13b36828a5b413041fcdb848e5d11d20306b81c`, and explicitly records
  `ppo_constructed_or_run: false`.
- The runner isolates J1/J2 in separate Isaac child processes because destroying and recreating a
  process-global SimulationContext blocked during the first attempt; this is runner lifecycle
  isolation, not a gate or metric change. It prints per-10-step status and merges the arm reports.
  Both repository preflight authorization flags are closed again. Environment/reward wiring is now
  verified, but this does not establish efficacy. The next boundary is a separately authorized,
  already-frozen fixed-budget J1/J2 PPO run; PPO, live ROS wiring, default switch and physical robot
  all remain unauthorized.
- Motion reward equations were extracted to the torch-only
  `joint_training_motion_core.py`. Tensor tests prove zero added cost for constant translation and
  constant yaw with zero acceleration/jerk, positive cost for roll/pitch/non-constant scan motion,
  and stronger weighting under localization vulnerability. This reduces the risk of learning to
  reject normal turning before the runtime preflight.
- Current verification split before the runtime preflight: system-Python suite `503 passed, 15 skipped` after excluding the
  clean-worktree publication runner and separately executed PyTorch files; Isaac-Python PyTorch
  subset `36 passed`; focused joint-training static set `24 passed`. No simulator was launched.
- The custom algorithm has completed a real RSL-RL synthetic rollout/storage/return/PPO update,
  not only an unbound helper test. Losses were finite, storage cleared correctly and the frozen
  model1450 reference remained unchanged. Model1450 action noise is exact-copied and frozen for both
  J1/J2 so exploration variance cannot bypass the mean-action anchor; the full 12-D actor mean remains
  trainable. The subsequent non-learning Isaac preflight passed as recorded above.

## Superseded research notes retained for traceability

- 使用者已確認這不是另開新題目，而是回去完成原本只做一半的新C：不把B或learned component
  governor當主要研究貢獻，正式完成localization-aware low-level motor adaptation。B仍是uniform
  confidence safety fallback／tradeoff參考且不要求新C全面勝過B，component translation/yaw limiter
  只作baseline。Residual contribution的valid matched-command path必須把exact original requested
  command交給frozen model1450，只能疊加bounded 12-D joint residual。
  無有益residual時使用original-command backbone-only而非risk stop，adaptation缺失／不可信／非法
  才fallback B，SLAM invalid/stale仍exact-zero。RMA只借用privileged teacher＋history adaptation
  的分層概念；runtime禁止GT/future/backend/intervention/route phase與hardcoded turn timer。
- 新方向的第一個必要gate是matched-speed low-level causal headroom，不是PPO。舊smooth/zero/
  antismooth 24-run evidence只有3/8 groups通過0.05 m/s speed guard，且lateral重複掉speed/progress，
  因此不能當成功介入、也不執行舊72-run expansion。新core與contract已實作；第一個fresh
  mechanism與gate現已凍結為history-phase touchdown displacement shaping，但history-only phase
  estimator、model1450 anti-collapse envelope與execution wiring尚未完成，沒有執行任何新run。
  teacher/adaptation/PPO/live/ROS/default/physical全為false，blocks581..584仍未使用。Command
  preservation只用來歸因residual，不是強迫不安全速度；完整controller採
  residual-first，scale1不可行時允許最小必要降速，但必須分開記錄且不得算成residual貢獻。
  詳細：`docs/validation/slam_localization_aware_motor_adaptation_v1.md`。
- 新C必須事前防四種退化：`STOP_COLLAPSE`（不願走）、`SHUFFLE_COLLAPSE`（小碎步／高頻接觸）、
  `POSTURE_COLLAPSE`（蹲低、站太開、joint/clearance/torque/energy/slip惡化）、
  `LIMITER_COLLAPSE`（效果其實來自降速或等同B/component scaling）。步態調整涵蓋腳步與完整
  body/sensor response：平均linear/yaw command tracking需保留，真正要降低的是不必要roll/pitch、
  yaw tracking error／高頻振動、linear/angular acceleration與jerk、LiDAR scan-time motion；不能為
  降低總角速度而拒絕正常旋轉。
- 論文目標是證明locomotion action能改變下一刻localization risk，而不是重新做一個SLAM
  confidence predictor。現有confidence已定義為未來0.5 s usability probability；被動資料中的
  motion-hazard關聯不等於action的causal controllability。
- 小型simulation intervention pilot是目前唯一合理的研究路徑：在相同state／command／terrain／
  friction下，對bounded 12-D joint residual做事前凍結、平衡交叉配對的zero-mean
  intervention，含exact-zero control；其pilot第一個smooth cell已觸發事前安全停止。
- FAST-LIO2與LIO-SAM必須分開報告；matched command是設計條件，matched speed應用於gate／
  分層而不是事後宣稱隨機因果。不得使用GT、future label、backend ID或intervention ID作
  runtime policy input。
- Uniform short-pulse blocks567..570已否定「只調整一個總速度倍率」足以支撐LIO-SAM target；
  fresh component-pulse blocks572..575則完成64/64並依事前gate得到`PASS`。它直接比較component
  arms與B-like `uniform_075`：LIO-SAM right curve的uniform/preserve-yaw/preserve-translation risk
  為`0.826/0.616/0.399`，後兩者moving speed亦高`0.075/0.140 m/s`；兩candidate均通過FAST
  safeguard，64 runs無fall/base contact。這只授權下一步offline component-conditioned risk model，
  不代表C policy、PPO或正式claim已完成；left/right asymmetry與FAST right-curve tradeoff必須保留。
- 第一版risk model失敗後，fresh richer identification已完成：block576 wiring為12/12
  `WIRING_PASS`，blocks577..580六arms共96/96為development `PASS`，且沒有跨block重複的
  arm-specific safety excess。簡化後的v2 component selector仍在LOBO gate得到`FAIL`：Brier
  `0.257138`差於state-only `0.250483`與scalar `0.254038`，LIO selected-minus-uniform hazard
  `+0.128289`，且選到兩個含safety event的control run。故blocks581..584保留驗證未執行，
  action-conditioned risk signal仍不存在；live、ROS wiring、C2/PPO、default與physical全不授權。
- V1 pilot第1個smooth cell發生fall與base contact並依當時gate停止；但單次simulation fall
  不能證明smooth-specific harm，因為同block的zero／antismooth尚未執行。該`FAIL`只對中止的
  v1 matrix有效，不能當成研究假設被否決。V2 safety semantics現已凍結，但live仍未授權；
  仍不建立student、不跑PPO或72-run。
- Fresh v2 pilot attempt2已完成24/24並得到`INCONCLUSIVE`：完整性與安全全通過、沒有任何
  smooth fall/base contact，但action/body-rate排序、backend hazard方向與moving-speed guard未
  同時通過；亦沒有任何事前FAIL條件。只允許另行審查是否授權預定72-run expansion。
- Post-pilot exploratory value diagnosis顯示目前smoothing intervention不值得執行72-run：8個
  matched blocks僅5/8降低action rate、5/8降低body rate、3/8通過speed guard；FAST與LIO
  lateral各兩個blocks都重複超出speed guard且progress下降，LIO lateral更在兩blocks都出現
  smooth action-rate反向增加。這是停止目前intervention的資源決策，不把frozen
  `INCONCLUSIVE`改寫成研究假設FAIL。
- 新closed-loop路線已拆成staged risk-aware governor。Phase 1/2離線骨架已完成：正常且fresh
  時合法risk decision可取代並高於B的continuous soft throttle；risk prediction缺失／非法時
  exact fallback到B；SLAM invalid、source/receipt stale或numeric invalid時exact zero hard stop。
  實作位於ROS-independent`risk_aware_governor_core.py`，尚未接policy node、沒有risk model，
  config明確禁止live／PPO／default switch／physical use。
- Phase 3/3b speed-scale causal protocol與runtime wiring已完成並通過validator：scale
  `1.00/0.75/0.50/0.25`為明確treatment，uniform XYZ scaling保持command curvature；fresh
  block560規劃8-cell wiring smoke，561..562規劃32-cell兩backend×curve/lateral causal pilot。
  Realized speed是intended mediator，gate改看speed/hazard dose response與risk-progress Pareto。
  四個Arm-B fixed-scale ONNX/TorchScript wrappers已hash-lock，export parity與valid scale／invalid
  exact-zero checks通過；trace validator會逐frame用hash-locked B重建action。Runner已接既有ROS
  launch、offline usability、map、stability、estimator replay、run record與causal analyzer，兩個
  plan schedules可解析。Phase 4 block560 wiring smoke已完成8/8並得到`WIRING_PASS`：兩backend、
  四scales records全通過，最大action重建誤差`7.75e-7`、invalid exact-zero全通過、無fall／base
  contact。Manifest與decision hash已鎖進release。其後另行授權的blocks561..562 causal pilot
  已完成32/32且完整性全通過，但frozen decision為`FAIL`：speed ordering通過3/4 strata，future
  0.5 s hazard ordering只通過1/4（門檻3/4）；雖然reduced-minus-full平均hazard在FAST/LIO皆為
  負值，非單調反轉使目前treatment不能支撐action-conditioned risk model。唯一fall/base
  contact發生於LIO curve block562的scale1.00 control，沒有reduced-scale-specific repeated harm。
  Pilot manifest／decision hashes已鎖進release；執行後live gate已關閉，所有training授權為false。
  Block診斷顯示FAST block561四arms hazard皆為0、block562承擔主要failure，LIO curve亦有強烈
  非單調block effect；whole-episode scaling在endpoint前已讓各arm走到不同route state。下一版
  應改成相同untreated prefix後的短時scale pulse或等價state-restored設計，而不是增加原設計重複數。

## Current unfinished order

1. The non-learning J1/J2 preflight is complete and its one-shot gates are closed. Do not rerun it or
   alter the frozen reward/curriculum/evaluation gates based on future outcome data.
2. Only after a separate explicit PPO authorization, run the frozen fixed-budget J1/J2 feasibility
   training, including the common iteration-50 safety/collapse futility audit. Do not first build the
   complete RMA teacher stack and do not select seeds by interim efficacy.
3. Evaluate J0/J1/J2 and B on fresh original-command, matched-speed free-yaw, translation, lateral,
   mixed and bounded-random commands; report FAST and LIO separately.
4. Go/no-go is fixed: continue to teacher/history adaptation only if J2 adds matched-speed
   body/LiDAR/SLAM value over J1; if only J1 works, report generic smoother locomotion without a
   localization-aware contribution; if neither works without collapse, stop this architecture rather
   than repeatedly changing rewards.
5. Teacher, adaptation, production ROS wiring, default switch, physical ANYmal-D, commit and push each
   require separate explicit authorization. Blocks581..584 and 602..605 remain forbidden.

## Superseded execution history retained for traceability

1. 凍結action-conditioned localization-risk intervention pilot protocol，明確定義intervention、
   matched conditions、SLAM outcome、mechanism endpoints、futility stop與資料角色。
2. Frozen protocol與純函式已通過validation；smooth／zero／antismooth三個experimental
   policies已hash-lock匯出並通過TorchScript/ONNX、zero Arm-B、invalid fallback及residual
   bound parity。6-run runner已接通real ROS backend launch、intervention trace、stability、
   offline 0.5 s usability、map consistency、estimator replay與run-level record；plan mode通過。
   預防性SLAM endpoint只計算「當下tracking仍有效且有requested motion」的ticks，判斷未來
   0.5 s是否變成unusable；已失效ticks不重複計入hazard，clock無法配對則完整性gate失敗。
   自動decision analyzer已凍結完整性、四strata方向、speed、safety與PASS／INCONCLUSIVE／FAIL
   規則；不完整或不合法records直接FAIL。Runner已硬性要求release逐stage授權、從raw
   records重算前一stage gate，且遇接線／非有限值／殘差超界立即停止；v1保留legacy單一
   smooth safety stop以重現舊結果，v2則採下面的paired／repeated safety semantics。不能在
   pilot非INCONCLUSIVE時執行72-run expansion。
3. Runner不再要求每個中間步驟先commit；live manifest會保存HEAD與所有project-owned dirty
   檔案的SHA-256／size／existence，完整段落結束後才一次整理commit。首次授權smoke在commit
   `0f6ebb6`的第1/6 cell後自動停止為`WIRING_FAIL`；只執行
   FAST-LIO2 smooth，沒有執行其餘五個cells。原因是trace bound tolerance不一致及estimator
   replay漏接既有`deployment/python_vendor`，不是因果鏈否定。修復後用同一bag離線驗證：
   trace五項全通過，estimator 837/837 exact stamps、最大誤差0.0 m/s；release live授權已重新
   關閉。Attempt2在新output root完成6/6並得到`WIRING_PASS`；每個cell的trace、stability、
   offline label、map、estimator replay與run record全通過。探索性機制檢查中LIO-SAM符合
   smooth < zero < antismooth action/body-rate順序，FAST-LIO2未符合action-rate順序；smooth
   body-rate在兩backend皆略低於zero，但LIO-SAM smooth-minus-zero speed約`-0.053 m/s`，略超
   事前0.05 guard，因此不能由smoke宣稱排除降速。
4. Frozen 24-run pilot已獲授權，但在第1個cell
   `fastlio2/curve_1_5_right_1_0/block_543/smooth`後立即停止：launch與trace／offline label／map／
   estimator replay皆通過，但`fall=true`、`base_contact=true`，stability為
   `event_order=body_instability_first`且driver passed。Runner輸出
   `smooth_arm_safety_failure`、executed 1/24、decision `FAIL`、next step
   `stop_action_conditioned_new_C_route`；剩餘23 cells未執行。
5. V2 safety semantics已在`configs/slam_action_risk_intervention_pilot_v2.yaml`凍結：wiring／
   non-finite／contract／bound failure立即停matrix；simulation fall／base contact終止並保留該
   episode但單次event不停止matrix；單一`(backend, profile, block)`中smooth有event而exact-zero沒有，
   只會阻止PASS並得到INCONCLUSIVE，需同backend/profile兩個不同blocks皆出現此paired excess
   才route-level FAIL，並在第二個完整matched triplet後停止剩餘matrix。實體機仍是一有跌倒
   風險立即停機，simulation結果不得放寬該規則。
6. 為避免看到v1 block543 smooth結果後再補matched arms造成事後設計，v1 542..550全列為v2
   forbidden；v2 pilot使用fresh blocks 552..553，expansion用554..559。Attempt2 block542的6/6
   wiring evidence由v2 release鎖定manifest hash並從raw records重算，不需重跑；若evidence缺失或
   改變則fail closed。Block551只保留為需重新smoke時的fresh contingency。
7. V2首次執行在restricted sandbox內因GPU不可見、Isaac cache不可寫而於第1 cell wiring gate
   停止，沒有run record且不是研究outcome；保留在`outputs/slam_action_risk_intervention_pilot_v2/`。
   GPU-visible attempt2位於`outputs/slam_action_risk_intervention_pilot_v2_attempt2/pilot/`，完成
   24/24且collection passed。Frozen decision為`INCONCLUSIVE`：action-rate排序2/4、body-rate
   排序1/4；smooth-minus-zero survival為FAST `+0.975 s`、LIO `+0.025 s`，但hazard分別
   `+0.00553`與`+0.000316`；lateral speed差為FAST `-0.136 m/s`、LIO `-0.364 m/s`而使speed
   guard失敗。無fall/base contact、無paired safety excess、無joint body/survival harm且有action
   separation，因此不是FAIL，也不支持PASS。
8. Post-pilot block diagnosis建議不執行既定72-run expansion：speed guard只有3/8 blocks通過；
   FAST lateral兩blocks為`-0.064/-0.209 m/s`，LIO lateral為`-0.322/-0.407 m/s`且四次progress
   全下降；LIO lateral smooth action-rate兩次反而增加`+6.93/+4.80 /s`。增加樣本不能修復
   unchanged intervention的command-preserving解讀。下一步應停止previous-action smoothing，
   另行設計可先證明speed/progress preserving的新介入與disjoint protocol；不得用現有24 runs
   選新參數後再當validation。
9. 目前live授權已重新關閉；不授權72-run、PPO、新model或任何參數／checkpoint／profile sweep。
   正式model1450與deterministic Arm-B safety行為維持不變。
10. 所有新結果更新本文件的current state，完整方法與數字寫入對應validation文件；不得把既有
   800當未來新C blind test。
11. Risk-aware governor分階段順序：Phase 1/2介面與hard fallback純函式、Phase 3 protocol、
   Phase 3b fixed-scale runtime wiring／artifacts及Phase 4 block560 8-cell smoke皆已完成。
   Blocks561..562的32-cell causal pilot亦已完成，但因hazard dose response只有1/4 strata單調而
   `FAIL`。依事前gate停止在Phase 5之前，不建立action-conditioned risk model、不進simulation
   closed loop／PPO／實體。下一步只能重新設計speed treatment與identifiability protocol，不能
   用這32 runs事後挑scale後再把同資料當validation。
12. Matched-prefix short-pulse v2已完成離線實作但未live：所有arms先用相同Arm B與command
   profile，到profile time 7.25 s才施加0.75 s的uniform XYZ scale，之後exact recovery；driver
   保存profile clock與pulse evidence。Trace逐frame驗證prefix／pulse／recovery命令、policy
   observation與hash-locked B action，analyzer在每個四-arm set事前比較body/joint/previous-action/
   confidence state。Fresh block563為8-cell wiring smoke，564..565為另行授權的32-cell pilot；
   blocks<=562全部禁用。ROS package build通過，完整相關回歸54 tests passed，live/model/PPO/
   physical授權全為false。詳細設計：`docs/validation/slam_speed_pulse_causal_v2.md`。
13. Short-pulse block563 wiring attempt1只跑FAST curve scale1.00後依gate停止為`WIRING_FAIL`，其餘
   7 cells未跑。Pulse／recovery命令、observation、Arm-B action、stability、offline label、map與
   estimator皆通過；失敗是scale1.00 pulse publish count被錯記0，以及prefix validator誤含正常
   command ramp的一tick相位差。已修成所有pulse窗口皆計數、prefix只在ramp結束後驗證；舊artifact
   重算除不可改寫的driver count外全通過，attempt1保留FAIL不重標。ROS rebuild及15 focused tests
   通過，live gate已關閉；重跑attempt2仍需另行授權。
14. 新C2方向已事前凍結為ceiling-aware constrained residual，而不是要求每個backend都勝過B：
    FAST-LIO2只作ceiling／noninferiority safeguard；預先指定的LIO-SAM curve／lateral degraded
    strata至少一個必須證明可控headroom，之後的candidate亦至少一個stratum須risk不劣於B、progress
    高5%且不增加停止或安全事件。C2以frozen model1450為backbone，正常輸入使用原始requested
    command，不先套B soft scale；輸出可為anisotropic XYZ scale加0.10 bounded joint residual，risk
    缺失exact fallback B、invalid/stale仍exact-zero。Reward直接沿用既有robust locomotion stack並
    追蹤原始requested command，明確禁用舊C的confidence-scaled velocity targets，避免停止成為容易
    得分的解。若LIO-SAM目標strata連續評估行為等同B或舊C，明確標記
    `COLLAPSED_TO_B_OR_OLD_C`。目前short-pulse headroom、risk model、training wiring、PPO與physical
    gates全為false；尚未產生或訓練C2 policy。C2 phase1純函式控制核心已完成：先驗證SLAM與完整
    candidate輸出，再從原始command選effective command，之後才讓frozen model1450 inference並疊加
    已驗證residual；candidate缺失／非法exact fallback B且不靜默clip，invalid/stale與risk stop皆為
    exact-zero command/residual但保留不同mode。policy-node與training wiring仍明確未完成。包含舊路線
    回歸在內75 tests通過。詳細契約：
    `configs/slam_constrained_residual_c2_v1.yaml`、
    `docs/validation/slam_constrained_residual_c2_v1.md`。
15. 使用者授權沿既定路線自動推進、僅在需要研究決策或偏離方向時詢問。Short-pulse attempt2
    執行FAST前4 arms後停止：前三個完整通過，scale0.25因50 Hz trace漏掉7.18 s tick而暴露原本只有
    20 ms的pre-pulse sample窗口；其他trace/stability/SLAM/map/estimator與safety皆通過。已改成
    sample最多0.12 s old並另限pairwise time skew 0.04 s；舊scale0.25在7.16 s sample離安全邊界
    0.03 s且離線重算全通過，但attempt2保留`WIRING_FAIL`。Fresh attempt3完成FAST四arms且各自完整
    無safety event，完整matched set仍因joint-velocity pairwise L-inf `2.945 rad/s`超過凍結`1.5`
    而停止；其他pre-state項與sample-time skew全通過。這證明分開啟動的same-seed ROS/Isaac run仍有
    瞬時步態相位漂移；不可看到結果後直接放寬gate。Attempt3保留`WIRING_FAIL`，LIO四arms未跑，
    live已關閉，下一步需要選擇保留strict gate改state-restored設計，或在development evidence下
    事前改成有物理依據的composite／RMS matching並以fresh blocks驗證。
16. Composite/RMS route已完成一個完整方法段落並證明「分開重啟、same-seed、要求瞬時prestate
    matched」不可靠。Attempt4完成8/8且無safety event：FAST通過，LIO只在base angular
    `0.225>0.20`與previous action `0.417>0.40`失敗；其pre-treatment-only資料與attempt2/3合成
    24個pairwise development comparisons，凍結joint-velocity RMS/L-inf `1.25/3.25`、base angular
    `0.25`與previous action `0.45`，未用post-pulse outcome。Fresh attempt5完成8/8並得
    `WIRING_PASS`。其32-cell pilot第1個FAST block564 scale100在pulse前已tracking invalid，eligible
    ticks為0，故保留integrity `FAIL`而非效果結論。將degradation onset與pulse同設7.25 s並退休
    blocks<=565後，fresh block566 attempt6仍完成8/8、無safety event且FAST matched，但LIO獨立runs
    的prestate再擴至body angular `0.357`、joint velocity RMS/L-inf `1.881/4.429`、previous action
    `0.482`，超出backend-neutral gate。不可繼續追新最大值放寬門檻；attempt6保留`WIRING_FAIL`，
    32-cell effect pilot未跑，live已關閉。下一個真正研究決策是改採大量單狗依序、randomized
    repeated runs加事前凍結covariate adjustment（建議），或投入高成本state restoration；尚無
    short-pulse effect、risk model或PPO授權。
17. 使用者同意先測randomized repeated-run、不行再調整。方法在收集前凍結：單一robot依序執行
    兩backend×兩profiles×四scales×四fresh blocks567..570共64 runs，balanced order；prestate只
    報告、不作排除，pre-pulse tracking loss以ITT risk=1保留。64/64 integrity通過且onset-alignment
    後zero-eligible為0。FAST curve risk隨scale1/.75/.5/.25為`0.500/0.474/0.336/0.250`，FAST
    lateral為`0.467/0.468/0.408/0.368`，兩個FAST safeguard皆通過。Prespecified LIO未通過：curve
    為`0.530/0.507/1.000/0.664`，只有0.75很小且不一致改善，強降速反而更差；lateral四arms皆0
    為ceiling。LIO supported headroom為0/1，frozen decision `FAIL`。共4次fall/base contact但沒有
    跨block重複scale-specific harm。這證明command action可影響FAST未來usability，但不能用目前
    uniform slowdown建立原定LIO action-risk model；live/risk-model/PPO仍全關閉。下一步若改方向，
    應事前設計mild／anisotropic intervention並用fresh data，不能用本64 runs挑完參數後當validation。
18. Anisotropic component-pulse段落已完成。核心與ROS driver支援XYZ獨立scale且保留scalar相容；
    protocol事前凍結control、uniform0.75、preserve-yaw、preserve-translation四arms。Restricted
    sandbox attempt1因DDS/GPU不可用保留environment `WIRING_FAIL`；GPU-visible attempt2 block571
    完成8/8 `WIRING_PASS`。Fresh blocks572..575的64/64 randomized pilot完整通過，frozen decision
    `PASS`，兩component arms皆為qualified candidate且無safety event。最強訊號在LIO-SAM right curve：
    preserve-translation相對uniform hazard改善`0.428`、moving speed增加`0.140 m/s`；preserve-yaw改善
    `0.211`且增加`0.075 m/s`。LIO left僅preserve-translation risk持平但速度增加`0.327 m/s`，
    preserve-yaw有一block惡化；不得宣稱普遍對稱效果。Live已關閉，下一步是offline component-
    conditioned risk model與blockwise validation；PPO、default、physical均未授權。完整記錄：
    `docs/validation/slam_component_pulse_causal_v1.md`。
19. 第一版offline component-conditioned risk model已依事前凍結gate完成，不含backend/profile/GT/
    future label/intervention ID，使用blocks572..575 leave-one-block-out比較state-only、B-like scalar
    與translation/yaw分離模型。Frozen decision為`FAIL`：component Brier `0.250866`，差於state-only
    `0.245041`與scalar `0.245622`。雖然其選擇在LIO相對uniform平均risk `-0.2171`、速度
    `+0.2528 m/s`，FAST相對control risk `0.0`且無safety event，不能用選擇結果覆蓋calibration
    failure。Action headroom已確認，但`action_conditioned_risk_signal_available`仍false；不得接ROS、
    C2/PPO或在相同64 runs上改gate重標PASS。下一個研究決策是fresh richer component-identification
    blocks，或重設state representation後保留全新blocks驗證。詳細記錄：
    `docs/validation/slam_component_risk_model_v1.md`。
20. 使用者同意fresh richer component-identification路線後，block576完成12/12 wiring
    `WIRING_PASS`，blocks577..580完成六arms、兩backend、左右curve共96/96 development runs並
    得`PASS`；六次safety event沒有重複arm-specific excess。這批資料只作identification/model
    development。事前保留blocks581..584作external validation。簡化interaction的v2 component
    model在LOBO仍`FAIL`：component/state/scalar Brier為`0.257138/0.250483/0.254038`，LIO
    selected-minus-uniform hazard為`+0.128289`，且兩個selected control含safety event。依凍結gate
    未執行reserved blocks、未接ROS/C2/PPO，live已關閉。有效結論是component treatment有時有效，
    但目前prestate無法可靠選擇新block的action；此component-model route停在研究決策點。完整記錄：
    `docs/validation/slam_component_identification_v1.md`、
    `docs/validation/slam_component_risk_model_v2.md`。
21. 使用者在比較學術貢獻後正式選擇method 2：不把learned XYZ scale selector包裝成最終C，
    改做command-preserving localization-aware low-level motor adaptation。新增ROS-independent
    `localization_aware_residual_core.py`與凍結architecture contract：valid adapted path exact保留
    original command、只加0.05 bounded 12-D residual；沒有有益residual時backbone-only不停車，
    adaptation failure fallback B，invalid/stale hard stop。現有previous-action smoothing因只有3/8
    matched sets守住speed guard，不構成low-level headroom。下一步先設計新的mechanism-specific、
    matched-speed fresh intervention；通過後才允許RMA-inspired privileged teacher/history module，
    再之後才可能PPO。所有訓練與live gate目前關閉。
22. 後續釐清method 2就是原本未完成C的low-level residual分支，不是另一個新模型。舊C因
    confidence-scaled velocity target、近期Risk v1/v2因候選action全是scales，才會反覆趨近B；
    不能再讓新模型透過scale、停止或小碎步取得SLAM reward。Primary residual claim以相同command
    的model1450+zero residual為對照；B與component limiter只報risk--progress tradeoff，不要求全面
    勝過B。完整runtime可在residual不足時最小必要降速，但該效果必須與low-level貢獻分開。
23. Residual分支的第一個fresh mechanism與decision gate已離線實作：不沿用previous-action
    smoothing或固定joint offset，而是由允許的joint position／velocity／previous-action history
    估計late-swing phase，再用foot XYZ Jacobian最小範數解抵消下一個20 ms tick的部分向下foot
    displacement，同時一階horizontal target為exact zero；primary/low dose為50%/25%，完整12-D
    residual以uniform rescale維持L-inf 0.05。Protocol先要求blocks585..596的36-run zero-residual
    model1450 baseline凍結shuffle/posture envelope，再規劃block597 6-run wiring、598..601 72-run
    pilot及602..605 72-run unchanged confirmation；581..584完全不排程。Core、protocol validator、
    candidate-blind hierarchical baseline-envelope builder、matched-motion/mechanism/SLAM analyzer及
    focused tests已完成。後續baseline execution與phase estimator結果見下一點。詳細：
    `docs/validation/slam_low_level_touchdown_headroom_v1.md`。
24. 使用者授權touchdown-headroom simulation baseline後，新增canonical joint/torque/foot position
    instrumentation、model1450 ONNX trace重建、hard-safety run reducer、hash-locked baseline-only
    release及fail-closed runner。Restricted attempt1因DDS/iceoryx/GPU/cache sandbox限制在首格前停止；
    GPU-visible retry1完整跑首格，但因validator把正式ONNX input `obs`誤寫為`observation`而保留
    integrity FAIL。修正為讀取graph唯一input後，retry2 blocks585..596完成36/36，三profiles各12
    runs的command exact、model1450 parity、driver、physical-safety與record gates全通過；hierarchical
    anti-collapse envelope已`frozen=true`，hash
    `c9c987497fe990367388b1308fd1967dbd6ddca4e9d691fc737da81e125aad88`。接著在讀phase labels前
    另凍結history-only estimator split/algorithm/gates：train585..592、calibration593..594、one-shot
    holdout595..596，只用joint position/velocity/strict previous action。線性logistic+ridge v1為
    `FAIL`且`frozen=false`：calibration precision/false-trigger為`0.6924/0.01822`（門檻
    `>=0.90/<=0.01`），holdout precision`0.8386`；recall與progress MAE雖通過也不能救援安全性。
    不得用holdout事後調threshold/gate，block597與所有residual effect runs完全未執行。下一步若
    繼續phase estimator，必須事前定義不同temporal architecture並使用fresh disjoint資料驗證；
    teacher/adaptation/PPO/ROS policy wiring/default/physical仍未授權。
25. Phase estimator的必要性是把「怎麼改joint」與「何時安全改」分離：touchdown residual若在
    stance或early swing誤觸發，可能造成slip、失衡、改變擺腿路徑或掉速。針對v1單frame線性
    判斷的false activation，已實作v2 causal三狀態tracker，強制stance→early→late、禁止直接
    stance→late，early需兩筆確認且至少追蹤三筆，late active tick仍需當下late證據、progress不得
    明顯倒退並限制最多8 ticks。Blocks585..596全部降格為v2 development；最終development-only
    precision/recall/false-trigger/progress-MAE為`0.9095/0.5728/0.00405/0.0921`，三profiles及四腳皆
    有預測，但不得算正式PASS。Exact config hash為
    `ae446427cefd634a59731cd3947d752b371cc737dd8fe67dc264cf033f11ceca`，candidate hash為
    `8cca511eaef21ec0b32ce2b28c618d3bad6f5434788e3f06b4b2caa3907f66e7`。Fresh phase-only
    validation已用事前排定blocks606..609、三profiles共12個exact-zero model1450 runs完成；12/12
    integrity/safety PASS且unchanged one-shot gate PASS。Fresh aggregate precision/recall/
    false-trigger/progress-MAE為`0.9138/0.5883/0.00394/0.1046`，各profile precision皆`>=0.8794`、
    recall皆`>=0.5051`且四腳皆有positive prediction。已產生`frozen=true` artifact，SHA-256為
    `943a04fbe2964fb86ec61c43e5675c469150e162913dc36c2ef4dd94767fd3cb`；phase gate允許下一步
    block597，但execution gate已重新關閉。Blocks581..584與597仍完全未使用；residual effect、
    teacher/adaptation/PPO/physical皆未執行／未授權。
26. 已完成block597前可離線完成的deployable kinematics與完整pure pipeline。ANYmal-D官方USD
    rest geometry依canonical model1450 mapping產生四腳base-frame position、per-rad與per-policy-action
    `3x12` Jacobian及由joint velocity推算的relative foot velocity；不使用simulator foot/contact runtime
    input。36-run/28,438-sample parity中position RMSE/max為`9.15e-7/5.68e-6 m`，vertical velocity
    RMSE為`0.09335 m/s`，Jacobian finite-difference max error`1.53e-10 m/rad`，全部通過frozen gate。
    原XYZ damped solve在完整replay暴露horizontal leakage `4.10e-5 m`超過事前`1e-5`契約，已依原
    mechanism改成horizontal Jacobian null-space內解vertical damped minimum-norm；不是看SLAM outcome
    調參。修後28,114-tick完整pipeline replay有2,128 active ticks，其餘25,986 exact-zero；command
    全exact、residual max`0.0500000007`（float32 tolerance）、horizontal leakage降至`5.91e-10 m`、
    vertical correction方向全正確。此為integrity PASS、effect claim=false；尚未證明realized gait、
    body/LiDAR或SLAM效果；phase v2 fresh gate現已通過，但ROS policy wiring仍未授權。最新驗證為
    focused 28 passed；依既有環境拆分的repository suite共515 passed、15 dependency/runtime skips，
    publication檔另9 passed及1個預期的dirty-worktree gate failure；ROS 2 package build通過。
27. Block597 simulation-only wiring smoke已完成。Attempt1因policy node把core正式欄位`applied_action`
    誤寫成`adapted_action`而在第一tick崩潰，沒有形成有效behavior evidence；完整失敗輸出保留且只修
    欄位名。Attempt2用相同block/seed/arm order完成FAST-LIO2與LIO-SAM各zero/25%/50%，6/6 integrity、
    command exactness、residual bound、horizontal leakage及safety皆PASS，正式決策`WIRING_PASS`、
    `claim_allowed=false`。25% arm每run有40個nonzero ticks，50% arm有35--42個，zero全exact-zero；
    residual max分別`0.03848/0.0500000007`。50%-vs-zero moving-speed差FAST/LIO為`-0.00367/-0.00232
    m/s`、yaw-rate差`-0.00257/+0.00136 rad/s`，沒有靠平均速度下降；但50%-vs-zero normalized-progress
    ratio僅FAST/LIO `0.934/0.950`，是pilot前必須保留的LIMITER/behavior warning。Touchdown speed/contact/
    body impulse有初步小幅下降，但LiDAR rotation與SLAM方向不一致：FAST hazard三arms皆0，LIO
    50%-vs-zero hazard約`+0.0050`。單一excluded wiring block不得作effect claim；下一步是
    blocks598..601的72-run pilot，
    尚未授權。Execution gate已重新關閉；production ROS wiring仍未授權。
28. Blocks598..601的72-run causal pilot已完成，72/72 raw simulations與run identities完整，無
    treatment-specific fall/base-contact。初版decision因pure-yaw的linear-only progress/stop reducer
    產生null而僅是integrity FAIL；v2修成yaw-distance progress後，發現anti-collapse仍誤用臨時絕對
    cadence/switch門檻；兩版records/decisions皆保留。正式v3改用候選前已凍結且hash-locked的model1450
    envelope，未看SLAM outcome調門檻，integrity PASS。正式決策為`INCONCLUSIVE`、`claim_allowed=false`、
    supported strata為0。24個backend/profile/block比較中21個matched-motion、僅3個mechanism PASS、
    6個primary arm通過全部anti-collapse，沒有任何block同時通過matched motion、mechanism、SLAM與
    anti-collapse。FAST/LIO mixed block600等3個比較未matched，其中LIO mixed block600 progress ratio
    僅`0.107`。Stratum mean touchdown speed只在FAST/LIO mixed略降，其餘四strata上升；SLAM hazard
    方向亦不一致。Fresh zero arms本身常超出舊absolute shuffle/posture envelope，顯示envelope外推也
    有限制，但即使不以此救援，mechanism repeatability仍不足。不得執行602..605 confirmation、不得
    事後調0/25/50%或phase thresholds重標本pilot；pilot execution gate已關閉。這個touchdown family
    尚未證明low-level causal headroom，不得進privileged teacher/adaptation/PPO。

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
