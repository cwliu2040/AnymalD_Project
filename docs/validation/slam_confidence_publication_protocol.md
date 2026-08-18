# SLAM confidence 論文數據收集 protocol v1

日期：2026-08-17

狀態：已凍結第一版實驗設計，但尚未授權正式收數。正式 Recovery v0.4.0
model1450 仍是唯一正式 policy；model48＋estimator15 仍是 experimental
candidate。機讀版本為
`configs/slam_confidence_publication_protocol.yaml`。

實作狀態（2026-08-17）：所有工作限定於`exp/slam-fastlio2`，不修改或合併
`main`。B/D ONNX exports已由clean code commit `9b015c7`重產並hash-lock，mechanism sidecar
亦已完成；block43的兩backend／兩conditions／四arms共16格excluded smoke全數通過。
正式collection仍須等待其餘readiness gates，因此目前資料不可混入formal dataset。

## 1. 研究問題與可宣稱範圍

主要研究問題是：在相同 SLAM backend、初始姿態、route、command trace、
seed 與 duration 下，confidence-conditioned model48 是否能在保留任務效率的
同時，降低機身晃動與足端滑動，並因此延長 SLAM tracking survival。

預註冊的因果鏈為：

```text
confidence 下降
  -> intent / structured gait 改變
  -> roll/pitch motion、action rate、foot slip 降低
  -> SLAM tracking survival 提高
  -> progress 與 completion efficiency 仍可接受
```

論文用語限於「具有安全結構約束的 confidence-conditioned learned gait
adaptation」。Intent 有 explicit auxiliary supervision，C-v10 另有 deterministic
envelope，因此不得描述為純 end-to-end PPO 自行發現完整減速策略。

同一 rosbag 的 replay 已固定機體運動與 sensor stream，只能證明 matched-input
backend 差異、資料管線可重現性與開迴路 action response；不得用它證明
`gait -> body motion -> SLAM` 的 closed-loop 因果。這個因果結論只能來自 live
closed-loop matrix。

## 2. 四個 policy arms

所有 arms 都使用 estimator15 作為唯一 locomotion body-velocity estimator；
simulator GT velocity 只供離線 metrics。兩個 backend 都只提供 pose 與
confidence，不提供 policy velocity。這可避免把 LIO/FAST velocity quality 混入
policy 比較。

| Arm | Policy | Confidence 使用方式 | Learned gait effect |
|---|---|---|---|
| A | frozen model1450 | 不 consume；只旁錄 | 無 |
| B | frozen model1450 | deterministic safe-command supervisor | 無 |
| C | model48＋estimator15 | supervised intent＋四個 PPO gait coordinates | 開啟 |
| D | model48 intent-only | 與 C 相同 intent；強制 applied structured delta=0 | 關閉 |

共同 deterministic safe scale 固定為：

```text
s = tracking_valid * clip((confidence - 0.2) / 0.8, 0, 1)
```

Arm B 必須把原 48-D command 乘上 `s` 後交給同一 frozen model1450。Arm D
仍計算並旁錄四個 head outputs，但 action path 在 structured projection 後強制
delta 為精確零；其輸出必須等於 model48 的 intent action。不得用 ROS node 外部
速度 clamp 冒充 D。

主要 contrasts 預先固定為：

1. C − B：learned intent/gait 相對相同 deterministic supervisor 的總增益。
2. C − D：四個 learned gait coordinates 的增量因果效果。
3. B − A：deterministic confidence supervisor 本身的效果。
4. 上述 contrasts 與 backend 的 interaction。

## 3. 凍結 artifacts

- 正式 model1450 checkpoint SHA：
  `3bbaff6247fdb59808af42a59e2ec9477c047531a3e2aa6cd0df6208a796feee`。
- Model48 checkpoint SHA：
  `888bc682eda65e2c427eccd445d8b9cb1e787635b5da86e6fa8a85b83fda4fa0`。
- Model48 deployment ONNX SHA：
  `c8225b05a2b8fd1a35fb919593a68326b0ddac3d955b69191708def5d071f4a5`。
- Estimator15 metadata SHA：
  `041c9f8a85d220120f6ed5de41bb8e58f7f95a03f3a03c9121898af18c25a1ff`。
- FAST-LIO2 confidence ID：`native-v1-1e6cf8347be1`。
- LIO-SAM confidence ID：`native-v1-edc098b0bd98`。

Arm B 與 D 尚未有 hash-locked deployment artifact；建立、parity 驗證與凍結
後，才可把 protocol 狀態改為 ready。若任何 policy、estimator、confidence
artifact 或計分公式改變，必須升 protocol version，不得在同一 formal matrix
中混用。

## 4. Live closed-loop matrix

Backend 固定為 FAST-LIO2 與 LIO-SAM，皆使用 native deskew 與各自已通過
holdout 的 confidence artifact。正式收數 headless 執行，不開 GUI/RViz。
Scene與初始姿態固定為 Factory、`(x,y,yaw)=(0,-18,0)`；每次launch固定
`simulation_steps=4000`。分析視窗以 profile driver 的simulation-time起訖為準，
profile結束後的額外idle只供flush，不計入progress或moving-speed。

凍結四條現有 command profiles：

- `lateral_right_1_5`：最難右移 cell。
- `curve_1_5_left_1_0`：左向 combined motion。
- `curve_1_5_right_1_0`：右向 combined motion。
- `warehouse_mapping_stress`：長時間、多 plateau、高速轉向與停止 recovery。

每條 profile 有兩個 perception conditions：

- `native`：不修改 LiDAR。
- `gradual_support_loss`：sensor-order-preserving 的 deterministic ring/FOV/density
  loss；3.0 s healthy、1.5 s ramp-down、4.5 s 低 support、3.0 s recovery，
  最低 support fraction 0.001；recovery後至profile結束維持full support。

第二項只能稱為「受控 support loss」，不可泛稱一般 LiDAR degradation。它沿用
confidence calibration 時已定義的 fault family，但 formal source runs 必須是新
capture，不得重用 training/calibration/holdout bags。

五個 paired block IDs／simulation seeds 固定為 `43..47`。Locomotion launch與matrix
runner現已把block ID明確傳入Isaac Lab `--seed`並寫入manifest；但完整excluded smoke
尚未證明reset後初始state與command trace逐run相等，所以正式收數仍未開放。每個
`backend × profile × perception condition` block 內，以由 block ID 決定的
balanced Latin square 排列 A/B/C/D，避免總讓某 policy 先跑。每 cell 五次，
總 live runs 為：

```text
2 backends × 4 profiles × 2 conditions × 4 policy arms × 5 blocks = 320
```

五個blocks是formal design的最低數量，不預設它一定足以估計低發生率fall risk。
第一個formal run前必須只用既有歷史資料做simulation-based power／CI-width分析；若
精度不足，只能在看新formal outcome前升版並增加所有cells的paired blocks，不能只替
某個arm加樣本。

每個 paired block 必須相同：

- simulator scene、初始 base pose、joint state 與 reset handshake；
- command trace 的每個 plateau/ramp、timestamp 與 SHA；
- simulation seed、duration 與 perception schedule；
- estimator、confidence artifact、sensor extrinsic 與 adapter contract；
- physics dt、policy rate、IMU/LiDAR rate與所有 launch arguments。

不同 policy 造成的實際 trajectory 差異是 treatment outcome，不得用事後 pose
teleport 或 command 調整把它消除。若 profile 因 policy failure未完成，progress
與 failure 仍保留，不補跑一個「較成功」的替代 trial。

## 5. Replay matrix 與證據邊界

每個 scheduled live run 都保存 raw `/lidar/points_raw`、`/imu/data`、GT `/odom`
及 contract 所需 topics。每份 raw bag 以隔離 ROS domain 分別由 FAST-LIO2 與
LIO-SAM replay 兩次。

Replay 用途：

- 相同 sensor motion 下的 backend ATE/RPE、tracking survival 與 confidence；
- confidence instrumentation 對 trajectory 的 non-interference；
- evaluator、timestamp alignment 與 derived metrics 的 repeatability；
- 對已記錄 observation/confidence 做 A/B/C/D 開迴路 action-response audit。

Replay 不得用來比較 policy 的 fall、slip、progress 或「步態改善 SLAM」，因為
bag 中的 physics 與 gait 已由 source live run 決定。跨 backend replay也必須保留
`source_backend` 與 `source_policy_arm`，不可把它當成完全無 carry-over 的 live
backend trial。

## 6. Metrics 與事件定義

所有連續 metrics 先以單一 run 彙整；50 Hz frames 不是獨立樣本。

### Safety

- Fall：hard termination 或 base height 低於 frozen stability threshold。
- Base contact：是否發生、peak force 與 force-time integral。
- Foot slip event：normal force ≥50 N、tangential speed ≥0.6 m/s，連續兩個
  50 Hz ticks；另報 stance-weighted slip RMS/integral，不能只報二元事件。
- Tilt：`sqrt(roll^2 + pitch^2)` 的 RMS、p95、max。
- Roll/pitch rate：`sqrt(wx^2 + wy^2)` 的 RMS、p95、max。
- Action rate：`||a(t)-a(t-1)||/dt` 的 RMS 與 p95。

### Efficiency

- Normalized progress：GT pose 沿 frozen reference route 的 progress／route
  length；hard failure 後不得把 reset 位移算成 progress。
- Completion：progress ≥0.95 且維持0.5 s、此前無 hard failure。
- Completion time：第一次達成 completion 的時間；未完成者不填虛假大值，另報
  timeout/censoring。
- Moving speed：requested planar speed ≥0.25 m/s 時的 GT planar speed。
- Stopped fraction：上述 active-command sample 中 actual speed <0.10 m/s。
- Confidence stop fraction：active-command sample 中 `safe_scale < 0.20`。
- False-stop fraction：confidence stop 同時 offline future-usability label `U=1`。
  `U` 仍採 confidence contract 的 GT-only 0.5 s future horizon，必須在 run 結束後
  才產生，不可進 runtime。
- Recovery latency：confidence 恢復並持續 valid、`C≥0.55` 0.5 s 起，至 actual
  speed 持續達 requested speed 50% 0.5 s；未恢復者 right-censor。

### SLAM

- Tracking survival：backend ready 後，第一次 invalid/LOST 持續 ≥0.25 s 的
  simulation time；run 結束仍 tracking 則 right-censor。
- Tracking-valid fraction、confidence distribution、reason mask 與 source age。
- Initial-SE(2)-aligned translation ATE RMSE。
- 1.0 s delta 的 translation/yaw RPE。
- Map consistency：估計 map 一次性 SE(2) alignment 後對 Factory reference
  surface 的 distance p50/p95，以及 duplicate-surface fraction。

目前 confidence 是短期 tracking usability predictor，不是 split-map detector。
Map metric 在通過 synthetic no-split/split validation 前只能列 exploratory；不可拿
RViz 主觀看圖或 confidence score 代替。

### Mechanism

每個 policy tick 必須可重建並記錄：

- confidence、tracking-valid、normalized age 與 safe scale；
- intent blend；
- raw/applied stride attenuation、crouch、stance width、action smoothing；
- structured delta L2、legacy action、safe action、intent action 與 final action。

現有 deployment ONNX 只輸出 final action。正式方案是對記錄的51-D observation
執行 offline diagnostic sidecar，從同一 checkpoint/config 重建中間量；sidecar
重建的 final action 必須逐 record 對上 runtime `raw_action`。這不改 live action
path，也不更換既有 model48 ONNX。若無法通過 parity，formal collection 不得開始。

## 7. 統計分析

Experimental unit 是 scheduled live run。主要報 paired effect size 與95% CI；policy safety
failure與SLAM registration failure保留為outcome，只有data-integrity failure可排除，避免
survivorship bias。
不把 frame、foot 或 point 當獨立 repetitions。

- 二元 outcome：paired risk difference 與 cluster bootstrap CI。
- 連續 outcome：paired mean difference、median difference 與 cluster bootstrap
  CI。
- Tracking survival：restricted mean survival time difference；run-end與安全停止
  使用明確 censoring。
- Bootstrap 10,000次，以 paired block/profile 為 cluster。
- 四個 primary contrasts 預先報告；secondary metrics 使用 Holm correction。
- 不論 CI 是否跨零，都報 raw distributions、effect size 與 CI，不只報 p-value。

Confirmatory population 是 controlled gradual-support-loss live runs，主要 learned-gait
contrast是C−D。預先指定的四個endpoints為 stance-weighted slip RMS、roll/pitch-rate
RMS、tracking restricted mean survival time與normalized progress/time。因果鏈只有在
degradation期間applied gait coordinates非零，C−D的slip與rate方向為負、tracking
RMST方向為正，且efficiency ratio的95% CI下界不低於0.90時，才可稱為完整支持；
否則逐段報告哪些link有證據、哪些沒有。不得看完結果後更換endpoint或margin。

Safety-efficiency Pareto 不用任意加權的單一safety composite。以相同efficiency axis
（normalized progress／elapsed simulation time）分別對hard-failure risk、stance-weighted
slip RMS與roll/pitch-rate RMS畫三個panels。

圖中必須顯示 dominated points、失敗與未完成 runs，不得只畫 Pareto frontier。
此外產生：

1. `confidence -> intent/gait -> slip/body motion -> SLAM state` 時間序列；
2. backend × policy 的 run-level box/violin points 與95% CI；
3. C vs D learned-gait ablation；
4. Tracking survival curve與RMST差；
5. native與controlled support-loss分層結果。

## 8. Run disposition 與防止 selection bias

- Infrastructure failure：同 block ID 重跑，但原 attempt manifest保留，且不得
  進 efficacy denominator。
- Policy/SLAM failure：就是 outcome，不可為了替換失敗而重跑。
- Timeout：保留；time-to-event metrics在timeout right-censor。
- Operator abort：只可因預註冊安全條件或外部 host fault排除，原因寫入manifest。
- 不作 efficacy early stopping；安全 early stop 可用，但必須列入 safety outcome。
- Smoke/pilot runs 永久標成 excluded，不可事後併入 formal matrix。

Infrastructure failure 至少包括 process未啟動、bag損壞、required topic缺失、時鐘
regression、artifact identity不符或 command trace mismatch；tracking lost、fall、
confidence fail-closed與未完成 route 均不是 infrastructure failure。

## 9. 正式收數前 readiness gates

以下全部完成後才可把機讀 config 的
`formal_collection_authorized` 改為 `true`：

1. A/B/C/D 均有 hash-locked artifact/metadata 與 action parity；D 的 structured
   delta 必須 exact zero。
2. 四組都確認使用同一 estimator15；A/B 不得偷用 GT velocity。
3. Mechanism sidecar 對 runtime action 通過逐 sample parity。
4. Live gradual-support adapter 通過 sensor-order、timeline、timestamp與 no-GT
   contract tests。
5. Route reference、initial pose、command trace serialization與 SHA 已凍結。
6. Offline usability label、false-stop evaluator與map-consistency synthetic gate通過。
7. 一個完整但排除於分析外的 smoke block跑完四arms、兩backends、兩conditions。
8. 歷史資料的power／precision分析確認五個paired blocks足夠；否則在formal前升版並
   對稱增加所有cells樣本。
9. Root project-owned tracked files為乾淨、可識別commit。Protected nested LIO-SAM
   user dirty 不修改、不還原；headless run不載入該RViz檔，仍記錄其path/content SHA。
10. 每個 run 的 topic rate、timestamp monotonicity、reset ACK、confidence identity、
   estimator identity與command trace equality gate通過。

2026-08-17已完成一個不納入正式分析的block43 smoke：`lateral_right_1_5`、兩backend、
native／gradual兩conditions、A/B/C/D共16格全數通過。每格均通過driver、confidence-aware
stability、policy observation dimension與required-topic raw-bag gate；B/C/D的runtime action
亦由checkpoint sidecar重建，最大absolute error為`1.66893e-6`（門檻`1e-5`），D structured
delta exact zero。機讀摘要為
`docs/validation/slam_confidence_publication_excluded_smoke_block43.json`。此結果只關閉第7項
readiness gate，不可併入formal efficacy dataset。其16格raw bags其後全數通過offline
future-usability／false-stop evaluator：共產生4,512個evaluation-grid labels，10,972／
10,972個policy ticks成功配對；1,497個具有已知label的confidence-stop samples中共有3個
false stops。這些數字只驗證量測鏈與揭露excluded smoke現象，不是正式效應估計。

Map-consistency evaluator以每格相同的`/lidar/points_raw`樣本分別配合offline GT pose與
canonical `/slam/odom`建立route-observed Factory reference／estimated surfaces，再做一次
SE(2)＋Z alignment，輸出reference distance p50/p95、off-reference與duplicate-surface
fractions；GT不進runtime。合成正常／0.25 m split-map gate四項檢查全過，機讀證據為
`docs/validation/slam_confidence_map_consistency_synthetic.json`。既有block43中兩backend的
native arm C bags亦已完成端到端執行；該次數值只作pipeline smoke，不設定或通過任何
efficacy threshold，正式比較須由完整paired matrix的分布與confidence interval決定。

每格現另由`build_slam_confidence_publication_run_record.py`把locomotion frames、offline
SLAM結果、map metric與mechanism sidecar聚合成單一`scheduled_live_run` record；原始frames
明載不得作independent replicates。`analyze_slam_confidence_publication.py`只接受完整配對，
輸出C–D、C–B、B–A的continuous difference、binary risk difference、C–D efficiency
ratio與backend interaction，95% interval固定以`(paired_block_id, profile)` cluster bootstrap
10,000次。正式claim還要求320個唯一identity逐格精確等於frozen schedule且所有run gates
通過。block43的16格已完成schema backfill與descriptive analysis，但因dataset role為
`excluded_smoke`且只有一個block，分析器固定輸出`formal_claim_allowed=false`與
`complete_support=false`。

Replay runner會對每個collection-valid scheduled live bag計算content fingerprint，再送入FAST-LIO2與
LIO-SAM native backend各兩次；完整formal live matrix因此對應1,280格replay。Replay
manifest固定`closed_loop_gait_causality_allowed=false`，只支援matched-input backend與
measurement variability，不能取代live gait causality。Infrastructure gate要求topic/count
schema一致、metrics finite、timestamps與backend evaluator通過；兩次數值差異完整保存，
但不以任意closeness threshold排除outcome。

第一個excluded replay smoke使用block43 FAST-source/native/arm C bag，兩backend×兩次共
4/4通過。FAST-LIO2兩次皆137 mapping samples、ATE `0.143664 m`；舊qualification的
`ATE>0.10 m`只記為outcome。LIO-SAM兩次皆138 samples，ATE為`0.038180/0.044856 m`，
差`0.006677 m`；yaw RMSE差`0.004102 deg`。機讀證據為
`docs/validation/slam_confidence_publication_replay_excluded_smoke.json`。這證明runner與
outcome/infrastructure分離可用，不是完整1,280格replay或backend優劣結論。

Sample-size decision不得使用formal outcomes或與formal相同的seeds。已完成160格
`excluded_pilot`：blocks/seeds `143..147`、四routes、gradual-support-loss、兩backend、
A/B/C/D；與formal `43..47`完全分離，且pilot永遠不得進formal efficacy dataset。
`justify_slam_confidence_sample_size.py`要求160個唯一run records精確匹配pilot schedule、
全部gates通過且role正確，才會用C–D paired effects做10,000次stratified simulation。
原先的log-efficiency ratio在pilot中因control progress接近零而出現最高約82倍ratio，屬
不可識別estimand。這是在任何formal data前發現，protocol v2透明修訂為paired
`normalized_progress` difference，NI margin `-0.10`、95% half-width `0.05`，並補上binary
risk-difference precision。Candidate blocks仍為`5,6,8,10,12,15,20,25,30`，provisional
選出15 blocks；但formal仍鎖定，因原support minimum 0.001使tracking event鎖在density
ramp，無法辨識gait對SLAM survival的因果效果。

Pilot raw collection來自clean commit `8a8ebc7`，160/160 cells完成。Disposition修正後從同一
批bags backfill出160 unique scheduled-run records，28個policy/SLAM failures及5個invalid-map
registrations保留為outcomes；estimator replay 160/160通過且最大速度誤差為`0.0 m/s`。
機讀摘要為`docs/validation/slam_confidence_publication_pilot_v1_summary.json`。下一步先跑
40-cell `excluded_calibration` coarse sweep已在commit `9aded5e`完成：40/40 collection-valid，
但support 0.005至0.10只讓FAST tracking event由3.5移到3.7 s、LIO-SAM由3.8移到3.9 s，
C–D最大差0.10 s，仍被1.5 s ramp鎖定，故不得凍結。機讀摘要為
`docs/validation/slam_confidence_challenge_calibration_v1_summary.json`。

Calibration v2在任何formal data前改用3.0 s healthy、6.0 s ramp-down、4.0 s low-support
hold、3.0 s recovery，support candidates為0.35、0.45、0.55、0.65、0.75。兩backend adapter
共用由launch/protocol顯式傳入的相同時序。`select_slam_confidence_challenge.py`只依完整資料
gate、每個backend/profile的event/censoring bracket、至少1.0 s跨support event-time span、
事件比例、phase-boundary距離及至少2.0 s反應窗選擇最低support；不使用C相對D的療效來挑
challenge，避免selection bias。只有selector通過並凍結條件後，才可重做最終sample-size
pilot與考慮formal authorization。

正式mechanism reconstruction所需的model48 checkpoint已從training `logs/`來源以byte-for-byte
相同SHA `888bc682...4fa0` curate至tracked
`checkpoints/anymal_d_locomotion_slam_confidence_sim_v1/model_48.pt`；protocol與release只依賴
此tracked路徑。Export metadata中原training source path仍保留為歷史provenance，不是formal
runtime dependency。

新bag contract另要求`/foot_contacts`與`/locomotion/estimated_odom`。每格會用同一
estimator15 ONNX、arm policy metadata的joint mapping，以及固定`0.025 s` tolerance的
source-stamp synchronizer重新建立IMU／joint／四腳contact 20-step history。Synchronizer等待
各input stream watermark通過joint stamp後選nearest source stamp，tie固定選較早stamp，故不受
跨topic DDS callback順序影響；逐exact joint stamp比對live estimator output，至少95%
replay outputs需匹配且最大速度誤差`<=1e-5 m/s`。Estimator replay不讀GT。舊block43 bags沒有
這兩個topics，因此不能拿來關閉此新gate；disjoint pilot的每格都必須通過後才能開始formal。
新schema FAST-LIO2與LIO-SAM A/B/C/D共8格excluded qualifications全部通過，合計5486個
replay outputs全部exact-stamp match且最大誤差皆為`0.0 m/s`，連同其他cell gates全過；機讀證據為
`docs/validation/slam_confidence_estimator15_stamp_sync_qualification.json`。這只證明execution
chain可用，不取代160格pilot的逐格gate。

Model48＋estimator15的五方向simulation recovery regression已另行鎖定。Forward、lateral
left、lateral right、reverse與combined各使用不同seed、512 environments與1000 steps；五份
behavior gate皆通過，合計2560 environments中6個hard termination（`0.00234375`）。Validator
同時鎖定report、curated checkpoint與estimator15 metadata SHA；機讀結果為
`docs/validation/slam_confidence_model48_estimator15_regression_gate.json`。此結果只關閉既有
recovery regression，不可用來略過新bag estimator replay、disjoint pilot或formal matrix。

## 10. 資料與 provenance

所有產物保存在 repository 內的
`outputs/slam_confidence_publication_v1/`，不提交大型 bags/runtime logs。每個 run
manifest至少包含：root commit/status、nested repository commits與dirty hashes、
protocol SHA、policy/estimator/confidence artifact identity、launch arguments、
initial pose、paired block ID、ROS domain、topic schema/rate摘要、command trace SHA、
raw bag SHA與 run disposition。

正式分析只讀 immutable manifest 列出的 files；不得用目錄 glob 自動吸入後來的
debug runs。Derived tables必須保留 source run IDs，圖表可由 run-level table重新產生。

目前可執行入口為
`scripts/validation/run_slam_confidence_publication_matrix.py`。它會先驗證四個arm與
estimator15的SHA、產生320-cell balanced schedule，並固定block ID等於simulation seed。
預設輸出永久標為`excluded_smoke`；只有顯式`--formal`、protocol已授權、所有artifact SHA
通過且tracked worktree乾淨時才允許正式收數；實際clean HEAD會寫入每次manifest。正式輸出亦
強制位於`outputs/slam_confidence_publication_v1/`；每個live run開啟raw sensor bag，供後續
matched-input replay使用。

## 11. Promotion 邊界

完成 protocol 只建立 publication data contract，不構成 model48 promotion。
完整 backend matrix、統計分析、failure review與 publication gate 通過前，正式
model1450 不變。即使模擬結果通過，仍不能宣稱 physical ANYmal-D sim-to-real
qualification；實體 sensor extrinsic、state estimator、low-level controller與 safety
controller 仍須獨立驗證。
