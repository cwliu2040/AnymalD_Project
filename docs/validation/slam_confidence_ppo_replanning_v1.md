# SLAM confidence PPO 重新規劃交接 v1

日期：2026-08-28

## 本文件目的

本文件不是新方法protocol，也不授權任何training、ROS wiring或實體執行。它集中整理截至目前
和「把SLAM confidence放進locomotion PPO」直接相關的證據、失敗分支與下一個新對話的規劃
邊界，避免把已失敗的方法換名稱、換reward或換proxy後重做。

## 使用者鎖定的正式研究目標

正式目標仍是把`confidence`、`tracking_valid`與`age`及其history放入locomotion PPO，讓policy
在維持requested command與realized motion的前提下，學會改變完整12-D low-level joint action，
降低不必要的body／LiDAR motion，進而減少FAST-LIO2／LIO-SAM的短期失準及map splitting／裂圖。

正式backend基準固定為upstream LIO-SAM native deskew與FAST-LIO2 native undistortion。既有
project-owned motion deskew只有RViz主觀上較少裂圖的exploratory觀察，沒有完成ATE、RPE、pose
jump、tracking survival或map consistency qualification；它不進正式方法、訓練label或主要claim。
Repository內仍保留其歷史，不得用刪除證據或假裝不存在的方式處理。

若論文要聲稱減少裂圖，future outcome不能只有目前的短期tracking confidence；還必須正式量測
pose jump、map consistency、off-reference與duplicate-surface fraction。目前confidence的既有正式
定位仍是短期tracking-usability predictor，不是已完成的split-map detector。

## 不變的actor與因果比較

- 新候選從乾淨model1450 fine-tune，不從零學走路，也不續接v1/v2/v3/v4失敗checkpoint。
- Actor runtime輸入可包含confidence／validity／age、original requested XYZ/yaw command、body、
  joint與previous-action history；禁止GT、future label、backend ID、intervention ID、route phase與
  hardcoded timer。
- Requested command不得因confidence縮放；invalid／stale hard stop與B fallback另屬safety contract。
- 主要comparators固定為J0=model1450、J1=相同訓練但neutral localization history、J2=真實
  confidence／validity／age history。`J1-J0`是generic locomotion effect，`J2-J1`才是confidence
  contribution。
- 所有low-level claim必須matched command、matched realized linear/yaw motion，並通過
  STOP／SHUFFLE／POSTURE／LIMITER collapse、slip、torque、energy、clearance、joint margin、fall與
  base-contact gates。

## 目前已證明

1. Locomotion action會因果影響未來SLAM。Uniform scale pulse及translation／yaw component pulse
   已在真實FAST-LIO2／LIO-SAM simulation backend runs中改變future usability；兩個motion component
   效果不同。這個問題不得再次從頭重驗。
2. B的deterministic confidence supervisor是強simple baseline：800-cell formal result中相對A降低
   slip與body rate並延長tracking survival，但有18 percentage-point completion代價。它證明motion
   intervention有效，不是最終low-level PPO貢獻。
3. Confidence／validity／age及history可以正確接入1068-D full locomotion actor；model1450
   48-to-1068 bootstrap可保持initial action exact parity。
4. Action-dependent delayed credit wiring已在Isaac內接通：action改變body／LiDAR proxy、再改變下一刻
   synthetic localization state及delayed reward。這只證明軟體因果dataflow，不證明proxy等於真實
   FAST-LIO2／LIO-SAM。
5. Model1450 ±0.05 deterministic action hard projection能大幅降低unrestricted PPO的collapse風險；
   v4 iteration50曾達到8/8 continuation safety。
6. 現有publication資料不是只有JSON。`outputs/slam_confidence_publication_v1`約76 GB，formal tree
   包含raw ROS 2 bags、run records與disposition；有效bags可含`/lidar/points_raw`、`/imu/data`、
   `/joint_states`、`/foot_contacts`、`/cmd_vel`、`/slam/odom`、`/slam_confidence`、GT-only `/odom`與
   `/clock`。但失敗attempt也有空bag，任何重用必須只接受disposition／fingerprint／topic gate完整
   的run，不能掃到檔案就視為有效樣本。

## 目前尚未證明

1. 在相同command與realized speed下，bounded 12-D low-level joint adaptation能跨forward、lateral、
   yaw與mixed motion重複改善body／LiDAR motion及真實SLAM。
2. Confidence／validity／age history本身足以讓runtime actor判斷當下哪一個low-level action會改善
   SLAM。既有component risk v1/v2都顯示「action有headroom」不等於「prestate能選對action」。
3. Motion-only synthetic transition可跨route／support代表真實backend。Actual-backend calibration
   v1已正式失敗。
4. 既有800-cell bags能直接教完整12-D新action。資料主要來自A/B/C/D舊policy、command scaling及
   舊gait interventions，可先作SLAM response／outcome development，但low-level action variation可能
   不足；而且已看過的formal outcomes不能重新包裝成新方法的真正blind final holdout。
5. Confidence-aware PPO能減少map splitting。既有map-consistency evaluator已通過synthetic
   no-split／0.25 m split測試，但新policy尚未在fresh native-backend runs證明duplicate-surface或
   pose-jump改善。

## 已嘗試方法與不可重做理由

| 分支 | 實際做法 | 結果 | 後續限制 |
|---|---|---|---|
| 舊C/model48 | Confidence進PPO，但velocity target先被confidence縮放 | 易學成降速／停止並趨近B；800-cell不支持額外gait價值 | 不可把scale target重新包裝成joint adaptation |
| Risk v1/v2 | 在control／uniform／translation／yaw scales中離線選擇 | 兩版皆FAIL，prestate不能可靠選action | 不做Risk v3，不接ROS／PPO，不用581..584救援 |
| Previous-action smoothing residual | smooth／zero／antismooth改joint action | 只有3/8 matched groups守住0.05 m/s speed guard，SLAM方向不一致 | 不跑舊72-run expansion，不再調smoothing alpha |
| Touchdown residual | Phase estimator加late-swing Jacobian vertical correction，0/25/50% | Wiring與phase integrity通過；72-run pilot為INCONCLUSIVE，mechanism與SLAM不重複 | 不跑602..605，不事後調phase／dose，不再人工發明第二個固定腳步公式 |
| Full-policy joint training v1 | Model1450 full actor加history與body/LiDAR rewards；localization為外生clock | J1有generic smoothing但tracking、stance、energy、slip惡化；J2沒有保留改善 | 外生confidence無action→future-confidence因果，整版淘汰 |
| Causal v2 calibration | Motion-only scan proxy擬合真實backend outcome | FAST coefficients退化為0；LIO mixed heldout MSE惡化5.09% | 不可只調translation／rotation scale後重跑 |
| Causal v2 direct PPO | Unrestricted full actor，synthetic action-dependent confidence | Iteration50 continuation safety僅2/8 | 不可用更多updates救援unrestricted actor |
| Constrained v3 | Full actor投影到model1450 ±0.05 | Safety改善為6/8；lateral slip及mixed yaw progress失敗 | 不可放寬audit gate或續訓 |
| Barrier v4 | V3加lateral-slip與mixed-yaw hinge barriers | Iteration50 safety 8/8但body/LiDAR 1/8；iteration300 safety回落7/8且mechanism 2/8 | Model299正式FAIL；model49只是安全checkpoint，不是研究成功；不再逐項增加人工barrier |

詳細數字與artifact見：

- `docs/validation/slam_confidence_joint_training_v1.md`
- `docs/validation/slam_confidence_causal_joint_training_v2.md`
- `docs/validation/slam_low_level_touchdown_headroom_v1.md`
- `docs/validation/slam_component_risk_model_v1.md`
- `docs/validation/slam_component_risk_model_v2.md`

## 新對話必須重新規劃的問題

新對話不可直接延續一個預設的六階段大架構，也不可立刻再跑PPO。完成repository與本文件必讀後，
應先重新比較最小可行路徑，並明確回答它與上述失敗分支的差異。可考慮但尚未決定的方向是：

- 用accepted native-backend bags作development，建立包含`motion × point-support／SLAM state`
  interaction的privileged future-outcome supervision；只在block／route heldout與backend-separated
  calibration真正通過後，才考慮asymmetric PPO。
- 或設計能直接使用真實native-backend episodic outcome、但計算可承受的訓練方式，避免未校準
  surrogate；必須先估算是否實際可執行。
- 若無法得到可泛化的low-level action supervision，停止新PPO分支，不能用teacher/student、更大
  network或更多reward terms掩蓋不可識別性。

下一個規劃必須有一個低成本、一次性的early kill gate，且不得重新測已證明的「motion會影響
SLAM」。真正需要回答的是：如何讓confidence history選到可重複、matched-speed、低階且對未來
native SLAM／map consistency有益的action。

## 目前執行狀態

所有PPO、teacher/student、actual-backend candidate evaluation、live ROS policy wiring、default
switch與physical robot gates均關閉。Blocks581..584及602..605不得使用。任何新training、live、
大型資料收集、commit或push都需要重新整理範圍並取得使用者明確授權。
