# Causal confidence-conditioned joint training v2

更新日期：2026-08-27

## 為什麼需要 v2

V1 雖然把 confidence、validity、age history 放進完整 locomotion actor，也依 localization
vulnerability 加重 body/LiDAR motion penalty，但模擬 confidence 是固定時間／environment-ID
循環，完全不受 policy action 影響。因此 PPO 看不到所需的因果鏈：

`action -> joint/body/LiDAR motion -> future localization -> delayed return`

V1 的 300-update 結果只能學到 generic smoothing；它不是 localization-aware learning，並已在
pre-SLAM motion gate 淘汰。

## V2 架構

V2 保留 model1450 完整 12-D locomotion policy 與 exact original XYZ/yaw command。Runtime actor
仍只接收可部署的 confidence、tracking validity、normalized age，以及 command/body/joint/
previous-action histories；沒有額外的「未來 confidence」輸入，也禁止 GT、future label、backend
ID、intervention ID、route phase 或 clock。

環境依目前 action 造成的 body acceleration、angular acceleration 與 roll/pitch angular velocity，
形成 LiDAR scan-time translation/rotation proxy，再更新下一刻 confidence/validity/age。25 steps
（0.5 s）後的 localization potential 差進入 delayed reward。J1 actor 將 localization history
neutralize，J2 actor 看見實際 causal history；兩者 reward 與其餘條件完全相同，所以未來的
`J2-J1` 才能歸因為 localization history 的價值。

目前不需要 teacher/student。只有在真實校準後發現 runtime history 無法推斷必要的 privileged
latent，才重新評估 RMA-style teacher/student。

## 非學習模擬 preflight

執行 seed1450、8 environments、100 steps，warmup 25 steps；J1 baseline、J2 baseline、J2
diagnostic perturbation 各得到 600 個 post-warmup samples。Perturbation 是交替 ±0.05 joint
action，只用來驗證 wiring，不是 controller 候選；三組皆保持 exact original command。

| J2 cell | normalized distortion | confidence | delayed reward |
|---|---:|---:|---:|
| baseline | 1.899145 | 0.369902 | -0.125977 |
| ±0.05 diagnostic perturbation | 1.988485 | 0.365000 | -0.129920 |

三項事前方向檢查均通過：擾動提高 scan distortion、降低 future confidence、降低 delayed reward。
Model1450 48-to-1068 bootstrap action max error為 0.0；沒有 termination、truncation、non-finite；
沒有建立或執行 PPO runner/optimizer。

機讀結果：
`outputs/slam_confidence_causal_joint_training_v2/nonlearning_preflight.json`，SHA-256
`4cbffd1a2fe345df986de96697fef1b14b2c09b97b118d47da237b241a31bd3e`。

## 結論與下一個 gate

目前只可聲稱 action-dependent dataflow 與 delayed credit assignment 已正確接通。Transition
參數仍是 uncalibrated wiring proxy，不能描述成 FAST-LIO2/LIO-SAM dynamics，也不能據此開始
研究訓練或宣稱有效改善 SLAM。

### Actual-backend calibration v1 result

為避免重跑已存在的真實 backend simulation，calibration v1重用touchdown pilot的72個完整runs：
兩backend × 三routes × 四blocks × 三arms。這些runs使用model1450 original command，並已有
FAST-LIO2/LIO-SAM future-usability labels及完整body trace。每個25%/50% arm只與相同backend、
route、block的zero arm比較；48/48 contrasts皆通過exact-command、integrity及linear/yaw
matched-motion門檻。

Lateral與pure-yaw routes只用於擬合non-negative scan-translation/rotation motion coefficients；
mixed route只作held-out。兩backend不得pooling，且事前要求相對zero-action-effect baseline至少
降低5% held-out MSE。

結果為`CALIBRATION_FAIL`：

- FAST-LIO2的translation與rotation coefficients均退化為0；held-out MSE改善0%，非零effect
  sign accuracy 0%。
- LIO-SAM在calibration routes得到positive translation coefficient，但mixed held-out MSE反而
  惡化5.09%；sign accuracy為62.5%，仍未通過主要MSE gate。

機讀結果位於
`outputs/slam_confidence_causal_joint_training_v2/calibration_v1/summary.json`，SHA-256
`6c7869db41b214aa90b20b61dd5780ebacbccf64028ddbac1ea1b04426fa8370`；精確執行版protocol snapshot
SHA-256為`947a4d9c6ecf0c3a912bb6d09ac37d60ed0212a2a61a25abfcce94fba54dd721`。

因此不能凍結目前motion-only transition，也不能開始PPO。結果表示相同scan-motion量在不同
route/support state下不對應相同SLAM outcome；下一版必須把當下scene/point-support或backend
diagnostic state作為training-only transition conditioning，測試`motion × support/state` interaction，
而不是再收集同一種excitation或只改translation/rotation scale。Runtime actor仍只能使用
confidence/validity/age history來間接推斷該latent。所有calibration execution、PPO、teacher/student、
live ROS、default switch與physical robot gates已關閉。

## User-authorized direct PPO experiment

使用者在了解motion-only proxy未校準的風險後，明確授權直接PPO。J1/J2均由model1450 exact
48-to-1068 bootstrap開始，使用4096 environments、24-step rollout、seed1450，先各跑50 updates。
兩臂使用相同original locomotion、behavior anchor、causal delayed reward與auxiliary
angular-acceleration/jerk/LiDAR-motion objectives；唯一actor差異仍是J1 neutral localization
history、J2 causal confidence/validity/age history。兩個`model_49.pt` tensors均finite。

依事前約定，iteration50只做collapse/safety audit，不以proxy efficacy決定是否延長。12個
J0/J1/J2 × forward/lateral/yaw/mixed fixed-command runs均無termination，但continuation gate失敗：

- J1相對J0四profiles皆有physical/gait noninferiority failure。主要包括stance width、minimum
  joint margin；lateral另有duty factor、foot clearance、slip與linear progress問題。
- J2相對J1在lateral與yaw守住continuation safety，但forward/mixed linear progress ratio只有
  `0.96684/0.96315`，低於凍結的`0.98`。

Audit artifact：
`outputs/slam_confidence_causal_joint_training_v2/iteration50_motion_audit/summary.json`，SHA-256
`c262506351820fec61894b3d1089a6ff33cc2d5e092583a1fee39e0f572af590`。

因此沒有續跑300 updates，也沒有執行FAST-LIO2/LIO-SAM evaluation。這次結果直接回答「先讓PPO
練看看」：PPO確實能跑且reward上升，但在50 updates內已改壞gait margin／posture或progress，不能
把更多updates當成解法。PPO、calibration、teacher/student與deployment gates均已關閉。

## Action-constrained full-policy PPO v3

為處理unrestricted PPO的posture/joint-margin drift，v3仍訓練完整1068-D actor，但把rollout
distribution mean與deployment inference action都平滑投影到matched-command frozen model1450
action的±0.05內。初始化action exact相同；舊v2 checkpoints禁止作warm start。J1/J2再次各以
seed1450、4096 envs、24-step rollout完成50 updates，checkpoints finite。

相同12-run continuation audit顯示明顯改善：8個J1-J0/J2-J1 profile comparisons中6個通過，
而unrestricted v2只有2個。J2-J1的forward/lateral/yaw/mixed四profiles現在全部通過matched
motion、anti-collapse與safety。

但凍結規則要求J1與J2全部通過，J1-J0仍有兩個near-boundary failures：

- lateral stance-foot slip差`+0.05255 m/s`，門檻為`<=+0.05`；
- mixed yaw progress ratio `0.97123`，門檻為`>=0.98`。

因此正式狀態是near-pass但不可續跑300，不能事後把兩個門檻放寬。Artifact：
`outputs/slam_confidence_constrained_joint_training_v3/iteration50_motion_audit/summary.json`，
SHA-256 `21ef7138ef5cb8e6d99aac2281432d87a89d397a131be17a07cf861766199191`。
所有training與deployment gates已關閉；沒有執行actual-backend SLAM evaluation。

## Constrained barrier PPO v4

V4不改v3的model1450 same-command ±0.05 action hard projection，也不改任何audit門檻；只在
training前固定兩個hinge-squared barriers：`|vy command| >= 0.5 m/s`時，支撐腳平面速度超過
`0.5 m/s`才加罰（weight `-1.0`）；mixed motion時yaw tracking absolute error超過
`0.10 rad/s`才加罰（weight `-2.0`）。J1/J2 reward與budget完全相同，且兩者都由乾淨
model1450開始，禁止沿用v3 checkpoints。

Seed1450、4096 environments、24-step rollout的iteration50 checkpoints均finite。Frozen
12-run audit的continuation safety由v3的6/8提升至8/8：J1-J0 lateral stance-foot slip由v3的
`+0.05255 m/s`變為`-0.08470 m/s`，mixed yaw progress ratio由`0.97123`變為`1.00908`；所有
comparisons均matched motion、anti-collapse及termination safety PASS，stopped fraction皆0。
但完整body/LiDAR mechanism只通過1/8，所以iteration50只證明安全續訓資格，不是SLAM efficacy。
Artifact SHA-256為`1ec3ef5bf16fa90e0e0f32c8a08f978f52c28d86eabbf96cdd74150effdf3e17`。

依事前規則由各自model_49接續到總計300 updates；model_299均finite。Final frozen audit仍全部
matched motion、無停止或termination，但continuation safety回落為7/8：J2-J1 forward
stance-foot slip差`+0.05794 m/s`，超過`+0.05`上限`0.00794 m/s`。完整body/LiDAR mechanism
只通過2/8，沒有形成跨forward/lateral/yaw/mixed可重複的localization-aware motion機制。
Final artifact：
`outputs/slam_confidence_constrained_barrier_v4/iteration300_motion_audit/summary.json`，SHA-256
`dba0e7787cfaa8294893e30086ed392ae9475c84a46456e59d4e3573f7a77227`。

因此model_299正式FAIL，model_49只保留為iteration50 safety checkpoint，不能提升為研究成功或
actual-backend候選。不得事後調barrier weight／audit gate後重標；PPO、actual-backend evaluation、
teacher/student、ROS、default與physical gates全部關閉。
