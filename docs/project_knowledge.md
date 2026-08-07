# Current project knowledge

更新日期：2026-08-06

這份文件保存跨對話補充知識，讓 Work locally 模式的新對話在直接閱讀
repository 的程式、設定與其他 `docs/` 時，也能知道目前正式產物、近期
診斷、驗證狀態與未完成事項。它不取代 repository 或其他文件；詳細設計與
歷史仍以 architecture、release、validation 文件及實際程式碼為準。

## Repository state

- 專案根目錄：`/home/ros/anymal_locomotion`
- 目前 branch：`exp/slam-fastlio2`。
- `main` 與 `origin/main` 仍停在 `98b43dd`（`驗證：完成 reset、地圖品質與
  閉環驗證`）；benchmark base branch
  `benchmark/slam-liosam-fastlio2` 與目前實驗 branch 都以
  `fee8c9f 建立 SLAM backend 比較基線` 為共同基礎。
- `exp/slam-fastlio2` 已建立並 push 到 `origin/exp/slam-fastlio2`。目前 HEAD
  為 `326f3b3 文件：補充原生 SLAM 啟動與壓力測試結果`；其前的
  `649fcda 實作：加入原地旋轉 SLAM 壓力測試` 包含 yaw-stress、renderer、
  effective-support diagnostics 與 pilot tooling。目前 root worktree 同時有
  本輪尚未提交的 native calibration adapter／launch／文件／測試修改，以及使用者手動調整的
  `deployment/ros2_ws/src/anymal_locomotion_ros2/config/
  slam_backend_compare.rviz`；後者不應覆蓋、stage 或還原。
  runtime 產物位於 project-local `logs/`／`outputs/`，未列入提交清單。
- 更早的核心修正仍位於歷史 commit，包括：
  `e34f023 修正：改用增量診斷並穩定視窗效能`、
  `a76981c 修正：更正 IMU 座標並加入狀態回放診斷`，以及
  `c53f7d1 修正：改善高速 LIO-SAM 去畸變與點雲傳輸`。
- 根目錄 `AGENTS.md` 是 `.gitignore` 中的本機工作規則，不可 stage 或
  push。本文件是可由 repository 共享的跨對話補充知識。
- Nested upstream LIO-SAM 位於
  `deployment/ros2_ws/src/lio_sam`，目前是 detached HEAD。
- Nested LIO-SAM 的 `config/rviz2.rviz` 有對話開始前就存在的 dirty；
  不可還原、stage 或提交。

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

- 使用者目前先比較 LIO-SAM 與 FAST-LIO2，並計畫把標準化的 SLAM
  tracking confidence 輸入 locomotion PPO，使 policy 在特徵不足或 tracking
  退化時學會降低速度、減少機身晃動。FAST-LIO2 的第一階段 adapter／replay
  baseline 已在 `exp/slam-fastlio2` 建立；confidence contract 與 PPO
  observation/training 尚未開始。
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
- PPO confidence training 應是獨立的
  `feature/ppo-slam-confidence` 實驗，先用可重播／可控制的 simulated
  confidence 驗證 observation 與行為，再與各 SLAM backend 做相同資料集的
  matrix comparison。不同 SLAM 的 raw ICP fitness／residual 不可直接當成
  可比較的 confidence；需要先定義 `[0, 1]` semantics、validity、age 與
  calibration。低 confidence 的 speed cap／stop safety fallback 仍應有
  deterministic supervisor，不可只依賴 PPO 自己學會保護。
- 最終只把通過 gate、需要長期維護的 SLAM adapter、confidence contract
  與 PPO training/config 合併回 `main`；正式 Recovery v0.4.0 baseline 在
  比較期間保持可重現，不切換 Recovery v0.5。

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
  `fastlio_cube_side_length`；預設仍是 `0.5/2/4/0.5/0.5/200`。這些只覆寫
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
  此方向／運動型態取捨，因此目前沒有可誠實鎖定的單一 static candidate。
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

詳細結果在 `docs/validation/slam_backend_comparison.md`。目前尚未加入
`slam_confidence`、`slam_tracking_valid`、confidence age，也尚未建立 PPO
confidence observation 或 safety supervisor。

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
- `exp/slam-fastlio2` 目前 Python pytest 為 `128 passed, 3 skipped`；
  `git diff --check` 與包含 LIO-SAM、Livox driver、FAST-LIO2、project package
  的 project-local ROS 2 workspace build 均通過；
  `scripts/setup_deployment.sh --check` 也通過。3 個 skip 都是 Isaac Sim
  runtime unavailable 的 external-project tests。
- FAST-LIO2 stationary／forward live smoke 的 output 分別位於：
  `logs/stability_benchmarks/fastlio2/stationary_smoke_timer` 與
  `logs/stability_benchmarks/fastlio2/forward_0_5_smoke_atol01`。第一次
  forward run 只因 default IMU parity `0.002` gate 超過
  `0.0028817` 而停止，沒有 termination 或 policy freshness failure；依照
  既有 LIO-SAM live benchmark 改用 `0.01` 後重跑通過，最後 error
  `0.0025872`。

## Unfinished work

依目前 gate 順序：

1. Loop-closure gate、FAST-LIO2 持久化、direct launch 與一般第二終端 DDS
   discovery 已完成；official per-backend RViz selector 也已加入。
2. FAST-LIO2 的 live continuous `/cmd_vel` `wz=2.0` 已完成虛擬 LiDAR
   point-order A/B；下一個 gate 是在 `sensor_order + staggered` 固定後，量測
   queue／throughput／timestamp age，並完成 estimator parameter calibration；
   不得先加入 project-based deskew。
3. FAST-LIO2 calibration 必須用獨立 calibration bags 選定
   `point_filter_num`、range／voxel／iteration 與 covariance 等參數，再以
   holdout bags 驗證並鎖定 config。現有 `point_filter_num=2` 只是候選，不是
   正式 baseline；調參前不做 LIO-SAM／FAST-LIO2 final comparison。
4. FAST-LIO2 通過 live／replay／holdout gate 後，才重做 paired backend
   comparison、point-density matrix 與 effective support／latency／queue
   analysis。
5. 資料與 calibration label 足夠後，定義 backend-neutral
   `/slam_confidence [0,1]`、`/slam_tracking_valid`、timestamp／age contract；
   不可直接把不同 backend 的 raw residual 當成可比較 confidence。
6. confidence gate 通過後，才建立 `feature/ppo-slam-confidence`，先做
   simulated confidence perturbation 與 PPO observation parity，再進行
   confidence-conditioned training 與 locomotion matrix；實體 ANYmal-D
   sensor extrinsic、
   low-level interface；IMU frame contract
   必須維持 sensor-local `base_link` semantics。目前尚無 driver／SDK、實測
   extrinsic 或 safety controller；準備清單見
   `docs/physical_anymal_d_integration.md`。
7. Recovery v0.5 model2402/model2420 維持實驗候選封存；除非正式 bridge
   gate 再次出現 policy failure，否則不切換 policy、不立即續訓。

GroundPlane open-loop prehistory 在 t=25 前失敗，不能當有效 counterfactual；
後續結論仍必須保留 Factory/contact history dependency。

## Next diagnostic gate

目前已正式分類為 ROS 2 Bridge IMU angular-velocity frame bug；可控 reset gate、
三次 map-quality benchmark、修正版 12/12 loop-closure matrix，以及 FAST-LIO2
point-order 的 replay/live A/B 都已完成各自的初步 gate。下一步不是 confidence
或 PPO，而是以 `sensor_order + staggered` 固定 input contract 後完成 FAST-LIO2
estimator parameter calibration 與 holdout validation。Recovery v0.4.0 仍是正式
policy。

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
