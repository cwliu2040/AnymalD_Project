# Current project knowledge

更新日期：2026-07-29

這份文件保存跨對話補充知識，讓 Work locally 模式的新對話在直接閱讀
repository 的程式、設定與其他 `docs/` 時，也能知道目前正式產物、近期
診斷、驗證狀態與未完成事項。它不取代 repository 或其他文件；詳細設計與
歷史仍以 architecture、release、validation 文件及實際程式碼為準。

## Repository state

- 專案根目錄：`/home/ros/anymal_locomotion`
- Branch：`main`
- 目前 baseline：
  `6624ded 文件：加入跨對話專案知識`。
- 本節所述 2026-07-29 場景與 IMU 修正尚未 commit；不可把它們當成
  `origin/main` 已包含的內容。
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
- 正式 v0.4 qualification：
  - 低速長圓周後 zero-command recovery 通過。
  - 純平移 3 m/s regression 通過。
  - Turning matrix 36/36。
  - True loop-closure matrix 12/12。

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

## Experimental Recovery v0.5

Recovery v0.5 training foundation 已加入，但所有候選都仍是實驗產物：

- Warehouse failure sequence replay。
- High-combined、combined-to-straight、reverse-yaw、rapid-zero sampling。
- Turning regression replay。
- 左右對稱 data augmentation。
- 40 s episode 與 recovery sampling。
- 純低 yaw、低速曲線與 3 m/s 曲線的條件式 rewards。

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

目前未提交的 refinery／IMU／watchdog 診斷：

- Pytest：94 passed、3 skipped。
- Python compileall：通過。
- `git diff --check`：通過。
- `scripts/setup_deployment.sh --check`：通過。
- Git LFS fsck：通過；新 `.usdc` 由 LFS 規則管理。
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
- 尚未完成：t=25 s state-transplant replay，以及含 episode reset 的 LIO
  負載下新版 IMU transient reporting runtime regression。

## Unfinished work

依目前 gate 順序：

1. 以 `refinery_fix_01` 的 formal state 建立 t=25 s state-transplant
   replay，避免 open-loop yaw 差累積成 3 m 路徑分岔。
2. State transplant 分別在 GroundPlane 與修正版 Factory 執行；只有
   Factory 失敗才繼續查局部接觸，兩者都失敗則歸到 policy recovery。
3. 對 state-transplant failure 做 watchdog deceleration A/B；正式
   bringup 預設在通過使用者路線與停止安全 regression 前維持不變。
4. 再決定 Recovery v0.5 是否需要加入「高速轉向歷史→直行／停止」的
   targeted reset distribution；不可只增加 iteration。
5. Locomotion 通過後才重跑 LIO-SAM map-quality regression。
6. 完成實體 sensor extrinsic 與 low-level interface。

## Next diagnostic gate

不需要使用者再手動重走同一路線才能前進；`refinery_fix_01` 已有足夠的
robot state、effective command、raw action 與 odometry。下一步實作
state transplant 時至少要恢復：

- base pose、body-frame linear/angular velocity；
- deterministic canonical joint position/velocity；
- external policy `previous_action`；
- trace 時刻後的 effective command history。

Transplant 必須先做 observation parity，確認第一個 policy input 與 formal
trace 一致，再開始閉迴路 rollout。不可只搬 base pose，否則 gait phase
仍不同。

## Handoff maintenance checklist

每次準備切換到新對話時：

1. 更新本文件的日期、Git baseline 與 dirty worktrees。
2. 同步正式 policy；實驗候選必須分開列出。
3. 更新最新完成的修正、診斷結論和 validation gate。
4. 移除已解決的 unfinished item，加入新發現的阻塞。
5. 寫清楚下一個 log、測試或使用者決策。
6. 再產生可貼入新對話的摘要。
