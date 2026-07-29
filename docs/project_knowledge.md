# Current project knowledge

更新日期：2026-07-29

這份文件保存跨對話補充知識，讓 Work locally 模式的新對話在直接閱讀
repository 的程式、設定與其他 `docs/` 時，也能知道目前正式產物、近期
診斷、驗證狀態與未完成事項。它不取代 repository 或其他文件；詳細設計與
歷史仍以 architecture、release、validation 文件及實際程式碼為準。

## Repository state

- 專案根目錄：`/home/ros/anymal_locomotion`
- Branch：`main`
- 最新實作功能 commit：
  `2a7159a 功能：加入倉庫失穩診斷與恢復訓練基礎`
- `origin/main` 已包含該功能 commit；目前 repository HEAD 應以
  `git log -1` 現場核對，後續可能另有文件更新 commit。
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
- 失穩發生在最後一段高速左轉；raw command 到 31.96 s 後才歸零。
- 既有 v0.4 turning matrix 沒有涵蓋約 `vx=2.3` 與 `wz=2.0` 同時長時間
  施加。
- 主要 locomotion 缺口是 active high-combined command capability，不是
  refinery、摩擦位置或 FloorB 雙層碰撞。
- Simulator episode reset 後 external policy `previous_action` 未同步
  reset，是使用者看到 reset 後「腳打結」的另一個問題。

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

`warehouse_mapping_stress` 重播使用者的高速直行、左右最大 yaw、最後高速
左轉與 zero-command settle：

- Profile 在 29.02 s 停止刷新 `/cmd_vel`。
- Policy 的 0.5 s command watchdog 負責產生真正 effective zero command。
- Collision-free policy-isolation spawn：
  `x=-28.0, y=-18.0, yaw=-0.3926990817`。
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

## Unfinished work

依目前 gate 順序：

1. 取得並分析使用者下一次「走到後面突然向前摔倒」的完整 diagnostics。
2. 區分 sustained straight 累積失穩、轉向後殘留狀態、watchdog
   zero-command recovery、action-state discontinuity 或場景物件接觸。
3. 改善 Recovery v0.5 的 3 m/s curve tracking 與 high-combined recovery。
4. 新候選先跑 `warehouse_mapping_stress` 單次。
5. 同 profile 三次。
6. Turning regression 36/36。
7. 最後才重跑 true loop-closure regression。
8. 釐清 200 Hz IMU `angular_velocity` contract 警告。
9. 完成實體 sensor extrinsic。
10. 完成實體 low-level interface。

## Next forward-fall log

建議使用新的 diagnostics directory：

```bash
ros2 launch anymal_locomotion_ros2 bringup.launch.py \
  enable_locomotion_diagnostics:=true \
  locomotion_diagnostics_dir:=/home/ros/anymal_locomotion/logs/formal_bringup/forward_fall_01
```

跌倒後正常結束 bringup，保留整個目錄。下一次分析至少需要：

- `locomotion_diagnostics.json`
- `policy_diagnostics.json`

收到 log 後直接分析，不要求使用者手動讀 log。優先對齊：

- 跌倒前後 effective command 與 watchdog 狀態。
- Pitch、height、base contact 與 termination 時序。
- 四隻腳的接觸力與 stance tangential speed。
- Raw action／previous action 是否跳變。
- Episode reset sequence 與 ACK。
- Robot world trajectory 是否接近已啟用碰撞的場景物件。

## Handoff maintenance checklist

每次準備切換到新對話時：

1. 更新本文件的日期、Git baseline 與 dirty worktrees。
2. 同步正式 policy；實驗候選必須分開列出。
3. 更新最新完成的修正、診斷結論和 validation gate。
4. 移除已解決的 unfinished item，加入新發現的阻塞。
5. 寫清楚下一個 log、測試或使用者決策。
6. 再產生可貼入新對話的摘要。
