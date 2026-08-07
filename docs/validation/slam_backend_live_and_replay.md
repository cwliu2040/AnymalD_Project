# SLAM backend live / native replay 操作手冊

這份手冊是給第一次親自比較 LIO-SAM 與 FAST-LIO2 的操作入口。它使用
host ROS 2 Humble，不需要 Docker。正式 Recovery v0.4.0 model1450 只用
simulator GT `/odom` 產生 joint command；選中的 SLAM backend 不會進入這個
motion driver。

## 先準備 overlay

LIO-SAM：

```bash
source /opt/ros/humble/setup.bash
source /home/ros/anymal_locomotion/deployment/ros2_ws/install/setup.bash
```

FAST-LIO2 現在與 LIO-SAM 一樣由 project workspace 的 pinned source 建置，
只需 source project overlay：

```bash
source /opt/ros/humble/setup.bash
source /home/ros/anymal_locomotion/deployment/ros2_ws/install/setup.bash
```

FAST-LIO2 source/build provenance 見
`docs/validation/fastlio2_source_manifest.yaml`。source 位於
`deployment/ros2_ws/src/fast_lio`，Livox ROS driver 與 SDK2 也固定在同一個
workspace 的 `src`。這些第三方 checkout 由 `.gitignore` 排除，版本與重建
流程則由 repository 追蹤。不要使用舊的 `/tmp/fastlio2_ros2_smoke`；它只是
既有 dirty smoke artifact。

新 checkout 或需要重建時，由 repository root 執行正式安裝入口（不需要
Docker）：

```bash
cd /home/ros/anymal_locomotion
./scripts/setup_deployment.sh
```

這個入口會依 `deployment/ros2_ws/fastlio2.repos` 匯入固定版本、初始化
ikd-Tree submodule、建立 Livox ROS 2 `package.xml`、套用 project-owned
`publish.map_en` patch、在 `deployment/ros2_ws/vendor` 建置 SDK2，最後一起
建置 project workspace。可用 `./scripts/setup_deployment.sh --check` 做唯讀
完整性檢查。

## 親自看 live 差異

第一次只想確認 FAST-LIO2 原生畫面時，優先使用直接入口。它 include candidate
原本的 `mapping_ouster64.launch.py` 與 `fastlio.rviz`；project wrapper 只補
simulator raw PointCloud2 轉換、正式 model1450 GT motion driver 與 teleop，
不啟動 benchmark、evaluator 或 odometry adapter：

```bash
ros2 launch anymal_locomotion_ros2 fastlio2_run.launch.py
```

upstream RViz 的 fixed frame 是 `camera_init`，主要觀看
`/cloud_registered`、`/Odometry` 與 `/path`。`/Laser_map` 因 upstream
無界累積問題刻意停用。

直接人工入口現在與正式 LIO-SAM bringup 使用相同的
`ROS_LOCALHOST_ONLY` shell contract。另一個 terminal 只需 source 相同 ROS 2
與 project overlay，即可使用一般 ROS 2 CLI：

```bash
source /opt/ros/humble/setup.bash
source /home/ros/anymal_locomotion/deployment/ros2_ws/install/setup.bash
ros2 topic list
```

以下 `slam_backend_compare.launch.py` 保留給同一 selector 下切換兩個 backend
與後續比較流程，不再作為第一次 FAST-LIO2 視覺 smoke 的入口。

啟動 LIO-SAM native deskew：

```bash
ROS_LOG_DIR=/tmp/anymal_locomotion_ros_logs \
ros2 launch anymal_locomotion_ros2 slam_backend_compare.launch.py \
  slam_backend:=liosam deskew_mode:=native
```

另開一個 terminal 啟動 FAST-LIO2 native deskew：

```bash
ROS_LOG_DIR=/tmp/anymal_locomotion_ros_logs \
ros2 launch anymal_locomotion_ros2 slam_backend_compare.launch.py \
  slam_backend:=fastlio2 deskew_mode:=native
```

兩者都會開對應 backend 的官方 RViz 與 teleop terminal。LIO-SAM 的 fixed frame
是 `map`，FAST-LIO2 的 fixed frame 是 `camera_init`；可觀察：

- LIO-SAM native odometry：`/lio_sam/mapping/odometry`
- FAST-LIO2 native odometry：`/Odometry`，frame 是 `camera_init -> body`
- LIO-SAM mapping path：`/lio_sam/mapping/path`
- FAST-LIO2 path：`/path`
- 相同 raw sensor：`/imu/data`、`/lidar/points_raw`

專案 config 將 FAST-LIO2 的 `/Laser_map` 累積 map publisher 設為
`publish.map_en: false`。候選 ROS 2 port 原本每秒累積並發布無界的全域 map；在
CycloneDDS/Iceoryx 下約 4.36 MB 就會超過預設 4 MB chunk 並使 node abort。這個
只影響 visualization 輸出，不影響 estimator 或 `/Odometry`；RViz 應使用
`/cloud_registered` 與 `/path` 觀察 native backend。若要重建 clean candidate，
必須先套用 `fastlio2_map_pub_downstream.patch`。

### FAST-LIO2 native calibration knobs

目前 native launch 已把演算法參數做成可重現的 launch override，預設值仍是
branch 上的候選基線：`fastlio_point_filter_num=2`、
`fastlio_max_iteration=4`、`fastlio_filter_size_surf=0.5`、
`fastlio_filter_size_map=0.5`、`fastlio_cube_side_length=200.0`、
`fastlio_blind=0.5`；replay/native 另可覆寫 `fastlio_acc_cov`、
`fastlio_gyr_cov`、`fastlio_b_acc_cov`、`fastlio_b_gyr_cov`。例如只改 input
stride：

```bash
ROS_LOG_DIR=/tmp/anymal_locomotion_ros_logs \
ros2 launch anymal_locomotion_ros2 slam_backend_compare.launch.py \
  slam_backend:=fastlio2 deskew_mode:=native \
  fastlio_point_filter_num:=1
```

這些參數只覆寫 FAST-LIO2 estimator；不會啟動 project-based deskew。
`fastlio_time_direction` 預設為 `clockwise`，只供離線 A/B 確認 per-point time
方向，不是正式調參旋鈕。`scan_line`、`timestamp_unit`、`scan_rate`、ring／
per-point time、IMU-LiDAR 外參、topic 與 frame 仍由 sensor contract 固定，不能
拿來當調參旋鈕。每次校準只改一個參數或一組預先登記的 candidate，先在
calibration bag 觀察 crash／queue／throughput 與 tracking，再用未參與選參數的
holdout bag 鎖定結果。

目前的 smoke sanity check（不是正式 qualification）結果如下：

| Candidate | `smoke_out_and_back` ATE / yaw | 判讀 |
| --- | ---: | --- |
| stride 2, iteration 4, voxel 0.5 | 0.0944 m / 0.2246° | 現有候選起點 |
| stride 1, iteration 4, voxel 0.5 | 0.1133 m / 0.1803° | ATE gate 失敗 |
| stride 2, iteration 3, voxel 0.5 | 0.1369 m / 0.3091° | ATE gate 失敗 |
| stride 2, iteration 4, voxel 0.3 | 0.0603 m / 0.0979° | 待 live／holdout 驗證 |
| stride 2, iteration 4, voxel 0.8 | 0.5099 m / 0.8247° | 明確失敗 |

`voxel=0.3` 在既有左右 `yaw_stress_2.0` holdout replay 也通過，但 replay
不能單獨代表 estimator 已完成 calibration；在 live／holdout gate 前仍不改
`fastlio2_anymal_ouster32.yaml` 的正式候選值。虛擬 RTX LiDAR 的 FAST-LIO2
entry point 另外固定 `fastlio_time_source=sensor_order`、
`fastlio_point_order=staggered`；`destaggered` 只能用顯式 launch override 做
官方排列 A/B，不能漏掉 point-order 參數而誤把兩種 input 當成同一 baseline。

### Broad motion calibration sweep

這一輪不是只測固定原地旋轉，而是以同一個 simulator、model1450、raw bag 與
native deskew，建立平滑加速、平移、側移、曲線、混合運動、左右轉向，以及
最後一個 command step 的急加速案例：

| Capture | command / profile | max linear speed | max yaw rate | 備註 |
| --- | --- | ---: | ---: | --- |
| `forward_0_5` | 平滑 `vx=0.5` | 0.533 m/s | 0.080 rad/s | 低速平移 |
| `forward_3_0` | 平滑 `vx=3.0` | 3.178 m/s | 0.236 rad/s | 高速前進 |
| `lateral_1_5` | 平滑 `vy=1.5` | 1.650 m/s | 0.271 rad/s | 側移 |
| `combined` | 平滑 `vx=1.5,wz=1.0` | 1.571 m/s | 1.216 rad/s | 混合平移／轉向 |
| `curve_3_0_left_0_5` | 平滑 `vx=3.0,wz=0.5` | 3.066 m/s | 0.678 rad/s | 曲線高速 |
| `warehouse_final_turn` | 瞬間切入 `vx=2.3,wz=2.0` | 2.372 m/s | 2.176 rad/s | 29 個 high-yaw scans |

`backward_2_0` 的 source wall-separation gate 本身失敗，所以只作 exploratory
資料，不用它選參數。其餘 capture 的 source timing、raw scan coverage 與
motion contract 通過。

以目前固定的 `point_filter_num=2`、`max_iteration=4`、`surf=0.3` 比較 map
voxel，可看到沒有一組靜態數值同時涵蓋所有運動：

| Candidate | 平滑平移／曲線 | 左右 `wz=2.0` | 急切換 `vx=2.3,wz=2.0` | 代表結果 |
| --- | --- | --- | --- | --- |
| `surf=0.3,map=0.3` | 前進通過；側移、混合、曲線失敗 | 左右通過 | 通過 | yaw／hard-step 較好 |
| `surf=0.3,map=0.4` | 側移、混合、曲線通過 | 左轉 yaw 1.202° 失敗；右轉通過 | ATE 0.1068 m 失敗 | 中間折衷，仍非全域解 |
| `surf=0.3,map=0.6` | `forward_0_5/3.0`、側移、混合、曲線、loop 均通過 | 左轉 ATE 0.1412 m、右轉 ATE 0.1788 m 失敗 | — | 平移較穩，yaw 較差 |
| `surf=0.3,map=0.58` | 曲線通過；側移 ATE 0.1342 m 失敗 | 左右均通過 | — | yaw 較好，側移較差 |

同一批 bag 的 LIO-SAM native control 全部通過：`forward_3_0` 0.0596 m /
0.078°、`lateral_1_5` 0.0430 m / 0.041°、`combined` 0.0509 m / 0.441°、
曲線 0.0576 m / 0.168°、左右 `wz=2.0` 分別 0.0278 m / 0.831° 與
0.0298 m / 0.773°。因此目前 FAST-LIO2 的側移／方向／急切換取捨不是
evaluator 或 source bag 共通失敗。

另外做了 stride、iteration、blind、IMU covariance 與 per-point timestamp
方向 A/B。`counterclockwise` 反向時間在 mixed case yaw 5.847°、左右 yaw stress
約 54°／52°，確認目前 OS1 synthetic time 的 `clockwise` 預設正確；反向只保留
作診斷，不是修正方案。其他 covariance／stride 組合沒有消除上述方向性失敗。
因此本輪結論是「已完成 broad calibration evidence，但尚未找到可鎖定的
單一 FAST-LIO2 native static candidate」，不把任何 `.3/.3`、`.3/.4` 或
`.3/.6` 寫入正式 YAML。

### 虛擬 RTX LiDAR contract A/B（2026-08-06）

原始 RTX scan buffer 不是帶有 Ouster native 欄位的 PointCloud2：它只有
`x/y/z`，但從 bag 可確認每個水平欄位的 ring 依 `0..31` 連續排列，欄位間以
ring wrap 分隔。adapter 的正式 `sensor_order` 因此用 wrap 重建 1024 欄位的
column time（同欄位共享 `t`），再轉成官方 Ouster native 的 ring-major
PointCloud2 layout；raw header 先扣除完整 0.1 s scan period，因為它是 scan end。
RTX sensor 設為 `NONCOMPENSATED`，兩個 backend 都只用 native deskew。

這不是改 FAST-LIO2 upstream，而是補足虛擬 sensor 缺少的 Ouster input
contract。per-emitter `fireTimeNs` 與 azimuth phase 保留為 diagnostics；在
同一個快速轉向 bag、`.4/.3` estimator candidate 下，column time ATE
`0.096243 m`（passed），fire-time A/B ATE `0.100346 m`（failed）。官方 driver
排列依 [Ouster point-cloud composer](https://github.com/ouster-lidar/ouster-ros/blob/master/src/point_cloud_compose.h)
核對；`sensor_order + staggered` 已另通過 live continuous `/cmd_vel` 的 input
ordering A/B，但 estimator candidate 仍須跨運動 holdout 驗證。

目前 ring-major `.4/.3` candidate 的 native replay metrics：

| Bag | FAST-LIO2 ATE / yaw RMSE | LIO-SAM control ATE / yaw RMSE |
| --- | ---: | ---: |
| `forward_3_0` | `0.062100 m / 0.4417°` | `0.060128 m / 0.0859°` |
| `warehouse_final_turn` | `0.096243 m / 0.6534°` | `0.046419 m / 0.3146°` |
| `combined` | `0.072520 m / 0.7422°` | 尚未重跑本輪 adapter A/B |

這些結果只證明 adapter contract 與候選參數能通過目前 replay gate，不代表
FAST-LIO2 已完成正式 calibration，也不授權把 `.4/.3` 寫回正式 YAML。

### Virtual RTX LiDAR point-order A/B（2026-08-06）

FAST-LIO2 的 Ouster handler 會先以 `points.back().t` 推定 scan end，再進行
後續排序。虛擬 RTX raw scan 的 header 是 scan end；adapter 重建的
`sensor_order` column time 範圍約為 `0..99.9 ms`。因此若先用官方
`destaggered` ring-major packing，最後一個輸入點可能落在約 `1 ms` 的早期
capture time；`staggered` 則保留 capture-column 順序，最後一個點接近 scan
end。這個差異是 FAST-LIO2 input contract 的實驗 A/B，不是 project-based
deskew，兩個 backend 都仍使用 native deskew。

目前可重現的結果：

| Test | `sensor_order + staggered` | `sensor_order + destaggered` |
| --- | ---: | ---: |
| `lateral_1_5` replay ATE | `0.124354 m` | `0.519142 m` |
| `yaw_stress_left_2_0` live | 0 termination、1,336 policy odometry records、`no_instability` | repeated policy resets、FAST `No Effective Points!`、10 terminations、joint-command timeout |

staggered live run 的 simulator 另有 IMU parity error `0.021747 > 0.01`；這是
bridge validation gate，並非 FAST-LIO2 process crash。所有 FAST-LIO2 native／
replay／live entry points 現在預設 `sensor_order + staggered`，而
`destaggered` 僅保留為明確的 diagnostic A/B。這個 input ordering 結論已通過
package build、adapter tests 與 live A/B，但 estimator 的 voxel／iteration／
covariance 尚未完成跨運動 holdout，因此不代表 FAST-LIO2 已經整體調好。

FAST-LIO2 的 `map -> camera_init`、`body -> base_link` 與機械
`base_link -> lidar_link` 只為共同 RViz 視圖提供 frame chain，不是 odometry
adapter，也不會改變 FAST-LIO2 estimator。第一階段不啟動
`fastlio_odom_adapter`。

`open_teleop_terminal:=false` 會刻意不開鍵盤；若要操作鍵盤請省略這個參數
或設成 `true`。`simulation_steps:=300` 是約數十秒的短 smoke，模擬完成後
launch 會正常自動關閉；長時間手動觀察請省略它或使用較大的值。LIO-SAM
內部 launch 也有自己的 `use_rviz` 參數，因此外層共用 RViz 開關命名為
`compare_use_rviz`，避免 nested launch 覆蓋比較器的 RViz condition。

若只想跑 backend、不開 GUI：

```bash
ros2 launch anymal_locomotion_ros2 slam_backend_compare.launch.py \
  slam_backend:=fastlio2 compare_use_rviz:=false \
  open_teleop_terminal:=false headless:=true simulation_steps:=300
```

`point_density` 預設 `1.0`。只有在受控退化實驗才改成 `0.75`、`0.50` 或
`0.25`。目前 `point_filter_num: 2` 仍只是校準起點，不是已完成 holdout
validation 的正式 baseline。

## 同一份 raw bag 的 native replay

目前可重現的短 smoke bag：

```bash
export BAG=/home/ros/anymal_locomotion/outputs/lio_sam_loop_closure/smoke_out_and_back/capture/bag
```

LIO-SAM native：

```bash
ros2 launch anymal_locomotion_ros2 slam_backend_native_replay.launch.py \
  slam_backend:=liosam deskew_mode:=native point_density:=1.0 \
  bag_path:=$BAG \
  output_path:=/home/ros/anymal_locomotion/logs/slam_backend_replay/liosam_native.json
```

FAST-LIO2 native：

```bash
ros2 launch anymal_locomotion_ros2 slam_backend_native_replay.launch.py \
  slam_backend:=fastlio2 deskew_mode:=native point_density:=1.0 \
  bag_path:=$BAG \
  output_path:=/home/ros/anymal_locomotion/logs/slam_backend_replay/fastlio2_native.json
```

這個 replay launch 對兩個 backend 使用同一份 `/clock`、`/odom`、`/imu/data`
與 `/lidar/points_raw`；evaluator 只在 offline 讀 `/odom` 產生 ATE/RMSE label。
FAST-LIO2 直接評估 `/Odometry`，不經 `/slam/odom`；因此這是 native backend
comparison，不是 locomotion integration comparison。

LIO-SAM project-deskew reference 可額外跑：

```bash
ros2 launch anymal_locomotion_ros2 slam_backend_native_replay.launch.py \
  slam_backend:=liosam deskew_mode:=project point_density:=1.0 \
  bag_path:=$BAG \
  output_path:=/home/ros/anymal_locomotion/logs/slam_backend_replay/liosam_project.json
```

`deskew_mode:=project` 對 FAST-LIO2 會被拒絕，避免把 candidate native deskew
誤標成 project deskew。

## Overnight headless pilot matrix

兩份 source bag 準備好後，可用以下 runner sequential 執行：

```bash
source /opt/ros/humble/setup.bash
source /home/ros/anymal_locomotion/deployment/ros2_ws/install/setup.bash

python3 scripts/validation/run_slam_backend_pilot.py \
  --factory-bag /home/ros/anymal_locomotion/outputs/lio_sam_loop_closure/smoke_out_and_back/capture/bag \
  --groundplane-bag /home/ros/anymal_locomotion/outputs/slam_backend_pilot/source_groundplane_forward_0_5/bag \
  --output-root /home/ros/anymal_locomotion/logs/slam_backend_pilot/matrix \
  --resume
```

預設是 2 scenes × 2 backends × 4 densities 的 16 個 unique cells；每個
100% baseline 與 50% 第一個 boundary 會重複三次，其餘密度一次，總共 32
次 sequential replay。每個 cell 都保留 `launch.log`、cell metadata 與
evaluator JSON。evaluator 明確回報的 backend/scene failure 會標為
`expected_backend_failure` 並繼續；timeout、missing result JSON 或 launch
在 evaluator 前崩潰會標為 `infrastructure_failure` 並停止整套。總結在
`summary.json`。

## 目前 gate 邊界

這些 launch 會驗證可啟動、topic/frame contract、raw point coverage、native
odometry 與 simulator/rosbag 時間對齊；它們不宣稱完成多場景、退化點雲、
queue-drop、latency 或 physical-clock throughput qualification。confidence
與 PPO observation 尚未在這個入口中加入。
