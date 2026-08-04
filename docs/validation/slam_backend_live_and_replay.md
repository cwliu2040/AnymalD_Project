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

兩者都會開 RViz 與 teleop terminal。RViz 的 fixed frame 是 `map`；可觀察：

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
`0.25`；FAST-LIO2 的正式 candidate 演算法參數仍固定為 config 裡的
`point_filter_num: 2`。

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
