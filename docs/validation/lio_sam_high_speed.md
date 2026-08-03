# Factory 高速 LIO-SAM 驗證

## 結論

Factory simulation 的高速直行裂圖已在 ROS 2 deployment 層修正。最終資料
路徑使用 CycloneDDS／Iceoryx shared memory 傳輸大型點雲，並在 upstream
Image Projection 與 Feature Extraction 之間加入 project-owned motion
deskew。沒有修改 upstream LIO-SAM、沒有以固定速度上限掩蓋問題，也沒有把
simulation ground truth 回饋給 SLAM。

2026-07-28 的 `forward_3_0` deterministic profile 連續三次通過：

| run | 實際最高線速度 | translation ATE RMSE | 最大 translation jump | yaw RMSE | vertical wall separation p95 最大值 | ≥2.5 m/s scans |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.129 m/s | 0.0155 m | 0.0424 m | 0.046° | 0.0688 m | 7 |
| 2 | 3.129 m/s | 0.0137 m | 0.0350 m | 0.052° | 0.0910 m | 7 |
| 3 | 3.054 m/s | 0.0145 m | 0.0375 m | 0.054° | 0.1004 m | 7 |

三輪的 raw cloud、adapted cloud 與 CloudInfo 都保持精確 10 Hz simulation
timestamp；benchmark ready 後的 IMU、odometry 與 motion-deskew coverage
缺失皆為 0。

## 根因與處理

1. 每個 OS1 scan 約三萬點。原本 CycloneDDS UDP fragmentation 在本機會丟失
   大型 PointCloud2；RouDi shared memory 後恢復完整 10 Hz。
2. Humble 所附 CycloneDDS／Iceoryx 對 transient-local history 有限制，因此
   `/tf_static` publisher 使用 UDP-only 設定，避免晚加入的 RViz 缺固定 TF。
3. Upstream ROS 2 LIO-SAM 沒有做 point translation deskew，旋轉也採逐軸
   Euler 累加。專案節點改用 raw IMU quaternion integration，加上 200 Hz
   incremental odometry translation interpolation，重建每個 point 到 scan
   start 時刻。
4. Map Optimization 調整為 8 cores、0.05 s process interval 與 0.4 m
   voxel，使 mapping correction 跟上 10 Hz scan。

## 驗收門檻

- scan timestamps 嚴格遞增，steady-state 每個 scan 都有完整 IMU／odometry
  bracket；任何 scan drop 直接失敗。
- translation ATE RMSE ≤ 0.10 m 或 path length 的 1%。
- yaw RMSE ≤ 1°。
- false pose jump ≤ 0.20 m 且 ≤ 2°。
- truth-registered vertical wall separation p95 ≤ 0.15 m。
- 每個 profile 三次有效 run；跌倒、碰撞、base tilt 超過 15° 或 height drop
  超過 0.15 m 均判為 invalid，不得算通過。
- loop closure 關閉，simulation time 判資料正確性；wall time 只評效能。

## 重跑

先完成 deployment build，然後從 repository root 執行：

```bash
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash

for run in 1 2 3; do
  ros2 launch anymal_locomotion_ros2 lio_benchmark.launch.py \
    profile:=forward_3_0 \
    output_dir:="$(pwd)/outputs/lio_sam_benchmarks/forward_3_0_run${run}"
done
```

每輪輸出 `metrics.json`；加上 `record_bag:=true` 才會錄 bag。原始結果保存在
Git ignored `outputs/`，避免把大量暫存資料納入版本控制。

## 尚未宣稱通過的範圍

`yaw_0_5`、`yaw_2_0` 與長時間 combined rotation 在目前 High-Speed policy
下會先讓 ANYmal 本體傾斜超過 15°，因此測試判為 invalid，而不是 LIO-SAM
failed。這些樣本不能用來宣稱旋轉 SLAM 通過或失敗；需先取得能穩定完成該
軌跡的 locomotion policy，再用相同門檻重跑。

本文件的高速旋轉樣本仍不在有效驗收範圍；loop closure 已由獨立的正式
v0.4.0 matrix 以 12/12 通過。實體 ANYmal-D sensor extrinsic 與實體
low-level interface 仍不在本文件的 simulation gate 範圍，詳見
`docs/physical_anymal_d_integration.md`。
