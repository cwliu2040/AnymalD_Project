# AGENTS.md

## 適用範圍

本規則適用於整個專案：

`~/anymal_locomotion`

所有專案擁有的訓練紀錄、設定、checkpoint 與匯出 policy，都必須保存在此
repository 內。

## 受保護的工作區

- 除非使用者明確要求，絕對不可修改 `~/IsaacLab`。
- 絕對不可修改舊工作區
  `~/Documents/anymal_project/anymal_ws`。
- `~/IsaacLab` 只能作為 framework、dependency，以及官方實作的參考來源。

## 架構邊界

- Isaac Lab training 與 ROS 2 deployment 必須視為不同的架構層。
- 不可在 Isaac Sim 或 Isaac Lab Python 環境中 import `rclpy`。
- 不可使用 UDP 作為最終 ROS 2 架構。
- Isaac Sim 與 ROS 2 之間必須使用 ROS 2 Bridge 和／或 Isaac Sim
  Action Graph 通訊。

## 訓練與 Policy 整合

- 維持 deterministic joint-name-to-policy-index mapping。
- 優先採用基於官方 Isaac Lab locomotion task 的最小變更。
- 設定、訓練產物、checkpoint、log 與匯出 policy 都必須保存在
  `~/anymal_locomotion`。

## Git 提交

- 後續 Git commit 訊息使用清楚、簡潔的繁體中文。
- 不為翻譯既有 commit 訊息而重寫已推送的 Git 歷史。

## 部署方向

- 所有介面與 policy 整合都必須考慮未來部署到實體 ANYmal-D 的
  sim-to-real 需求。
- 模擬器專屬邏輯必須與 ROS 2 deployment、實體硬體整合邏輯分離。
