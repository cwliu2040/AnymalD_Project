# ANYmal-D Locomotion

這是一個基於 Isaac Sim 5.1、Isaac Lab v2.3.2 與 RSL-RL PPO 的
ANYmal-D locomotion External Project。目標是建立乾淨、可維護，並能逐步
延伸至實體 ANYmal-D 的 sim-to-real 系統。

Flat Locomotion v1 以官方 ANYmal-D Flat task 為基準，採用
Manager-Based workflow 與 48 維 proprioceptive observation。LiDAR、
RGB-D、SLAM 與 navigation 不放入 v1 policy observation；LIO-SAM 是獨立的
ROS 2 感知層，不會改變既有 policy 契約。

## 目前基準

- 作業系統：Ubuntu 22.04 x86_64
- 專案根目錄：可放在任意使用者目錄，launch 會自動解析
- Isaac Sim：`5.1.0`
- Isaac Lab：`v2.3.2`
- Isaac Lab commit：`37ddf626871758333d6ed89cf64ad702aef127d0`
- RL：Manager-Based、single-agent、RSL-RL PPO
- Train task：`Isaac-Velocity-Flat-Anymal-D-Locomotion-v0`
- Play task：`Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0`

Isaac Lab 必須固定在支援的 tag，不以持續變動的 `origin/main` 作為基準。

## 新電腦安裝

目標環境是有 NVIDIA RTX GPU 的 Ubuntu 22.04 x86_64。Isaac Sim、Isaac
Lab、ROS 2 與 GPU driver 是系統 dependency，不會複製進 repository；
Factory 地圖、貼圖、High-Speed v0.2.0 deployment policy、ROS launch 與
專案設定則全部由 Git／Git LFS 提供。

### 1. NVIDIA driver 與基本工具

先安裝符合 [Isaac Sim 5.1 system requirements][isaac-sim-requirements] 的
NVIDIA driver，重新開機後確認：

```bash
nvidia-smi
```

安裝 Git、Git LFS 與建置工具：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake curl git git-lfs python3-colcon-common-extensions \
  python3-pip python3-rosdep python3-vcstool software-properties-common unzip
git lfs install
```

### 2. ROS 2 Humble

若尚未安裝 ROS 2 apt repository：

```bash
sudo add-apt-repository universe
sudo apt update
sudo apt install -y curl
sudo curl -sSL \
  https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
```

安裝 ROS 2、RViz、teleop 與專案建置工具：

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop ros-dev-tools ros-humble-teleop-twist-keyboard \
  gnome-terminal
source /opt/ros/humble/setup.bash
sudo rosdep init
rosdep update
```

若 `rosdep init` 回報設定已存在，保留既有設定並繼續執行
`rosdep update` 即可。完整 ROS 安裝基準以
[ROS 2 Humble Ubuntu deb 文件][ros-humble-install] 為準。

### 3. Isaac Sim 5.1

從 [NVIDIA Isaac Sim 5.1 download][isaac-sim-download] 下載 Linux x86_64
workstation zip，檔名應為
`isaac-sim-standalone-5.1.0-linux-x86_64.zip`，然後：

```bash
mkdir -p "${HOME}/isaacsim"
unzip "${HOME}/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip" \
  -d "${HOME}/isaacsim"
cd "${HOME}/isaacsim"
./post_install.sh
./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```

第一次啟動可先執行 `./isaac-sim.selector.sh`，讓 shader cache 完成初始化。

### 4. Isaac Lab v2.3.2

Isaac Lab 固定使用 `v2.3.2`，預設安裝位置是 `~/IsaacLab`：

```bash
cd "${HOME}"
git clone --branch v2.3.2 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git
cd "${HOME}/IsaacLab"
ln -s "${HOME}/isaacsim" _isaac_sim
./isaaclab.sh --install rsl_rl
git rev-parse HEAD
```

最後一行應為
`37ddf626871758333d6ed89cf64ad702aef127d0`。若 Isaac Lab 安裝在其他位置，
啟動前設定 `export ISAACLAB_ROOT=/實際/IsaacLab/路徑`；專案不會修改
Isaac Lab。安裝方式依 [Isaac Lab v2.3.2 binary installation][isaac-lab-install]
固定。

### 5. Clone 與一鍵建置

安裝上述系統 dependency 後，新電腦只需要：

```bash
cd "${HOME}"
git clone https://github.com/cwliu2040/AnymalD_Project.git anymal_locomotion
cd "${HOME}/anymal_locomotion"
./scripts/setup_deployment.sh
```

`setup_deployment.sh` 會：

- 下載並驗證 Git LFS 中的 Factory 資產與 deployment policy。
- 依 `lio_sam.repos` 下載固定 commit 的 upstream LIO-SAM。
- 用 `rosdep` 安裝 ROS package dependency。
- 將 ONNX 1.20.1 安裝到 repository 內的 `deployment/python_vendor/`。
- 建置 `deployment/ros2_ws` 並驗證必要檔案與版本。

它不會修改 `~/.bashrc`、Isaac Lab、Isaac Sim 或 upstream LIO-SAM。完成後：

```bash
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash
ros2 launch anymal_locomotion_ros2 bringup.launch.py
```

repository 可以 clone 到其他路徑；launch 會自動尋找 Factory、policy 與
Python vendor directory。Isaac Lab 若不在 `~/IsaacLab`，只需設定
`ISAACLAB_ROOT`，不必改程式。

## v1 Policy 契約

- Observation：48 維
- Action：12 維 joint-position action
- Policy 頻率：50 Hz
- Physics 頻率：200 Hz
- Action scale：0.5
- Command：body-frame `[vx, vy, wz]`
- Command 範圍：`vx = [-2.0, 3.0]`、`vy = [-1.5, 1.5]`、
  `wz = [-2.0, 2.0]`
- Joint order：依 `configs/policy_contract.yaml` 固定，runtime 與 ROS array
  一律依 joint name remap
- Terrain：flat plane
- Policy 不包含 height scan、LiDAR 或 camera

## 目錄

```text
anymal_locomotion/
├── source/anymal_locomotion/   # Isaac Lab extension，不 import rclpy
├── scripts/rsl_rl/             # train / play
├── scripts/validation/         # runtime 與契約驗證
├── configs/                    # policy 與 artifact 契約
├── assets/maps/factory/        # project-local Factory OpenUSD 與相依資產
├── deployment/ros2_ws/         # 外部 ROS 2 policy/runtime
├── action_graph/               # ROS 2 Bridge / Action Graph 契約
├── logs/                       # RSL-RL run 與 TensorBoard
├── checkpoints/                # 挑選後保留的 checkpoint
├── exported/                   # TorchScript / ONNX 與 metadata
├── tests/
└── docs/
```

Isaac Lab 與 RSL-RL 都是 dependency，不會複製進本專案。

## 驗證

靜態測試：

```bash
cd <repository-root>
PYTHONPATH=source/anymal_locomotion python3 -m pytest -q tests
```

Isaac Sim runtime smoke test：

```bash
cd <repository-root>
PYTHONPATH=source/anymal_locomotion \
  "${ISAACLAB_ROOT:-${HOME}/IsaacLab}/isaaclab.sh" -p \
  scripts/validation/validate_project.py --headless
```

ROS 2 Action Graph smoke test：

```bash
TERM=xterm-256color PYTHONPATH=source/anymal_locomotion \
  "${ISAACLAB_ROOT:-${HOME}/IsaacLab}/isaaclab.sh" -p \
  scripts/validation/validate_ros2_bridge.py \
  --headless --device cuda:0 --steps 250 \
  --external-control --validate-observation-parity
```

目前已在 RTX 5080 主機完成驗證：

- 48 observations
- 12 actions
- 50 Hz policy
- 無 height scanner
- deterministic canonical-to-runtime joint mapping
- 完整 runtime smoke test 通過
- 外部 ROS 2 已接收 `/clock`、`/joint_states`、`/odom`、`/tf`、`/imu/data`
- GPU PhysX、ROS 2 Bridge 與 ONNX closed loop 已完成整合驗證
- 專案擁有的 RTX LiDAR、PointCloud2 adapter 與 LIO-SAM 已完成 60 秒
  端到端驗證；deskew、mapping odometry 與 `map → base_link` TF 可正常輸出
- ROS 組成的 48 維 observation 與 Isaac Lab observation 逐項一致
- 官方 ±1.0 baseline 已完成 300 iterations
- High-Speed v0.2.0 已由 baseline checkpoint 接續完成 1,000 iterations
  （最終 timeout 92.55%、XY velocity error 0.376 m/s）

## 單一 Launch：Factory 建圖與鍵盤控制

這是日常人工操作的正式入口。第一次執行先完成上方「新電腦安裝」與
`./scripts/setup_deployment.sh`；之後只需：

```bash
cd <repository-root>
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash

ros2 launch anymal_locomotion_ros2 bringup.launch.py
```

此 launch 會繼承啟動 shell 的 `ROS_DOMAIN_ID`，若未設定則使用 `1`；
ONNX policy 路徑、Factory map、`cuda:0`、LIO-SAM、RViz2、RTX LiDAR、
simulation parity tolerance 與 episode timeout 設定均已有預設值，不必
在每次啟動時逐項傳入。它會開啟 Isaac Sim、RViz2，並另外開一個 GNOME
Terminal 執行官方 `teleop_twist_keyboard`。

每個新 terminal 都應 source ROS 2 與本專案 workspace；專案安裝流程不會
修改使用者的 `.bashrc`。

Factory 地圖及其相依 OpenUSD 資產位於
`assets/maps/factory/`。ROS 2 deployment host 以 Factory USD terrain
取代原本的 Flat plane，ANYmal 預設出生於 `(x=0, y=-18, yaw=0)`；訓練
task 與 48 維 policy 契約仍維持官方 Flat baseline。Factory collision
統一使用 `static=1.0`、`dynamic=1.0`、`multiply` 的地面材質，與訓練時
腳端 `0.8/0.6` 材質組合後，接觸摩擦對齊為 `0.8/0.6`。

保持 teleop terminal 焦點，按住按鍵控制：

- `i/,`：前進／後退
- `j/l`：向左／向右旋轉
- `Shift+j`／`Shift+l`：向左／向右側移
- `k`：停止
- `q/z`：同時提高／降低線速度與角速度
- `w/x`：提高／降低線速度
- `e/c`：提高／降低角速度

官方 teleop 直接發布 `/cmd_vel`，command 只 clamp 到 High-Speed policy
原本的訓練範圍。高速時的 SLAM 裂圖不使用固定速度 filter 掩蓋；後續方向是
整合多種 3D SLAM 的定位／建圖信心度，再把信心訊號納入 PPO 控制，讓 policy
在特徵不足時主動降速、信心足夠時恢復高速。

模擬視窗中的綠色箭頭是 `/cmd_vel` 目標，藍色箭頭是實際速度。若沒有
動作，確認 teleop terminal 保持焦點，並檢查 `/cmd_vel` 與
`/joint_command` 是否持續發布。

重新啟動 simulation time 前，必須先在主 launch terminal 按 `Ctrl-C`
關閉整組 process，並確認自動開啟的 teleop terminal 也已關閉，避免舊
LIO-SAM publisher 把兩套 correction 送入 GTSAM。這份 quick start 與
[deployment/ros2_ws/README.md](deployment/ros2_ws/README.md) 的
「單一 Bringup Launch」互相對應；未來若 executable、路徑、參數、按鍵或
啟動 process 改變，必須在同一批修改中同步更新兩處。

## 訓練

訓練入口：

```bash
cd <repository-root>
PYTHONPATH=source/anymal_locomotion \
  "${ISAACLAB_ROOT:-${HOME}/IsaacLab}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-D-Locomotion-v0 \
  --headless \
  --seed 42
```

High-Speed v0.2.0 預設訓練 1,000 iterations，command range 與官方 Spot
example 相同。訓練範圍不等於已驗證的最高速度，也不等於實體機安全速度；
後續須用固定速度 evaluation 驗證，實體機部署另套經安全審查的限制。

訓練產物會寫入：

`logs/rsl_rl/anymal_d_locomotion_v1/`

每個 run 會保存 resolved environment/agent config、版本與 seed manifest、
TensorBoard events 以及 RSL-RL checkpoint。Checkpoint 是可續訓或匯出
policy 的模型存檔，不會提交到 GitHub。

## Play 與匯出

有 checkpoint 後，可用 play task 載入：

```bash
PYTHONPATH=source/anymal_locomotion \
  "${ISAACLAB_ROOT:-${HOME}/IsaacLab}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0 \
  --checkpoint <checkpoint-path>
```

固定速度量測使用 `evaluate.py`，例如以 128 個環境測試 3 m/s、20 秒：

```bash
PYTHONPATH=source/anymal_locomotion \
  "${ISAACLAB_ROOT:-${HOME}/IsaacLab}/isaaclab.sh" -p scripts/rsl_rl/evaluate.py \
  --headless --num_envs 128 --steps 1000 --warmup_steps 100 \
  --vx 3.0 --vy 0.0 --wz 0.0 --checkpoint <checkpoint-path>
```

結果寫入 `logs/evaluation/`；目前量測結果見
[High-Speed v0.2.0 Evaluation](docs/high_speed_evaluation.md)。

匯出產物放在 `exported/anymal_d_locomotion_v1/`，包含 TorchScript、ONNX
與 `policy_metadata.yaml`。

目前提升為 deployment candidate 的 High-Speed v0.2.0 checkpoint、artifact
hash 與 checkpoint／JIT／ONNX output parity 結果見
[High-Speed Policy v0.2.0](docs/policy_release_v0.2.0.md)。

## 架構限制

- Training 與 Isaac Sim Python 不 import `rclpy`。
- ROS 2 policy node 必須在 Isaac Sim process 外執行。
- Isaac Sim 通訊使用 ROS 2 Bridge / Action Graph。
- UDP 不作為最終架構。
- 未來 custom USD 必須通過 joint contract 驗證。

詳細內容請見：

- [系統架構](docs/architecture.md)
- [ROS 2 模擬部署對齊紀錄](docs/ros2_deployment_decisions.md)
- [Baseline 分析](docs/baseline_analysis.md)
- [ROS 2 Policy Runtime v0.2](deployment/ros2_ws/README.md)

[isaac-sim-requirements]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
[isaac-sim-download]: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html
[isaac-lab-install]: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/binaries_installation.html
[ros-humble-install]: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
