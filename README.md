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
NVIDIA driver。安裝或更新 driver 後必須重新開機，不能只登出再登入：

```bash
sudo reboot
```

重新登入後確認 driver 與 GPU：

```bash
nvidia-smi
```

安裝 Git、Git LFS 與建置工具：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake curl git git-lfs locales \
  python3-colcon-common-extensions python3-pip python3-rosdep python3-vcstool \
  software-properties-common unzip
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
  ros-humble-gtsam ros-humble-rmw-cyclonedds-cpp \
  ros-humble-iceoryx-posh gnome-terminal
source /opt/ros/humble/setup.bash
sudo rosdep init
rosdep update
```

若 `rosdep init` 回報設定已存在，保留既有設定並繼續執行
`rosdep update` 即可。完整 ROS 安裝基準以
[ROS 2 Humble Ubuntu deb 文件][ros-humble-install] 為準。

README 不要求在 `~/.bashrc` 新增 ROS 載入函式或
`RMW_IMPLEMENTATION`。需要 ROS 的 terminal 直接執行
`source /opt/ros/humble/setup.bash` 即可。若現有 `.bashrc` 已自動 source
ROS 且使用正常，也不必特地刪除；下方 Isaac Lab 安裝命令會只針對該次命令
清除 ROS／venv 路徑，避免誤用 Python。

ROS 2 Humble 使用 Ubuntu 22.04 的 system Python，Isaac Sim 5.1 則使用自己
內建的 Python 3.11。不需要更換 `/usr/bin/python3`、修改 alternatives，或
把 ROS Python package 裝進 Isaac Sim Python。

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

### 4. Isaac Lab v2.3.2（Git clone source）

這裡不使用 Conda。Isaac Sim 是上一節解壓的 binary；Isaac Lab 則從 GitHub
clone `v2.3.2` source，`isaaclab.sh` 再用 Isaac Sim 內建的 Python 3.11
安裝 editable extensions。預設安裝位置是 `~/IsaacLab`：

```bash
cd "${HOME}"
git clone --branch v2.3.2 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git
cd "${HOME}/IsaacLab"
ln -sfn "${HOME}/isaacsim" _isaac_sim
git rev-parse HEAD
readlink -f _isaac_sim
./_isaac_sim/python.sh --version

TERM=xterm-256color env \
  -u CONDA_PREFIX \
  -u VIRTUAL_ENV \
  -u PYTHONPATH \
  -u LD_LIBRARY_PATH \
  ./isaaclab.sh -p -m pip install "setuptools==69.5.1" "wheel==0.41.3"
TERM=xterm-256color env \
  -u CONDA_PREFIX \
  -u VIRTUAL_ENV \
  -u PYTHONPATH \
  -u LD_LIBRARY_PATH \
  ./isaaclab.sh -p -m pip install --no-build-isolation "flatdict==4.0.1"
TERM=xterm-256color env \
  -u CONDA_PREFIX \
  -u VIRTUAL_ENV \
  -u PYTHONPATH \
  -u LD_LIBRARY_PATH \
  ./isaaclab.sh --install rsl_rl
TERM=xterm-256color env \
  -u CONDA_PREFIX \
  -u VIRTUAL_ENV \
  -u PYTHONPATH \
  -u LD_LIBRARY_PATH \
  ./isaaclab.sh -p -c \
  'from importlib.metadata import version; print("flatdict=" + version("flatdict")); print("isaaclab=" + version("isaaclab")); print("rsl-rl-lib=" + version("rsl-rl-lib"))'
```

`git rev-parse HEAD` 應為
`37ddf626871758333d6ed89cf64ad702aef127d0`，Python 應為 `3.11.x`，
`readlink` 應指向剛安裝的 Isaac Sim 5.1。上述 `env -u` 只清除該次命令
繼承的 Conda、venv、ROS Python 與 library path，不會刪除或修改那些環境；
`TERM` 則避免 `isaaclab.sh` 的 terminal 初始化在部分 terminal 中提前失敗。
前兩個 pip 命令處理 `flatdict==4.0.1` 舊版 source package 與新版
setuptools 不相容的問題。無 pip cache 的乾淨 Python 3.11 環境若直接執行
Isaac Lab install，可能會在這裡出現 `ModuleNotFoundError: pkg_resources`。
最後的版本檢查應輸出 `flatdict=4.0.1`、`isaaclab=0.54.2` 與
`rsl-rl-lib=3.1.2`。

安裝過程可能出現 pip dependency resolver 對 Isaac Sim pre-bundled package
（例如 FastAPI／Starlette）的衝突提示。若該段後面仍顯示
`Successfully installed`、install command exit code 為 `0`，且上述三個版本
檢查通過，這不是安裝中止；不要為了消除提示而在 Isaac Sim Python 中任意
upgrade package。真正失敗會是非零 exit code，且不會完成版本檢查。

若 Isaac Lab 安裝在其他位置，啟動前設定
`export ISAACLAB_ROOT=/實際/IsaacLab/路徑`；專案不會修改 Isaac Lab source。
安裝方式依 [Isaac Lab v2.3.2 binary installation][isaac-lab-install] 固定。

若先前已在 `flatdict` 中途失敗，可直接從兩個 pip 相容性命令重新開始，不需
重新解壓 Isaac Sim 或 clone Isaac Lab。其他問題請保留「第一個 `ERROR` 及其
前後輸出」；後面的錯誤通常只是連鎖結果。

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

`rosdep` 會使用 apt 補齊仍缺少的 system dependency，因此第一次執行仍可能
要求 sudo 密碼；前面預先安裝 `ros-humble-gtsam` 可避免 LIO-SAM 到這一步
才要求安裝 GTSAM。ONNX 使用 `pip --target` 安裝在 repository 的
`deployment/python_vendor/`，不會覆寫 system Python package。

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
`./scripts/setup_deployment.sh`。

實驗室先替每位使用者、每台機器人或每組實驗分配一個 `0` 到 `101` 的
domain。若同一個帳號固定只使用一組 ROS 系統，可將分配值寫入
`~/.bashrc`；例如分配到 `37`：

```bash
# ~/.bashrc；37 只是範例，請換成實驗室實際分配值。
export ROS_DOMAIN_ID=37
```

存檔後開新 terminal，或執行 `source ~/.bashrc`。同一套 ANYmal、RViz、
teleop 與檢查 terminal 必須使用相同值；同一實體網路上的不同使用者、
機器人或實驗則使用不同值，避免 DDS 互相 discovery。若同一帳號需要切換
多組機器人，不要在 `.bashrc` 固定 domain，改在每個 terminal 執行
`export ROS_DOMAIN_ID=<該組分配值>`。

設定完成後啟動：

```bash
cd <repository-root>
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset PYTHONPATH LD_LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source /opt/ros/humble/setup.bash
source deployment/ros2_ws/install/setup.bash

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-未設定}"
ros2 launch anymal_locomotion_ros2 bringup.launch.py
```

若所有 ROS process 都只在同一台電腦，可另外設定
`export ROS_LOCALHOST_ONLY=1` 阻止跨電腦 discovery。需要與實體 ANYmal 或
另一台電腦通訊時，必須 unset 它或設為 `0`。

此 launch 會繼承啟動 shell 的 `ROS_DOMAIN_ID`；為維持單機向後相容，未設定
時才回退到 `1`，共享網路不應依賴此 fallback。也可以在命令列傳入
`ros_domain_id:=<該組分配值>`，但寫入個人 `.bashrc` 或先 export 可讓後續
執行的 `ros2 topic` 命令自然使用相同 domain。

ONNX policy 路徑、Factory map、`cuda:0`、LIO-SAM、RViz2、RTX LiDAR、
simulation parity tolerance 與 episode timeout 設定均已有預設值，不必
在每次啟動時逐項傳入。它會開啟 Isaac Sim、RViz2，並另外開一個 GNOME
Terminal 執行官方 `teleop_twist_keyboard`。Launch 也會啟動 Iceoryx
RouDi，讓本機多 MB PointCloud2 經 CycloneDDS shared memory 傳輸；
`/tf_static` 另外保留 UDP transient history，晚啟動的 RViz 仍能取得固定
sensor transform。使用者不需設定 `RMW_IMPLEMENTATION`。

每個新 terminal 都應 source ROS 2 與本專案 workspace；專案安裝流程不會
修改使用者的 `.bashrc`。

第一次開啟 Factory／RTX LiDAR 時需要建立 shader cache，wall time 下觀察到
的 topic 頻率可能暫時偏低；cache 穩定後再量測。這不代表 200 Hz physics／
50 Hz policy 契約已改變。

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
原本的訓練範圍。高速直行裂圖已在 ROS 2 deployment 層以完整 scan motion
deskew 與無遺失的 shared-memory 點雲路徑處理，不使用固定速度 filter
掩蓋。Factory 的 3 m/s 自動軌跡已連續三次通過定量門檻；結果與重跑方式見
[高速 LIO-SAM 驗證](docs/validation/lio_sam_high_speed.md)。後續仍會整合
多種 3D SLAM 的定位／建圖信心度，再把信心訊號納入 PPO 控制，讓 policy
在特徵不足時主動降速、信心足夠時恢復高速。

若桌面環境沒有 GNOME Terminal，或 launch 無法自動開啟 teleop，改用：

```bash
ros2 launch anymal_locomotion_ros2 bringup.launch.py \
  open_teleop_terminal:=false
```

再從另一個已 source 相同 workspace、且使用相同 `ROS_DOMAIN_ID` 的 terminal
執行：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args \
  -p speed:=0.5 \
  -p turn:=0.5 \
  -r cmd_vel:=/cmd_vel
```

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

目前正式 deployment 預設為 Recovery v0.4.0；它保留 High-Speed v0.2.0
速度範圍，並通過長時序停止恢復、36 組轉彎與 12 組 true loop-closure
qualification。Checkpoint、artifact hash 與 parity 結果見
[Recovery Policy v0.4.0](docs/policy_release_v0.4.0.md)。舊版基準仍記錄於
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
