# Repository cleanup manifest

建立日期：2026-08-28
Cleanup branch：`exp/slam-confidence-v2`
Baseline：`bd989e975d4804b0dfffd5508d677fdc90195e40`

本清單只整理既有研究技術債，不改變研究方法、不授權新 PPO、live、default 或實體機器人執行。
判定依據是 `docs/project_knowledge.md`、`docs/project_history.md`、主題 validation 結論、static
import/call grep，以及目前 launch／policy contract。歷史結論保留在 Git history、project history 與
主題 Markdown；active tree 不再保留已停止方法的完整可執行堆疊。

| Path / group | KEEP / REMOVE / UNCERTAIN | Reason | Current dependency |
| --- | --- | --- | --- |
| `docs/project_knowledge.md` | KEEP | 跨對話正式 decision ledger；本輪唯讀 | 所有後續研究規劃 |
| `docs/project_history.md` | KEEP | 完整歷史 archive | retired 方法與 artifact provenance |
| `docs/slam_confidence_contract.md` | KEEP | canonical confidence／validity／age／label contract | 雙 backend、ROS message、policy observation |
| `configs/slam_confidence_contract.yaml`, `slam_confidence_calibration.yaml`, backend split configs | KEEP | 正式 confidence schema、calibration 與 group split | FAST-LIO2／LIO-SAM extractor 與校準 |
| `configs/policy_contract*.yaml` | KEEP | 48-D model1450 與 confidence observation contract | training、export、ROS runtime |
| FAST-LIO2/LIO-SAM confidence cores, nodes, odom adapters and launch/config | KEEP | current backend-neutral architecture | `/slam/odom`, atomic `/slam_confidence` |
| `SlamConfidence.msg`, confidence label/calibration/observation cores | KEEP | current interface and offline-label implementation | runtime consumer與formal calibration |
| model1450 Recovery v0.4.0 checkpoint/export and policy runtime | KEEP | formal locomotion baseline | current deployment and all comparators |
| `full_policy_bootstrap.py`, `joint_training_contract.py`, `joint_training_motion_core.py`, behavior anchor, 1068-D history policy | UNCERTAIN | v1–v4結果失敗，但這些是目前唯一可重用的完整12-D/history infrastructure；刪除會預先決定下一研究架構 | future bounded full-policy entry point and matched-motion evaluator |
| v1 generic joint-training config/runner and audit evaluator | UNCERTAIN | execution gate closed；config是失敗方法，但 evaluator含已修正progress／anti-collapse定義 | frozen evidence reproduction and next cheap gate |
| causal motion-only proxy and v2/v3/v4 experiment configs/preflight tests | REMOVE | calibration／safety／final gates已正式 FAIL，禁止重跑或事後調參 | no current backend or deployment dependency |
| old safe-command, teacher, residual, gait-mode, gait/intent/structured/transition policy stack | REMOVE | retired/superseded；active formal policy不是這些task | no current model1450/confidence-core dependency |
| action-risk, speed-scale/pulse, component-pulse/risk/identification executable stacks | REMOVE | causal conclusions已記錄，selector/model routes已停止 | no current runtime dependency |
| constrained C2 and localization-aware residual prototypes | REMOVE | never promoted; current route已停止 | no production policy-node dependency |
| touchdown phase/kinematics/residual experiment stack | REMOVE | frozen pilot `INCONCLUSIVE`，family retired；blocks602..605禁止執行 | no current policy or backend dependency |
| retired method summary Markdown in `docs/validation/` | KEEP | negative results仍影響研究去重 | decision ledger citations |
| per-model model9/19/40/48/49/60/99 JSON, iteration-0/debug/export-parity dumps | REMOVE, except release dependencies | repeated generated evidence；但 `slam_confidence_sim_release_v1.yaml` hash-lock 的 model48 behavior/parity records 必須保留 | formal A/B/C/D artifact validation |
| `configs/slam_confidence_sim_release_v1.yaml` and every referenced artifact | KEEP | formal publication release manifest and direct hash-locked dependencies | publication matrix validation |
| confidence behavior/holdout configs referenced by retained formal records | KEEP | provenance dependency even though the old training route is retired | formal A/B/C/D evidence reconstruction |
| formal confidence holdout, DDS fault, backend, publication protocol/statistics evidence | KEEP | protects current contract and paper baseline | calibration IDs, publication conclusions |
| publication A/B/C/D exports and model48 formal artifact | KEEP | formal 800-cell comparator reproducibility／potential unique source | publication evidence only; not promotion |
| speed/action intervention exported arm bundles | REMOVE | experiment-only generated policies; conclusions retained in Markdown/Git | no active runtime dependency |
| estimator15 artifact and estimator contract/runtime | KEEP | shared experimental policy-state artifact; status clearly non-production | retained publication and diagnostic paths |
| project-owned motion deskew | UNCERTAIN | not part of formal native-backend claim, but still a supported explicit compatibility/diagnostic path and older deployment evidence | optional LIO launch and tests |
| root `build/`, `install/`, `log/`, `lidar_type` | KEEP outside cleanup | pre-existing untracked runtime artifacts; user work must not be overwritten | local build/runtime only |
| nested LIO-SAM/FAST-LIO2 checkout dirty files | KEEP untouched | protected user/upstream state | backend source/runtime |

## Removal rule

只有同時符合下列條件的項目才實際移除：正式文件已標記 retired／superseded／failed；目前
model1450、confidence contract、雙 backend、publication baseline或deployment沒有 import、call、
launch或artifact dependency；重要結論已由保留的 Markdown 或 formal evidence 覆蓋。無法確認
唯一 source-of-truth 或仍可能是下一階段共用基礎者維持 `UNCERTAIN`，本輪不刪。

## Cleanup 後簡化 tree

```text
anymal_locomotion/
├── README.md, pyproject.toml
├── configs/                         # 29：policy/confidence/backend/publication + retained provenance
├── source/anymal_locomotion/
│   └── anymal_locomotion/           # baseline, generic full-policy/history, estimator, simulation boundary
├── deployment/ros2_ws/src/
│   ├── anymal_locomotion_interfaces # SlamConfidence + FootContactState
│   └── anymal_locomotion_ros2       # 41 Python cores/nodes + current launch/config
├── scripts/
│   ├── rsl_rl/                      # train, play, evaluate, CLI
│   └── validation/                  # 44 current backend/confidence/publication/deployment tools
├── tests/                           # 50 current contract/backend/policy/publication tests
├── docs/
│   ├── project_knowledge.md         # cleanup 中未修改
│   ├── project_history.md
│   ├── slam_confidence_contract.md
│   ├── repo_cleanup_manifest.md
│   └── validation/                  # 48 formal evidence and decision-summary files
├── checkpoints/                     # formal publication comparator artifact retained
├── exported/                        # model1450/publication/estimator artifacts retained
├── assets/, action_graph/
└── build/, install/, log/, lidar_type # pre-existing untracked runtime state; untouched
```

## Current entry points

- model1450 deployment：`deployment/ros2_ws/src/anymal_locomotion_ros2/launch/bringup.launch.py`
- native backend comparison/replay：`slam_backend_compare.launch.py`、
  `slam_backend_native_replay.launch.py`
- confidence contract/calibration：`configs/slam_confidence_contract.yaml`、
  `configs/slam_confidence_calibration.yaml`、`scripts/validation/calibrate_slam_confidence.py`
- formal publication reproduction：`configs/slam_confidence_publication_protocol.yaml`、
  `configs/slam_confidence_sim_release_v1.yaml`、`run_slam_confidence_publication_matrix.py`、
  `run_slam_confidence_publication_replay.py`
- generic full-policy infrastructure（execution gate closed）：`scripts/rsl_rl/train.py`、
  `full_policy_bootstrap.py`、`joint_training_contract.py`、`policies/joint_training.py`、
  `evaluate_slam_confidence_joint_training_block.py`
