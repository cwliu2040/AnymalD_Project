# ANYmal-D Locomotion

Clean Isaac Lab External Project for a maintainable ANYmal-D locomotion stack
with a future sim-to-real deployment path.

The v1 task is a project-owned, Manager-Based RSL-RL PPO baseline derived from
the official Isaac Lab ANYmal-D Flat task. LiDAR, RGB-D, SLAM, and navigation
remain outside the Flat locomotion policy observation.

## Supported baseline

- Project root: `/home/ros/anymal_locomotion`
- Isaac Sim: `5.1.0`
- Isaac Lab: `v2.3.2`
- Isaac Lab tag commit: `37ddf626871758333d6ed89cf64ad702aef127d0`
- Historical rollback reference:
  `cbf51abb5e98d1b3d497c8c73dc989e9f3628b89`
- RL workflow: Manager-Based, single-agent
- RL library: RSL-RL PPO

Do not develop against `origin/main`. Pin the supported Isaac Lab tag.

## Task IDs

- Train: `Isaac-Velocity-Flat-Anymal-D-Locomotion-v0`
- Play: `Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0`

The project task keeps the official Flat rewards, PPO parameters, environment
count, timing, action scale, and official ANYmal-D asset. Documented deviations
are:

- direct `[vx, vy, wz]` commands (`heading_command=False`);
- deterministic canonical joint ordering for actions and joint observations;
- fail-fast runtime joint-contract validation;
- project-local experiment/artifact paths.

## Directory structure

```text
anymal_locomotion/
├── source/anymal_locomotion/   # Isaac Lab extension; never imports rclpy
├── scripts/
│   ├── rsl_rl/                 # v2.3.2-based train/play entry points
│   └── validation/
├── deployment/ros2_ws/         # future external ROS 2 policy/runtime
├── action_graph/                # future ROS 2 Bridge / Action Graph assets
├── configs/                     # policy, metadata, and artifact contracts
├── logs/                        # generated RSL-RL runs
├── checkpoints/                 # curated/promoted checkpoints
├── exported/                    # policies plus metadata
├── tests/
└── docs/
```

Isaac Lab and RSL-RL remain dependencies. Their source is not vendored here.

## Installation assumptions

1. `/home/ros/IsaacLab` is checked out at `v2.3.2`.
2. Isaac Sim 5.1.0 is linked/installed for that checkout.
3. The NVIDIA driver and GPU are available.
4. Commands are run from this project root unless stated otherwise.

Install this extension in editable mode:

```bash
cd /home/ros/anymal_locomotion
/home/ros/IsaacLab/isaaclab.sh -p -m pip install -e source/anymal_locomotion
```

This install command has not been run in the current workspace. Tests use
`PYTHONPATH` so the Isaac Sim environment is not mutated.

## Validation

Static contract and dependency tests:

```bash
cd /home/ros/anymal_locomotion
PYTHONPATH=source/anymal_locomotion python3 -m pytest -q tests
```

One-environment Isaac Sim runtime smoke test:

```bash
cd /home/ros/anymal_locomotion
PYTHONPATH=source/anymal_locomotion \
  /home/ros/IsaacLab/isaaclab.sh -p scripts/validation/validate_project.py --headless
```

The runtime test instantiates one environment and validates registration,
48 observations, 12 actions, no height scanner, 50 Hz policy rate, and runtime
joint-name remapping. It does not start PPO training.

Current validation status:

- static suite: verified;
- Python compilation and YAML/JSON syntax: verified;
- official USD joint-name extraction: verified;
- Isaac Sim runtime initialization: verified on the project host with an
  RTX 5080; manager dimensions are 48 observations and 12 actions at 50 Hz;
- complete Isaac Sim smoke test: verified, including deterministic
  canonical-to-runtime joint remapping.

## Future training

After runtime smoke passes and training is explicitly approved:

```bash
cd /home/ros/anymal_locomotion
PYTHONPATH=source/anymal_locomotion \
  /home/ros/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-D-Locomotion-v0 \
  --headless \
  --seed 42
```

This command is prepared but has not been executed. It targets the project-local
`logs/rsl_rl/anymal_d_locomotion_v1/` directory and saves resolved environment,
agent, seed/version manifest, TensorBoard events, and RSL-RL checkpoints.
Training intentionally fails before environment creation if the project has no
committed Git revision, so create/review an initial commit before running it.

## Future play and export

After a project-owned checkpoint exists:

```bash
cd /home/ros/anymal_locomotion
PYTHONPATH=source/anymal_locomotion \
  /home/ros/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0 \
  --checkpoint /home/ros/anymal_locomotion/logs/rsl_rl/anymal_d_locomotion_v1/<run>/model_299.pt
```

The v2.3.2 play flow exports TorchScript and ONNX into
`exported/anymal_d_locomotion_v1/<run>/` and writes
`policy_metadata.yaml`. This flow is implemented but remains unverified until a
new project checkpoint and a working Isaac Sim runtime are available.

## Architecture boundaries

- Training and Isaac Sim Python never import `rclpy`.
- The future ROS 2 policy node runs outside the Isaac Sim process.
- Isaac Sim ROS communication uses ROS 2 Bridge / Action Graph.
- UDP is not a final architecture.
- ROS `JointState` arrays are remapped by joint name.
- A future custom USD must pass the canonical joint contract or introduce a
  reviewed, versioned schema update.

See [architecture](docs/architecture.md) and
[baseline analysis](docs/baseline_analysis.md).
