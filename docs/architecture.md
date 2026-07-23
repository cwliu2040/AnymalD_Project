# ANYmal-D Locomotion Architecture

## Scope

Flat Locomotion v1 is a 50 Hz proprioceptive policy. Its inputs are base
velocity, angular velocity, projected gravity, direct velocity command, joint
position/velocity, and previous action. LiDAR, RGB-D, terrain perception, SLAM,
and navigation are intentionally outside this policy.

## Lifecycle

```text
Isaac Lab v2.3.2 training
  (Manager-Based, RSL-RL PPO, no rclpy)
            |
            v
Project-local checkpoint and resolved configs
            |
            v
Policy export (TorchScript / ONNX + policy_metadata.yaml)
            |
            v
External ROS 2 policy node
  /cmd_vel + estimated state -> 48-D observation -> 12-D action
            |
            +---------------- simulation ----------------+
            |                                             |
            v                                             |
ROS 2 messages <-> ROS 2 Bridge / Action Graph <-> Isaac Sim
            |
            +---------------- hardware ------------------+
            |
            v
Reviewed hardware adapter / safety controller -> physical ANYmal-D
```

The physical branch does not use Isaac Sim Action Graph. It reuses the same
versioned policy contract through a robot-specific, safety-reviewed hardware
adapter. The physical low-level control interface is not yet confirmed.

## Training layer

Owned by `source/anymal_locomotion` and `scripts/rsl_rl`.

- Depends on Isaac Lab v2.3.2 and RSL-RL; it does not vendor either source tree.
- Does not import `rclpy`.
- Uses official ANYmal-D USD for the first smoke/regression baseline.
- Uses direct body-frame `[vx, vy, wz]` command semantics.
- Writes all runs under this repository.
- Exports a policy only together with a committed Git revision and metadata.

## Policy contract

The source of truth is `configs/policy_contract.yaml`.

- Observation: 48 values in a fixed seven-term concatenation order.
- Action: 12 joint-position actions.
- Action transformation:
  `target_position = default_position + 0.5 * policy_action`.
- Policy period: 0.02 s.
- Command limits: ±1.0 m/s for x/y and ±1.0 rad/s for yaw.
- Actor/critic observation normalization: disabled.

Actions, joint position observations, and joint velocity observations use the
same canonical name order. Runtime and ROS arrays must be remapped by name.

## Simulation deployment boundary

The future external ROS 2 policy node will:

1. receive `/cmd_vel` as `geometry_msgs/msg/Twist`;
2. receive timestamped IMU, joint state, and state-estimation data;
3. assemble the 48-D observation according to the exported metadata;
4. run inference outside Isaac Sim;
5. publish the reviewed low-level command interface.

Isaac Sim will use built-in ROS 2 Bridge / Action Graph nodes for message
transport. No Isaac Sim or Isaac Lab Python module may import `rclpy`, and no
private command-manager tensor mutation is part of the architecture.

Exact low-level command message types remain open until the physical ANYmal-D
control interface is confirmed.

## State estimation

The official task observes simulator ground-truth base linear velocity. A real
deployment requires an estimator with an explicit frame and timestamp contract.
Before ROS 2 policy implementation, the project must confirm:

- base velocity estimator and body/world frame;
- IMU orientation/angular-velocity conventions;
- odometry source, update rate, and covariance handling;
- synchronization and stale-data timeouts;
- safety clamp, rate limit, and emergency stop behavior.

## Perception and navigation

```text
LiDAR / RGB-D
      |
      v
ROS 2 Bridge / sensor drivers
      |
      v
SLAM / terrain perception / Nav2
      |
      v
/cmd_vel
      |
      v
External locomotion policy node
```

LiDAR and camera tensors are not part of Flat Locomotion v1 observation. Rough
and perceptive locomotion will be separate tasks and policy versions.

## Future custom USD

The official Isaac Sim 5.1 ANYmal-D USD is the current canonical reference.
Custom USD integration requires a reviewed project-owned `ArticulationCfg` and
validation of:

- joint names, axes, signs, limits, and default positions;
- base, foot, IMU, LiDAR, and camera frames;
- inertial and collision properties;
- contact body names;
- actuator model compatibility.

Custom joint names will not be guessed. Any required mapping change must update
the versioned contract and associated tests.
