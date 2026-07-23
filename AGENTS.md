# AGENTS.md

## Scope

These instructions apply to the entire repository rooted at:

`~/anymal_locomotion`

All project-owned training logs, configurations, checkpoints, and exported
policies must remain inside this repository.

## Protected Workspaces

- NEVER modify `~/IsaacLab` unless the user explicitly requests it.
- NEVER modify the legacy workspace at
  `~/Documents/anymal_project/anymal_ws`.
- Use `~/IsaacLab` only as a framework/dependency and as a reference for
  official implementations.

## Architecture Boundaries

- Treat Isaac Lab training and ROS 2 deployment as separate architectural
  layers.
- Do not import `rclpy` inside Isaac Sim or Isaac Lab Python environments.
- Do not use UDP as the final ROS 2 architecture.
- Use ROS 2 Bridge and/or Isaac Sim Action Graph for communication between
  Isaac Sim and ROS 2.

## Training and Policy Integration

- Maintain a deterministic mapping from joint names to policy indices.
- Prefer minimal changes based on official Isaac Lab locomotion tasks.
- Keep configurations, training artifacts, checkpoints, logs, and exported
  policies under `~/anymal_locomotion`.

## Deployment Direction

- Design all interfaces and policy integration with future sim-to-real
  deployment to a physical ANYmal-D in mind.
- Keep simulator-specific concerns isolated from ROS 2 deployment and hardware
  integration concerns.
