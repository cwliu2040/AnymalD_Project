#!/usr/bin/env python3
"""Launch a one-environment Isaac Sim smoke test without PPO training."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate the ANYmal-D Locomotion v1 runtime contract.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import anymal_locomotion.tasks  # noqa: F401
from anymal_locomotion.policy_contract import (
    CANONICAL_JOINT_ORDER,
    POLICY_CONTRACT,
    validate_runtime_joint_names,
)
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import TASK_ID
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main() -> None:
    env_cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = 42
    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        canonical_to_runtime = validate_runtime_joint_names(robot.joint_names)
        policy_shape = base_env.observation_manager.group_obs_dim["policy"]
        if not isinstance(policy_shape, tuple):
            raise RuntimeError(f"Expected concatenated policy observations, received shape metadata: {policy_shape}")
        observation_dim = policy_shape[-1]
        action_dim = base_env.action_manager.total_action_dim
        policy_rate_hz = 1.0 / base_env.step_dt
        command_ranges = env_cfg.commands.base_velocity.ranges
        contract_limits = POLICY_CONTRACT["command"]["limits"]

        assert observation_dim == POLICY_CONTRACT["observation"]["dimension"] == 48
        assert action_dim == POLICY_CONTRACT["action"]["dimension"] == 12
        assert env_cfg.scene.height_scanner is None
        assert env_cfg.observations.policy.height_scan is None
        assert policy_rate_hz == 50.0
        assert agent_cfg.max_iterations == 1000
        assert command_ranges.lin_vel_x == tuple(contract_limits["vx"])
        assert command_ranges.lin_vel_y == tuple(contract_limits["vy"])
        assert command_ranges.ang_vel_z == tuple(contract_limits["wz"])

        print(f"PASS task={TASK_ID}", flush=True)
        print(
            f"PASS observation={observation_dim} action={action_dim} policy_rate={policy_rate_hz:g}Hz",
            flush=True,
        )
        print(f"PASS runtime_joints={robot.joint_names}", flush=True)
        print(f"PASS canonical_joints={list(CANONICAL_JOINT_ORDER)}", flush=True)
        print(f"PASS canonical_to_runtime={list(canonical_to_runtime)}", flush=True)
        print(
            "PASS command_ranges="
            f"vx={command_ranges.lin_vel_x} vy={command_ranges.lin_vel_y} wz={command_ranges.ang_vel_z}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        simulation_app.close(skip_cleanup=True)
        raise
    else:
        simulation_app.close()
