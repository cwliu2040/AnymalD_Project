#!/usr/bin/env python3
"""Launch a one-environment Isaac Sim smoke test without PPO training."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate the ANYmal-D Locomotion v1 runtime contract.")
parser.add_argument(
    "--task-variant",
    choices=("baseline", "robust"),
    default="baseline",
    help="Project task configuration to validate.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import anymal_locomotion.tasks  # noqa: F401
from anymal_locomotion.policy_contract import (
    CANONICAL_JOINT_ORDER,
    POLICY_CONTRACT,
    validate_runtime_joint_names,
)
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import (
    ROBUST_TASK_ID,
    TASK_ID,
)
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import (
    mdp as project_mdp,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _validate_symmetry() -> None:
    policy = torch.arange(96, device=args_cli.device, dtype=torch.float32).reshape(2, 48)
    actions = torch.arange(24, device=args_cli.device, dtype=torch.float32).reshape(2, 12)

    left_right_policy = project_mdp.symmetry._transform_observation(
        policy, left_right=True, front_back=False
    )
    front_back_policy = project_mdp.symmetry._transform_observation(
        policy, left_right=False, front_back=True
    )
    assert torch.equal(
        project_mdp.symmetry._transform_observation(
            left_right_policy, left_right=True, front_back=False
        ),
        policy,
    )
    assert torch.equal(
        project_mdp.symmetry._transform_observation(
            front_back_policy, left_right=False, front_back=True
        ),
        policy,
    )
    assert torch.equal(
        project_mdp.symmetry._left_right(project_mdp.symmetry._left_right(actions)),
        actions,
    )
    assert torch.equal(
        project_mdp.symmetry._front_back(project_mdp.symmetry._front_back(actions)),
        actions,
    )
    assert torch.equal(
        project_mdp.symmetry._front_back(project_mdp.symmetry._left_right(actions)),
        project_mdp.symmetry._left_right(project_mdp.symmetry._front_back(actions)),
    )


def _is_edge_command(command: torch.Tensor) -> torch.Tensor:
    modes = (
        ((2.5, 3.0), (-0.15, 0.15), (-0.15, 0.15)),
        ((-2.0, -1.5), (-0.15, 0.15), (-0.15, 0.15)),
        ((-0.2, 0.2), (1.2, 1.5), (-0.15, 0.15)),
        ((-0.2, 0.2), (-1.5, -1.2), (-0.15, 0.15)),
        ((-0.15, 0.15), (-0.15, 0.15), (1.5, 2.0)),
        ((-0.15, 0.15), (-0.15, 0.15), (-2.0, -1.5)),
        ((0.5, 3.0), (-0.2, 0.2), (-1.0, 1.0)),
    )
    matches = torch.zeros(command.shape[0], dtype=torch.bool, device=command.device)
    for mode in modes:
        mode_matches = torch.ones_like(matches)
        for component, (lower, upper) in enumerate(mode):
            mode_matches &= command[:, component] >= lower
            mode_matches &= command[:, component] <= upper
        matches |= mode_matches
    return matches


def main() -> None:
    task_id = ROBUST_TASK_ID if args_cli.task_variant == "robust" else TASK_ID
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 64 if args_cli.task_variant == "robust" else 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = 42
    env = gym.make(task_id, cfg=env_cfg)
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
        assert agent_cfg.max_iterations == (
            300 if args_cli.task_variant == "robust" else 1000
        )
        assert command_ranges.lin_vel_x == tuple(contract_limits["vx"])
        assert command_ranges.lin_vel_y == tuple(contract_limits["vy"])
        assert command_ranges.ang_vel_z == tuple(contract_limits["wz"])

        if args_cli.task_variant == "robust":
            command_term = base_env.command_manager.get_term("base_velocity")
            assert isinstance(
                command_term,
                project_mdp.EdgeBiasedVelocityCommand,
            )
            assert command_term.cfg.edge_probability == 0.6
            assert agent_cfg.algorithm.learning_rate == 3.0e-4
            assert agent_cfg.algorithm.entropy_coef == 0.003
            assert agent_cfg.algorithm.symmetry_cfg.use_data_augmentation
            original_probability = command_term.cfg.edge_probability
            command_term.cfg.edge_probability = 1.0
            command_term._resample_command(
                torch.arange(base_env.num_envs, device=args_cli.device)
            )
            assert torch.all(_is_edge_command(command_term.vel_command_b))
            command_term.cfg.edge_probability = original_probability
            _validate_symmetry()
            print(
                "PASS robust=edge-biased-command+feet-slide-reward"
                "+canonical-symmetry",
                flush=True,
            )

        print(f"PASS task={task_id}", flush=True)
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
