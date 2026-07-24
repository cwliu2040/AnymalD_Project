# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint with one fixed body-frame velocity command."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate ANYmal-D velocity tracking with a fixed command.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0",
    help="Registered Isaac Lab play task.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RSL-RL agent configuration entry point.",
)
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel evaluation environments.")
parser.add_argument("--seed", type=int, default=42, help="Deterministic evaluation seed.")
parser.add_argument("--vx", type=float, default=0.0, help="Fixed body-frame x velocity command in m/s.")
parser.add_argument("--vy", type=float, default=0.0, help="Fixed body-frame y velocity command in m/s.")
parser.add_argument("--wz", type=float, default=0.0, help="Fixed yaw-rate command in rad/s.")
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Evaluation horizon in policy steps; defaults to one complete episode.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=100,
    help="Initial policy steps excluded from velocity statistics.",
)
parser.add_argument("--output", type=str, default=None, help="Optional JSON result path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.steps is not None and args_cli.steps <= 0:
    parser.error("--steps must be positive")
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps must be non-negative")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Imports below this point require the running Isaac Sim application."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401

import anymal_locomotion.tasks  # noqa: F401
from anymal_locomotion.artifacts import LOG_ROOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_output_path(checkpoint: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    command = f"vx_{args_cli.vx:+.2f}_vy_{args_cli.vy:+.2f}_wz_{args_cli.wz:+.2f}"
    safe_command = command.replace("+", "p").replace("-", "m").replace(".", "p")
    return LOG_ROOT / "evaluation" / checkpoint.parent.name / f"{stamp}_{safe_command}.json"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Run one fixed-command evaluation and write a machine-readable result."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    command_cfg.ranges.lin_vel_y = (args_cli.vy, args_cli.vy)
    command_cfg.ranges.ang_vel_z = (args_cli.wz, args_cli.wz)
    command_cfg.ranges.heading = None

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None

    checkpoint = Path(retrieve_file_path(args_cli.checkpoint)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    try:
        if agent_cfg.class_name != "OnPolicyRunner":
            raise ValueError(f"Only OnPolicyRunner is supported, got: {agent_cfg.class_name}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = runner.alg.policy

        horizon_steps = args_cli.steps if args_cli.steps is not None else env.max_episode_length
        if args_cli.warmup_steps >= horizon_steps:
            raise ValueError("--warmup_steps must be smaller than the evaluation horizon")

        device = env.unwrapped.device
        target = torch.tensor([args_cli.vx, args_cli.vy, args_cli.wz], device=device)
        ever_failed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        first_failure_step = torch.full(
            (env.num_envs,),
            horizon_steps,
            dtype=torch.long,
            device=device,
        )
        actual_sum = torch.zeros(3, device=device)
        absolute_error_sum = torch.zeros(3, device=device)
        squared_error_sum = torch.zeros(3, device=device)
        reward_sum = torch.tensor(0.0, device=device)
        valid_sample_count = torch.tensor(0, dtype=torch.long, device=device)
        termination_events = torch.tensor(0, dtype=torch.long, device=device)
        timeout_events = torch.tensor(0, dtype=torch.long, device=device)

        obs = env.get_observations()
        for step in range(horizon_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)

            dones_bool = dones.to(dtype=torch.bool)
            terminated = env.unwrapped.reset_terminated.to(dtype=torch.bool)
            timed_out = env.unwrapped.reset_time_outs.to(dtype=torch.bool)
            new_failures = terminated & ~ever_failed
            first_failure_step[new_failures] = step + 1
            ever_failed |= terminated
            termination_events += terminated.sum()
            timeout_events += timed_out.sum()

            if step >= args_cli.warmup_steps:
                valid = ~dones_bool & ~ever_failed
                if torch.any(valid):
                    robot = env.unwrapped.scene["robot"]
                    actual = torch.stack(
                        (
                            robot.data.root_lin_vel_b[:, 0],
                            robot.data.root_lin_vel_b[:, 1],
                            robot.data.root_ang_vel_b[:, 2],
                        ),
                        dim=1,
                    )
                    error = actual - target
                    actual_sum += actual[valid].sum(dim=0)
                    absolute_error_sum += error[valid].abs().sum(dim=0)
                    squared_error_sum += error[valid].square().sum(dim=0)
                    reward_sum += rewards[valid].sum()
                    valid_sample_count += valid.sum()

            policy_nn.reset(dones)

        sample_count = int(valid_sample_count.item())
        if sample_count == 0:
            raise RuntimeError("No valid post-warmup samples were collected")

        mean_actual = actual_sum / sample_count
        mean_absolute_error = absolute_error_sum / sample_count
        root_mean_squared_error = torch.sqrt(squared_error_sum / sample_count)
        step_dt = float(env.unwrapped.step_dt)

        result = {
            "schema_version": 1,
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "seed": args_cli.seed,
            "num_envs": env.num_envs,
            "command": {"vx_mps": args_cli.vx, "vy_mps": args_cli.vy, "wz_radps": args_cli.wz},
            "horizon": {
                "steps": horizon_steps,
                "seconds": horizon_steps * step_dt,
                "warmup_steps": args_cli.warmup_steps,
            },
            "tracking": {
                "valid_sample_count": sample_count,
                "mean_actual": {
                    "vx_mps": float(mean_actual[0].item()),
                    "vy_mps": float(mean_actual[1].item()),
                    "wz_radps": float(mean_actual[2].item()),
                },
                "mean_absolute_error": {
                    "vx_mps": float(mean_absolute_error[0].item()),
                    "vy_mps": float(mean_absolute_error[1].item()),
                    "wz_radps": float(mean_absolute_error[2].item()),
                },
                "root_mean_squared_error": {
                    "vx_mps": float(root_mean_squared_error[0].item()),
                    "vy_mps": float(root_mean_squared_error[1].item()),
                    "wz_radps": float(root_mean_squared_error[2].item()),
                },
                "mean_reward_per_step": float((reward_sum / sample_count).item()),
            },
            "stability": {
                "survival_rate": float((~ever_failed).float().mean().item()),
                "mean_time_to_first_failure_seconds": float(first_failure_step.float().mean().item() * step_dt),
                "termination_events": int(termination_events.item()),
                "timeout_events": int(timeout_events.item()),
            },
        }

        output_path = Path(args_cli.output).resolve() if args_cli.output else _default_output_path(checkpoint)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print("\n[INFO] Fixed-command evaluation completed.")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[INFO] Result written to: {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
