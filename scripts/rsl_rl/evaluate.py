# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint with one fixed body-frame velocity command."""

import argparse
import math
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
    "--ramp-up-start-s",
    type=float,
    default=0.0,
    help="Hold zero command for this duration before a smooth ramp.",
)
parser.add_argument(
    "--ramp-up-duration-s",
    type=float,
    default=0.0,
    help="Cubic smoothstep duration from zero to the requested command.",
)
parser.add_argument(
    "--hold-duration-s",
    type=float,
    default=None,
    help=(
        "Optional full-command hold duration. When set, the command ramps "
        "back to zero instead of remaining active for the whole horizon."
    ),
)
parser.add_argument(
    "--ramp-down-duration-s",
    type=float,
    default=None,
    help="Ramp-down duration; defaults to --ramp-up-duration-s.",
)
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
parser.add_argument(
    "--inference-backend",
    choices=("checkpoint", "torchscript", "onnx"),
    default="checkpoint",
    help="Policy implementation used inside the otherwise identical Isaac Lab evaluation.",
)
parser.add_argument(
    "--policy-artifact",
    type=str,
    default=None,
    help="Required TorchScript or ONNX artifact for the selected exported backend.",
)
parser.add_argument(
    "--factory-usd-path",
    type=str,
    default=None,
    help="Optional project-owned Factory USD replacing the Flat plane.",
)
parser.add_argument("--spawn-x", type=float, default=0.0)
parser.add_argument("--spawn-y", type=float, default=-18.0)
parser.add_argument("--spawn-yaw", type=float, default=0.0)
parser.add_argument(
    "--enhanced-determinism",
    action="store_true",
    help="Enable the same PhysX mode used by Factory deployment.",
)
parser.add_argument(
    "--ground-only-collision",
    action="store_true",
    help="Diagnostic mode: disable Factory colliders except its infinite plane.",
)
parser.add_argument(
    "--disable-factory-collision-group",
    action="append",
    default=[],
    help="Diagnostic runtime-only top-level Factory collision group to disable.",
)
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
if (
    not math.isfinite(args_cli.ramp_up_start_s)
    or not math.isfinite(args_cli.ramp_up_duration_s)
    or args_cli.ramp_up_start_s < 0.0
    or args_cli.ramp_up_duration_s < 0.0
):
    parser.error("ramp timing must be finite and non-negative")
if args_cli.hold_duration_s is not None and (
    not math.isfinite(args_cli.hold_duration_s)
    or args_cli.hold_duration_s < 0.0
):
    parser.error("--hold-duration-s must be finite and non-negative")
if args_cli.ramp_down_duration_s is not None and (
    not math.isfinite(args_cli.ramp_down_duration_s)
    or args_cli.ramp_down_duration_s < 0.0
):
    parser.error("--ramp-down-duration-s must be finite and non-negative")
if args_cli.inference_backend != "checkpoint" and not args_cli.policy_artifact:
    parser.error("--policy-artifact is required for exported inference backends")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Imports below this point require the running Isaac Sim application."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import omni.usd
import torch
from pxr import UsdPhysics
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat
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
    env_cfg.sim.physx.enable_enhanced_determinism = (
        args_cli.enhanced_determinism
    )
    factory_usd_path = (
        Path(args_cli.factory_usd_path).expanduser().resolve()
        if args_cli.factory_usd_path
        else None
    )
    if factory_usd_path is not None:
        if not factory_usd_path.is_file():
            raise FileNotFoundError(
                f"Factory USD does not exist: {factory_usd_path}"
            )
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/Factory",
            terrain_type="usd",
            usd_path=str(factory_usd_path),
            env_spacing=2.5,
            collision_group=-1,
            debug_vis=False,
        )
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (args_cli.spawn_x, args_cli.spawn_x),
            "y": (args_cli.spawn_y, args_cli.spawn_y),
            "yaw": (args_cli.spawn_yaw, args_cli.spawn_yaw),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            name: (0.0, 0.0)
            for name in (
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
            )
        }

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
    disabled_factory_collision_count = 0
    disabled_collision_groups = tuple(
        str(value) for value in args_cli.disable_factory_collision_group
    )
    if args_cli.ground_only_collision or disabled_collision_groups:
        if factory_usd_path is None:
            raise ValueError(
                "Factory collision filtering requires --factory-usd-path"
            )
        stage = omni.usd.get_context().get_stage()
        ground_path = (
            "/World/Factory/terrain/GroundPlane/CollisionPlane"
        )
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            relative_path = prim_path.removeprefix(
                "/World/Factory/terrain/"
            )
            group = relative_path.split("/", maxsplit=1)[0]
            should_disable = (
                args_cli.ground_only_collision
                and prim_path != ground_path
            ) or group in disabled_collision_groups
            if (
                prim_path.startswith("/World/Factory/")
                and should_disable
                and prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                UsdPhysics.CollisionAPI(
                    prim
                ).CreateCollisionEnabledAttr(False)
                disabled_factory_collision_count += 1
        gym_env.reset()
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    try:
        if agent_cfg.class_name != "OnPolicyRunner":
            raise ValueError(f"Only OnPolicyRunner is supported, got: {agent_cfg.class_name}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        checkpoint_policy = runner.get_inference_policy(
            device=env.unwrapped.device
        )
        policy_nn = runner.alg.policy
        policy_artifact = (
            Path(args_cli.policy_artifact).expanduser().resolve()
            if args_cli.policy_artifact
            else None
        )
        if policy_artifact is not None and not policy_artifact.is_file():
            raise FileNotFoundError(
                f"Policy artifact does not exist: {policy_artifact}"
            )
        if args_cli.inference_backend == "torchscript":
            exported_policy = torch.jit.load(
                str(policy_artifact),
                map_location=env.unwrapped.device,
            ).eval()
        elif args_cli.inference_backend == "onnx":
            from onnx.reference import ReferenceEvaluator

            exported_policy = ReferenceEvaluator(str(policy_artifact))
            if (
                len(exported_policy.input_names) != 1
                or len(exported_policy.output_names) != 1
            ):
                raise ValueError("ONNX policy must have one input and output")
            onnx_input_name = exported_policy.input_names[0]

        def infer(observations):
            if args_cli.inference_backend == "checkpoint":
                return checkpoint_policy(observations)
            policy_observations = (
                observations["policy"]
                if "policy" in observations
                else observations
            )
            if args_cli.inference_backend == "torchscript":
                return exported_policy(policy_observations)
            inputs = policy_observations.detach().cpu().numpy()
            outputs = exported_policy.run(
                None,
                {onnx_input_name: inputs},
            )[0]
            return torch.as_tensor(
                outputs,
                dtype=policy_observations.dtype,
                device=policy_observations.device,
            )

        horizon_steps = args_cli.steps if args_cli.steps is not None else env.max_episode_length
        if args_cli.warmup_steps >= horizon_steps:
            raise ValueError("--warmup_steps must be smaller than the evaluation horizon")

        device = env.unwrapped.device
        target = torch.tensor([args_cli.vx, args_cli.vy, args_cli.wz], device=device)
        command_tensor = env.unwrapped.command_manager.get_command(
            "base_velocity"
        )

        def command_at(time_s: float) -> torch.Tensor:
            if time_s < args_cli.ramp_up_start_s:
                scale = 0.0
            elif args_cli.ramp_up_duration_s == 0.0:
                scale = 1.0
            elif time_s < (
                args_cli.ramp_up_start_s + args_cli.ramp_up_duration_s
            ):
                phase = (
                    time_s - args_cli.ramp_up_start_s
                ) / args_cli.ramp_up_duration_s
                scale = phase * phase * (3.0 - 2.0 * phase)
            else:
                scale = 1.0
            if args_cli.hold_duration_s is not None:
                hold_start = (
                    args_cli.ramp_up_start_s
                    + args_cli.ramp_up_duration_s
                )
                ramp_down_start = hold_start + args_cli.hold_duration_s
                ramp_down_duration = (
                    args_cli.ramp_down_duration_s
                    if args_cli.ramp_down_duration_s is not None
                    else args_cli.ramp_up_duration_s
                )
                if time_s >= ramp_down_start + ramp_down_duration:
                    scale = 0.0
                elif time_s >= ramp_down_start:
                    if ramp_down_duration == 0.0:
                        scale = 0.0
                    else:
                        phase = (
                            time_s - ramp_down_start
                        ) / ramp_down_duration
                        smooth = phase * phase * (3.0 - 2.0 * phase)
                        scale = 1.0 - smooth
            return scale * target

        command_tensor.copy_(command_at(0.0).expand_as(command_tensor))
        ever_failed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        first_failure_step = torch.full(
            (env.num_envs,),
            horizon_steps,
            dtype=torch.long,
            device=device,
        )
        first_failure_pose = torch.full(
            (env.num_envs, 6),
            float("nan"),
            device=device,
        )
        maximum_absolute_roll_pitch = torch.zeros(
            (env.num_envs, 2),
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
        robot = env.unwrapped.scene["robot"]
        for step in range(horizon_steps):
            scheduled_command = command_at(step * float(env.unwrapped.step_dt))
            command_tensor.copy_(
                scheduled_command.expand_as(command_tensor)
            )
            pre_step_position = robot.data.root_pos_w.clone()
            pre_step_quaternion = robot.data.root_quat_w.clone()
            pre_step_roll, pre_step_pitch, pre_step_yaw = (
                euler_xyz_from_quat(pre_step_quaternion)
            )
            maximum_absolute_roll_pitch = torch.maximum(
                maximum_absolute_roll_pitch,
                torch.stack(
                    (pre_step_roll.abs(), pre_step_pitch.abs()),
                    dim=1,
                ),
            )
            policy_observations = (
                obs["policy"] if "policy" in obs else obs
            )
            with torch.inference_mode():
                policy_observations[:, 9:12].copy_(
                    scheduled_command.expand(env.num_envs, 3)
                )
                actions = infer(obs)
                obs, rewards, dones, _ = env.step(actions)

            dones_bool = dones.to(dtype=torch.bool)
            terminated = env.unwrapped.reset_terminated.to(dtype=torch.bool)
            timed_out = env.unwrapped.reset_time_outs.to(dtype=torch.bool)
            new_failures = terminated & ~ever_failed
            first_failure_step[new_failures] = step + 1
            first_failure_pose[new_failures] = torch.cat(
                (
                    pre_step_position[new_failures],
                    pre_step_roll[new_failures, None],
                    pre_step_pitch[new_failures, None],
                    pre_step_yaw[new_failures, None],
                ),
                dim=1,
            )
            ever_failed |= terminated
            termination_events += terminated.sum()
            timeout_events += timed_out.sum()

            if step >= args_cli.warmup_steps:
                valid = ~dones_bool & ~ever_failed
                if torch.any(valid):
                    actual = torch.stack(
                        (
                            robot.data.root_lin_vel_b[:, 0],
                            robot.data.root_lin_vel_b[:, 1],
                            robot.data.root_ang_vel_b[:, 2],
                        ),
                        dim=1,
                    )
                    error = actual - scheduled_command
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
        failure_indices = torch.nonzero(ever_failed).flatten().tolist()
        failure_states = [
            {
                "environment_index": int(index),
                "time_seconds": float(
                    first_failure_step[index].item() * step_dt
                ),
                "position_m": {
                    axis: float(first_failure_pose[index, offset].item())
                    for offset, axis in enumerate(("x", "y", "z"))
                },
                "orientation_rad": {
                    axis: float(first_failure_pose[index, offset].item())
                    for offset, axis in enumerate(
                        ("roll", "pitch", "yaw"),
                        start=3,
                    )
                },
            }
            for index in failure_indices
        ]

        result = {
            "schema_version": 1,
            "task": args_cli.task,
            "factory_usd_path": (
                str(factory_usd_path)
                if factory_usd_path is not None
                else None
            ),
            "enhanced_determinism": args_cli.enhanced_determinism,
            "ground_only_collision": args_cli.ground_only_collision,
            "disabled_factory_collision_groups": list(
                disabled_collision_groups
            ),
            "disabled_factory_collision_count": (
                disabled_factory_collision_count
            ),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "inference": {
                "backend": args_cli.inference_backend,
                "artifact": (
                    str(policy_artifact)
                    if policy_artifact is not None
                    else str(checkpoint)
                ),
                "artifact_sha256": _sha256(
                    policy_artifact or checkpoint
                ),
            },
            "seed": args_cli.seed,
            "num_envs": env.num_envs,
            "command": {"vx_mps": args_cli.vx, "vy_mps": args_cli.vy, "wz_radps": args_cli.wz},
            "horizon": {
                "steps": horizon_steps,
                "seconds": horizon_steps * step_dt,
                "warmup_steps": args_cli.warmup_steps,
            },
            "command_schedule": {
                "ramp_up_start_s": args_cli.ramp_up_start_s,
                "ramp_up_duration_s": args_cli.ramp_up_duration_s,
                "hold_duration_s": args_cli.hold_duration_s,
                "ramp_down_duration_s": (
                    args_cli.ramp_down_duration_s
                    if args_cli.ramp_down_duration_s is not None
                    else (
                        args_cli.ramp_up_duration_s
                        if args_cli.hold_duration_s is not None
                        else None
                    )
                ),
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
                "maximum_absolute_roll_rad": float(
                    maximum_absolute_roll_pitch[:, 0].max().item()
                ),
                "maximum_absolute_pitch_rad": float(
                    maximum_absolute_roll_pitch[:, 1].max().item()
                ),
                "first_failure_states": failure_states,
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
