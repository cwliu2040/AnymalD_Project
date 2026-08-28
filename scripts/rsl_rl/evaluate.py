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
    "--motion-audit",
    action="store_true",
    help="Collect fixed-command gait, safety, body-motion and LiDAR scan-motion metrics.",
)
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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
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
        motion_audit = None
        if args_cli.motion_audit:
            contact_sensor = env.unwrapped.scene["contact_forces"]
            contact_foot_ids, contact_foot_names = contact_sensor.find_bodies(
                ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"],
                preserve_order=True,
            )
            robot_foot_ids, robot_foot_names = robot.find_bodies(
                ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"],
                preserve_order=True,
            )
            expected_feet = ("LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT")
            if tuple(contact_foot_names) != expected_feet or tuple(robot_foot_names) != expected_feet:
                raise RuntimeError(
                    "motion-audit foot mapping changed: "
                    f"contact={contact_foot_names}, robot={robot_foot_names}"
                )
            previous_contact = (
                torch.linalg.vector_norm(
                    contact_sensor.data.net_forces_w[:, contact_foot_ids], dim=-1
                )
                > 1.0
            )
            previous_linear_velocity = robot.data.root_lin_vel_b.clone()
            previous_angular_velocity = robot.data.root_ang_vel_b.clone()
            previous_linear_acceleration = torch.zeros_like(previous_linear_velocity)
            previous_joint_velocity = robot.data.joint_vel.clone()
            previous_actions = torch.zeros(
                (env.num_envs, env.num_actions), device=device
            )
            motion_audit = {
                "valid_samples": torch.tensor(0, dtype=torch.long, device=device),
                "roll_pitch_rate_sq": torch.tensor(0.0, device=device),
                "yaw_error_sq": torch.tensor(0.0, device=device),
                "linear_error_sq": torch.tensor(0.0, device=device),
                "linear_acceleration_sq": torch.tensor(0.0, device=device),
                "angular_acceleration_sq": torch.tensor(0.0, device=device),
                "linear_jerk_sq": torch.tensor(0.0, device=device),
                "lidar_translation_sq": torch.tensor(0.0, device=device),
                "lidar_rotation_sq": torch.tensor(0.0, device=device),
                "action_rate_sq": torch.tensor(0.0, device=device),
                "joint_acceleration_sq": torch.tensor(0.0, device=device),
                "torque_sq": torch.tensor(0.0, device=device),
                "mechanical_energy_j": torch.tensor(0.0, device=device),
                "positive_progress": torch.tensor(0.0, device=device),
                "positive_linear_progress": torch.tensor(0.0, device=device),
                "positive_yaw_progress": torch.tensor(0.0, device=device),
                "planar_speed_sum": torch.tensor(0.0, device=device),
                "absolute_yaw_rate_sum": torch.tensor(0.0, device=device),
                "stopped_samples": torch.tensor(0, dtype=torch.long, device=device),
                "contact_samples": torch.tensor(0, dtype=torch.long, device=device),
                "contact_switches": torch.tensor(0, dtype=torch.long, device=device),
                "touchdowns": torch.tensor(0, dtype=torch.long, device=device),
                "stance_slip_speed_sq": torch.tensor(0.0, device=device),
                "stance_width_sum": torch.tensor(0.0, device=device),
                "body_height_sum": torch.tensor(0.0, device=device),
                "swing_clearance_sum": torch.tensor(0.0, device=device),
                "swing_samples": torch.tensor(0, dtype=torch.long, device=device),
                "minimum_joint_margin": torch.tensor(float("inf"), device=device),
            }
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

            if motion_audit is not None:
                dt = float(env.unwrapped.step_dt)
                linear_velocity = robot.data.root_lin_vel_b
                angular_velocity = robot.data.root_ang_vel_b
                linear_acceleration = (
                    linear_velocity - previous_linear_velocity
                ) / dt
                angular_acceleration = (
                    angular_velocity - previous_angular_velocity
                ) / dt
                linear_jerk = (
                    linear_acceleration - previous_linear_acceleration
                ) / dt
                joint_acceleration = (
                    robot.data.joint_vel - previous_joint_velocity
                ) / dt
                action_rate = (actions - previous_actions) / dt
                linear_error = linear_velocity[:, :2] - scheduled_command[:2]
                yaw_error = angular_velocity[:, 2] - scheduled_command[2]
                lidar_translation = 0.5 * linear_acceleration * 0.10**2
                lidar_rotation = 0.5 * angular_acceleration * 0.10**2
                lidar_rotation = lidar_rotation.clone()
                lidar_rotation[:, :2] += angular_velocity[:, :2] * 0.10
                contact = (
                    torch.linalg.vector_norm(
                        contact_sensor.data.net_forces_w[:, contact_foot_ids], dim=-1
                    )
                    > 1.0
                )
                switches = contact != previous_contact
                touchdowns = contact & ~previous_contact

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

                    if motion_audit is not None:
                        foot_planar_speed = torch.linalg.vector_norm(
                            robot.data.body_lin_vel_w[:, robot_foot_ids, :2], dim=-1
                        )
                        valid_feet = valid[:, None].expand_as(contact)
                        stance = valid_feet & contact
                        swing = valid_feet & ~contact

                        foot_relative_w = (
                            robot.data.body_pos_w[:, robot_foot_ids]
                            - robot.data.root_pos_w[:, None, :]
                        )
                        root_quaternion = robot.data.root_quat_w[:, None, :].expand(
                            -1, len(robot_foot_ids), -1
                        )
                        foot_relative_b = quat_apply_inverse(
                            root_quaternion.reshape(-1, 4),
                            foot_relative_w.reshape(-1, 3),
                        ).reshape(env.num_envs, len(robot_foot_ids), 3)
                        stance_width = torch.abs(
                            foot_relative_b[:, :2, 1].mean(dim=1)
                            - foot_relative_b[:, 2:, 1].mean(dim=1)
                        )
                        terrain_height = env.unwrapped.scene.env_origins[:, 2]
                        foot_clearance = (
                            robot.data.body_pos_w[:, robot_foot_ids, 2]
                            - terrain_height[:, None]
                        ).clamp_min(0.0)
                        body_height = robot.data.root_pos_w[:, 2] - terrain_height
                        lower = robot.data.soft_joint_pos_limits[..., 0]
                        upper = robot.data.soft_joint_pos_limits[..., 1]
                        joint_margin = torch.minimum(
                            robot.data.joint_pos - lower,
                            upper - robot.data.joint_pos,
                        )
                        torque = robot.data.applied_torque
                        power = torch.abs(torque * robot.data.joint_vel).sum(dim=1)
                        command_xy = scheduled_command[:2]
                        command_speed = torch.linalg.vector_norm(command_xy)
                        if float(command_speed.item()) >= 0.25:
                            command_direction = command_xy / command_speed
                            progress_rate = torch.sum(
                                linear_velocity[:, :2] * command_direction, dim=1
                            )
                        elif abs(float(scheduled_command[2].item())) >= 0.25:
                            progress_rate = (
                                angular_velocity[:, 2]
                                * torch.sign(scheduled_command[2])
                            )
                        else:
                            progress_rate = torch.zeros(env.num_envs, device=device)
                        linear_progress_rate = torch.zeros(env.num_envs, device=device)
                        if float(command_speed.item()) >= 0.25:
                            linear_progress_rate = torch.sum(
                                linear_velocity[:, :2] * (command_xy / command_speed),
                                dim=1,
                            )
                        yaw_progress_rate = torch.zeros(env.num_envs, device=device)
                        if abs(float(scheduled_command[2].item())) >= 0.25:
                            yaw_progress_rate = (
                                angular_velocity[:, 2] * torch.sign(scheduled_command[2])
                            )
                        stopped = (
                            torch.linalg.vector_norm(linear_velocity[:, :2], dim=1)
                            < 0.10
                        ) & (torch.abs(angular_velocity[:, 2]) < 0.10)

                        motion_audit["valid_samples"] += valid.sum()
                        motion_audit["roll_pitch_rate_sq"] += torch.sum(
                            torch.square(angular_velocity[valid, :2])
                        )
                        motion_audit["yaw_error_sq"] += torch.sum(
                            torch.square(yaw_error[valid])
                        )
                        motion_audit["linear_error_sq"] += torch.sum(
                            torch.sum(torch.square(linear_error[valid]), dim=1)
                        )
                        for name, value in (
                            ("linear_acceleration_sq", linear_acceleration),
                            ("angular_acceleration_sq", angular_acceleration),
                            ("linear_jerk_sq", linear_jerk),
                            ("lidar_translation_sq", lidar_translation),
                            ("lidar_rotation_sq", lidar_rotation),
                        ):
                            motion_audit[name] += torch.sum(
                                torch.sum(torch.square(value[valid]), dim=1)
                            )
                        motion_audit["action_rate_sq"] += torch.sum(
                            torch.square(action_rate[valid])
                        )
                        motion_audit["joint_acceleration_sq"] += torch.sum(
                            torch.square(joint_acceleration[valid])
                        )
                        motion_audit["torque_sq"] += torch.sum(
                            torch.square(torque[valid])
                        )
                        motion_audit["mechanical_energy_j"] += (
                            torch.sum(power[valid]) * dt
                        )
                        motion_audit["positive_progress"] += (
                            torch.sum(torch.clamp_min(progress_rate[valid], 0.0)) * dt
                        )
                        motion_audit["positive_linear_progress"] += (
                            torch.sum(torch.clamp_min(linear_progress_rate[valid], 0.0)) * dt
                        )
                        motion_audit["positive_yaw_progress"] += (
                            torch.sum(torch.clamp_min(yaw_progress_rate[valid], 0.0)) * dt
                        )
                        motion_audit["planar_speed_sum"] += torch.sum(
                            torch.linalg.vector_norm(linear_velocity[valid, :2], dim=1)
                        )
                        motion_audit["absolute_yaw_rate_sum"] += torch.sum(
                            torch.abs(angular_velocity[valid, 2])
                        )
                        motion_audit["stopped_samples"] += torch.sum(stopped & valid)
                        motion_audit["contact_samples"] += torch.sum(stance)
                        motion_audit["contact_switches"] += torch.sum(
                            switches & valid_feet
                        )
                        motion_audit["touchdowns"] += torch.sum(
                            touchdowns & valid_feet
                        )
                        motion_audit["stance_slip_speed_sq"] += torch.sum(
                            torch.square(foot_planar_speed[stance])
                        )
                        motion_audit["stance_width_sum"] += torch.sum(
                            stance_width[valid]
                        )
                        motion_audit["body_height_sum"] += torch.sum(body_height[valid])
                        motion_audit["swing_clearance_sum"] += torch.sum(
                            foot_clearance[swing]
                        )
                        motion_audit["swing_samples"] += torch.sum(swing)
                        motion_audit["minimum_joint_margin"] = torch.minimum(
                            motion_audit["minimum_joint_margin"],
                            torch.min(joint_margin[valid]),
                        )

            if motion_audit is not None:
                previous_contact.copy_(contact)
                previous_linear_velocity.copy_(robot.data.root_lin_vel_b)
                previous_angular_velocity.copy_(robot.data.root_ang_vel_b)
                previous_linear_acceleration.copy_(linear_acceleration)
                previous_joint_velocity.copy_(robot.data.joint_vel)
                previous_actions.copy_(actions)
                reset_rows = dones_bool
                previous_linear_acceleration[reset_rows] = 0.0
                previous_actions[reset_rows] = 0.0

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
        if motion_audit is not None:
            audit_count = int(motion_audit["valid_samples"].item())
            joint_count = audit_count * int(robot.data.joint_pos.shape[1])
            foot_count = audit_count * 4
            contact_count = int(motion_audit["contact_samples"].item())
            swing_count = int(motion_audit["swing_samples"].item())
            progress = float(motion_audit["positive_progress"].item())

            def rms_sum(name: str, denominator: int) -> float:
                return math.sqrt(float(motion_audit[name].item()) / max(denominator, 1))

            result["motion_audit"] = {
                "valid_sample_count": audit_count,
                "body_lidar_metrics": {
                    "roll_pitch_rate_rms_radps": rms_sum("roll_pitch_rate_sq", audit_count),
                    "yaw_tracking_error_rms_radps": rms_sum("yaw_error_sq", audit_count),
                    "linear_tracking_error_rms_mps": rms_sum("linear_error_sq", audit_count),
                    "linear_acceleration_rms_mps2": rms_sum("linear_acceleration_sq", audit_count),
                    "angular_acceleration_rms_radps2": rms_sum("angular_acceleration_sq", audit_count),
                    "linear_jerk_rms_mps3": rms_sum("linear_jerk_sq", audit_count),
                    "lidar_scan_translation_error_rms_m": rms_sum("lidar_translation_sq", audit_count),
                    "lidar_scan_rotation_error_rms_rad": rms_sum("lidar_rotation_sq", audit_count),
                },
                "gait_safety_metrics": {
                    "cadence_hz": float(motion_audit["touchdowns"].item()) / max(audit_count * step_dt * 4, 1.0e-9),
                    "contact_switch_rate_hz": float(motion_audit["contact_switches"].item()) / max(audit_count * step_dt * 4, 1.0e-9),
                    "duty_factor": contact_count / max(foot_count, 1),
                    "body_height_m": float(motion_audit["body_height_sum"].item()) / max(audit_count, 1),
                    "stance_width_m": float(motion_audit["stance_width_sum"].item()) / max(audit_count, 1),
                    "minimum_joint_margin_rad": float(motion_audit["minimum_joint_margin"].item()),
                    "foot_clearance_m": float(motion_audit["swing_clearance_sum"].item()) / max(swing_count, 1),
                    "torque_rms_nm": rms_sum("torque_sq", joint_count),
                    "energy_per_progress": float(motion_audit["mechanical_energy_j"].item()) / max(progress, 1.0e-9),
                    "stance_foot_slip_rms_mps": rms_sum("stance_slip_speed_sq", contact_count),
                    "stopped_fraction": int(motion_audit["stopped_samples"].item()) / max(audit_count, 1),
                },
                "additional_metrics": {
                    "mean_planar_speed_mps": float(motion_audit["planar_speed_sum"].item()) / max(audit_count, 1),
                    "mean_absolute_yaw_rate_radps": float(motion_audit["absolute_yaw_rate_sum"].item()) / max(audit_count, 1),
                    "action_rate_rms_per_s": rms_sum("action_rate_sq", joint_count),
                    "joint_acceleration_rms_radps2": rms_sum("joint_acceleration_sq", joint_count),
                    "positive_command_aligned_progress": progress,
                    "positive_linear_progress_m": float(motion_audit["positive_linear_progress"].item()),
                    "positive_yaw_progress_rad": float(motion_audit["positive_yaw_progress"].item()),
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
