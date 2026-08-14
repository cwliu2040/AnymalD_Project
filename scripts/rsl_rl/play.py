# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--behavior_gate_config",
    type=Path,
    default=None,
    help="Machine-readable fixed confidence-state behavior gate configuration.",
)
parser.add_argument(
    "--confidence_supervisor_baseline",
    action="store_true",
    help=(
        "Evaluate a 48-D policy with the deterministic confidence profile applied "
        "only as an external command limiter."
    ),
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--evaluation_steps",
    type=int,
    default=None,
    help="Stop after this many inference steps and report metrics by confidence state.",
)
parser.add_argument(
    "--evaluation_output",
    type=Path,
    default=None,
    help="Optional project-local JSON path for confidence-state evaluation metrics.",
)
parser.add_argument(
    "--velocity_estimator_dataset_output",
    type=Path,
    default=None,
    help="Write simulator proprioception and GT velocity labels to a project-local NPZ.",
)
parser.add_argument(
    "--velocity_estimator_contact_threshold_n",
    type=float,
    default=1.0,
    help="Normal-force threshold used to create four binary contact inputs.",
)
parser.add_argument(
    "--velocity_estimator_fixed_command",
    type=float,
    nargs=3,
    metavar=("VX", "VY", "WZ"),
    default=None,
    help="Freeze one body command for a reproducible estimator capture.",
)
parser.add_argument(
    "--velocity_estimator_metadata",
    type=Path,
    default=None,
    help="Replace policy base velocity with a qualified project-local estimator artifact.",
)
parser.add_argument(
    "--export_only",
    action="store_true",
    help="Export the resolved checkpoint and exit without stepping simulation.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
behavior_gate_cfg = None
behavior_gate_path = None
if args_cli.behavior_gate_config is not None:
    behavior_gate_path = args_cli.behavior_gate_config.expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    if not behavior_gate_path.is_relative_to(project_root):
        parser.error("--behavior_gate_config must be inside the project repository")
    behavior_gate_cfg = yaml.safe_load(behavior_gate_path.read_text(encoding="utf-8"))
    args_cli.seed = int(behavior_gate_cfg["seed"])
    args_cli.num_envs = int(behavior_gate_cfg["num_envs"])
    args_cli.evaluation_steps = int(behavior_gate_cfg["steps"])
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import anymal_locomotion.tasks  # noqa: F401
from anymal_locomotion.artifacts import EXPORT_ROOT, LOG_ROOT, PROJECT_ROOT, assert_project_local_path
from anymal_locomotion.policy_contract import (
    POLICY_CONTRACT_PATH,
    SLAM_CONFIDENCE_POLICY_CONTRACT_PATH,
    write_export_metadata,
)
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d.mdp import (
    confidence_phase,
    simulated_slam_confidence,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.velocity_estimator_dataset_output is not None:
        if args_cli.evaluation_steps is None or args_cli.evaluation_steps <= 0:
            raise ValueError("velocity estimator collection requires --evaluation_steps")
        if env_cfg.scene.num_envs < 2:
            raise ValueError("velocity estimator collection requires at least two environments")
        if args_cli.velocity_estimator_contact_threshold_n <= 0.0:
            raise ValueError("contact threshold must be positive")
        # The first deployable estimator gate represents ordinary flat
        # navigation.  Recovery v0.4.0 already learned with these robustness
        # disturbances; do not let synthetic pushes dominate the estimator's
        # normal-operation accuracy gate.
        env_cfg.events.push_robot = None
        env_cfg.events.base_external_force_torque = None
        if args_cli.velocity_estimator_fixed_command is not None:
            vx, vy, wz = (float(value) for value in args_cli.velocity_estimator_fixed_command)
            command_cfg = env_cfg.commands.base_velocity
            command_cfg.heading_command = False
            command_cfg.rel_heading_envs = 0.0
            command_cfg.rel_standing_envs = 0.0
            command_cfg.resampling_time_range = (1000.0, 1000.0)
            command_cfg.ranges.lin_vel_x = (vx, vx)
            command_cfg.ranges.lin_vel_y = (vy, vy)
            command_cfg.ranges.ang_vel_z = (wz, wz)
            command_cfg.ranges.heading = None
    if args_cli.velocity_estimator_metadata is not None:
        estimator_metadata_path = args_cli.velocity_estimator_metadata.expanduser().resolve()
        if not estimator_metadata_path.is_relative_to(PROJECT_ROOT):
            raise ValueError("velocity estimator metadata must be inside the repository")
        # Closed-loop qualification uses the same clean flat operating envelope
        # as the estimator accuracy gate. Robustness pushes remain a later gate.
        env_cfg.events.push_robot = None
        env_cfg.events.base_external_force_torque = None
    if behavior_gate_cfg is not None:
        if "SlamConfidence" not in args_cli.task and not args_cli.confidence_supervisor_baseline:
            raise ValueError(
                "behavior gate requires a 51-D SlamConfidence task or "
                "--confidence_supervisor_baseline for a 48-D policy"
            )
        command_cfg = env_cfg.commands.base_velocity
        command = behavior_gate_cfg["command"]
        command_cfg.heading_command = False
        command_cfg.rel_heading_envs = 0.0
        command_cfg.rel_standing_envs = 0.0
        command_cfg.resampling_time_range = (1000.0, 1000.0)
        command_cfg.ranges.lin_vel_x = (float(command["vx"]), float(command["vx"]))
        command_cfg.ranges.lin_vel_y = (float(command["vy"]), float(command["vy"]))
        command_cfg.ranges.ang_vel_z = (float(command["wz"]), float(command["wz"]))
        command_cfg.ranges.heading = None
        env_cfg.observations.policy.enable_corruption = False
        if "SlamConfidence" in args_cli.task:
            env_cfg.observations.policy.slam_confidence.params["cycle_s"] = float(
                behavior_gate_cfg["cycle_s"]
            )
            env_cfg.observations.policy.slam_confidence.params[
                "phase_offset_mode"
            ] = "synchronized"
            velocity_command_term = env_cfg.observations.policy.velocity_commands
            if velocity_command_term.func is simulated_slam_confidence:
                raise RuntimeError("velocity command term cannot be the confidence observation")
            if "phase_offset_mode" in velocity_command_term.params:
                velocity_command_term.params["phase_offset_mode"] = "synchronized"
        env_cfg.events.base_external_force_torque = None
        env_cfg.events.push_robot = None

    # specify directory for logging experiments
    log_root_path = str(LOG_ROOT / "rsl_rl" / agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # Read-only evaluations and dataset collection must not rewrite an existing
    # policy export. Normal interactive play/export behavior remains unchanged.
    if (
        args_cli.velocity_estimator_dataset_output is None
        and args_cli.evaluation_steps is None
    ):
        export_model_dir = str(EXPORT_ROOT / agent_cfg.experiment_name / os.path.basename(log_dir))
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        resolved_agent_path = Path(export_model_dir) / "resolved_agent.yaml"
        resolved_agent_path.write_text(
            yaml.safe_dump(agent_cfg.to_dict(), sort_keys=False),
            encoding="utf-8",
        )
        contract_path = (
            SLAM_CONFIDENCE_POLICY_CONTRACT_PATH
            if "SlamConfidence" in args_cli.task
            else POLICY_CONTRACT_PATH
        )
        actor = getattr(policy_nn, "actor", None)
        extension_metadata = None
        if actor is not None and getattr(actor, "intent_auxiliary", False):
            extension_metadata = {
                "architecture": "frozen_model1450_aux_intent_structured_gait",
                "confidence_offset": int(actor.confidence_offset),
                "legacy_observation_dimension": int(actor.legacy_observation_dim),
                "gait_coordinates": [
                    "stride_attenuation",
                    "crouch",
                    "stance_width",
                    "action_smoothing",
                ],
                "gait_parameter_limits": [
                    float(value)
                    for value in actor.gait_parameter_limits.detach().cpu().tolist()
                ],
                "nonnegative_stride_smoothing": bool(
                    actor.nonnegative_stride_smoothing
                ),
                "suppress_gait_when_tracking_invalid": bool(
                    actor.suppress_gait_when_tracking_invalid
                ),
                "gait_delta_safe_scale_power": float(
                    actor.gait_delta_safe_scale_power
                ),
                "intent_blend_max": float(actor.intent_blend_max),
                "degraded_stride_envelope": {
                    "minimum_scale": float(actor.degraded_stride_min_scale),
                    "confidence_low": float(
                        actor.degraded_stride_confidence_low
                    ),
                    "confidence_high": float(
                        actor.degraded_stride_confidence_high
                    ),
                    "age_to_confidence_loss_ratio_max": float(
                        actor.degraded_stride_age_ratio_max
                    ),
                    "power": float(actor.degraded_stride_envelope_power),
                    "tracking_valid_only": True,
                },
                "runtime_ground_truth_inputs": False,
                "resolved_agent_config": {
                    "path": resolved_agent_path.name,
                    "sha256": _sha256(resolved_agent_path),
                },
            }
        write_export_metadata(
            export_model_dir,
            resume_path,
            contract_path=contract_path,
            policy_extension=extension_metadata,
        )
        if args_cli.export_only:
            print(f"[INFO] Export-only completed: {export_model_dir}")
            env.close()
            return

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    metric_sums = {
        name: {
            "count": 0,
            "linear_squared_error": 0.0,
            "yaw_absolute_error": 0.0,
            "planar_speed": 0.0,
            "absolute_yaw_rate": 0.0,
        }
        for name in ("healthy", "degraded", "invalid")
    }
    window_sums = {}
    if behavior_gate_cfg is not None:
        window_sums = {
            name: {
                "count": 0,
                "linear_squared_error": 0.0,
                "yaw_absolute_error": 0.0,
                "planar_speed": 0.0,
                "absolute_yaw_rate": 0.0,
                "speed_above_0_50": 0,
                "action_squared": 0.0,
                "action_rate_squared": 0.0,
                "body_tilt_squared": 0.0,
                "body_roll_pitch_rate_squared": 0.0,
                "absolute_vertical_speed": 0.0,
                "stance_foot_samples": 0,
                "stance_foot_slip_speed": 0.0,
                "effective_command_scale": 0.0,
                "gait_mode_counts": [0, 0, 0, 0],
                "policy_intent_blend": 0.0,
                "policy_gait_coordinates": [0.0, 0.0, 0.0, 0.0],
            }
            for name in behavior_gate_cfg["windows"]
        }
    hard_termination_count = 0
    hard_terminated_envs = torch.zeros(
        env.num_envs,
        dtype=torch.bool,
        device=env.unwrapped.device,
    )
    hard_termination_phase_counts = {
        "healthy": 0,
        "deceleration": 0,
        "invalid": 0,
        "recovery": 0,
    }
    timeout_count = 0
    estimator_batches: dict[str, list[np.ndarray]] | None = None
    estimator_episode_ids: torch.Tensor | None = None
    estimator_base_body_id: int | None = None
    estimator_contact_foot_ids: list[int] | None = None
    robot_foot_ids: list[int] | None = None
    estimator_model = None
    estimator_history = None
    estimator_history_count = None
    estimator_squared_error_sum = np.zeros(3, dtype=np.float64)
    estimator_absolute_error_sum = np.zeros(3, dtype=np.float64)
    estimator_error_samples = 0
    estimator_warmup_steps = 0
    if behavior_gate_cfg is not None or (
        args_cli.velocity_estimator_dataset_output is not None
        or args_cli.velocity_estimator_metadata is not None
    ):
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        contact_sensor = base_env.scene["contact_forces"]
        base_ids, base_names = robot.find_bodies("base")
        contact_foot_ids, contact_foot_names = contact_sensor.find_bodies(
            ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"],
            preserve_order=True,
        )
        resolved_robot_foot_ids, robot_foot_names = robot.find_bodies(
            ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"],
            preserve_order=True,
        )
        if base_names != ["base"] or tuple(contact_foot_names) != (
            "LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"
        ) or tuple(robot_foot_names) != (
            "LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"
        ):
            raise RuntimeError(
                "evaluation body mapping failed: "
                f"base={base_names}, contact_feet={contact_foot_names}, "
                f"robot_feet={robot_foot_names}"
            )
        estimator_base_body_id = int(base_ids[0])
        estimator_contact_foot_ids = list(contact_foot_ids)
        robot_foot_ids = list(resolved_robot_foot_ids)
        estimator_episode_ids = torch.zeros(
            env.num_envs, dtype=torch.int64, device=base_env.device
        )
        if args_cli.velocity_estimator_metadata is not None:
            metadata = json.loads(estimator_metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("schema_version") != 1
                or metadata.get("contract_id") != "anymal-d-proprioceptive-velocity-v1"
                or metadata.get("history_length") != 20
                or metadata.get("step_dimension") != 37
            ):
                raise ValueError("velocity estimator metadata contract mismatch")
            torchscript = metadata.get("artifacts", {}).get("torchscript", {})
            estimator_model_path = estimator_metadata_path.parent / str(
                torchscript.get("path", "")
            )
            if (
                not estimator_model_path.is_file()
                or _sha256(estimator_model_path) != torchscript.get("sha256")
            ):
                raise ValueError("velocity estimator TorchScript digest mismatch")
            estimator_model = torch.jit.load(
                str(estimator_model_path), map_location=base_env.device
            ).eval()
            estimator_history = torch.zeros(
                (env.num_envs, 20, 37),
                dtype=torch.float32,
                device=base_env.device,
            )
            estimator_history_count = torch.zeros(
                env.num_envs, dtype=torch.int64, device=base_env.device
            )
        if args_cli.velocity_estimator_dataset_output is not None:
            estimator_batches = {
                name: []
                for name in (
                    "steps",
                    "labels",
                    "environment_ids",
                    "episode_ids",
                    "timestamps_s",
                )
            }
    previous_evaluation_actions = torch.zeros(
        (env.num_envs, int(env.unwrapped.action_manager.total_action_dim)),
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            policy_observation = obs["policy"]
            if args_cli.confidence_supervisor_baseline:
                confidence_observation = simulated_slam_confidence(
                    env.unwrapped,
                    cycle_s=float(behavior_gate_cfg["cycle_s"]),
                    phase_offset_mode="synchronized",
                )
                confidence = confidence_observation[:, 0]
                valid = confidence_observation[:, 1]
                age = confidence_observation[:, 2]
                confidence_scale = valid * torch.clamp(
                    (confidence - 0.2) / 0.8, 0.0, 1.0
                )
            elif policy_observation.shape[1] >= 51:
                confidence = policy_observation[:, 48]
                valid = policy_observation[:, 49]
                age = policy_observation[:, 50]
                confidence_scale = valid * torch.clamp((confidence - 0.2) / 0.8, 0.0, 1.0)
            else:
                confidence = torch.ones(env.num_envs, device=policy_observation.device)
                valid = torch.ones_like(confidence)
                age = torch.zeros_like(confidence)
                confidence_scale = torch.ones_like(confidence)
            command = env.unwrapped.command_manager.get_command("base_velocity").clone()
            evaluation_phase = (
                confidence_phase(
                    env.unwrapped,
                    cycle_s=float(behavior_gate_cfg["cycle_s"]),
                )
                if behavior_gate_cfg is not None
                else None
            )
            # Build the deployable estimator input before policy inference.
            # Simulator GT velocity is only compared after inference; it is
            # never part of these 37 input columns.
            if estimator_model is not None:
                assert estimator_history is not None
                assert estimator_history_count is not None
                assert estimator_base_body_id is not None
                assert estimator_contact_foot_ids is not None
                base_env = env.unwrapped
                robot = base_env.scene["robot"]
                contact_sensor = base_env.scene["contact_forces"]
                gravity_w = torch.as_tensor(
                    base_env.cfg.sim.gravity,
                    device=base_env.device,
                    dtype=torch.float32,
                ).expand(env.num_envs, -1)
                specific_force_b = quat_apply_inverse(
                    robot.data.root_quat_w,
                    robot.data.body_lin_acc_w[:, estimator_base_body_id, :] - gravity_w,
                )
                contacts = (
                    torch.abs(
                        contact_sensor.data.net_forces_w[
                            :, estimator_contact_foot_ids, 2
                        ]
                    )
                    > args_cli.velocity_estimator_contact_threshold_n
                ).to(dtype=torch.float32)
                estimator_step = torch.cat(
                    (
                        policy_observation[:, 3:6],
                        specific_force_b,
                        policy_observation[:, 6:9],
                        policy_observation[:, 12:24],
                        policy_observation[:, 24:36],
                        contacts,
                    ),
                    dim=1,
                )
                estimator_history[:, :-1, :] = estimator_history[:, 1:, :].clone()
                estimator_history[:, -1, :] = estimator_step
                estimator_history_count = torch.clamp(
                    estimator_history_count + 1, max=20
                )
                ready = estimator_history_count == 20
                estimated_velocity = estimator_model(
                    estimator_history.reshape(env.num_envs, -1)
                )
                if (
                    estimated_velocity.shape != (env.num_envs, 3)
                    or not torch.all(torch.isfinite(estimated_velocity))
                    or torch.any(torch.abs(estimated_velocity) > 8.0)
                ):
                    raise RuntimeError("velocity estimator produced an invalid closed-loop output")
                ready_error = estimated_velocity[ready] - robot.data.root_lin_vel_b[ready]
                if ready_error.numel():
                    estimator_absolute_error_sum += (
                        torch.sum(torch.abs(ready_error), dim=0).cpu().numpy()
                    )
                    estimator_squared_error_sum += (
                        torch.sum(torch.square(ready_error), dim=0).cpu().numpy()
                    )
                    estimator_error_samples += int(ready.sum().item())
                policy_observation = policy_observation.clone()
                policy_observation[ready, 0:3] = estimated_velocity[ready]
                if args_cli.confidence_supervisor_baseline:
                    policy_observation[:, 9:12] *= confidence_scale.unsqueeze(-1)
                policy_input = dict(obs)
                policy_input["policy"] = policy_observation
                actions = policy(policy_input)
                actions[~ready] = 0.0
                estimator_warmup_steps += int((~ready).any().item())
            else:
                if args_cli.confidence_supervisor_baseline:
                    policy_observation = policy_observation.clone()
                    policy_observation[:, 9:12] *= confidence_scale.unsqueeze(-1)
                    policy_input = dict(obs)
                    policy_input["policy"] = policy_observation
                    actions = policy(policy_input)
                else:
                    actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            if estimator_history is not None and torch.any(dones):
                estimator_history[dones] = 0.0
                estimator_history_count[dones] = 0
            if estimator_batches is not None:
                assert estimator_episode_ids is not None
                assert estimator_base_body_id is not None
                assert estimator_contact_foot_ids is not None
                estimator_episode_ids += dones.to(dtype=torch.int64)
                base_env = env.unwrapped
                robot = base_env.scene["robot"]
                contact_sensor = base_env.scene["contact_forces"]
                gravity_w = torch.as_tensor(
                    base_env.cfg.sim.gravity,
                    device=base_env.device,
                    dtype=torch.float32,
                ).expand(env.num_envs, -1)
                specific_force_b = quat_apply_inverse(
                    robot.data.root_quat_w,
                    robot.data.body_lin_acc_w[:, estimator_base_body_id, :] - gravity_w,
                )
                normal_force = torch.abs(
                    contact_sensor.data.net_forces_w[:, estimator_contact_foot_ids, 2]
                )
                contacts = (
                    normal_force > args_cli.velocity_estimator_contact_threshold_n
                ).to(dtype=torch.float32)
                next_policy_observation = obs["policy"]
                estimator_step = torch.cat(
                    (
                        next_policy_observation[:, 3:6],
                        specific_force_b,
                        next_policy_observation[:, 6:9],
                        next_policy_observation[:, 12:24],
                        next_policy_observation[:, 24:36],
                        contacts,
                    ),
                    dim=1,
                )
                if estimator_step.shape[1] != 37:
                    raise RuntimeError(f"estimator step has shape {estimator_step.shape}")
                estimator_batches["steps"].append(estimator_step.cpu().numpy().astype(np.float32))
                estimator_batches["labels"].append(
                    robot.data.root_lin_vel_b.cpu().numpy().astype(np.float32)
                )
                estimator_batches["environment_ids"].append(
                    np.arange(env.num_envs, dtype=np.int64)
                )
                estimator_batches["episode_ids"].append(
                    estimator_episode_ids.cpu().numpy().astype(np.int64)
                )
                estimator_batches["timestamps_s"].append(
                    np.full(env.num_envs, (timestep + 1) * dt, dtype=np.float64)
                )
            velocity_linear = env.unwrapped.scene["robot"].data.root_lin_vel_b[:, :2]
            velocity_yaw = env.unwrapped.scene["robot"].data.root_ang_vel_b[:, 2]
            target_linear = command[:, :2] * confidence_scale.unsqueeze(-1)
            target_yaw = command[:, 2] * confidence_scale
            planar_speed = torch.linalg.vector_norm(velocity_linear, dim=1)
            masks = {
                "healthy": (valid >= 0.5) & (confidence >= 0.8) & (age <= 0.3),
                "degraded": (valid >= 0.5) & ~((confidence >= 0.8) & (age <= 0.3)),
                "invalid": valid < 0.5,
            }
            for name, mask in masks.items():
                count = int(mask.sum())
                if count == 0:
                    continue
                values = metric_sums[name]
                values["count"] += count
                values["linear_squared_error"] += float(
                    torch.sum(torch.sum(torch.square(target_linear[mask] - velocity_linear[mask]), dim=1))
                )
                values["yaw_absolute_error"] += float(torch.sum(torch.abs(target_yaw[mask] - velocity_yaw[mask])))
                values["planar_speed"] += float(torch.sum(planar_speed[mask]))
                values["absolute_yaw_rate"] += float(torch.sum(torch.abs(velocity_yaw[mask])))
            if behavior_gate_cfg is not None:
                assert estimator_contact_foot_ids is not None
                assert robot_foot_ids is not None
                robot = env.unwrapped.scene["robot"]
                contact_sensor = env.unwrapped.scene["contact_forces"]
                action_delta = actions - previous_evaluation_actions
                body_tilt = robot.data.projected_gravity_b[:, :2]
                body_roll_pitch_rate = robot.data.root_ang_vel_b[:, :2]
                absolute_vertical_speed = torch.abs(robot.data.root_lin_vel_b[:, 2])
                actor = getattr(policy_nn, "actor", None)
                policy_intent_blend = None
                policy_gait_coordinates = None
                if getattr(actor, "intent_auxiliary", False):
                    legacy_action = actor.backbone(
                        policy_observation[:, : actor.legacy_observation_dim]
                    )
                    policy_intent_blend = actor.intent_blend(
                        policy_observation, legacy_action
                    ).squeeze(-1)
                    confidence_input = policy_observation[:, actor.confidence_offset]
                    valid_input = policy_observation[:, actor.confidence_offset + 1]
                    safe_scale_input = torch.clamp(valid_input, 0.0, 1.0) * torch.clamp(
                        (confidence_input - 0.2) / 0.8, 0.0, 1.0
                    )
                    safe_observation = policy_observation[
                        :, : actor.legacy_observation_dim
                    ].clone()
                    safe_observation[
                        :, actor.command_offset : actor.command_offset
                        + actor.command_dimension
                    ] *= safe_scale_input.unsqueeze(-1)
                    safe_action = actor.backbone(safe_observation)
                    intent_action = legacy_action + policy_intent_blend.unsqueeze(-1) * (
                        safe_action - legacy_action
                    )
                    policy_gait_coordinates = actor.gait_coordinates(
                        policy_observation, intent_action
                    )
                contact_mask = (
                    torch.abs(
                        contact_sensor.data.net_forces_w[
                            :, estimator_contact_foot_ids, 2
                        ]
                    )
                    > args_cli.velocity_estimator_contact_threshold_n
                )
                foot_planar_speed = torch.linalg.vector_norm(
                    robot.data.body_lin_vel_w[:, robot_foot_ids, :2], dim=2
                )
                for name, window in behavior_gate_cfg["windows"].items():
                    start, end = (float(value) for value in window["phase"])
                    mask = (evaluation_phase >= start) & (evaluation_phase < end)
                    count = int(mask.sum())
                    if count == 0:
                        continue
                    values = window_sums[name]
                    values["count"] += count
                    values["linear_squared_error"] += float(
                        torch.sum(
                            torch.sum(
                                torch.square(target_linear[mask] - velocity_linear[mask]),
                                dim=1,
                            )
                        )
                    )
                    values["yaw_absolute_error"] += float(
                        torch.sum(torch.abs(target_yaw[mask] - velocity_yaw[mask]))
                    )
                    values["planar_speed"] += float(torch.sum(planar_speed[mask]))
                    values["absolute_yaw_rate"] += float(
                        torch.sum(torch.abs(velocity_yaw[mask]))
                    )
                    values["speed_above_0_50"] += int(
                        torch.sum(planar_speed[mask] > 0.50)
                    )
                    values["action_squared"] += float(
                        torch.sum(torch.square(actions[mask]))
                    )
                    values["action_rate_squared"] += float(
                        torch.sum(torch.square(action_delta[mask]))
                    )
                    values["body_tilt_squared"] += float(
                        torch.sum(torch.square(body_tilt[mask]))
                    )
                    values["body_roll_pitch_rate_squared"] += float(
                        torch.sum(torch.square(body_roll_pitch_rate[mask]))
                    )
                    values["absolute_vertical_speed"] += float(
                        torch.sum(absolute_vertical_speed[mask])
                    )
                    window_contacts = contact_mask[mask]
                    values["stance_foot_samples"] += int(window_contacts.sum())
                    values["stance_foot_slip_speed"] += float(
                        torch.sum(foot_planar_speed[mask][window_contacts])
                    )
                    raw_command_norm = torch.linalg.vector_norm(command[mask], dim=1)
                    effective_command_norm = torch.linalg.vector_norm(
                        policy_observation[mask, 9:12], dim=1
                    )
                    values["effective_command_scale"] += float(
                        torch.sum(
                            torch.where(
                                raw_command_norm > 1.0e-6,
                                effective_command_norm / raw_command_norm,
                                torch.ones_like(raw_command_norm),
                            )
                        )
                    )
                    if policy_intent_blend is not None:
                        values["policy_intent_blend"] += float(
                            torch.sum(policy_intent_blend[mask])
                        )
                    if policy_gait_coordinates is not None:
                        coordinate_sums = torch.sum(
                            policy_gait_coordinates[mask], dim=0
                        ).tolist()
                        values["policy_gait_coordinates"] = [
                            total + float(value)
                            for total, value in zip(
                                values["policy_gait_coordinates"], coordinate_sums
                            )
                        ]
                    governor = getattr(
                        env.unwrapped,
                        "_slam_confidence_gait_mode_governor",
                        None,
                    )
                    if governor is not None:
                        for mode_index in range(4):
                            values["gait_mode_counts"][mode_index] += int(
                                torch.sum(governor.mode[mask] == mode_index)
                            )
                previous_evaluation_actions.copy_(actions)
                previous_evaluation_actions[dones] = 0.0
            hard_reset = env.unwrapped.reset_terminated
            hard_termination_count += int(hard_reset.sum())
            hard_terminated_envs |= env.unwrapped.reset_terminated
            if evaluation_phase is not None and torch.any(hard_reset):
                phase_masks = {
                    "healthy": evaluation_phase < 0.30,
                    "deceleration": (evaluation_phase >= 0.30)
                    & (evaluation_phase < 0.50),
                    "invalid": (evaluation_phase >= 0.50)
                    & (evaluation_phase < 0.70),
                    "recovery": evaluation_phase >= 0.70,
                }
                for phase_name, phase_mask in phase_masks.items():
                    hard_termination_phase_counts[phase_name] += int(
                        torch.sum(hard_reset & phase_mask)
                    )
            timeout_count += int(env.unwrapped.reset_time_outs.sum())
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_length:
                break
        if args_cli.evaluation_steps is not None and timestep >= args_cli.evaluation_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.evaluation_steps is not None:
        report = {"checkpoint": str(resume_path), "steps": timestep, "num_envs": env.num_envs, "states": {}}
        if estimator_model is not None:
            if estimator_error_samples == 0:
                raise RuntimeError("closed-loop estimator produced no ready samples")
            report["velocity_estimator"] = {
                "metadata": str(estimator_metadata_path.relative_to(PROJECT_ROOT)),
                "samples": estimator_error_samples,
                "warmup_policy_steps": estimator_warmup_steps,
                "axis_mae_mps": (
                    estimator_absolute_error_sum / estimator_error_samples
                ).tolist(),
                "axis_rmse_mps": np.sqrt(
                    estimator_squared_error_sum / estimator_error_samples
                ).tolist(),
                "ground_truth_role": "offline_evaluator_only",
            }
        report["termination_summary"] = {
            "hard_termination_count": hard_termination_count,
            "hard_terminated_env_count": int(hard_terminated_envs.sum().item()),
            "hard_terminated_env_fraction": float(
                hard_terminated_envs.sum().item() / env.num_envs
            ),
            "hard_termination_phase_counts": hard_termination_phase_counts,
            "timeout_count": timeout_count,
        }
        for name, sums in metric_sums.items():
            count = sums["count"]
            report["states"][name] = {
                "samples": count,
                "linear_rmse": (sums["linear_squared_error"] / count) ** 0.5 if count else None,
                "yaw_mae": sums["yaw_absolute_error"] / count if count else None,
                "mean_planar_speed": sums["planar_speed"] / count if count else None,
                "mean_absolute_yaw_rate": sums["absolute_yaw_rate"] / count if count else None,
            }
        if behavior_gate_cfg is not None:
            report["behavior_gate_profile"] = {
                "path": str(behavior_gate_path),
                "sha256": _sha256(behavior_gate_path),
                "profile_id": behavior_gate_cfg["profile_id"],
                "seed": int(behavior_gate_cfg["seed"]),
                "command": behavior_gate_cfg["command"],
                "cycle_s": float(behavior_gate_cfg["cycle_s"]),
            }
            report["windows"] = {}
            for name, sums in window_sums.items():
                count = sums["count"]
                if count == 0:
                    raise RuntimeError(f"behavior window {name} has no samples")
                report["windows"][name] = {
                    "samples": count,
                    "linear_rmse": (sums["linear_squared_error"] / count) ** 0.5,
                    "yaw_mae": sums["yaw_absolute_error"] / count,
                    "mean_planar_speed": sums["planar_speed"] / count,
                    "mean_absolute_yaw_rate": sums["absolute_yaw_rate"] / count,
                    "speed_above_0_50_fraction": sums["speed_above_0_50"] / count,
                    "action_rms": (
                        sums["action_squared"] / (count * actions.shape[1])
                    ) ** 0.5,
                    "action_rate_rms": (
                        sums["action_rate_squared"] / (count * actions.shape[1])
                    ) ** 0.5,
                    "body_tilt_rms": (sums["body_tilt_squared"] / count) ** 0.5,
                    "body_roll_pitch_rate_rms_radps": (
                        sums["body_roll_pitch_rate_squared"] / count
                    ) ** 0.5,
                    "mean_absolute_vertical_speed_mps": (
                        sums["absolute_vertical_speed"] / count
                    ),
                    "mean_stance_foot_slip_speed_mps": (
                        sums["stance_foot_slip_speed"] / sums["stance_foot_samples"]
                        if sums["stance_foot_samples"]
                        else None
                    ),
                    "stance_foot_samples": sums["stance_foot_samples"],
                    "mean_effective_command_scale": (
                        sums["effective_command_scale"] / count
                    ),
                    "gait_mode_counts": sums["gait_mode_counts"],
                    "mean_policy_intent_blend": (
                        sums["policy_intent_blend"] / count
                    ),
                    "mean_policy_gait_coordinates": [
                        value / count for value in sums["policy_gait_coordinates"]
                    ],
                }
            windows = report["windows"]
            thresholds = behavior_gate_cfg["gates"]
            healthy_speed = windows["healthy"]["mean_planar_speed"]
            degraded_ratio = (
                windows["degraded_late"]["mean_planar_speed"] / healthy_speed
            )
            hard_termination_fraction = hard_termination_count / (timestep * env.num_envs)
            hard_terminated_env_count = int(hard_terminated_envs.sum())
            hard_terminated_env_fraction = hard_terminated_env_count / env.num_envs
            checks = {
                "healthy_linear_tracking": windows["healthy"]["linear_rmse"]
                <= float(thresholds["healthy_linear_rmse_max_mps"]),
                "healthy_speed_retained": healthy_speed
                >= float(thresholds["healthy_mean_planar_speed_min_mps"]),
                "degraded_deceleration": degraded_ratio
                <= float(thresholds["degraded_speed_ratio_to_healthy_max"]),
                "degraded_target_tracking": windows["degraded_late"]["linear_rmse"]
                <= float(thresholds["degraded_linear_rmse_max_mps"]),
                "invalid_mean_stop": windows["invalid_settled"]["mean_planar_speed"]
                <= float(thresholds["invalid_mean_planar_speed_max_mps"]),
                "invalid_tail_stop": windows["invalid_settled"]["speed_above_0_50_fraction"]
                <= float(thresholds["invalid_speed_above_0_50_fraction_max"]),
                "invalid_yaw_stop": windows["invalid_settled"]["mean_absolute_yaw_rate"]
                <= float(thresholds["invalid_mean_abs_yaw_rate_max_radps"]),
                "recovery_linear_tracking": windows["recovery_settled"]["linear_rmse"]
                <= float(thresholds["recovery_linear_rmse_max_mps"]),
                "recovery_speed_reacquired": windows["recovery_settled"]["mean_planar_speed"]
                >= float(thresholds["recovery_mean_planar_speed_min_mps"]),
                "hard_termination_rate": hard_terminated_env_fraction
                <= float(thresholds["hard_termination_fraction_max"]),
            }
            optional_yaw_checks = {
                "healthy_yaw_tracking": (
                    "healthy",
                    "healthy_yaw_mae_max_radps",
                ),
                "degraded_yaw_tracking": (
                    "degraded_late",
                    "degraded_yaw_mae_max_radps",
                ),
                "recovery_yaw_tracking": (
                    "recovery_settled",
                    "recovery_yaw_mae_max_radps",
                ),
            }
            for check_name, (window_name, threshold_name) in optional_yaw_checks.items():
                if threshold_name in thresholds:
                    checks[check_name] = windows[window_name]["yaw_mae"] <= float(
                        thresholds[threshold_name]
                    )
            report["behavior_gate"] = {
                "passed": all(checks.values()),
                "checks": checks,
                "derived": {
                    "degraded_speed_ratio_to_healthy": degraded_ratio,
                    "hard_termination_count": hard_termination_count,
                    "hard_termination_fraction": hard_termination_fraction,
                    "hard_terminated_env_count": hard_terminated_env_count,
                    "hard_terminated_env_fraction": hard_terminated_env_fraction,
                    "timeout_count": timeout_count,
                },
                "thresholds": thresholds,
            }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args_cli.evaluation_output is not None:
            output_path = assert_project_local_path(args_cli.evaluation_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
            print(f"[INFO] Wrote confidence evaluation: {output_path}")

    if estimator_batches is not None:
        output_path = assert_project_local_path(args_cli.velocity_estimator_dataset_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: np.concatenate(batches, axis=0)
            for name, batches in estimator_batches.items()
        }
        arrays.update(
            {
                "schema_version": np.asarray(1, dtype=np.int64),
                "contract_id": np.asarray("anymal-d-proprioceptive-velocity-v1"),
                "checkpoint_path": np.asarray(str(Path(resume_path).resolve().relative_to(PROJECT_ROOT))),
                "checkpoint_sha256": np.asarray(_sha256(Path(resume_path).resolve())),
                "seed": np.asarray(agent_cfg.seed, dtype=np.int64),
                "sample_period_s": np.asarray(dt, dtype=np.float64),
                "ground_truth_role": np.asarray("supervised_label_and_offline_evaluator_only"),
            }
        )
        np.savez_compressed(output_path, **arrays)
        print(
            f"[INFO] Wrote velocity estimator dataset: {output_path} "
            f"rows={arrays['steps'].shape[0]}",
            flush=True,
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
