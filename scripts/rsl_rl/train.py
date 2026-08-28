# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--actor-only-warm-start",
    type=Path,
    default=None,
    help=(
        "Initialize only std and actor.* from a 48-D checkpoint into a wider "
        "dense actor; critic and optimizer remain freshly initialized."
    ),
)
parser.add_argument(
    "--bootstrap-only",
    action="store_true",
    help="Save bootstrap_model_0.pt after actor-only warm-start and exit before learning.",
)
parser.add_argument(
    "--joint-training-protocol",
    type=Path,
    default=None,
    help="Required fail-closed protocol for J1/J2 full-policy training.",
)
parser.add_argument(
    "--authorized-joint-training",
    action="store_true",
    help="One-shot acknowledgment required in addition to both protocol PPO gates.",
)
parser.add_argument(
    "--velocity-estimator-metadata",
    type=Path,
    default=None,
    help=(
        "Train policy observations through a qualified repository-local "
        "proprioceptive velocity estimator artifact."
    ),
)
parser.add_argument(
    "--resume-checkpoint-path",
    type=Path,
    default=None,
    help=(
        "Resume from an explicit repository-local checkpoint while writing the "
        "continuation to the current task's experiment directory."
    ),
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.bootstrap_only and args_cli.actor_only_warm_start is None:
    parser.error("--bootstrap-only requires --actor-only-warm-start")
if args_cli.actor_only_warm_start is not None and args_cli.resume:
    parser.error("--actor-only-warm-start and --resume are mutually exclusive")
if args_cli.resume_checkpoint_path is not None and not args_cli.resume:
    parser.error("--resume-checkpoint-path requires --resume")

_joint_training_tasks = {
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-JointTraining-J1-v0",
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-JointTraining-J2-v0",
}
_joint_training_protocol = None
if args_cli.task in _joint_training_tasks:
    if args_cli.joint_training_protocol is None or not args_cli.authorized_joint_training:
        parser.error("J1/J2 training requires its protocol and one-shot authorization")
    _protocol_path = args_cli.joint_training_protocol.expanduser().resolve()
    _protocol = yaml.safe_load(_protocol_path.read_text(encoding="utf-8"))
    _joint_training_protocol = _protocol
    _budget = _protocol["training_budget"]
    _gates = _protocol["execution_gates"]
    _authorized_total_iterations = int(
        _budget.get("current_authorized_iterations_per_arm", _budget["maximum_iterations_per_seed"])
    )
    if not (_budget["execution_authorized"] and _gates["ppo_training_authorized"]):
        parser.error("J1/J2 PPO authorization gates are closed")
    _preflight_passed = _gates.get("nonlearning_preflight_passed") is True or (
        _protocol.get("nonlearning_preflight", {}).get("result", {}).get("status") == "passed"
    )
    if not _preflight_passed:
        parser.error("J1/J2 training requires a passing non-learning preflight")
    if any(
        _gates.get(name, False)
        for name in (
            "live_ros_wiring_authorized",
            "default_switch_authorized",
            "physical_robot_authorized",
            "blocks_581_584_allowed",
            "touchdown_blocks_602_605_allowed",
            "blocks_602_605_allowed",
        )
    ):
        parser.error("J1/J2 training requires all unrelated execution gates to remain closed")
    if args_cli.num_envs != int(_budget["num_environments_per_seed"]):
        parser.error("J1/J2 training environment count differs from the frozen protocol")
    if args_cli.seed not in [int(seed) for seed in _budget["independent_training_seeds"]]:
        parser.error("J1/J2 training seed is outside the frozen protocol")
    if args_cli.max_iterations is None or not (
        1 <= args_cli.max_iterations <= _authorized_total_iterations
    ):
        parser.error("J1/J2 iteration count exceeds the currently authorized total")
    if not args_cli.resume:
        _source = Path(_protocol["initialization_and_anchoring"]["source_checkpoint"])
        _source = (_protocol_path.parents[1] / _source).resolve()
        if args_cli.actor_only_warm_start is None:
            parser.error("fresh J1/J2 training requires the frozen model1450 warm-start")
        if args_cli.actor_only_warm_start.expanduser().resolve() != _source:
            parser.error("J1/J2 warm-start checkpoint differs from the frozen protocol")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import anymal_locomotion.tasks  # noqa: F401
from anymal_locomotion.artifacts import LOG_ROOT, PROJECT_ROOT, build_run_manifest
from anymal_locomotion.full_policy_bootstrap import (
    bootstrap_dense_actor_state,
    bootstrap_frozen_reference_actor_state,
    validate_fresh_optimizer_state,
)
from anymal_locomotion.velocity_estimator_training import (
    VelocityEstimatorTrainingWrapper,
    validate_velocity_estimator_artifact,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(state: dict[str, torch.Tensor], keys: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(keys):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def actor_only_warm_start(policy, optimizer, source_path: Path) -> dict[str, object]:
    """Copy the formal 48-D actor into a fresh wider-input runner."""
    source_path = source_path.expanduser().resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source = checkpoint["model_state_dict"]
    target = policy.state_dict()
    actor_keys = [name for name in target if name == "std" or name.startswith("actor.")]
    critic_keys = [name for name in target if name.startswith("critic.")]
    if tuple(source["actor.0.weight"].shape) != (128, 48):
        raise ValueError("source is not the expected 48-D model1450 actor")

    critic_before = _state_sha256(target, critic_keys)
    bootstrap_report = bootstrap_dense_actor_state(source, target)
    reference_actor_keys = [
        name for name in target if name.startswith("reference_actor.")
    ]
    if reference_actor_keys:
        bootstrap_frozen_reference_actor_state(source, target)
    critic_after = _state_sha256(policy.state_dict(), critic_keys)
    if critic_after != critic_before:
        raise RuntimeError("actor-only warm-start modified critic parameters")
    validate_fresh_optimizer_state(optimizer.state)
    target_input_dim = int(target["actor.0.weight"].shape[1])

    return {
        "schema_version": 1,
        "kind": f"actor_only_48_to_{target_input_dim}_warm_start",
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_iteration_ignored": int(checkpoint.get("iter", -1)),
        "target_iteration": 0,
        "copied_parameters": bootstrap_report["copied_parameters"],
        "actor_state_sha256": _state_sha256(policy.state_dict(), actor_keys),
        "reference_actor_state_sha256": (
            _state_sha256(policy.state_dict(), reference_actor_keys)
            if reference_actor_keys
            else None
        ),
        "behavior_reference_initialization": (
            "exact_frozen_model1450_actor" if reference_actor_keys else None
        ),
        "critic_state_sha256": critic_after,
        "critic_initialization": "fresh_runner_seeded_initialization",
        "optimizer_initialization": "fresh_empty_adam_state",
        "source_actor_input_dimension": 48,
        "target_actor_input_dimension": target_input_dim,
        "new_input_columns": bootstrap_report["new_input_columns"],
        "new_input_columns_initialization": "exact_zero",
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = str(LOG_ROOT / "rsl_rl" / agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    run_manifest = build_run_manifest(task_id=args_cli.task, seed=agent_cfg.seed, log_dir=log_dir)
    if _joint_training_protocol is not None:
        run_manifest["joint_training_protocol"] = {
            "path": str(_protocol_path),
            "sha256": _sha256(_protocol_path),
            "protocol_id": _joint_training_protocol["protocol_id"],
        }
    estimator_artifact = None
    if args_cli.velocity_estimator_metadata is not None:
        estimator_artifact = validate_velocity_estimator_artifact(
            args_cli.velocity_estimator_metadata
        )
        run_manifest["policy_observation_velocity_estimator"] = estimator_artifact

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if args_cli.resume_checkpoint_path is None:
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
        else:
            resume_path_obj = args_cli.resume_checkpoint_path.expanduser().resolve()
            if not resume_path_obj.is_relative_to(PROJECT_ROOT):
                raise ValueError("resume checkpoint must be inside the repository")
            if not resume_path_obj.is_file():
                raise FileNotFoundError(resume_path_obj)
            resume_path = str(resume_path_obj)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    if args_cli.velocity_estimator_metadata is None:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    else:
        env = VelocityEstimatorTrainingWrapper(
            env,
            metadata_path=args_cli.velocity_estimator_metadata,
            clip_actions=agent_cfg.clip_actions,
        )

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    warm_start_report = None
    if args_cli.actor_only_warm_start is not None:
        warm_start_report = actor_only_warm_start(
            runner.alg.policy,
            runner.alg.optimizer,
            args_cli.actor_only_warm_start,
        )
        runner.current_learning_iteration = 0
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        if args_cli.task in _joint_training_tasks:
            if not isinstance(runner, OnPolicyRunner):
                raise TypeError("J1/J2 continuation requires OnPolicyRunner")
            if _joint_training_protocol is None:
                raise RuntimeError("J1/J2 continuation protocol was not loaded")

            # RSL-RL checkpoints store the index of the last completed update.
            # Advance to the next index so continuation neither repeats an update
            # nor silently exceeds the protocol's fixed per-seed budget.
            loaded_iteration = int(runner.current_learning_iteration)
            completed_iterations = loaded_iteration + 1
            maximum_iterations = int(
                _joint_training_protocol["training_budget"].get(
                    "current_authorized_iterations_per_arm",
                    _joint_training_protocol["training_budget"]["maximum_iterations_per_seed"],
                )
            )
            if completed_iterations + int(agent_cfg.max_iterations) > maximum_iterations:
                raise ValueError(
                    "J1/J2 continuation exceeds the frozen iteration budget: "
                    f"{completed_iterations} completed + {agent_cfg.max_iterations} requested "
                    f"> {maximum_iterations}"
                )
            runner.current_learning_iteration = completed_iterations

            # A newly constructed Isaac Lab environment otherwise restarts its
            # curriculum at step zero. Restore the global rollout-step counter
            # and immediately re-apply every already-reached curriculum term.
            rollout_steps = int(agent_cfg.num_steps_per_env)
            restored_common_steps = completed_iterations * rollout_steps
            base_env = env.unwrapped
            base_env.common_step_counter = restored_common_steps
            all_env_ids = torch.arange(base_env.num_envs, device=base_env.device)
            base_env.curriculum_manager.compute(env_ids=all_env_ids)
            run_manifest["joint_training_continuation"] = {
                "checkpoint": str(resume_path),
                "checkpoint_sha256": _sha256(Path(resume_path)),
                "loaded_last_completed_iteration": loaded_iteration,
                "next_learning_iteration": completed_iterations,
                "restored_common_step_counter": restored_common_steps,
                "requested_additional_iterations": int(agent_cfg.max_iterations),
                "maximum_total_iterations": maximum_iterations,
            }

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "run_manifest.yaml"), run_manifest)
    if warm_start_report is not None:
        warm_start_report_path = Path(log_dir) / "params" / "actor_only_warm_start.json"
        warm_start_report_path.write_text(
            json.dumps(warm_start_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        runner.save(
            os.path.join(log_dir, "bootstrap_model_0.pt"),
            infos={"actor_only_warm_start": warm_start_report},
        )
        print(
            "[INFO] Saved actor-only iteration-0 checkpoint: "
            f"{log_dir}/bootstrap_model_0.pt"
        )
    if args_cli.bootstrap_only:
        print("[INFO] Bootstrap-only requested; exiting before PPO learning.")
        env.close()
        return

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
