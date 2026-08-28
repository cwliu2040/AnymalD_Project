#!/usr/bin/env python3
"""Non-learning causal-state wiring preflight for joint-training v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DEFAULT = Path("configs/slam_confidence_causal_joint_training_v2.yaml")
OUTPUT_DEFAULT = Path("outputs/slam_confidence_causal_joint_training_v2/nonlearning_preflight.json")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
parser.add_argument("--authorized-nonlearning-preflight", action="store_true")
parser.add_argument(
    "--cell",
    choices=("all", "J1_baseline", "J2_baseline", "J2_perturbed"),
    default="all",
    help=argparse.SUPPRESS,
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True


def _resolved(path: Path) -> Path:
    value = path.expanduser()
    return (PROJECT_ROOT / value).resolve() if not value.is_absolute() else value.resolve()


protocol_path = _resolved(args_cli.protocol)
preview = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
authorized = bool(args_cli.authorized_nonlearning_preflight)
authorized &= bool(preview["nonlearning_preflight"]["execution_authorized"])
authorized &= bool(preview["execution_gates"]["nonlearning_isaac_wiring_preflight_authorized"])
if not authorized:
    parser.error("causal v2 non-learning preflight gates are closed")
if any(
    preview["execution_gates"][name]
    for name in (
        "ppo_training_authorized",
        "teacher_training_authorized",
        "student_training_authorized",
        "live_ros_wiring_authorized",
        "physical_robot_authorized",
    )
):
    parser.error("causal v2 preflight requires all learning/live gates closed")

if args_cli.cell == "all":
    output = _resolved(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cells = []
    for cell in ("J1_baseline", "J2_baseline", "J2_perturbed"):
        cell_output = output.with_name(f"{output.stem}.{cell}{output.suffix}")
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--protocol", str(protocol_path),
                "--output", str(cell_output),
                "--authorized-nonlearning-preflight",
                "--cell", cell,
                "--headless",
                "--device", str(args_cli.device),
            ],
            check=False,
        )
        if completed.returncode != 0 or not cell_output.is_file():
            raise SystemExit(f"causal preflight cell failed: {cell}")
        cells.append(json.loads(cell_output.read_text(encoding="utf-8")))
    by_cell = {item["cell"]: item for item in cells}
    baseline = by_cell["J2_baseline"]["causal_metrics"]
    perturbed = by_cell["J2_perturbed"]["causal_metrics"]
    causal_checks = {
        "perturbed_scan_distortion_exceeds_baseline": (
            perturbed["mean_normalized_distortion"]
            > baseline["mean_normalized_distortion"]
        ),
        "perturbed_future_confidence_below_baseline": (
            perturbed["mean_confidence"] < baseline["mean_confidence"]
        ),
        "perturbed_delayed_reward_below_baseline": (
            perturbed["mean_delayed_localization_reward"]
            < baseline["mean_delayed_localization_reward"]
        ),
    }
    result = {
        "schema_version": 1,
        "protocol_id": preview["protocol_id"],
        "ppo_constructed_or_run": False,
        "passed": all(item["passed"] for item in cells) and all(causal_checks.values()),
        "causal_checks": causal_checks,
        "cells": cells,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    raise SystemExit(0 if result["passed"] else 1)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.networks import MLP

import anymal_locomotion.tasks  # noqa: F401
from anymal_locomotion.full_policy_bootstrap import bootstrap_dense_actor_state
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import (
    CAUSAL_JOINT_TRAINING_J1_TASK_ID,
    CAUSAL_JOINT_TRAINING_J2_TASK_ID,
)
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import mdp
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


class _DenseActor(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.actor = MLP(input_dim, 12, [128, 128, 128], "elu")
        self.std = torch.nn.Parameter(torch.ones(12))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actors(checkpoint_path: Path, device: str) -> tuple[_DenseActor, _DenseActor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint["model_state_dict"]
    source = _DenseActor(48).to(device)
    target = _DenseActor(1068).to(device)
    source.load_state_dict(
        {name: value for name, value in source_state.items() if name == "std" or name.startswith("actor.")},
        strict=True,
    )
    bootstrap_dense_actor_state(source_state, target.state_dict(), expected_target_input_dim=1068)
    return source.eval(), target.eval()


def main() -> None:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    cfg = protocol["nonlearning_preflight"]
    arm = args_cli.cell.split("_", maxsplit=1)[0]
    perturbed = args_cli.cell.endswith("perturbed")
    task_id = (
        CAUSAL_JOINT_TRAINING_J1_TASK_ID
        if arm == "J1"
        else CAUSAL_JOINT_TRAINING_J2_TASK_ID
    )
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = int(cfg["num_environments"])
    env_cfg.seed = int(cfg["seed"])
    env_cfg.sim.device = args_cli.device
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    checkpoint_path = _resolved(Path(cfg["source_checkpoint"]))
    source_actor, target_actor = _actors(checkpoint_path, args_cli.device)
    env = gym.make(task_id, cfg=env_cfg)
    try:
        observation, _ = env.reset()
        base_env = env.unwrapped
        policy_observation = observation["policy"]
        parity_max = 0.0
        original_command_exact = True
        localization_exact = True
        finite = True
        terminated_count = 0
        truncated_count = 0
        sample_count = 0
        distortion_sum = 0.0
        confidence_sum = 0.0
        reward_sum = 0.0
        perturbation = float(cfg["perturbation"]["alternating_joint_action_linf"])
        pattern = torch.tensor(
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            device=args_cli.device,
        )
        for step in range(int(cfg["total_steps"])):
            command = base_env.command_manager.get_command("base_velocity")
            original_command_exact &= bool(torch.equal(policy_observation[:, 9:12], command))
            history = policy_observation[:, 48:].reshape(base_env.num_envs, 20, 51)
            if arm == "J1":
                expected = torch.tensor([1.0, 1.0, 0.0], device=args_cli.device)
                localization_exact &= bool(torch.equal(history[:, :, 48:51], expected.expand_as(history[:, :, 48:51])))
            else:
                localization_exact &= bool(
                    torch.equal(history[:, -1, 48:51], mdp.causal_slam_state(base_env))
                )
            with torch.no_grad():
                source_action = source_actor.actor(policy_observation[:, :48])
                target_action = target_actor.actor(policy_observation)
            parity_max = max(parity_max, float(torch.max(torch.abs(source_action - target_action)).item()))
            action = target_action
            if perturbed:
                action = action + (perturbation if step % 2 == 0 else -perturbation) * pattern
            observation, reward, terminated, truncated, _ = env.step(action)
            policy_observation = observation["policy"]
            terminated_count += int(torch.count_nonzero(terminated).item())
            truncated_count += int(torch.count_nonzero(truncated).item())
            finite &= bool(torch.all(torch.isfinite(policy_observation)))
            finite &= bool(torch.all(torch.isfinite(reward)))
            if step >= int(cfg["warmup_steps"]):
                cache = base_env._causal_slam_dynamics_cache
                term_cfg = base_env.reward_manager.get_term_cfg("causal_slam_delayed_advantage")
                delayed = term_cfg.func(base_env, **term_cfg.params)
                distortion_sum += float(torch.sum(cache["normalized_distortion"]).item())
                confidence_sum += float(torch.sum(cache["state"][:, 0]).item())
                reward_sum += float(torch.sum(delayed).item())
                sample_count += base_env.num_envs
        passed = bool(
            parity_max <= 1.0e-6
            and original_command_exact
            and localization_exact
            and finite
            and terminated_count == 0
            and truncated_count == 0
            and sample_count > 0
        )
        result = {
            "cell": args_cli.cell,
            "arm": arm,
            "perturbed": perturbed,
            "passed": passed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "bootstrap_action_parity_max_abs": parity_max,
            "original_command_exact": original_command_exact,
            "localization_history_exact": localization_exact,
            "finite": finite,
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "causal_metrics": {
                "samples": sample_count,
                "mean_normalized_distortion": distortion_sum / sample_count,
                "mean_confidence": confidence_sum / sample_count,
                "mean_delayed_localization_reward": reward_sum / sample_count,
            },
        }
        output = _resolved(args_cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if not passed:
            raise SystemExit(1)
    finally:
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=True)
