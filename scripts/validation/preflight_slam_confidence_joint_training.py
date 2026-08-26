#!/usr/bin/env python3
"""Run the execution-gated, non-learning J1/J2 Isaac preflight.

This script never constructs an optimizer, PPO algorithm or OnPolicyRunner. It
uses the model1450 deterministic actor for a fixed 100-step runtime contract
check and writes one machine-readable report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--protocol",
    type=Path,
    default=Path("configs/slam_confidence_joint_training_v1.yaml"),
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("outputs/slam_confidence_joint_training_v1/nonlearning_preflight.json"),
)
parser.add_argument(
    "--authorized-nonlearning-preflight",
    action="store_true",
    help="Required in addition to the protocol authorization gate; does not authorize PPO.",
)
parser.add_argument(
    "--arm",
    choices=("all", "J1", "J2"),
    default="all",
    help=argparse.SUPPRESS,
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Fail closed before constructing AppLauncher. Both the repository gate and
# the one-shot CLI acknowledgment are required for any simulator process.
_preview_protocol_path = Path(args_cli.protocol).expanduser()
if not _preview_protocol_path.is_absolute():
    _preview_protocol_path = (PROJECT_ROOT / _preview_protocol_path).resolve()
_preview_protocol = yaml.safe_load(_preview_protocol_path.read_text(encoding="utf-8"))
_preview_authorized = bool(args_cli.authorized_nonlearning_preflight)
_preview_authorized &= bool(_preview_protocol["nonlearning_preflight"]["execution_authorized"])
_preview_authorized &= bool(_preview_protocol["execution_gates"]["nonlearning_preflight_authorized"])
if not _preview_authorized:
    parser.error("non-learning preflight authorization gates are closed; simulator was not launched")
if _preview_protocol["execution_gates"]["ppo_training_authorized"]:
    parser.error("non-learning preflight requires PPO training to remain unauthorized")

# Isaac Sim owns a process-global SimulationContext. Closing one environment
# and constructing the other arm in the same process can block in PhysX stage
# teardown, so keep the two arms in isolated child processes. The outer process
# never launches Isaac Sim and only merges their machine-readable reports.
if args_cli.arm == "all":
    output_path = Path(args_cli.output).expanduser()
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arm_results = []
    for arm in ("J1", "J2"):
        arm_output = output_path.with_name(f"{output_path.stem}.{arm}{output_path.suffix}")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--protocol",
            str(_preview_protocol_path),
            "--output",
            str(arm_output),
            "--authorized-nonlearning-preflight",
            "--arm",
            arm,
            "--headless",
            "--device",
            str(args_cli.device),
        ]
        completed = subprocess.run(command, check=False)
        if not arm_output.is_file():
            raise SystemExit(
                f"{arm} preflight exited {completed.returncode} without writing {arm_output}"
            )
        arm_result = json.loads(arm_output.read_text(encoding="utf-8"))
        arm_results.append(arm_result)
    combined = {
        key: value
        for key, value in arm_results[0].items()
        if key not in ("passed", "arms")
    }
    combined["passed"] = all(result["passed"] for result in arm_results)
    combined["arms"] = [result["arms"][0] for result in arm_results]
    output_path.write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(combined, indent=2, sort_keys=True), flush=True)
    raise SystemExit(0 if combined["passed"] else 1)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.networks import MLP

import anymal_locomotion.tasks  # noqa: F401
from anymal_locomotion.full_policy_bootstrap import bootstrap_dense_actor_state
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import (
    JOINT_TRAINING_J1_TASK_ID,
    JOINT_TRAINING_J2_TASK_ID,
)
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import (
    mdp as project_mdp,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


CUSTOM_REWARD_TERMS = (
    "angular_acceleration_l2",
    "linear_jerk_l2",
    "lidar_scan_translation_distortion_l2",
    "lidar_scan_rotation_distortion_l2",
)


class _DenseActorContainer(torch.nn.Module):
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


def _resolved_project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (PROJECT_ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def _load_protocol() -> tuple[dict, Path]:
    protocol_path = _resolved_project_path(args_cli.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    authorized = bool(protocol["execution_gates"]["nonlearning_preflight_authorized"])
    authorized &= bool(protocol["nonlearning_preflight"]["execution_authorized"])
    if not args_cli.authorized_nonlearning_preflight or not authorized:
        raise RuntimeError(
            "non-learning preflight is closed: require the CLI acknowledgment and both protocol gates"
        )
    if protocol["execution_gates"]["ppo_training_authorized"]:
        raise RuntimeError("preflight protocol must not authorize PPO training")
    return protocol, protocol_path


def _actor_from_checkpoint(checkpoint_path: Path, device: str) -> tuple[_DenseActorContainer, _DenseActorContainer]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint["model_state_dict"]
    source = _DenseActorContainer(48).to(device)
    target = _DenseActorContainer(1068).to(device)
    source.load_state_dict(
        {
            name: value
            for name, value in source_state.items()
            if name == "std" or name.startswith("actor.")
        },
        strict=True,
    )
    bootstrap_dense_actor_state(
        source_state,
        target.state_dict(),
        expected_target_input_dim=1068,
    )
    source.eval()
    target.eval()
    return source, target


def _term_layout(base_env) -> list[list[object]]:
    names = base_env.observation_manager.active_terms["policy"]
    dims = base_env.observation_manager.group_obs_term_dim["policy"]
    return [[name, int(np.prod(shape))] for name, shape in zip(names, dims)]


def _validate_observation(
    base_env,
    policy_observation: torch.Tensor,
    arm: str,
) -> dict[str, object]:
    if tuple(policy_observation.shape) != (base_env.num_envs, 1068):
        raise RuntimeError(f"{arm} policy observation has shape {tuple(policy_observation.shape)}")
    command = base_env.command_manager.get_command("base_velocity")
    command_exact = torch.equal(policy_observation[:, 9:12], command)
    history = policy_observation[:, 48:].reshape(base_env.num_envs, 20, 51)
    buffer = base_env.observation_manager._group_obs_term_history_buffer["policy"][
        "history_frame"
    ].buffer
    history_buffer_exact = torch.equal(history, buffer)
    startup = bool(torch.all(base_env.episode_length_buf == 0))
    startup_padding_exact = (
        bool(torch.equal(history, history[:, :1].expand_as(history)))
        if startup
        else True
    )
    localization = history[:, :, 48:51]
    if arm == "J1":
        expected = torch.tensor([1.0, 1.0, 0.0], device=localization.device)
        localization_passed = torch.equal(localization, expected.expand_as(localization))
    else:
        confidence_and_age = torch.all((localization[:, :, (0, 2)] >= 0.0) & (localization[:, :, (0, 2)] <= 1.0))
        validity = torch.all((localization[:, :, 1] == 0.0) | (localization[:, :, 1] == 1.0))
        expected_newest = project_mdp.simulated_slam_confidence(
            base_env, cycle_s=10.0, phase_offset_mode="distributed"
        )
        localization_passed = bool(confidence_and_age and validity and torch.equal(localization[:, -1], expected_newest))
    finite = bool(torch.all(torch.isfinite(policy_observation)))
    return {
        "finite": finite,
        "original_command_exact": command_exact,
        "history_buffer_flattening_exact": history_buffer_exact,
        "startup_repeat_oldest_padding_exact": startup_padding_exact,
        "localization_contract_passed": bool(localization_passed),
    }


def _run_arm(
    arm: str,
    task_id: str,
    protocol: dict,
    source_actor: _DenseActorContainer,
    target_actor: _DenseActorContainer,
) -> dict[str, object]:
    cfg = protocol["nonlearning_preflight"]
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = int(cfg["num_environments"])
    env_cfg.seed = int(cfg["seed"])
    env_cfg.sim.device = args_cli.device
    env = gym.make(task_id, cfg=env_cfg)
    raw_terms: dict[str, list[float]] = {name: [] for name in CUSTOM_REWARD_TERMS}
    parity_max_abs = 0.0
    checks: list[dict[str, object]] = []
    terminated_count = 0
    truncated_count = 0
    started_at = time.monotonic()
    print(
        f"[PREFLIGHT] {arm} starting: {cfg['num_environments']} envs, "
        f"{cfg['total_steps']} steps",
        flush=True,
    )
    try:
        base_env = env.unwrapped
        observation, _ = env.reset()
        policy_observation = observation["policy"]
        layout = _term_layout(base_env)
        expected_layout = cfg["required_term_layout"]
        if layout != expected_layout:
            raise RuntimeError(f"{arm} observation layout {layout} != {expected_layout}")
        if base_env.action_manager.total_action_dim != int(cfg["expected_action_dimension"]):
            raise RuntimeError(f"{arm} action dimension changed")
        policy_rate_hz = 1.0 / float(base_env.step_dt)
        if not np.isclose(policy_rate_hz, float(cfg["expected_policy_rate_hz"]), rtol=0.0, atol=1.0e-9):
            raise RuntimeError(f"{arm} policy rate changed to {policy_rate_hz}")
        if agent_cfg.max_iterations != protocol["training_budget"]["maximum_iterations_per_seed"]:
            raise RuntimeError(f"{arm} runner budget drifted")

        for step in range(int(cfg["total_steps"])):
            check = _validate_observation(base_env, policy_observation, arm)
            checks.append(check)
            with torch.no_grad():
                source_action = source_actor.actor(policy_observation[:, :48])
                target_action = target_actor.actor(policy_observation)
            parity_max_abs = max(
                parity_max_abs,
                float(torch.max(torch.abs(source_action - target_action)).cpu()),
            )
            if not torch.all(torch.isfinite(target_action)):
                raise RuntimeError(f"{arm} produced a nonfinite bootstrap action")
            observation, reward, terminated, truncated, _ = env.step(target_action)
            policy_observation = observation["policy"]
            terminated_count += int(torch.count_nonzero(terminated).cpu())
            truncated_count += int(torch.count_nonzero(truncated).cpu())
            if not torch.all(torch.isfinite(reward)):
                raise RuntimeError(f"{arm} produced a nonfinite reward")
            if step >= int(cfg["reward_scale_warmup_steps"]):
                for name in CUSTOM_REWARD_TERMS:
                    term_cfg = base_env.reward_manager.get_term_cfg(name)
                    value = term_cfg.func(base_env, **term_cfg.params)
                    if not torch.all(torch.isfinite(value)):
                        raise RuntimeError(f"{arm} raw reward term {name} is nonfinite")
                    raw_terms[name].extend(value.detach().cpu().tolist())
            completed_steps = step + 1
            if completed_steps % 10 == 0 or completed_steps == int(cfg["total_steps"]):
                print(
                    f"[PREFLIGHT] {arm} {completed_steps}/{cfg['total_steps']} steps "
                    f"({time.monotonic() - started_at:.1f}s)",
                    flush=True,
                )

        final_weights = protocol["reward_contract"]["implemented_motion_weights_after_curriculum"]
        dt = float(base_env.step_dt)
        projected_by_term = []
        term_summaries = {}
        weight_names = {
            "angular_acceleration_l2": "angular_acceleration_l2",
            "linear_jerk_l2": "linear_jerk_l2",
            "lidar_scan_translation_distortion_l2": "lidar_scan_translation_distortion_l2",
            "lidar_scan_rotation_distortion_l2": "lidar_scan_rotation_distortion_l2",
        }
        for name, values in raw_terms.items():
            array = np.asarray(values, dtype=np.float64)
            weighted = np.abs(array * float(final_weights[weight_names[name]]) * dt)
            projected_by_term.append(weighted)
            term_summaries[name] = {
                "raw_p50": float(np.percentile(array, 50)),
                "raw_p99": float(np.percentile(array, 99)),
                "projected_abs_per_step_p99": float(np.percentile(weighted, 99)),
                "projected_abs_per_step_max": float(np.max(weighted)),
            }
        projected_array = np.sum(np.stack(projected_by_term, axis=0), axis=0)
        projected_p99 = float(np.percentile(projected_array, 99))
        projected_max = float(np.max(projected_array))
        passed = (
            parity_max_abs <= float(cfg["maximum_runtime_bootstrap_action_parity_abs"])
            and all(all(bool(value) for value in check.values()) for check in checks)
            and terminated_count == 0
            and truncated_count == 0
            and projected_p99 <= float(cfg["maximum_new_reward_projected_p99_abs_per_step"])
            and projected_max <= float(cfg["maximum_new_reward_projected_max_abs_per_step"])
        )
        return {
            "arm": arm,
            "task_id": task_id,
            "passed": passed,
            "observation_layout": layout,
            "observation_checks_all_passed": all(
                all(bool(value) for value in check.values()) for check in checks
            ),
            "bootstrap_action_parity_max_abs": parity_max_abs,
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "projected_new_reward_abs_per_step_p99": projected_p99,
            "projected_new_reward_abs_per_step_max": projected_max,
            "reward_terms": term_summaries,
        }
    finally:
        # The per-arm process exits immediately after writing its report. Avoid
        # the process-global PhysX teardown path here; SimulationApp performs a
        # skip-cleanup shutdown below, and the OS releases child resources.
        pass


def main() -> None:
    protocol, protocol_path = _load_protocol()
    cfg = protocol["nonlearning_preflight"]
    checkpoint_path = _resolved_project_path(cfg["source_checkpoint"])
    source_actor, target_actor = _actor_from_checkpoint(checkpoint_path, args_cli.device)
    arms_by_name = {
        "J1": JOINT_TRAINING_J1_TASK_ID,
        "J2": JOINT_TRAINING_J2_TASK_ID,
    }
    arms = ((args_cli.arm, arms_by_name[args_cli.arm]),)
    reports = [
        _run_arm(arm, task_id, protocol, source_actor, target_actor)
        for arm, task_id in arms
    ]
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "ppo_constructed_or_run": False,
        "passed": all(report["passed"] for report in reports),
        "arms": reports,
    }
    output_path = _resolved_project_path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=True)
