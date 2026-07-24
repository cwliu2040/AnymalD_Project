#!/usr/bin/env python3
"""Create and run the project-owned ROS 2 Bridge graph in one Play environment."""

from __future__ import annotations

import argparse
import math
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=250, help="Number of 50 Hz environment steps.")
parser.add_argument("--real-time", action="store_true", help="Pace the smoke host at 50 Hz.")
parser.add_argument(
    "--external-control",
    action="store_true",
    help="Route ROS joint commands through the ManagerBasedEnv action manager.",
)
parser.add_argument(
    "--external-control-warmup-steps",
    type=int,
    default=150,
    help="Hold the default joint pose while the external policy receives initial state.",
)
parser.add_argument(
    "--joint-command-timeout-steps",
    type=int,
    default=5,
    help="Return to zero raw action after this many 50 Hz steps without a new command.",
)
parser.add_argument(
    "--validate-observation-parity",
    action="store_true",
    help="Compare the 48-D Action Graph observation contract with Isaac Lab every controlled step.",
)
parser.add_argument(
    "--observation-parity-atol",
    type=float,
    default=1.0e-4,
    help="Maximum absolute difference allowed for observation parity.",
)
parser.add_argument(
    "--straight-line-check",
    action="store_true",
    help="Require a continuous [0.5, 0, 0] command and enforce the 10-second straight-line contract.",
)
parser.add_argument(
    "--straight-line-duration-s",
    type=float,
    default=10.0,
    help="Straight-line measurement duration in simulation seconds.",
)
parser.add_argument(
    "--disable-episode-timeout",
    action="store_true",
    help="Keep an interactive external-control session from resetting at the task time limit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.steps <= 0:
    parser.error("--steps must be positive")
if args_cli.external_control_warmup_steps < 0:
    parser.error("--external-control-warmup-steps must be non-negative")
if args_cli.joint_command_timeout_steps <= 0:
    parser.error("--joint-command-timeout-steps must be positive")
if args_cli.observation_parity_atol <= 0.0:
    parser.error("--observation-parity-atol must be positive")
if args_cli.straight_line_duration_s <= 0.0:
    parser.error("--straight-line-duration-s must be positive")
if (args_cli.validate_observation_parity or args_cli.straight_line_check) and not args_cli.external_control:
    parser.error("observation parity and straight-line checks require --external-control")
required_kit_args = (
    "--enable omni.graph.core "
    "--enable omni.graph.nodes "
    "--enable isaacsim.core.nodes "
    "--enable isaacsim.ros2.bridge "
    "--/exts/isaacsim.ros2.bridge/ros_distro=system_default"
)
args_cli.kit_args = f"{args_cli.kit_args} {required_kit_args}".strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import omni.usd
import torch
from pxr import UsdPhysics

import anymal_locomotion.tasks  # noqa: F401
from anymal_locomotion.policy_contract import (
    CANONICAL_JOINT_ORDER,
    POLICY_CONTRACT,
    validate_runtime_joint_names,
)
from anymal_locomotion.simulation.ros2_bridge import (
    create_ros2_policy_bridge,
    read_base_state,
    read_joint_position_command,
    read_velocity_command,
    trigger_policy_step,
    write_base_orientation,
    write_joint_state,
)
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import PLAY_TASK_ID
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _find_articulation_root(env_prim_path: str) -> str:
    stage = omni.usd.get_context().get_stage()
    roots = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(env_prim_path)
        and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one articulation root under {env_prim_path}, received {roots}")
    return roots[0]


def _tick_action_graph(base_env, bridge) -> None:
    """Run one policy-rate graph impulse without advancing physics or rendering."""
    base_env.sim.forward()
    trigger_policy_step(bridge)


def _raw_policy_action(
    names: tuple[str, ...],
    targets: tuple[float, ...],
    *,
    device: str | torch.device,
) -> torch.Tensor:
    canonical_to_received = validate_runtime_joint_names(names)
    canonical_targets = [targets[index] for index in canonical_to_received]
    defaults = [
        float(joint["default_position"])
        for joint in POLICY_CONTRACT["joints"]
    ]
    action_scale = float(POLICY_CONTRACT["action"]["scale"])
    raw_action = [
        (target - default) / action_scale
        for target, default in zip(canonical_targets, defaults, strict=True)
    ]
    if tuple(names[index] for index in canonical_to_received) != CANONICAL_JOINT_ORDER:
        raise RuntimeError("ROS joint command remapping did not produce canonical order")
    return torch.tensor([raw_action], dtype=torch.float32, device=device)


def _projected_gravity_from_xyzw(
    orientation_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    quaternion = np.asarray(orientation_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-9:
        raise RuntimeError("Action Graph base orientation has near-zero norm")
    x, y, z, w = quaternion / norm
    return -np.asarray(
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=np.float32,
    )


def _clamp_external_command(command) -> np.ndarray:
    values = np.asarray(
        [command.linear[0], command.linear[1], command.angular[2]],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"/cmd_vel contains NaN or Inf: {values.tolist()}")
    limits = POLICY_CONTRACT["command"]["limits"]
    lower = np.asarray(
        [limits["vx"][0], limits["vy"][0], limits["wz"][0]],
        dtype=np.float32,
    )
    upper = np.asarray(
        [limits["vx"][1], limits["vy"][1], limits["wz"][1]],
        dtype=np.float32,
    )
    return np.clip(values, lower, upper)


def _set_internal_velocity_command(base_env, command: np.ndarray) -> None:
    command_tensor = base_env.command_manager.get_command("base_velocity")
    command_tensor.copy_(
        torch.as_tensor(command, dtype=command_tensor.dtype, device=command_tensor.device)
        .unsqueeze(0)
        .expand_as(command_tensor)
    )


def _write_bridge_joint_state(
    bridge,
    robot,
    canonical_joint_indices: tuple[int, ...],
    *,
    timestamp_s: float,
) -> None:
    positions = (
        robot.data.joint_pos[0, list(canonical_joint_indices)]
        .detach()
        .cpu()
        .numpy()
    )
    velocities = (
        robot.data.joint_vel[0, list(canonical_joint_indices)]
        .detach()
        .cpu()
        .numpy()
    )
    write_joint_state(
        bridge,
        CANONICAL_JOINT_ORDER,
        positions,
        velocities,
        timestamp_s=timestamp_s,
    )


def _build_bridge_observation(
    bridge,
    robot,
    canonical_joint_indices: tuple[int, ...],
    command: np.ndarray,
    previous_action: torch.Tensor,
):
    base_state = read_base_state(bridge)
    joint_positions = (
        robot.data.joint_pos[0, list(canonical_joint_indices)]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    joint_velocities = (
        robot.data.joint_vel[0, list(canonical_joint_indices)]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    default_positions = np.asarray(
        [joint["default_position"] for joint in POLICY_CONTRACT["joints"]],
        dtype=np.float32,
    )
    observation = np.concatenate(
        (
            np.asarray(base_state.linear_velocity, dtype=np.float32),
            np.asarray(base_state.angular_velocity, dtype=np.float32),
            _projected_gravity_from_xyzw(base_state.orientation_xyzw),
            command.astype(np.float32, copy=False),
            joint_positions - default_positions,
            joint_velocities,
            previous_action[0].detach().cpu().numpy().astype(np.float32, copy=False),
        )
    )
    if observation.shape != (48,):
        raise RuntimeError(f"Bridge observation has unexpected shape {observation.shape}")
    return observation, base_state


_OBSERVATION_TERMS = (
    ("base_linear_velocity", 0, 3),
    ("base_angular_velocity", 3, 6),
    ("projected_gravity", 6, 9),
    ("velocity_command", 9, 12),
    ("joint_position", 12, 24),
    ("joint_velocity", 24, 36),
    ("previous_action", 36, 48),
)


def _observation_term_errors(
    bridge_observation: np.ndarray,
    isaac_observation: np.ndarray,
) -> dict[str, float]:
    return {
        name: float(np.max(np.abs(bridge_observation[start:end] - isaac_observation[start:end])))
        for name, start, end in _OBSERVATION_TERMS
    }


def _yaw_from_wxyz(quaternion: torch.Tensor) -> float:
    w, x, y, z = (float(value) for value in quaternion.detach().cpu())
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def main() -> None:
    env_cfg = load_cfg_from_registry(PLAY_TASK_ID, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = 42
    if args_cli.disable_episode_timeout:
        env_cfg.terminations.time_out = None
    if args_cli.external_control:
        command_cfg = env_cfg.commands.base_velocity
        command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        command_cfg.rel_standing_envs = 0.0
        command_cfg.rel_heading_envs = 0.0
        command_cfg.heading_command = False
        command_cfg.ranges.lin_vel_x = (0.0, 0.0)
        command_cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_cfg.ranges.ang_vel_z = (0.0, 0.0)
        command_cfg.ranges.heading = None
    env = gym.make(PLAY_TASK_ID, cfg=env_cfg)
    try:
        base_env = env.unwrapped
        articulation_root = _find_articulation_root(base_env.scene.env_prim_paths[0])
        bridge = create_ros2_policy_bridge(
            articulation_root,
            connect_articulation_controller=not args_cli.external_control,
        )
        graph_prim = omni.usd.get_context().get_stage().GetPrimAtPath(bridge.graph_path)
        if not graph_prim.IsValid():
            raise RuntimeError(f"ROS 2 graph was not created: {bridge.graph_path}")

        print(f"PASS graph={bridge.graph_path}", flush=True)
        print(f"PASS articulation_root={bridge.articulation_root_path}", flush=True)
        print(
            "PASS topics="
            f"{bridge.command_topic},{bridge.joint_state_topic},{bridge.imu_topic},"
            f"{bridge.odometry_topic},{bridge.joint_command_topic}",
            flush=True,
        )
        print(f"PASS ros_domain_id={bridge.domain_id}", flush=True)

        env.reset()
        robot = base_env.scene["robot"]
        canonical_joint_indices = validate_runtime_joint_names(robot.joint_names)
        external_velocity_command = np.zeros(3, dtype=np.float32)
        if args_cli.external_control:
            write_base_orientation(bridge, robot.data.root_quat_w[0])
            _write_bridge_joint_state(
                bridge,
                robot,
                canonical_joint_indices,
                timestamp_s=0.0,
            )
            _tick_action_graph(base_env, bridge)
            external_velocity_command = _clamp_external_command(
                read_velocity_command(bridge)
            )
            _set_internal_velocity_command(base_env, external_velocity_command)
        actions = torch.zeros(
            (base_env.num_envs, base_env.action_manager.total_action_dim),
            device=base_env.device,
        )
        straight_line_steps = math.ceil(args_cli.straight_line_duration_s / base_env.step_dt)
        straight_line_start_step = args_cli.external_control_warmup_steps
        straight_line_end_step = straight_line_start_step + straight_line_steps
        if args_cli.straight_line_check and args_cli.steps < straight_line_end_step:
            raise ValueError(
                "--steps must cover external-control warmup plus straight-line duration: "
                f"received {args_cli.steps}, need at least {straight_line_end_step}"
            )
        received_command_count = 0
        last_command_timestamp = float("-inf")
        steps_without_new_command = 0
        terminated_count = 0
        truncated_count = 0
        parity_max_errors = {name: 0.0 for name, _, _ in _OBSERVATION_TERMS}
        parity_worst_values: dict[str, tuple[list[float], list[float]]] = {}
        parity_angular_debug: dict[str, list[float]] = {}
        parity_samples = 0
        straight_start_position: np.ndarray | None = None
        straight_start_yaw: float | None = None
        straight_end_position: np.ndarray | None = None
        straight_end_yaw: float | None = None
        straight_command_error = 0.0
        straight_terminated_count = 0
        straight_truncated_count = 0
        controlled_wall_start: float | None = None
        for step_index in range(args_cli.steps):
            start = time.monotonic()
            if step_index == args_cli.external_control_warmup_steps:
                controlled_wall_start = start
            command_for_step = external_velocity_command.copy()
            if args_cli.external_control:
                _set_internal_velocity_command(base_env, command_for_step)
            action_for_step = actions.clone()
            if args_cli.straight_line_check and step_index == straight_line_start_step:
                straight_start_position = (
                    robot.data.root_pos_w[0, :2].detach().cpu().numpy().astype(np.float64, copy=True)
                )
                straight_start_yaw = _yaw_from_wxyz(robot.data.root_quat_w[0])

            step_result = env.step(action_for_step)
            step_terminated = 0
            step_truncated = 0
            if len(step_result) == 5:
                step_terminated = int(step_result[2].sum().item())
                step_truncated = int(step_result[3].sum().item())
                terminated_count += step_terminated
                truncated_count += step_truncated
                if (
                    args_cli.straight_line_check
                    and straight_line_start_step <= step_index < straight_line_end_step
                ):
                    straight_terminated_count += step_terminated
                    straight_truncated_count += step_truncated

            # The command manager is normally a random command generator. External-control
            # mode overwrites it before Kit renders the green target arrow.
            if args_cli.external_control:
                _set_internal_velocity_command(base_env, command_for_step)
                write_base_orientation(bridge, robot.data.root_quat_w[0])
                _write_bridge_joint_state(
                    bridge,
                    robot,
                    canonical_joint_indices,
                    timestamp_s=(step_index + 1) * base_env.step_dt,
                )
            _tick_action_graph(base_env, bridge)
            if (
                args_cli.validate_observation_parity
                and step_index >= args_cli.external_control_warmup_steps
                and step_terminated == 0
                and step_truncated == 0
            ):
                bridge_observation, bridge_base_state = _build_bridge_observation(
                    bridge,
                    robot,
                    canonical_joint_indices,
                    command_for_step,
                    action_for_step,
                )
                observation_group = step_result[0]
                if not isinstance(observation_group, dict) or "policy" not in observation_group:
                    raise RuntimeError("Isaac Lab step did not return a policy observation group")
                isaac_observation = (
                    observation_group["policy"][0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                term_errors = _observation_term_errors(
                    bridge_observation,
                    isaac_observation,
                )
                for name, term_start, term_end in _OBSERVATION_TERMS:
                    error = term_errors[name]
                    if error > parity_max_errors[name]:
                        parity_max_errors[name] = error
                        parity_worst_values[name] = (
                            bridge_observation[term_start:term_end].astype(float).tolist(),
                            isaac_observation[term_start:term_end].astype(float).tolist(),
                        )
                        if name == "base_angular_velocity":
                            parity_angular_debug = {
                                "compute_odometry": list(
                                    bridge_base_state.world_angular_velocity
                                ),
                                "orientation_xyzw": list(
                                    bridge_base_state.orientation_xyzw
                                ),
                                "isaac_world": robot.data.root_ang_vel_w[0]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(float)
                                .tolist(),
                            }
                parity_samples += 1

            if args_cli.external_control:
                external_velocity_command = _clamp_external_command(
                    read_velocity_command(bridge)
                )
                _set_internal_velocity_command(base_env, external_velocity_command)
            if (
                args_cli.external_control
                and step_index >= args_cli.external_control_warmup_steps
            ):
                command = read_joint_position_command(bridge)
                if (
                    command is not None
                    and command.timestamp > last_command_timestamp
                ):
                    actions = _raw_policy_action(
                        command.names,
                        command.positions,
                        device=base_env.device,
                    )
                    last_command_timestamp = command.timestamp
                    steps_without_new_command = 0
                    received_command_count += 1
                else:
                    steps_without_new_command += 1
                    if (
                        steps_without_new_command
                        >= args_cli.joint_command_timeout_steps
                    ):
                        actions.zero_()
            if (
                args_cli.straight_line_check
                and straight_line_start_step <= step_index < straight_line_end_step
            ):
                straight_command_error = max(
                    straight_command_error,
                    float(
                        np.max(
                            np.abs(
                                command_for_step
                                - np.asarray([0.5, 0.0, 0.0], dtype=np.float32)
                            )
                        )
                    ),
                )
            if args_cli.straight_line_check and step_index + 1 == straight_line_end_step:
                straight_end_position = (
                    robot.data.root_pos_w[0, :2].detach().cpu().numpy().astype(np.float64, copy=True)
                )
                straight_end_yaw = _yaw_from_wxyz(robot.data.root_quat_w[0])
            if args_cli.real_time:
                remaining = base_env.step_dt - (time.monotonic() - start)
                if remaining > 0.0:
                    time.sleep(remaining)
        controlled_wall_seconds = (
            time.monotonic() - controlled_wall_start
            if controlled_wall_start is not None
            else 0.0
        )
        if args_cli.external_control and received_command_count == 0:
            raise RuntimeError(
                "External control was requested, but no fresh /joint_command "
                "message was received"
            )
        if args_cli.validate_observation_parity:
            if parity_samples == 0:
                raise RuntimeError("Observation parity was requested but no samples were compared")
            print(f"PASS observation_parity_samples={parity_samples}", flush=True)
            for name, error in parity_max_errors.items():
                print(f"PASS observation_parity_{name}_max_abs_error={error:.9g}", flush=True)
            failed_terms = {
                name: error
                for name, error in parity_max_errors.items()
                if error > args_cli.observation_parity_atol
            }
            if failed_terms:
                for name in failed_terms:
                    bridge_values, isaac_values = parity_worst_values[name]
                    print(
                        f"FAIL observation_parity_{name}_bridge={bridge_values}",
                        flush=True,
                    )
                    print(
                        f"FAIL observation_parity_{name}_isaac={isaac_values}",
                        flush=True,
                    )
                if "base_angular_velocity" in failed_terms:
                    print(
                        f"FAIL observation_parity_angular_debug={parity_angular_debug}",
                        flush=True,
                    )
                raise RuntimeError(
                    "48-D observation parity failed: "
                    f"atol={args_cli.observation_parity_atol}, errors={failed_terms}"
                )
        if args_cli.straight_line_check:
            if (
                straight_start_position is None
                or straight_start_yaw is None
                or straight_end_position is None
                or straight_end_yaw is None
            ):
                raise RuntimeError("Straight-line measurement window was incomplete")
            world_displacement = straight_end_position - straight_start_position
            cos_yaw = math.cos(straight_start_yaw)
            sin_yaw = math.sin(straight_start_yaw)
            forward_displacement = (
                cos_yaw * world_displacement[0] + sin_yaw * world_displacement[1]
            )
            lateral_displacement = (
                -sin_yaw * world_displacement[0] + cos_yaw * world_displacement[1]
            )
            heading_change = _wrap_angle(straight_end_yaw - straight_start_yaw)
            print(
                f"PASS straight_forward_displacement_m={forward_displacement:.6f}",
                flush=True,
            )
            print(
                f"PASS straight_lateral_displacement_m={lateral_displacement:.6f}",
                flush=True,
            )
            print(
                f"PASS straight_heading_change_deg={math.degrees(heading_change):.6f}",
                flush=True,
            )
            print(
                f"PASS straight_command_max_abs_error={straight_command_error:.9g}",
                flush=True,
            )
            failures = []
            if straight_command_error > 1.0e-6:
                failures.append(f"command_error={straight_command_error}")
            if straight_terminated_count != 0:
                failures.append(f"terminations={straight_terminated_count}")
            if straight_truncated_count != 0:
                failures.append(f"truncations={straight_truncated_count}")
            if forward_displacement < 4.0:
                failures.append(f"forward={forward_displacement:.6f}m < 4.0m")
            if abs(lateral_displacement) > 0.3:
                failures.append(f"lateral={lateral_displacement:.6f}m")
            if abs(math.degrees(heading_change)) > 10.0:
                failures.append(f"heading={math.degrees(heading_change):.6f}deg")
            if failures:
                raise RuntimeError(f"Straight-line contract failed: {failures}")
        print(f"PASS simulation_steps={args_cli.steps}", flush=True)
        if args_cli.external_control:
            print(
                "PASS external_control_warmup_steps="
                f"{args_cli.external_control_warmup_steps}",
                flush=True,
            )
            print(
                f"PASS received_joint_commands={received_command_count}",
                flush=True,
            )
            controlled_steps = max(
                0,
                args_cli.steps - args_cli.external_control_warmup_steps,
            )
            if controlled_steps > 0 and controlled_wall_seconds > 0.0:
                controlled_sim_seconds = controlled_steps * base_env.step_dt
                print(
                    "PASS controlled_real_time_factor="
                    f"{controlled_sim_seconds / controlled_wall_seconds:.6f}",
                    flush=True,
                )
                print(
                    "PASS controlled_loop_wall_hz="
                    f"{controlled_steps / controlled_wall_seconds:.6f}",
                    flush=True,
                )
            print(f"PASS terminated_count={terminated_count}", flush=True)
            print(f"PASS truncated_count={truncated_count}", flush=True)
        print(
            f"PASS final_base_height={float(robot.data.root_pos_w[0, 2]):.6f}",
            flush=True,
        )
        print(
            "PASS final_base_position="
            f"{robot.data.root_pos_w[0].detach().cpu().tolist()}",
            flush=True,
        )
        print(
            "PASS final_projected_gravity="
            f"{robot.data.projected_gravity_b[0].detach().cpu().tolist()}",
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
