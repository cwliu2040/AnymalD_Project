#!/usr/bin/env python3
"""Create and run the project-owned ROS 2 Bridge graph in one Play environment."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACTORY_USD_PATH = (
    PROJECT_ROOT / "assets" / "maps" / "factory" / "Factory_Layout.usd"
)
DEFAULT_STABILITY_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "stability_diagnostics.yaml"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=250, help="Number of 50 Hz environment steps.")
parser.add_argument("--seed", type=int, default=42, help="Isaac Lab environment seed.")
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
    help=(
        "Initial policy steps excluded from controlled timing metrics while "
        "the external policy history becomes ready."
    ),
)
parser.add_argument(
    "--joint-command-timeout-steps",
    type=int,
    default=5,
    help="Return to zero raw action after this many 50 Hz steps without a new command.",
)
parser.add_argument(
    "--joint-command-wait-timeout-s",
    type=float,
    default=0.02,
    help=(
        "Maximum wall time spent refreshing the Action Graph subscriber for "
        "the next timestamped policy command."
    ),
)
parser.add_argument(
    "--episode-reset-ack-timeout-s",
    type=float,
    default=2.0,
    help=(
        "Maximum wall time for the external policy to acknowledge an explicit "
        "simulator episode reset."
    ),
)
parser.add_argument(
    "--controlled-episode-reset-step",
    type=int,
    default=-1,
    help=(
        "Reset the simulator exactly once before this 0-based policy step and "
        "verify the ROS 2 reset ACK; -1 disables the controlled reset."
    ),
)
parser.add_argument(
    "--validate-observation-parity",
    action="store_true",
    help="Compare the 48-D Action Graph observation contract with Isaac Lab every controlled step.",
)
parser.add_argument(
    "--policy-parity-artifact",
    type=Path,
    default=None,
    help=(
        "Optional project-local TorchScript policy used to compare the "
        "external ROS action against the same prior-step observation."
    ),
)
parser.add_argument(
    "--observation-parity-atol",
    type=float,
    default=1.0e-4,
    help="Maximum absolute difference allowed for observation parity.",
)
parser.add_argument(
    "--imu-observation-parity-atol",
    type=float,
    default=2.0e-3,
    help="Maximum IMU projected-gravity parity error.",
)
parser.add_argument(
    "--imu-angular-velocity-parity-atol",
    type=float,
    default=None,
    help=(
        "Maximum physics-IMU angular-velocity error. Defaults to "
        "--imu-observation-parity-atol when omitted."
    ),
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
parser.add_argument(
    "--enhanced-determinism",
    action="store_true",
    help="Enable PhysX enhanced determinism for repeatable benchmark runs.",
)
parser.add_argument(
    "--enhanced-determinism-value",
    choices=("true", "false"),
    default=None,
    help="Explicit boolean override used by parameterized launch files.",
)
parser.add_argument(
    "--enable-lio-sam",
    action="store_true",
    help=(
        "Create the project RTX LiDAR, publish full scans, render every policy "
        "step, and leave odom->base_link TF ownership to LIO-SAM."
    ),
)
parser.add_argument(
    "--factory-usd-path",
    type=Path,
    default=DEFAULT_FACTORY_USD_PATH,
    help="Project-local Factory USD used instead of the Flat plane.",
)
parser.add_argument(
    "--factory-friction",
    type=float,
    default=1.0,
    help=(
        "Runtime-only Factory static and dynamic friction override. "
        "The source USD remains unchanged."
    ),
)
parser.add_argument(
    "--spawn-x",
    type=float,
    default=0.0,
    help="Initial ANYmal x position in the Factory map.",
)
parser.add_argument(
    "--spawn-y",
    type=float,
    default=-18.0,
    help="Initial ANYmal y position in the Factory map.",
)
parser.add_argument(
    "--spawn-yaw",
    type=float,
    default=0.0,
    help="Initial ANYmal yaw in the Factory map.",
)
parser.add_argument(
    "--state-transplant-manifest",
    type=str,
    default="",
    help=(
        "Optional project-local state manifest. Its z/orientation, body "
        "velocities, canonical joint state and policy history are restored; "
        "--spawn-x/--spawn-y select the target world location."
    ),
)
parser.add_argument(
    "--locomotion-diagnostics-output",
    type=Path,
    default=None,
    help=(
        "Optional project-local JSON trace containing four-foot contact, "
        "stance slip, base attitude and command-tracking diagnostics."
    ),
)
parser.add_argument(
    "--diagnostics-flush-steps",
    type=int,
    default=25,
    help=(
        "Flush the incremental diagnostic JSONL trace after this many "
        "policy steps."
    ),
)
parser.add_argument(
    "--benchmark-completion-file",
    type=Path,
    default=None,
    help=(
        "Optional project-local file written by a benchmark driver.  The "
        "simulation exits cleanly after a new version of the file appears."
    ),
)
parser.add_argument(
    "--locomotion-profile",
    type=str,
    default="unspecified",
    help="Profile label stored in the diagnostic metadata.",
)
parser.add_argument(
    "--stability-diagnostics-config",
    type=Path,
    default=DEFAULT_STABILITY_CONFIG_PATH,
    help="Frozen project-owned event-order thresholds.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.steps <= 0:
    parser.error("--steps must be positive")
if args_cli.controlled_episode_reset_step == -1:
    args_cli.controlled_episode_reset_step = None
elif args_cli.controlled_episode_reset_step < 0:
    parser.error("--controlled-episode-reset-step must be -1 or non-negative")
if args_cli.external_control_warmup_steps < 0:
    parser.error("--external-control-warmup-steps must be non-negative")
if args_cli.joint_command_timeout_steps <= 0:
    parser.error("--joint-command-timeout-steps must be positive")
if (
    not math.isfinite(args_cli.joint_command_wait_timeout_s)
    or args_cli.joint_command_wait_timeout_s < 0.0
):
    parser.error("--joint-command-wait-timeout-s must be finite and non-negative")
if (
    not math.isfinite(args_cli.episode_reset_ack_timeout_s)
    or args_cli.episode_reset_ack_timeout_s <= 0.0
):
    parser.error("--episode-reset-ack-timeout-s must be finite and positive")
if args_cli.observation_parity_atol <= 0.0:
    parser.error("--observation-parity-atol must be positive")
if args_cli.imu_observation_parity_atol <= 0.0:
    parser.error("--imu-observation-parity-atol must be positive")
if args_cli.imu_angular_velocity_parity_atol is None:
    args_cli.imu_angular_velocity_parity_atol = (
        args_cli.imu_observation_parity_atol
    )
elif args_cli.imu_angular_velocity_parity_atol <= 0.0:
    parser.error("--imu-angular-velocity-parity-atol must be positive")
if args_cli.straight_line_duration_s <= 0.0:
    parser.error("--straight-line-duration-s must be positive")
if args_cli.diagnostics_flush_steps <= 0:
    parser.error("--diagnostics-flush-steps must be positive")
if (args_cli.validate_observation_parity or args_cli.straight_line_check) and not args_cli.external_control:
    parser.error("observation parity and straight-line checks require --external-control")
if args_cli.locomotion_diagnostics_output is not None and not args_cli.external_control:
    parser.error("locomotion diagnostics require --external-control")
if args_cli.policy_parity_artifact is not None and not args_cli.external_control:
    parser.error("policy action parity requires --external-control")
if args_cli.controlled_episode_reset_step is not None:
    if not args_cli.external_control:
        parser.error(
            "--controlled-episode-reset-step requires --external-control"
        )
    if args_cli.controlled_episode_reset_step >= args_cli.steps:
        parser.error(
            "--controlled-episode-reset-step must be smaller than --steps"
        )
    if args_cli.controlled_episode_reset_step <= args_cli.external_control_warmup_steps:
        parser.error(
            "--controlled-episode-reset-step must be after external-control warmup"
        )
args_cli.factory_usd_path = args_cli.factory_usd_path.expanduser().resolve()
if not args_cli.factory_usd_path.is_file():
    parser.error(f"Factory USD does not exist: {args_cli.factory_usd_path}")
if not all(math.isfinite(value) for value in (args_cli.spawn_x, args_cli.spawn_y, args_cli.spawn_yaw)):
    parser.error("Factory spawn pose must contain finite values")
if not math.isfinite(args_cli.factory_friction) or args_cli.factory_friction <= 0.0:
    parser.error("--factory-friction must be finite and positive")
if args_cli.locomotion_diagnostics_output is not None:
    args_cli.locomotion_diagnostics_output = (
        args_cli.locomotion_diagnostics_output.expanduser().resolve()
    )
    if not args_cli.locomotion_diagnostics_output.is_relative_to(PROJECT_ROOT):
        parser.error(
            "locomotion diagnostic output must remain inside the project: "
            f"{args_cli.locomotion_diagnostics_output}"
        )
if args_cli.state_transplant_manifest:
    args_cli.state_transplant_manifest = (
        Path(args_cli.state_transplant_manifest).expanduser().resolve()
    )
    if (
        not args_cli.state_transplant_manifest.is_relative_to(PROJECT_ROOT)
        or not args_cli.state_transplant_manifest.is_file()
    ):
        parser.error(
            "state transplant manifest must be an existing project-local file: "
            f"{args_cli.state_transplant_manifest}"
        )
else:
    args_cli.state_transplant_manifest = None
action_replay_rendering_required = False
if args_cli.state_transplant_manifest is not None:
    early_transplant_document = json.loads(
        args_cli.state_transplant_manifest.read_text(encoding="utf-8")
    )
    early_replay = early_transplant_document.get(
        "deterministic_action_replay"
    )
    action_replay_rendering_required = bool(
        isinstance(early_replay, dict)
        and early_replay.get("lio_sam_enabled", False)
    )
if args_cli.benchmark_completion_file is not None:
    args_cli.benchmark_completion_file = (
        args_cli.benchmark_completion_file.expanduser().resolve()
    )
    if not args_cli.benchmark_completion_file.is_relative_to(PROJECT_ROOT):
        parser.error(
            "benchmark completion file must remain inside the project: "
            f"{args_cli.benchmark_completion_file}"
        )
if args_cli.policy_parity_artifact is not None:
    args_cli.policy_parity_artifact = (
        args_cli.policy_parity_artifact.expanduser().resolve()
    )
    if (
        not args_cli.policy_parity_artifact.is_relative_to(PROJECT_ROOT)
        or not args_cli.policy_parity_artifact.is_file()
    ):
        parser.error(
            "policy parity artifact must be an existing project-local file: "
            f"{args_cli.policy_parity_artifact}"
        )
args_cli.stability_diagnostics_config = (
    args_cli.stability_diagnostics_config.expanduser().resolve()
)
if (
    args_cli.locomotion_diagnostics_output is not None
    and not args_cli.stability_diagnostics_config.is_file()
):
    parser.error(
        "stability diagnostic config does not exist: "
        f"{args_cli.stability_diagnostics_config}"
    )
if args_cli.enable_lio_sam or action_replay_rendering_required:
    # RTX sensors are Hydra render products. Headless Isaac Lab otherwise uses
    # NO_RENDERING and silently creates the sensor without producing frames.
    args_cli.enable_cameras = True
enhanced_determinism_enabled = (
    args_cli.enhanced_determinism
    if args_cli.enhanced_determinism_value is None
    else args_cli.enhanced_determinism_value == "true"
)
required_kit_args = (
    "--enable omni.graph.core "
    "--enable omni.graph.nodes "
    "--enable isaacsim.core.nodes "
    "--enable isaacsim.sensors.physics "
    "--enable isaacsim.sensors.rtx "
    "--enable isaacsim.ros2.bridge "
    "--/exts/isaacsim.ros2.bridge/ros_distro=system_default"
)
if args_cli.enable_lio_sam or action_replay_rendering_required:
    # Isaac Sim 5.1 disables Motion BVH by default. A rotating LiDAR mounted
    # on a moving robot needs it for correct intra-scan motion effects.
    # Its SimulationManager also emits a known non-fatal time-interpolation
    # warning for every RTX frame. Filter only that source at the producer so
    # the GUI terminal and Kit log do not grow without bound; errors and all
    # other warning sources remain visible.
    required_kit_args += (
        " --/renderer/raytracingMotion/enabled=true"
        " --/renderer/raytracingMotion/enableHydraEngineMasking=true"
        " --/renderer/raytracingMotion/enabledForHydraEngines=0,1,2,3,4"
        " --/log/channels/isaacsim.core.simulation_manager.plugin=error"
    )
args_cli.kit_args = f"{args_cli.kit_args} {required_kit_args}".strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import omni.usd
import torch
import yaml
from pxr import UsdPhysics, UsdShade

import anymal_locomotion.tasks  # noqa: F401
from anymal_locomotion.policy_contract import (
    CANONICAL_JOINT_ORDER,
    POLICY_CONTRACT,
    validate_runtime_joint_names,
)
from anymal_locomotion.simulation.physics_imu import PhysicsImuSpawnerCfg
from anymal_locomotion.simulation.rtx_lidar import create_rtx_lidar_sensor
from anymal_locomotion.simulation.ros2_bridge import (
    create_ros2_policy_bridge,
    publish_episode_reset,
    read_base_state,
    read_episode_reset_ack,
    read_imu_state,
    read_joint_position_command,
    read_velocity_command,
    trigger_command_step,
    trigger_policy_step,
    write_base_orientation,
    write_joint_state,
)
from anymal_locomotion.stability_diagnostics import (
    DiagnosticThresholds,
    DiagnosticTraceWriter,
    build_diagnostic_report,
    load_diagnostic_trace,
    quaternion_to_rpy_wxyz,
)
from anymal_locomotion.state_transplant import load_state_transplant
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import PLAY_TASK_ID
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


_FOOT_NAMES = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")


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


def _set_factory_friction(friction: float) -> None:
    """Apply a session-layer material override without editing the map asset."""
    stage = omni.usd.get_context().get_stage()
    material = stage.GetPrimAtPath(
        "/World/Factory/terrain/FactoryPhysicsMaterial"
    )
    if not material.IsValid():
        raise RuntimeError("Factory physics material is missing")
    material.GetAttribute("physics:staticFriction").Set(float(friction))
    material.GetAttribute("physics:dynamicFriction").Set(float(friction))


def _validate_factory_physics_material(
    expected_friction: float,
) -> tuple[float, float, str, int]:
    stage = omni.usd.get_context().get_stage()
    terrain_path = "/World/Factory/terrain"
    material_path = f"{terrain_path}/FactoryPhysicsMaterial"
    terrain = stage.GetPrimAtPath(terrain_path)
    material = stage.GetPrimAtPath(material_path)
    if not terrain.IsValid() or not material.IsValid():
        raise RuntimeError(
            "Factory terrain physics material is missing: "
            f"terrain={terrain.IsValid()}, material={material.IsValid()}"
        )
    static_friction = material.GetAttribute("physics:staticFriction").Get()
    dynamic_friction = material.GetAttribute("physics:dynamicFriction").Get()
    combine_mode = material.GetAttribute(
        "physxMaterial:frictionCombineMode"
    ).Get()
    binding = terrain.GetRelationship("material:binding:physics")
    if (
        static_friction is None
        or dynamic_friction is None
        or not math.isclose(
            float(static_friction),
            expected_friction,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        or not math.isclose(
            float(dynamic_friction),
            expected_friction,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        or combine_mode != "multiply"
        or [str(path) for path in binding.GetTargets()] != [material_path]
        or binding.GetMetadata("bindMaterialAs") != "strongerThanDescendants"
    ):
        raise RuntimeError(
            "Factory physics material does not match the training terrain: "
            f"static={static_friction}, dynamic={dynamic_friction}, "
            f"combine={combine_mode}, targets={binding.GetTargets()}, "
            f"strength={binding.GetMetadata('bindMaterialAs')}"
        )
    collision_prims = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(f"{terrain_path}/")
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    unresolved = []
    for prim in collision_prims:
        bound_material, _ = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial("physics")
        if not bound_material or str(bound_material.GetPath()) != material_path:
            unresolved.append(str(prim.GetPath()))
            if len(unresolved) >= 5:
                break
    if not collision_prims or unresolved:
        raise RuntimeError(
            "Factory collision prims did not resolve the project physics material: "
            f"collision_count={len(collision_prims)}, unresolved={unresolved}"
        )
    return (
        float(static_friction),
        float(dynamic_friction),
        str(combine_mode),
        len(collision_prims),
    )


def _tick_action_graph(base_env, bridge, *, render_lidar: bool = False) -> None:
    """Run one policy-rate graph impulse and optionally one RTX render."""
    base_env.sim.forward()
    trigger_policy_step(bridge)
    if render_lidar:
        base_env.sim.render()


def _wait_for_episode_reset_ack(
    bridge,
    sequence: int,
    *,
    timeout_s: float,
) -> None:
    """Publish one reset sequence until the external policy acknowledges it."""
    reset_deadline = time.monotonic() + timeout_s
    while True:
        publish_episode_reset(bridge, sequence)
        trigger_command_step(bridge)
        if read_episode_reset_ack(bridge) >= sequence:
            return
        if time.monotonic() >= reset_deadline:
            raise RuntimeError(
                "External policy did not acknowledge simulator "
                f"episode reset {sequence}"
            )
        time.sleep(0.001)


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
    imu_state = read_imu_state(bridge)
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
            np.asarray(imu_state.angular_velocity, dtype=np.float32),
            _projected_gravity_from_xyzw(imu_state.orientation_xyzw),
            command.astype(np.float32, copy=False),
            joint_positions - default_positions,
            joint_velocities,
            previous_action[0].detach().cpu().numpy().astype(np.float32, copy=False),
        )
    )
    if observation.shape != (48,):
        raise RuntimeError(f"Bridge observation has unexpected shape {observation.shape}")
    return observation, base_state, imu_state


def _read_raw_imu_angular_velocity(bridge) -> list[float]:
    """Read the untransformed IsaacReadIMU angular-velocity output."""
    import omni.graph.core as og

    value = og.Controller.get(
        og.Controller.attribute(
            f"{bridge.graph_path}/ReadImuSensor.outputs:angVel"
        )
    )
    if value is None or len(value) != 3:
        raise RuntimeError("Physics IMU node did not produce raw angVel")
    return [float(component) for component in value]


def _build_native_transplant_observation(
    robot,
    canonical_joint_indices: tuple[int, ...],
    command: np.ndarray,
    previous_action: tuple[float, ...],
) -> np.ndarray:
    defaults = np.asarray(
        [joint["default_position"] for joint in POLICY_CONTRACT["joints"]],
        dtype=np.float32,
    )
    canonical_ids = list(canonical_joint_indices)
    return np.concatenate(
        (
            robot.data.root_lin_vel_b[0].detach().cpu().numpy(),
            robot.data.root_ang_vel_b[0].detach().cpu().numpy(),
            robot.data.projected_gravity_b[0].detach().cpu().numpy(),
            command,
            (
                robot.data.joint_pos[0, canonical_ids]
                .detach()
                .cpu()
                .numpy()
                - defaults
            ),
            robot.data.joint_vel[0, canonical_ids]
            .detach()
            .cpu()
            .numpy(),
            np.asarray(previous_action, dtype=np.float32),
        )
    ).astype(np.float32, copy=False)


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


def _rotate_body_vector_to_world(
    vector: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    """Rotate one body-frame vector with a normalized wxyz quaternion."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w = quaternion[0]
    xyz = quaternion[1:]
    body = np.asarray(vector, dtype=np.float64)
    return (
        body
        + 2.0 * w * np.cross(xyz, body)
        + 2.0 * np.cross(xyz, np.cross(xyz, body))
    )


def _write_diagnostic_report(
    path: Path,
    *,
    samples: list[dict[str, Any]],
    metadata: dict[str, Any],
    thresholds: DiagnosticThresholds,
) -> None:
    report = build_diagnostic_report(
        samples=samples,
        metadata=metadata,
        thresholds=thresholds,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _diagnostic_sample(
    *,
    robot,
    contact_sensor,
    robot_foot_ids: list[int],
    contact_foot_ids: list[int],
    base_contact_id: int,
    time_s: float,
    command: np.ndarray,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    quaternion = (
        robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
    )
    roll, pitch, yaw = quaternion_to_rpy_wxyz(quaternion.tolist())
    actual_velocity = torch.stack(
        (
            robot.data.root_lin_vel_b[0, 0],
            robot.data.root_lin_vel_b[0, 1],
            robot.data.root_ang_vel_b[0, 2],
        )
    ).detach().cpu().numpy()
    actual_linear_velocity_body = (
        robot.data.root_lin_vel_b[0].detach().cpu().numpy()
    )
    foot_velocities = (
        robot.data.body_lin_vel_w[0, robot_foot_ids]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    contact_forces = (
        contact_sensor.data.net_forces_w[0, contact_foot_ids]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    base_contact_force = float(
        torch.linalg.vector_norm(
            contact_sensor.data.net_forces_w_history[0, :, base_contact_id],
            dim=-1,
        )
        .max()
        .item()
    )
    feet = {}
    for index, name in enumerate(_FOOT_NAMES):
        force = contact_forces[index]
        velocity = foot_velocities[index]
        feet[name] = {
            "force_w_n": force.astype(float).tolist(),
            "normal_force_n": abs(float(force[2])),
            "tangential_force_n": float(np.linalg.norm(force[:2])),
            "velocity_w_mps": velocity.astype(float).tolist(),
            "tangential_speed_mps": float(np.linalg.norm(velocity[:2])),
        }
    return {
        "time_s": float(time_s),
        "command": command.astype(float).tolist(),
        "actual_velocity": actual_velocity.astype(float).tolist(),
        "actual_linear_velocity_body_mps": (
            actual_linear_velocity_body.astype(float).tolist()
        ),
        "base_position_w_m": (
            robot.data.root_pos_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        ),
        "base_height_m": float(robot.data.root_pos_w[0, 2].item()),
        "roll_rad": roll,
        "pitch_rad": pitch,
        "yaw_rad": yaw,
        "base_contact_force_n": base_contact_force,
        "terminated": terminated,
        "truncated": truncated,
        "feet": feet,
    }


def _actuator_lstm_checkpoint(robot) -> dict[str, Any] | None:
    """Capture public ActuatorNetLSTM recurrent buffers at one policy boundary."""
    checkpoints: dict[str, Any] = {}
    for name, actuator in robot.actuators.items():
        if not (
            hasattr(actuator, "sea_hidden_state_per_env")
            and hasattr(actuator, "sea_cell_state_per_env")
        ):
            continue
        checkpoints[name] = {
            "hidden_state": (
                actuator.sea_hidden_state_per_env[:, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            ),
            "cell_state": (
                actuator.sea_cell_state_per_env[:, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            ),
        }
    return checkpoints or None


def _restore_actuator_lstm_checkpoint(robot, checkpoint: dict[str, Any]) -> None:
    """Restore public ActuatorNetLSTM buffers captured from the same task."""
    if set(checkpoint) != {
        name
        for name, actuator in robot.actuators.items()
        if hasattr(actuator, "sea_hidden_state_per_env")
        and hasattr(actuator, "sea_cell_state_per_env")
    }:
        raise RuntimeError(
            "state-transplant actuator names do not match the runtime task: "
            f"manifest={sorted(checkpoint)}, runtime={sorted(robot.actuators)}"
        )
    for name, state in checkpoint.items():
        actuator = robot.actuators[name]
        for key, target in (
            ("hidden_state", actuator.sea_hidden_state_per_env[:, 0]),
            ("cell_state", actuator.sea_cell_state_per_env[:, 0]),
        ):
            source = torch.as_tensor(
                state[key],
                dtype=target.dtype,
                device=target.device,
            )
            if source.shape != target.shape:
                raise RuntimeError(
                    f"actuator {name} {key} shape differs: "
                    f"manifest={tuple(source.shape)}, runtime={tuple(target.shape)}"
                )
            target.copy_(source)


def main() -> None:
    transplant = (
        load_state_transplant(args_cli.state_transplant_manifest)
        if args_cli.state_transplant_manifest is not None
        else None
    )
    diagnostic_thresholds: DiagnosticThresholds | None = None
    if args_cli.locomotion_diagnostics_output is not None:
        diagnostic_config = yaml.safe_load(
            args_cli.stability_diagnostics_config.read_text(encoding="utf-8")
        )
        if (
            not isinstance(diagnostic_config, dict)
            or diagnostic_config.get("schema_version") != 1
            or not isinstance(diagnostic_config.get("classification"), dict)
        ):
            raise ValueError(
                "stability diagnostic config must contain schema_version 1 "
                "and a classification mapping"
            )
        diagnostic_thresholds = DiagnosticThresholds.from_mapping(
            diagnostic_config["classification"]
        )
    env_cfg = load_cfg_from_registry(PLAY_TASK_ID, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/Factory",
        terrain_type="usd",
        usd_path=str(args_cli.factory_usd_path),
        env_spacing=2.5,
        collision_group=-1,
        debug_vis=False,
    )
    env_cfg.scene.ros2_imu_sensor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/imu_sensor",
        spawn=PhysicsImuSpawnerCfg(sensor_period=0.005),
    )
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.physx.enable_enhanced_determinism = (
        enhanced_determinism_enabled
    )
    env_cfg.seed = args_cli.seed
    if args_cli.locomotion_diagnostics_output is not None:
        # Preserve one complete 50 Hz control interval of 200 Hz contact data.
        env_cfg.scene.contact_forces.history_length = 4
    # Isaac Compute Odometry reports orientation relative to the reset pose.
    # The ROS bridge composes this pose with the configured initial base
    # orientation before its 200 Hz world-to-base IMU projection.
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (args_cli.spawn_x, args_cli.spawn_x),
        "y": (args_cli.spawn_y, args_cli.spawn_y),
        "yaw": (args_cli.spawn_yaw, args_cli.spawn_yaw),
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
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
    diagnostic_trace_writer: DiagnosticTraceWriter | None = None
    diagnostic_sample_count = 0
    diagnostic_metadata: dict[str, Any] = {
        "factory_usd_path": str(args_cli.factory_usd_path),
        "seed": args_cli.seed,
        "spawn": {
            "x": args_cli.spawn_x,
            "y": args_cli.spawn_y,
            "yaw": args_cli.spawn_yaw,
        },
        "enhanced_determinism": enhanced_determinism_enabled,
        "factory_friction": float(args_cli.factory_friction),
        "external_control_warmup_steps": args_cli.external_control_warmup_steps,
        "controlled_episode_reset_step": args_cli.controlled_episode_reset_step,
        "lio_sam_enabled": bool(args_cli.enable_lio_sam),
        "profile": args_cli.locomotion_profile,
        "policy_parity_artifact": (
            str(args_cli.policy_parity_artifact)
            if args_cli.policy_parity_artifact is not None
            else None
        ),
        "stability_diagnostics_config": str(
            args_cli.stability_diagnostics_config
        ),
        "state_transplant_manifest": (
            str(args_cli.state_transplant_manifest)
            if args_cli.state_transplant_manifest is not None
            else None
        ),
    }
    if args_cli.locomotion_diagnostics_output is not None:
        diagnostic_trace_writer = DiagnosticTraceWriter(
            args_cli.locomotion_diagnostics_output
        )
        diagnostic_metadata["incremental_trace_path"] = str(
            diagnostic_trace_writer.path
        )
    completion_file_not_before_wall_s = time.time()
    benchmark_completion_observed = False
    try:
        base_env = env.unwrapped
        policy_parity_model = (
            torch.jit.load(
                str(args_cli.policy_parity_artifact),
                map_location=base_env.device,
            ).eval()
            if args_cli.policy_parity_artifact is not None
            else None
        )
        articulation_root = _find_articulation_root(base_env.scene.env_prim_paths[0])
        lidar = (
            create_rtx_lidar_sensor(articulation_root)
            if args_cli.enable_lio_sam
            or action_replay_rendering_required
            else None
        )
        bridge = create_ros2_policy_bridge(
            articulation_root,
            initial_base_orientation_wxyz=(
                (
                    math.cos(0.5 * transplant.replay_spawn[2]),
                    0.0,
                    0.0,
                    math.sin(0.5 * transplant.replay_spawn[2]),
                )
                if transplant is not None
                and transplant.action_history
                and transplant.replay_spawn is not None
                else transplant.base_orientation_wxyz
                if transplant is not None
                else (
                    math.cos(0.5 * args_cli.spawn_yaw),
                    0.0,
                    0.0,
                    math.sin(0.5 * args_cli.spawn_yaw),
                )
            ),
            connect_articulation_controller=not args_cli.external_control,
            publish_ground_truth_tf=not args_cli.enable_lio_sam,
        )
        physics_dt = float(base_env.sim.get_physics_dt())
        if not math.isclose(
            physics_dt,
            bridge.imu_update_period_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "Physics and IMU periods must match for true 200 Hz sampling: "
                f"physics_dt={physics_dt}, imu_period={bridge.imu_update_period_s}"
            )
        graph_prim = omni.usd.get_context().get_stage().GetPrimAtPath(bridge.graph_path)
        if not graph_prim.IsValid():
            raise RuntimeError(f"ROS 2 graph was not created: {bridge.graph_path}")
        _set_factory_friction(args_cli.factory_friction)
        (
            factory_static,
            factory_dynamic,
            factory_combine,
            factory_collision_count,
        ) = _validate_factory_physics_material(args_cli.factory_friction)

        print(f"PASS graph={bridge.graph_path}", flush=True)
        print(f"PASS articulation_root={bridge.articulation_root_path}", flush=True)
        print(f"PASS factory_usd_path={args_cli.factory_usd_path}", flush=True)
        print("PASS factory_terrain_prim=/World/Factory/terrain", flush=True)
        print(
            "PASS factory_physics_material="
            f"static:{factory_static},dynamic:{factory_dynamic},"
            f"combine:{factory_combine},"
            f"effective_with_robot:"
            f"{factory_static * 0.8:.3f}/{factory_dynamic * 0.6:.3f},"
            f"resolved_collision_prims:{factory_collision_count}",
            flush=True,
        )
        print(
            "PASS factory_spawn_pose="
            f"({args_cli.spawn_x},{args_cli.spawn_y},{args_cli.spawn_yaw})",
            flush=True,
        )
        print(f"PASS imu_sensor_prim={bridge.imu_sensor_path}", flush=True)
        print(f"PASS imu_parent_prim={bridge.imu_parent_path}", flush=True)
        print(f"PASS imu_frame_id={bridge.imu_frame_id}", flush=True)
        print(
            f"PASS imu_mount_translation_xyz={bridge.imu_mount_translation_xyz}",
            flush=True,
        )
        print(
            f"PASS imu_mount_orientation_wxyz={bridge.imu_mount_orientation_wxyz}",
            flush=True,
        )
        print(
            f"PASS imu_update_period_s={bridge.imu_update_period_s:.6f}",
            flush=True,
        )
        print(f"PASS physics_dt_s={physics_dt:.6f}", flush=True)
        print(
            f"PASS publishes_ground_truth_tf={bridge.publishes_ground_truth_tf}",
            flush=True,
        )
        if lidar is not None:
            print(f"PASS lidar_mount_prim={lidar.mount_path}", flush=True)
            print(f"PASS lidar_sensor_prim={lidar.sensor_path}", flush=True)
            print(
                f"PASS lidar_render_product={lidar.render_product_path}",
                flush=True,
            )
            print(f"PASS lidar_ros2_writer={lidar.ros2_writer_name}", flush=True)
            print(f"PASS lidar_frame_id={lidar.frame_id}", flush=True)
            print(f"PASS lidar_raw_topic={lidar.raw_topic}", flush=True)
            print(
                f"PASS lidar_profile={lidar.config}/{lidar.variant}",
                flush=True,
            )
            print(
                f"PASS lidar_mount_translation_xyz={lidar.mount_translation_xyz}",
                flush=True,
            )
        print(
            "PASS topics="
            f"{bridge.command_topic},{bridge.joint_state_topic},{bridge.imu_topic},"
            f"{bridge.odometry_topic},{bridge.tf_topic},"
            f"{lidar.raw_topic if lidar is not None else None},"
            f"{bridge.joint_command_topic}",
            flush=True,
        )
        print(f"PASS ros_domain_id={bridge.domain_id}", flush=True)

        env.reset()
        robot = base_env.scene["robot"]
        contact_sensor = base_env.scene["contact_forces"]
        robot_foot_ids, robot_foot_names = robot.find_bodies(
            list(_FOOT_NAMES),
            preserve_order=True,
        )
        contact_foot_ids, contact_foot_names = contact_sensor.find_bodies(
            list(_FOOT_NAMES),
            preserve_order=True,
        )
        base_contact_ids, base_contact_names = contact_sensor.find_bodies("base")
        if (
            tuple(robot_foot_names) != _FOOT_NAMES
            or tuple(contact_foot_names) != _FOOT_NAMES
            or base_contact_names != ["base"]
        ):
            raise RuntimeError(
                "Deterministic diagnostic body mapping failed: "
                f"robot_feet={robot_foot_names}, "
                f"contact_feet={contact_foot_names}, "
                f"base={base_contact_names}"
            )
        diagnostic_metadata.update(
            {
                "physics_dt_s": float(base_env.sim.get_physics_dt()),
                "policy_dt_s": float(base_env.step_dt),
                "foot_names": list(_FOOT_NAMES),
                "base_contact_name": "base",
            }
        )
        canonical_joint_indices = validate_runtime_joint_names(robot.joint_names)
        if transplant is not None and transplant.action_history:
            if transplant.replay_spawn is None:
                raise RuntimeError(
                    "action-history transplant is missing its source spawn"
                )
            replay_offset = (
                args_cli.spawn_x - transplant.replay_spawn[0],
                args_cli.spawn_y - transplant.replay_spawn[1],
            )
            replay_terminated_count = 0
            replay_truncated_count = 0
            for replay_action in transplant.action_history:
                replay_result = env.step(
                    torch.as_tensor(
                        [replay_action],
                        dtype=torch.float32,
                        device=base_env.device,
                    )
                )
                if len(replay_result) == 5:
                    replay_terminated_count += int(
                        replay_result[2].sum().item()
                    )
                    replay_truncated_count += int(
                        replay_result[3].sum().item()
                    )
                if transplant.action_replay_lio_sam_enabled:
                    base_env.sim.render()
            if replay_terminated_count or replay_truncated_count:
                raise RuntimeError(
                    "deterministic action pre-rollout reset unexpectedly: "
                    f"terminated={replay_terminated_count}, "
                    f"truncated={replay_truncated_count}"
                )
            expected_position = np.asarray(
                (
                    transplant.base_position_w_m[0] + replay_offset[0],
                    transplant.base_position_w_m[1] + replay_offset[1],
                    transplant.base_position_w_m[2],
                ),
                dtype=np.float32,
            )
            replay_position_error = float(
                np.max(
                    np.abs(
                        robot.data.root_pos_w[0]
                        .detach()
                        .cpu()
                        .numpy()
                        - expected_position
                    )
                )
            )
            print(
                "PASS state_transplant_action_history_steps="
                f"{len(transplant.action_history)}",
                flush=True,
            )
            print(
                "INFO state_transplant_action_replay_position_max_abs_error="
                f"{replay_position_error:.9g}",
                flush=True,
            )
        elif transplant is not None:
            root_state = robot.data.root_state_w.clone()
            root_state[0, :3] = torch.as_tensor(
                (
                    args_cli.spawn_x,
                    args_cli.spawn_y,
                    transplant.base_position_w_m[2],
                ),
                dtype=root_state.dtype,
                device=root_state.device,
            )
            root_state[0, 3:7] = torch.as_tensor(
                transplant.base_orientation_wxyz,
                dtype=root_state.dtype,
                device=root_state.device,
            )
            root_state[0, 7:10] = torch.as_tensor(
                _rotate_body_vector_to_world(
                    transplant.base_linear_velocity_b_mps,
                    transplant.base_orientation_wxyz,
                ),
                dtype=root_state.dtype,
                device=root_state.device,
            )
            root_state[0, 10:13] = torch.as_tensor(
                _rotate_body_vector_to_world(
                    transplant.base_angular_velocity_b_rps,
                    transplant.base_orientation_wxyz,
                ),
                dtype=root_state.dtype,
                device=root_state.device,
            )
            joint_positions = robot.data.joint_pos.clone()
            joint_velocities = robot.data.joint_vel.clone()
            canonical_ids = list(canonical_joint_indices)
            joint_positions[0, canonical_ids] = torch.as_tensor(
                transplant.joint_positions,
                dtype=joint_positions.dtype,
                device=joint_positions.device,
            )
            joint_velocities[0, canonical_ids] = torch.as_tensor(
                transplant.joint_velocities,
                dtype=joint_velocities.dtype,
                device=joint_velocities.device,
            )
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(
                joint_positions,
                joint_velocities,
            )
            if transplant.actuator_lstm_state is not None:
                _restore_actuator_lstm_checkpoint(
                    robot,
                    transplant.actuator_lstm_state,
                )
            base_env.sim.forward()
        if transplant is not None:
            initial_observation = _build_native_transplant_observation(
                robot,
                canonical_joint_indices,
                np.asarray(
                    transplant.effective_command,
                    dtype=np.float32,
                ),
                transplant.previous_action,
            )
            initial_observation_error = float(
                np.max(
                    np.abs(
                        initial_observation
                        - np.asarray(
                            transplant.initial_observation,
                            dtype=np.float32,
                        )
                    )
                )
            )
            if (
                not transplant.action_history
                and initial_observation_error > 2.0e-4
            ):
                raise RuntimeError(
                    "state-transplant initial 48-D policy input parity failed: "
                    f"max_abs_error={initial_observation_error}, atol=0.0002"
                )
            print(
                "PASS state_transplant="
                f"source_time:{transplant.source_time_s},"
                f"policy_clock:{transplant.policy_clock_s},"
                f"target_xy:({args_cli.spawn_x},{args_cli.spawn_y})",
                flush=True,
            )
            print(
                (
                    "INFO"
                    if transplant.action_history
                    else "PASS"
                )
                + " state_transplant_initial_observation_max_abs_error="
                f"{initial_observation_error:.9g}",
                flush=True,
            )
            print(
                "PASS state_transplant_actuator_lstm_state_restored="
                f"{transplant.actuator_lstm_state is not None}",
                flush=True,
            )
        external_velocity_command = np.zeros(3, dtype=np.float32)
        bridge_time_offset_s = (
            transplant.source_time_s
            if transplant is not None and transplant.action_history
            else 0.0
        )
        if args_cli.external_control:
            if transplant is not None:
                external_velocity_command = np.asarray(
                    transplant.effective_command,
                    dtype=np.float32,
                )
            else:
                trigger_command_step(bridge)
                external_velocity_command = _clamp_external_command(
                    read_velocity_command(bridge)
                )
            _set_internal_velocity_command(
                base_env,
                external_velocity_command,
            )
            write_base_orientation(bridge, robot.data.root_quat_w[0])
            _write_bridge_joint_state(
                bridge,
                robot,
                canonical_joint_indices,
                timestamp_s=bridge_time_offset_s,
            )
            _tick_action_graph(
                base_env,
                bridge,
                render_lidar=(
                    args_cli.enable_lio_sam
                    or action_replay_rendering_required
                ),
            )
        actions = torch.zeros(
            (base_env.num_envs, base_env.action_manager.total_action_dim),
            device=base_env.device,
        )
        if transplant is not None and not transplant.action_history:
            actions[0] = torch.as_tensor(
                transplant.bootstrap_action,
                dtype=actions.dtype,
                device=actions.device,
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
        received_controlled_command_count = 0
        last_command_timestamp = float("-inf")
        last_applied_command_timestamp: float | None = None
        controlled_action_age_steps: list[float] = []
        controlled_command_wait_s: list[float] = []
        command_wait_timeout_count = 0
        policy_input_for_next_step: torch.Tensor | None = None
        policy_action_parity_errors: list[float] = []
        steps_without_new_command = 0
        terminated_count = 0
        truncated_count = 0
        episode_reset_sequence = 0
        controlled_episode_reset_count = 0
        parity_max_errors = {name: 0.0 for name, _, _ in _OBSERVATION_TERMS}
        parity_worst_values: dict[str, tuple[list[float], list[float]]] = {}
        parity_angular_debug: dict[str, list[float]] = {}
        parity_samples = 0
        first_imu_sensor_time: float | None = None
        last_imu_sensor_time = float("-inf")
        imu_sensor_samples = 0
        imu_timestamp_step_error = 0.0
        imu_angular_velocity_max_error = 0.0
        imu_angular_velocity_worst_sample: dict[str, object] | None = None
        imu_projected_gravity_max_error = 0.0
        imu_reset_transient_angular_velocity_max_error = 0.0
        imu_reset_transient_projected_gravity_max_error = 0.0
        imu_reset_grace_steps_remaining = 0
        imu_orientation_norm_max_error = 0.0
        final_imu_state = None
        straight_start_position: np.ndarray | None = None
        straight_start_yaw: float | None = None
        straight_end_position: np.ndarray | None = None
        straight_end_yaw: float | None = None
        straight_command_error = 0.0
        straight_terminated_count = 0
        straight_truncated_count = 0
        controlled_wall_start: float | None = None
        completed_step_count = 0
        for step_index in range(args_cli.steps):
            start = time.monotonic()
            if (
                args_cli.controlled_episode_reset_step is not None
                and step_index == args_cli.controlled_episode_reset_step
            ):
                reset_result = env.reset()
                if policy_parity_model is not None:
                    reset_observation_group = reset_result[0]
                    if (
                        not isinstance(reset_observation_group, dict)
                        or "policy" not in reset_observation_group
                    ):
                        raise RuntimeError(
                            "Controlled episode reset did not return the policy "
                            "observation group"
                        )
                    policy_input_for_next_step = (
                        reset_observation_group["policy"].detach().clone()
                    )
                episode_reset_sequence += 1
                _wait_for_episode_reset_ack(
                    bridge,
                    episode_reset_sequence,
                    timeout_s=args_cli.episode_reset_ack_timeout_s,
                )
                controlled_episode_reset_count += 1
                imu_reset_grace_steps_remaining = 4
                external_velocity_command = _clamp_external_command(
                    read_velocity_command(bridge)
                )
                _set_internal_velocity_command(
                    base_env,
                    external_velocity_command,
                )
                actions.zero_()
                steps_without_new_command = 0
                last_applied_command_timestamp = None
                write_base_orientation(bridge, robot.data.root_quat_w[0])
                _write_bridge_joint_state(
                    bridge,
                    robot,
                    canonical_joint_indices,
                    timestamp_s=(
                        bridge_time_offset_s
                        + (step_index + 1) * base_env.step_dt
                    ),
                )
                _tick_action_graph(
                    base_env,
                    bridge,
                    render_lidar=(
                        args_cli.enable_lio_sam
                        or action_replay_rendering_required
                    ),
                )
            if step_index == args_cli.external_control_warmup_steps:
                controlled_wall_start = start
            if args_cli.external_control:
                wait_start = time.monotonic()
                wait_deadline = (
                    wait_start + args_cli.joint_command_wait_timeout_s
                )
                command = None
                while True:
                    trigger_command_step(bridge)
                    command = read_joint_position_command(bridge)
                    if (
                        command is not None
                        and command.timestamp > last_command_timestamp
                    ):
                        break
                    if time.monotonic() >= wait_deadline:
                        if (
                            step_index
                            >= args_cli.external_control_warmup_steps
                        ):
                            command_wait_timeout_count += 1
                        break
                    time.sleep(0.0005)
                if step_index >= args_cli.external_control_warmup_steps:
                    controlled_command_wait_s.append(
                        time.monotonic() - wait_start
                    )
                external_velocity_command = _clamp_external_command(
                    read_velocity_command(bridge)
                )
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
                    last_applied_command_timestamp = command.timestamp
                    steps_without_new_command = 0
                    received_command_count += 1
                    if (
                        step_index
                        >= args_cli.external_control_warmup_steps
                    ):
                        received_controlled_command_count += 1
                else:
                    steps_without_new_command += 1
                    if (
                        steps_without_new_command
                        >= args_cli.joint_command_timeout_steps
                    ):
                        actions.zero_()
                if (
                    last_applied_command_timestamp is not None
                    and step_index
                    >= args_cli.external_control_warmup_steps
                ):
                    controlled_action_age_steps.append(
                        (
                            step_index * base_env.step_dt
                            - last_applied_command_timestamp
                        )
                        / base_env.step_dt
                    )
                if (
                    policy_parity_model is not None
                    and policy_input_for_next_step is not None
                    and step_index
                    >= args_cli.external_control_warmup_steps
                ):
                    parity_input = policy_input_for_next_step.clone()
                    parity_input[:, 9:12] = torch.as_tensor(
                        external_velocity_command,
                        dtype=parity_input.dtype,
                        device=parity_input.device,
                    )
                    with torch.inference_mode():
                        expected_action = policy_parity_model(parity_input)
                    policy_action_parity_errors.append(
                        float(
                            torch.max(
                                torch.abs(actions - expected_action)
                            ).item()
                        )
                    )
            command_for_step = external_velocity_command.copy()
            if args_cli.external_control:
                _set_internal_velocity_command(base_env, command_for_step)
            if (
                transplant is not None
                and step_index < len(transplant.bridge_bootstrap_actions)
            ):
                actions[0] = torch.as_tensor(
                    transplant.bridge_bootstrap_actions[step_index],
                    dtype=actions.dtype,
                    device=actions.device,
                )
            action_for_step = actions.clone()
            if args_cli.straight_line_check and step_index == straight_line_start_step:
                straight_start_position = (
                    robot.data.root_pos_w[0, :2].detach().cpu().numpy().astype(np.float64, copy=True)
                )
                straight_start_yaw = _yaw_from_wxyz(robot.data.root_quat_w[0])

            step_result = env.step(action_for_step)
            completed_step_count = step_index + 1
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
            if policy_parity_model is not None:
                observation_group = step_result[0]
                if (
                    not isinstance(observation_group, dict)
                    or "policy" not in observation_group
                ):
                    raise RuntimeError(
                        "Policy action parity requires the policy observation group"
                    )
                policy_input_for_next_step = (
                    observation_group["policy"].detach().clone()
                )

            if (
                args_cli.external_control
                and (step_terminated != 0 or step_truncated != 0)
            ):
                episode_reset_sequence += 1
                _wait_for_episode_reset_ack(
                    bridge,
                    episode_reset_sequence,
                    timeout_s=args_cli.episode_reset_ack_timeout_s,
                )

            # The command manager is normally a random command generator. External-control
            # mode overwrites it before Kit renders the green target arrow.
            if args_cli.external_control:
                _set_internal_velocity_command(base_env, command_for_step)
                write_base_orientation(bridge, robot.data.root_quat_w[0])
                _write_bridge_joint_state(
                    bridge,
                    robot,
                    canonical_joint_indices,
                    timestamp_s=(
                        bridge_time_offset_s
                        + (step_index + 1) * base_env.step_dt
                    ),
                )
            _tick_action_graph(
                base_env,
                bridge,
                render_lidar=(
                    args_cli.enable_lio_sam
                    or action_replay_rendering_required
                ),
            )
            if args_cli.locomotion_diagnostics_output is not None:
                if diagnostic_trace_writer is None:
                    raise RuntimeError("diagnostic trace writer was not initialized")
                diagnostic_sample = _diagnostic_sample(
                    robot=robot,
                    contact_sensor=contact_sensor,
                    robot_foot_ids=robot_foot_ids,
                    contact_foot_ids=contact_foot_ids,
                    base_contact_id=base_contact_ids[0],
                    time_s=(step_index + 1) * base_env.step_dt,
                    command=command_for_step,
                    terminated=bool(step_terminated),
                    truncated=bool(step_truncated),
                )
                diagnostic_sample["applied_raw_action"] = (
                    action_for_step[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                    .tolist()
                )
                if (
                    (diagnostic_sample_count + 1)
                    % args_cli.diagnostics_flush_steps
                    == 0
                ):
                    actuator_checkpoint = _actuator_lstm_checkpoint(robot)
                    if actuator_checkpoint is not None:
                        diagnostic_sample["actuator_lstm_state"] = (
                            actuator_checkpoint
                        )
                if policy_input_for_next_step is not None:
                    diagnostic_sample["native_policy_observation"] = (
                        policy_input_for_next_step[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(float)
                        .tolist()
                    )
                diagnostic_trace_writer.append(diagnostic_sample)
                diagnostic_sample_count += 1
                if (
                    diagnostic_sample_count % args_cli.diagnostics_flush_steps
                    == 0
                ):
                    diagnostic_trace_writer.flush()
            imu_state = read_imu_state(bridge)
            if imu_state.sensor_time <= last_imu_sensor_time:
                raise RuntimeError(
                    "IMU sensor timestamp did not increase monotonically: "
                    f"previous={last_imu_sensor_time}, current={imu_state.sensor_time}"
                )
            if first_imu_sensor_time is None:
                first_imu_sensor_time = imu_state.sensor_time
            else:
                imu_timestamp_step_error = max(
                    imu_timestamp_step_error,
                    abs(
                        (imu_state.sensor_time - last_imu_sensor_time)
                        - base_env.step_dt
                    ),
                )
            last_imu_sensor_time = imu_state.sensor_time
            imu_sensor_samples += 1
            final_imu_state = imu_state
            angular_velocity_error = float(
                np.max(
                    np.abs(
                        np.asarray(
                            imu_state.angular_velocity,
                            dtype=np.float32,
                        )
                        - robot.data.root_ang_vel_b[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                )
            )
            if angular_velocity_error > imu_angular_velocity_max_error:
                imu_angular_velocity_worst_sample = {
                    "step_index": step_index,
                    "simulation_time_s": (step_index + 1) * base_env.step_dt,
                    "sensor_time_s": imu_state.sensor_time,
                    "sensor_angular_velocity": list(imu_state.angular_velocity),
                    "isaac_root_angular_velocity_b": (
                        robot.data.root_ang_vel_b[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(float)
                        .tolist()
                    ),
                    "compute_odometry_angular_velocity_b": list(
                        read_base_state(bridge).angular_velocity
                    ),
                }
            projected_gravity_error = float(
                np.max(
                    np.abs(
                        _projected_gravity_from_xyzw(
                            imu_state.orientation_xyzw
                        )
                        - robot.data.projected_gravity_b[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                )
            )
            imu_reset_transient_sample = False
            if step_terminated != 0 or step_truncated != 0:
                # PhysX IMU data can retain the pre-reset sample for one
                # or more policy ticks after the articulation has reset under
                # RTX rendering load. Keep the termination sample visible as a
                # separate metric and start the bounded post-reset window.
                imu_reset_grace_steps_remaining = 4
                imu_reset_transient_angular_velocity_max_error = max(
                    imu_reset_transient_angular_velocity_max_error,
                    angular_velocity_error,
                )
                imu_reset_transient_projected_gravity_max_error = max(
                    imu_reset_transient_projected_gravity_max_error,
                    projected_gravity_error,
                )
            elif imu_reset_grace_steps_remaining > 0:
                if (
                    angular_velocity_error
                    <= args_cli.imu_angular_velocity_parity_atol
                    and projected_gravity_error
                    <= args_cli.imu_observation_parity_atol
                ):
                    imu_reset_grace_steps_remaining = 0
                else:
                    imu_reset_transient_sample = True
                    imu_reset_transient_angular_velocity_max_error = max(
                        imu_reset_transient_angular_velocity_max_error,
                        angular_velocity_error,
                    )
                    imu_reset_transient_projected_gravity_max_error = max(
                        imu_reset_transient_projected_gravity_max_error,
                        projected_gravity_error,
                    )
                    imu_reset_grace_steps_remaining -= 1
            else:
                imu_angular_velocity_max_error = max(
                    imu_angular_velocity_max_error,
                    angular_velocity_error,
                )
                imu_projected_gravity_max_error = max(
                    imu_projected_gravity_max_error,
                    projected_gravity_error,
                )
            imu_orientation_norm_max_error = max(
                imu_orientation_norm_max_error,
                abs(
                    float(np.linalg.norm(imu_state.orientation_xyzw))
                    - 1.0
                ),
            )
            if (
                args_cli.validate_observation_parity
                and step_index >= args_cli.external_control_warmup_steps
                and step_terminated == 0
                and step_truncated == 0
                and not imu_reset_transient_sample
            ):
                (
                    bridge_observation,
                    bridge_base_state,
                    bridge_imu_state,
                ) = _build_bridge_observation(
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
                                "step_index": [float(step_index)],
                                "compute_odometry": list(
                                    bridge_base_state.world_angular_velocity
                                ),
                                "orientation_xyzw": list(
                                    bridge_imu_state.orientation_xyzw
                                ),
                                "imu_sensor_time": [
                                    bridge_imu_state.sensor_time
                                ],
                                "raw_imu_angular_velocity": (
                                    _read_raw_imu_angular_velocity(bridge)
                                ),
                                "isaac_world": robot.data.root_ang_vel_w[0]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(float)
                                .tolist(),
                            }
                parity_samples += 1

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
            if (
                args_cli.benchmark_completion_file is not None
                and args_cli.benchmark_completion_file.is_file()
                and args_cli.benchmark_completion_file.stat().st_mtime
                >= completion_file_not_before_wall_s
            ):
                benchmark_completion_observed = True
                print(
                    "PASS benchmark_completion_file="
                    f"{args_cli.benchmark_completion_file}",
                    flush=True,
                )
                break
        if (
            args_cli.benchmark_completion_file is not None
            and not benchmark_completion_observed
        ):
            raise RuntimeError(
                "Simulation reached --steps before the benchmark completion "
                f"file was refreshed: {args_cli.benchmark_completion_file}"
            )
        controlled_wall_seconds = (
            time.monotonic() - controlled_wall_start
            if controlled_wall_start is not None
            else 0.0
        )
        controlled_steps = max(
            0,
            completed_step_count - args_cli.external_control_warmup_steps,
        )
        if args_cli.external_control:
            diagnostic_metadata["external_control_timing"] = {
                "controlled_steps": controlled_steps,
                "received_controlled_joint_commands": (
                    received_controlled_command_count
                ),
                "joint_command_freshness_ratio": (
                    received_controlled_command_count / controlled_steps
                    if controlled_steps > 0
                    else None
                ),
                "action_age_steps": {
                    "sample_count": len(controlled_action_age_steps),
                    "min": (
                        min(controlled_action_age_steps)
                        if controlled_action_age_steps
                        else None
                    ),
                    "mean": (
                        sum(controlled_action_age_steps)
                        / len(controlled_action_age_steps)
                        if controlled_action_age_steps
                        else None
                    ),
                    "max": (
                        max(controlled_action_age_steps)
                        if controlled_action_age_steps
                        else None
                    ),
                },
                "fresh_command_wait_s": {
                    "sample_count": len(controlled_command_wait_s),
                    "mean": (
                        sum(controlled_command_wait_s)
                        / len(controlled_command_wait_s)
                        if controlled_command_wait_s
                        else None
                    ),
                    "max": (
                        max(controlled_command_wait_s)
                        if controlled_command_wait_s
                        else None
                    ),
                    "timeout_count": command_wait_timeout_count,
                },
                "action_alignment_probe_max_abs_error": {
                    "sample_count": len(policy_action_parity_errors),
                    "mean": (
                        sum(policy_action_parity_errors)
                        / len(policy_action_parity_errors)
                        if policy_action_parity_errors
                        else None
                    ),
                    "max": (
                        max(policy_action_parity_errors)
                        if policy_action_parity_errors
                        else None
                    ),
                },
            }
        if args_cli.external_control and received_command_count == 0:
            raise RuntimeError(
                "External control was requested, but no fresh /joint_command "
                "message was received"
            )
        if args_cli.external_control and command_wait_timeout_count != 0:
            raise RuntimeError(
                "Fresh /joint_command wait timed out during controlled steps: "
                f"count={command_wait_timeout_count}"
            )
        if args_cli.controlled_episode_reset_step is not None:
            if controlled_episode_reset_count != 1:
                raise RuntimeError(
                    "Controlled episode reset did not execute exactly once: "
                    f"count={controlled_episode_reset_count}"
                )
            print(
                "PASS controlled_episode_reset_step="
                f"{args_cli.controlled_episode_reset_step}",
                flush=True,
            )
            print(
                "PASS controlled_episode_reset_count="
                f"{controlled_episode_reset_count}",
                flush=True,
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
                if error
                > (
                    args_cli.imu_angular_velocity_parity_atol
                    if name == "base_angular_velocity"
                    else (
                        args_cli.imu_observation_parity_atol
                        if name == "projected_gravity"
                        else args_cli.observation_parity_atol
                    )
                )
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
                    f"atol={args_cli.observation_parity_atol}, "
                    "imu_angular_atol="
                    f"{args_cli.imu_angular_velocity_parity_atol}, "
                    "imu_gravity_atol="
                    f"{args_cli.imu_observation_parity_atol}, "
                    f"errors={failed_terms}"
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
        imu_contract_failures = {}
        if imu_timestamp_step_error > 1.0e-4:
            imu_contract_failures["timestamp_step_error"] = imu_timestamp_step_error
        if (
            imu_angular_velocity_max_error
            > args_cli.imu_angular_velocity_parity_atol
        ):
            imu_contract_failures["angular_velocity_error"] = (
                imu_angular_velocity_max_error
            )
        if imu_projected_gravity_max_error > args_cli.imu_observation_parity_atol:
            imu_contract_failures["projected_gravity_error"] = (
                imu_projected_gravity_max_error
            )
        if imu_orientation_norm_max_error > 1.0e-6:
            imu_contract_failures["orientation_norm_error"] = (
                imu_orientation_norm_max_error
            )
        if imu_contract_failures:
            if "angular_velocity_error" in imu_contract_failures:
                print(
                    "FAIL imu_angular_velocity_worst_sample="
                    f"{imu_angular_velocity_worst_sample}",
                    flush=True,
                )
            raise RuntimeError(
                f"200 Hz IMU contract failed: {imu_contract_failures}"
            )
        print(f"PASS simulation_steps={args_cli.steps}", flush=True)
        print(f"PASS imu_monotonic_policy_samples={imu_sensor_samples}", flush=True)
        print(f"PASS imu_final_sensor_time_s={last_imu_sensor_time:.6f}", flush=True)
        print(
            f"PASS imu_timestamp_step_max_error_s={imu_timestamp_step_error:.9g}",
            flush=True,
        )
        print(
            "PASS imu_angular_velocity_max_error="
            f"{imu_angular_velocity_max_error:.9g}",
            flush=True,
        )
        print(
            "PASS imu_projected_gravity_max_error="
            f"{imu_projected_gravity_max_error:.9g}",
            flush=True,
        )
        print(
            "INFO imu_reset_transient_angular_velocity_max_error="
            f"{imu_reset_transient_angular_velocity_max_error:.9g}",
            flush=True,
        )
        print(
            "INFO imu_reset_transient_projected_gravity_max_error="
            f"{imu_reset_transient_projected_gravity_max_error:.9g}",
            flush=True,
        )
        print(
            "PASS imu_orientation_norm_max_error="
            f"{imu_orientation_norm_max_error:.9g}",
            flush=True,
        )
        if final_imu_state is None:
            raise RuntimeError("Physics IMU did not produce any samples")
        print(
            f"PASS imu_final_orientation_xyzw={final_imu_state.orientation_xyzw}",
            flush=True,
        )
        print(
            f"PASS imu_final_angular_velocity={final_imu_state.angular_velocity}",
            flush=True,
        )
        print(
            f"PASS imu_final_linear_acceleration={final_imu_state.linear_acceleration}",
            flush=True,
        )
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
            print(
                "PASS received_controlled_joint_commands="
                f"{received_controlled_command_count}",
                flush=True,
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
                print(
                    "PASS joint_command_freshness_ratio="
                    f"{received_controlled_command_count / controlled_steps:.9f}",
                    flush=True,
                )
            if controlled_action_age_steps:
                print(
                    "PASS joint_command_action_age_steps="
                    f"min={min(controlled_action_age_steps):.6f},"
                    f"mean={sum(controlled_action_age_steps) / len(controlled_action_age_steps):.6f},"
                    f"max={max(controlled_action_age_steps):.6f}",
                    flush=True,
                )
            if policy_action_parity_errors:
                print(
                    "INFO policy_action_alignment_probe_max_abs_error="
                    f"{max(policy_action_parity_errors):.9g}",
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
        try:
            if diagnostic_trace_writer is not None:
                diagnostic_trace_writer.close()
                if diagnostic_trace_writer.sample_count:
                    _write_diagnostic_report(
                        diagnostic_trace_writer.report_path,
                        samples=load_diagnostic_trace(diagnostic_trace_writer.path),
                        metadata=diagnostic_metadata,
                        thresholds=diagnostic_thresholds,
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
