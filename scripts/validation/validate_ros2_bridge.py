#!/usr/bin/env python3
"""Create and run the project-owned ROS 2 Bridge graph in one Play environment."""

from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACTORY_USD_PATH = (
    PROJECT_ROOT / "assets" / "maps" / "factory" / "Factory_Layout.usd"
)

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
    "--imu-observation-parity-atol",
    type=float,
    default=2.0e-3,
    help="Maximum IMU angular-velocity and projected-gravity parity error.",
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
if args_cli.imu_observation_parity_atol <= 0.0:
    parser.error("--imu-observation-parity-atol must be positive")
if args_cli.straight_line_duration_s <= 0.0:
    parser.error("--straight-line-duration-s must be positive")
if (args_cli.validate_observation_parity or args_cli.straight_line_check) and not args_cli.external_control:
    parser.error("observation parity and straight-line checks require --external-control")
args_cli.factory_usd_path = args_cli.factory_usd_path.expanduser().resolve()
if not args_cli.factory_usd_path.is_file():
    parser.error(f"Factory USD does not exist: {args_cli.factory_usd_path}")
if not all(math.isfinite(value) for value in (args_cli.spawn_x, args_cli.spawn_y, args_cli.spawn_yaw)):
    parser.error("Factory spawn pose must contain finite values")
if args_cli.enable_lio_sam:
    # RTX sensors are Hydra render products. Headless Isaac Lab otherwise uses
    # NO_RENDERING and silently creates the sensor without producing frames.
    args_cli.enable_cameras = True
required_kit_args = (
    "--enable omni.graph.core "
    "--enable omni.graph.nodes "
    "--enable isaacsim.core.nodes "
    "--enable isaacsim.sensors.physics "
    "--enable isaacsim.sensors.rtx "
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
    read_base_state,
    read_imu_state,
    read_joint_position_command,
    read_velocity_command,
    trigger_policy_step,
    write_base_orientation,
    write_joint_state,
)
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import PLAY_TASK_ID
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
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


def _validate_factory_physics_material() -> tuple[float, float, str, int]:
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
        or not math.isclose(float(static_friction), 1.0)
        or not math.isclose(float(dynamic_friction), 1.0)
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
    env_cfg.seed = 42
    # Isaac Compute Odometry reports orientation relative to the reset pose.
    # The ROS deployment host therefore uses an odom-aligned initial base pose
    # so its 200 Hz world-to-base IMU projection has no hidden yaw offset.
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
    try:
        base_env = env.unwrapped
        articulation_root = _find_articulation_root(base_env.scene.env_prim_paths[0])
        lidar = (
            create_rtx_lidar_sensor(articulation_root)
            if args_cli.enable_lio_sam
            else None
        )
        bridge = create_ros2_policy_bridge(
            articulation_root,
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
        (
            factory_static,
            factory_dynamic,
            factory_combine,
            factory_collision_count,
        ) = _validate_factory_physics_material()

        print(f"PASS graph={bridge.graph_path}", flush=True)
        print(f"PASS articulation_root={bridge.articulation_root_path}", flush=True)
        print(f"PASS factory_usd_path={args_cli.factory_usd_path}", flush=True)
        print("PASS factory_terrain_prim=/World/Factory/terrain", flush=True)
        print(
            "PASS factory_physics_material="
            f"static:{factory_static},dynamic:{factory_dynamic},"
            f"combine:{factory_combine},effective_with_robot:0.8/0.6,"
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
            _tick_action_graph(
                base_env,
                bridge,
                render_lidar=args_cli.enable_lio_sam,
            )
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
        first_imu_sensor_time: float | None = None
        last_imu_sensor_time = float("-inf")
        imu_sensor_samples = 0
        imu_timestamp_step_error = 0.0
        imu_angular_velocity_max_error = 0.0
        imu_projected_gravity_max_error = 0.0
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
            _tick_action_graph(
                base_env,
                bridge,
                render_lidar=args_cli.enable_lio_sam,
            )
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
            if step_terminated == 0 and step_truncated == 0:
                imu_angular_velocity_max_error = max(
                    imu_angular_velocity_max_error,
                    float(
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
                    ),
                )
                imu_projected_gravity_max_error = max(
                    imu_projected_gravity_max_error,
                    float(
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
                    ),
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
                                "compute_odometry": list(
                                    bridge_base_state.world_angular_velocity
                                ),
                                "orientation_xyzw": list(
                                    bridge_imu_state.orientation_xyzw
                                ),
                                "imu_sensor_time": [bridge_imu_state.sensor_time],
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
                if error
                > (
                    args_cli.imu_observation_parity_atol
                    if name in ("base_angular_velocity", "projected_gravity")
                    else args_cli.observation_parity_atol
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
                    f"imu_atol={args_cli.imu_observation_parity_atol}, "
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
        if imu_angular_velocity_max_error > args_cli.imu_observation_parity_atol:
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
