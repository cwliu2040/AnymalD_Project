#!/usr/bin/env python3
"""Validate the Factory infinite ground plane with a deterministic drop grid."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACTORY_USD_PATH = (
    PROJECT_ROOT / "assets" / "maps" / "factory" / "Factory_Layout.usd"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "logs" / "validation" / "factory_ground_collision.json"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--factory-usd-path",
    type=Path,
    default=DEFAULT_FACTORY_USD_PATH,
)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
parser.add_argument(
    "--grid-size",
    type=int,
    default=9,
    help="Odd number of probes along each map axis.",
)
parser.add_argument(
    "--extent-m",
    type=float,
    default=24.0,
    help="Probe coordinates span [-extent, +extent] on both axes.",
)
parser.add_argument("--drop-height-m", type=float, default=1.0)
parser.add_argument("--cube-size-m", type=float, default=0.1)
parser.add_argument("--settle-seconds", type=float, default=3.0)
parser.add_argument(
    "--max-final-vertical-speed-mps",
    type=float,
    default=0.1,
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.grid_size < 3 or args_cli.grid_size % 2 == 0:
    parser.error("--grid-size must be an odd integer of at least three")
for name in (
    "extent_m",
    "drop_height_m",
    "cube_size_m",
    "settle_seconds",
    "max_final_vertical_speed_mps",
):
    value = float(getattr(args_cli, name))
    if not math.isfinite(value) or value <= 0.0:
        parser.error(f"--{name.replace('_', '-')} must be positive")
if args_cli.drop_height_m <= args_cli.cube_size_m:
    parser.error("--drop-height-m must exceed --cube-size-m")

args_cli.factory_usd_path = args_cli.factory_usd_path.expanduser().resolve()
args_cli.output = args_cli.output.expanduser().resolve()
if not args_cli.factory_usd_path.is_file():
    parser.error(f"Factory USD does not exist: {args_cli.factory_usd_path}")
if not args_cli.output.is_relative_to(PROJECT_ROOT):
    parser.error(f"output must remain inside {PROJECT_ROOT}: {args_cli.output}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from pxr import UsdPhysics


def _grid_coordinates() -> list[tuple[float, float]]:
    axis = np.linspace(
        -args_cli.extent_m,
        args_cli.extent_m,
        args_cli.grid_size,
    )
    return [(float(x), float(y)) for x in axis for y in axis]


def _validate_collision_plane() -> dict[str, object]:
    stage = omni.usd.get_context().get_stage()
    candidates = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith("/World/Factory/")
        and prim.GetTypeName() == "Plane"
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one Factory PhysicsCollisionAPI Plane, "
            f"received {[str(prim.GetPath()) for prim in candidates]}"
        )
    plane = candidates[0]
    axis = plane.GetAttribute("axis").Get()
    collision_enabled = plane.GetAttribute("physics:collisionEnabled").Get()
    if axis != "Z" or not plane.IsActive() or collision_enabled is False:
        raise RuntimeError(
            "Factory collision plane is not an active Z-up collider: "
            f"path={plane.GetPath()}, axis={axis}, active={plane.IsActive()}, "
            f"collision_enabled={collision_enabled}"
        )
    return {
        "path": str(plane.GetPath()),
        "type_name": str(plane.GetTypeName()),
        "axis": str(axis),
        "active": bool(plane.IsActive()),
        "collision_enabled": collision_enabled is not False,
        "infinite_xy": True,
    }


def _create_probes(
    coordinates: list[tuple[float, float]],
) -> RigidObjectCollection:
    sim_utils.create_prim("/World/GroundCollisionProbes", "Xform")
    spawn = sim_utils.CuboidCfg(
        size=(
            args_cli.cube_size_m,
            args_cli.cube_size_m,
            args_cli.cube_size_m,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    configs = {
        f"probe_{index:03d}": RigidObjectCfg(
            prim_path=f"/World/GroundCollisionProbes/Probe_{index:03d}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x, y, args_cli.drop_height_m)
            ),
        )
        for index, (x, y) in enumerate(coordinates)
    }
    return RigidObjectCollection(
        RigidObjectCollectionCfg(rigid_objects=configs)
    )


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim_cfg.physx.enable_enhanced_determinism = True
    sim = sim_utils.SimulationContext(sim_cfg)
    TerrainImporter(
        TerrainImporterCfg(
            prim_path="/World/Factory",
            terrain_type="usd",
            usd_path=str(args_cli.factory_usd_path),
            num_envs=1,
            env_spacing=2.5,
            collision_group=-1,
            debug_vis=False,
        )
    )
    plane = _validate_collision_plane()
    coordinates = _grid_coordinates()
    probes = _create_probes(coordinates)

    sim.reset()
    initial_positions = (
        probes.data.object_pos_w[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
    )
    step_count = math.ceil(args_cli.settle_seconds / sim.get_physics_dt())
    minimum_center_z = initial_positions[:, 2].copy()
    for _ in range(step_count):
        sim.step()
        probes.update(sim.get_physics_dt())
        positions = (
            probes.data.object_pos_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        minimum_center_z = np.minimum(minimum_center_z, positions[:, 2])

    final_positions = (
        probes.data.object_pos_w[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
    )
    final_velocities = (
        probes.data.object_lin_vel_w[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
    )
    half_size = args_cli.cube_size_m / 2.0
    lower_center_limit = half_size - 0.02
    probe_results = []
    failures = []
    descended_count = 0
    for index, (x, y) in enumerate(coordinates):
        initial_z = float(initial_positions[index, 2])
        final_z = float(final_positions[index, 2])
        min_z = float(minimum_center_z[index])
        vertical_speed = float(final_velocities[index, 2])
        descended = initial_z - final_z
        descended_count += int(descended >= 0.25)
        reasons = []
        if not all(
            math.isfinite(value)
            for value in (initial_z, final_z, min_z, vertical_speed)
        ):
            reasons.append("non_finite_state")
        if min_z < lower_center_limit:
            reasons.append(
                f"ground_penetration:{min_z:.6f}<{lower_center_limit:.6f}"
            )
        if abs(vertical_speed) > args_cli.max_final_vertical_speed_mps:
            reasons.append(
                "not_settled:"
                f"{vertical_speed:.6f}mps>"
                f"{args_cli.max_final_vertical_speed_mps:.6f}mps"
            )
        result = {
            "name": f"probe_{index:03d}",
            "x_m": x,
            "y_m": y,
            "initial_center_z_m": initial_z,
            "final_center_z_m": final_z,
            "minimum_center_z_m": min_z,
            "final_vertical_speed_mps": vertical_speed,
            "descended_m": descended,
            "passed": not reasons,
            "failure_reasons": reasons,
        }
        probe_results.append(result)
        if reasons:
            failures.append(result)

    minimum_descended_count = math.ceil(0.5 * len(coordinates))
    if descended_count < minimum_descended_count:
        failures.append(
            {
                "failure_reasons": [
                    "insufficient_free_fall_probes:"
                    f"{descended_count}<{minimum_descended_count}"
                ]
            }
        )

    report = {
        "schema_version": 1,
        "factory_usd_path": str(args_cli.factory_usd_path),
        "plane": plane,
        "configuration": {
            "grid_size": args_cli.grid_size,
            "probe_count": len(coordinates),
            "extent_m": args_cli.extent_m,
            "drop_height_m": args_cli.drop_height_m,
            "cube_size_m": args_cli.cube_size_m,
            "settle_seconds": args_cli.settle_seconds,
            "physics_dt_s": sim.get_physics_dt(),
            "enhanced_determinism": True,
        },
        "summary": {
            "passed": not failures,
            "passed_probe_count": sum(
                result["passed"] for result in probe_results
            ),
            "failed_probe_count": sum(
                not result["passed"] for result in probe_results
            ),
            "descended_at_least_0_25_m_count": descended_count,
            "minimum_center_z_m": float(minimum_center_z.min()),
            "maximum_final_vertical_speed_mps": float(
                np.max(np.abs(final_velocities[:, 2]))
            ),
        },
        "probes": probe_results,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(
            "Factory ground collision grid failed: "
            f"{len(failures)} failure records; report={args_cli.output}"
        )
    print(f"PASS factory_collision_plane={plane}", flush=True)
    print(f"PASS ground_probe_count={len(coordinates)}", flush=True)
    print(f"PASS ground_probe_output={args_cli.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        simulation_app.close(skip_cleanup=True)
        raise
    else:
        simulation_app.close()
