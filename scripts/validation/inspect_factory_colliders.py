#!/usr/bin/env python3
"""Inventory Factory collider groups and colliders near a world-space corridor."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACTORY = (
    PROJECT_ROOT / "assets" / "maps" / "factory" / "Factory_Layout.usd"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--factory-usd-path", type=Path, default=DEFAULT_FACTORY)
parser.add_argument(
    "--corridor",
    type=float,
    nargs=6,
    metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    default=(-1.0, 10.0, -19.0, -17.0, -0.1, 2.0),
)
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_ROOT
    / "logs"
    / "validation"
    / "factory_collider_inventory.json",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics


def _project_path(path: Path, *, must_exist: bool) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        parser.error(f"path must remain inside {PROJECT_ROOT}: {resolved}")
    if must_exist and not resolved.is_file():
        parser.error(f"file does not exist: {resolved}")
    return resolved


def _intersects(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    corridor: tuple[float, ...],
) -> bool:
    return (
        maximum[0] >= corridor[0]
        and minimum[0] <= corridor[1]
        and maximum[1] >= corridor[2]
        and minimum[1] <= corridor[3]
        and maximum[2] >= corridor[4]
        and minimum[2] <= corridor[5]
    )


def main() -> None:
    factory_path = _project_path(args.factory_usd_path, must_exist=True)
    output_path = _project_path(args.output, must_exist=False)
    corridor = tuple(float(value) for value in args.corridor)
    stage = Usd.Stage.Open(str(factory_path))
    if stage is None:
        raise RuntimeError(f"Failed to open Factory stage: {factory_path}")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=True,
    )
    group_counts: Counter[str] = Counter()
    near_corridor = []
    unbounded = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        components = [component for component in path.split("/") if component]
        group = components[1] if len(components) > 1 else "<root>"
        group_counts[group] += 1
        aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if aligned_range.IsEmpty():
            unbounded.append(path)
            continue
        minimum = tuple(float(value) for value in aligned_range.GetMin())
        maximum = tuple(float(value) for value in aligned_range.GetMax())
        if _intersects(minimum, maximum, corridor):
            near_corridor.append(
                {
                    "path": path,
                    "group": group,
                    "aabb_min": minimum,
                    "aabb_max": maximum,
                }
            )
    report = {
        "schema_version": 1,
        "factory_usd_path": str(factory_path),
        "corridor": {
            "x": [corridor[0], corridor[1]],
            "y": [corridor[2], corridor[3]],
            "z": [corridor[4], corridor[5]],
        },
        "collision_prim_count": sum(group_counts.values()),
        "group_counts": dict(sorted(group_counts.items())),
        "unbounded_collision_prims": unbounded,
        "near_corridor_count": len(near_corridor),
        "near_corridor": near_corridor,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Collider inventory written to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
