#!/usr/bin/env python3
"""Export the reviewable GroundPlane USDA to the runtime USD layer."""

from __future__ import annotations

from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher({"headless": True}).app

from pxr import Usd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "assets" / "maps" / "ground_plane" / "GroundPlane.usda"
OUTPUT = PROJECT_ROOT / "assets" / "maps" / "ground_plane" / "GroundPlane.usd"


def main() -> None:
    stage = Usd.Stage.Open(str(SOURCE))
    if stage is None or not stage.GetDefaultPrim():
        raise RuntimeError(f"failed to open default prim in {SOURCE}")
    if not stage.GetRootLayer().Export(str(OUTPUT)):
        raise RuntimeError(f"failed to export {OUTPUT}")
    print(f"Wrote {OUTPUT}")


try:
    if __name__ == "__main__":
        main()
finally:
    simulation_app.close()
