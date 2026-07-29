#!/usr/bin/env python3
"""Build the refinery source asset without its coplanar ground rectangle."""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher({"headless": True}).app

from pxr import Usd, UsdGeom

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "assets/maps/factory/source/source.usdc"
OUTPUT = (
    PROJECT_ROOT
    / "assets/maps/factory/source/source_without_coplanar_floor.usdc"
)
FACTORY_SOURCE = PROJECT_ROOT / "assets/maps/factory/Factory_Layout.usda"
FACTORY_RUNTIME = PROJECT_ROOT / "assets/maps/factory/Factory_Layout.usd"
MESH_PATH = (
    "/source/Meshes/Sketchfab_model/refinery_fbx/RootNode/refinery/"
    "refinery_refineria_0/refinery_refineria_0"
)
REMOVED_FACE_INDICES = (46822, 46823)


def main() -> None:
    shutil.copy2(SOURCE, OUTPUT)
    stage = Usd.Stage.Open(str(OUTPUT))
    if stage is None:
        raise RuntimeError(f"failed to open copied refinery source: {OUTPUT}")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(MESH_PATH))
    if not mesh:
        raise RuntimeError(f"refinery mesh does not exist: {MESH_PATH}")

    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    points = mesh.GetPointsAttr().Get()
    if len(counts) != 50346:
        raise RuntimeError(
            f"unexpected refinery face count: {len(counts)} != 50346"
        )

    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    removed_vertices = []
    mesh_to_source = UsdGeom.XformCache().GetLocalToWorldTransform(
        mesh.GetPrim()
    )
    for face_index in REMOVED_FACE_INDICES:
        start = offsets[face_index]
        end = offsets[face_index + 1]
        removed_vertices.append(
            [
                mesh_to_source.Transform(points[index])
                for index in indices[start:end]
            ]
        )
    if any(len(face) != 3 for face in removed_vertices):
        raise RuntimeError("expected two triangular refinery floor faces")
    if any(
        abs(float(point[1])) > 2.0e-4
        for face in removed_vertices
        for point in face
    ):
        raise RuntimeError(
            "target refinery faces are no longer the coplanar local-Y floor"
        )

    kept_counts = [
        count
        for face_index, count in enumerate(counts)
        if face_index not in REMOVED_FACE_INDICES
    ]
    kept_indices = []
    for face_index in range(len(counts)):
        if face_index in REMOVED_FACE_INDICES:
            continue
        kept_indices.extend(indices[offsets[face_index] : offsets[face_index + 1]])
    if not mesh.GetFaceVertexCountsAttr().Set(kept_counts):
        raise RuntimeError("failed to author corrected refinery face counts")
    if not mesh.GetFaceVertexIndicesAttr().Set(kept_indices):
        raise RuntimeError("failed to author corrected refinery face indices")
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"failed to save corrected refinery source: {OUTPUT}")
    stage = None
    verification_stage = Usd.Stage.Open(str(OUTPUT))
    verification_mesh = UsdGeom.Mesh(
        verification_stage.GetPrimAtPath(MESH_PATH)
    )
    corrected_face_count = len(
        verification_mesh.GetFaceVertexCountsAttr().Get()
    )
    if corrected_face_count != 50344:
        raise RuntimeError(
            f"corrected refinery face count is {corrected_face_count}, "
            "expected 50344"
        )

    factory_stage = Usd.Stage.Open(str(FACTORY_SOURCE))
    if factory_stage is None:
        raise RuntimeError(f"failed to open Factory source layer: {FACTORY_SOURCE}")
    if not factory_stage.GetRootLayer().Export(str(FACTORY_RUNTIME)):
        raise RuntimeError(f"failed to export Factory runtime layer: {FACTORY_RUNTIME}")

    print(f"Wrote {OUTPUT}")
    print(f"Wrote {FACTORY_RUNTIME}")
    print(f"Removed refinery faces: {REMOVED_FACE_INDICES}")


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
