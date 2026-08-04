#!/usr/bin/env python3
"""Run the deterministic native LIO-SAM/FAST-LIO2 pilot matrix.

The runner deliberately treats a written evaluator JSON with ``status`` equal
to ``failed`` as an expected backend cell result.  A timeout, missing result,
or launch failure before the evaluator writes its JSON is an infrastructure
failure and stops the suite so an overnight run cannot silently manufacture a
partial qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPLAY_LAUNCH = "anymal_locomotion_ros2"
REPLAY_FILE = "slam_backend_native_replay.launch.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs" / "slam_backend_pilot" / "matrix"


@dataclass(frozen=True)
class Cell:
    scene: str
    bag_path: Path
    backend: str
    density_label: str
    density: float
    repeat: int

    @property
    def case_name(self) -> str:
        return (
            f"{self.scene}__{self.backend}__density_{self.density_label}"
            f"__run_{self.repeat:02d}"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_density(value: str) -> tuple[str, float]:
    density = float(value)
    if not 0.0 < density <= 1.0:
        raise argparse.ArgumentTypeError(
            f"point density must be in (0, 1], received {value!r}"
        )
    label = f"{density * 100:.0f}"
    return label, density


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _preflight(args: argparse.Namespace) -> None:
    if shutil.which("ros2") is None:
        raise RuntimeError("ros2 executable is not on PATH")
    for name, path in (
        ("factory bag", args.factory_bag),
        ("groundplane bag", args.groundplane_bag),
    ):
        if not path.is_dir() or not (path / "metadata.yaml").is_file():
            raise RuntimeError(
                f"{name} must be a rosbag2 directory with metadata.yaml: {path}"
            )
    if "fastlio2" in args.backends:
        probe = subprocess.run(
            ["ros2", "pkg", "prefix", "fast_lio"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "FAST-LIO2 is not in the sourced overlay; source "
                "./scripts/setup_deployment.sh first"
            )


def _build_cells(args: argparse.Namespace) -> list[Cell]:
    scenes = (
        ("factory", args.factory_bag),
        ("groundplane", args.groundplane_bag),
    )
    densities = [_parse_density(value) for value in args.point_densities]
    cells: list[Cell] = []
    for scene, bag_path in scenes:
        for backend in args.backends:
            for density_label, density in densities:
                repeats = args.standard_repeats
                if density == 1.0:
                    repeats = args.baseline_repeats
                elif density == args.boundary_density:
                    repeats = args.boundary_repeats
                for repeat in range(1, repeats + 1):
                    cells.append(
                        Cell(
                            scene=scene,
                            bag_path=bag_path,
                            backend=backend,
                            density_label=density_label,
                            density=density,
                            repeat=repeat,
                        )
                    )
    return cells


def _run_cell(cell: Cell, args: argparse.Namespace) -> dict[str, Any]:
    case_dir = args.output_root / cell.case_name
    result_path = case_dir / "replay.json"
    log_path = case_dir / "launch.log"
    metadata_path = case_dir / "cell.json"
    metadata = {
        "schema_version": 1,
        "started_at": _utc_now(),
        "scene": cell.scene,
        "bag_path": str(cell.bag_path),
        "backend": cell.backend,
        "deskew_mode": "native",
        "point_density": cell.density,
        "point_density_label": cell.density_label,
        "repeat": cell.repeat,
        "result_path": str(result_path),
        "launch_log": str(log_path),
    }
    _write_json(metadata_path, metadata)

    environment = os.environ.copy()
    environment.update(
        {
            "ROS_DOMAIN_ID": str(args.ros_domain_id),
            "ROS_LOCALHOST_ONLY": "1",
            "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
            "TERM": "xterm-256color",
        }
    )
    command = [
        "ros2",
        "launch",
        REPLAY_LAUNCH,
        REPLAY_FILE,
        f"slam_backend:={cell.backend}",
        "deskew_mode:=native",
        f"point_density:={cell.density:.2f}",
        f"bag_rate:={args.bag_rate}",
        f"bag_path:={cell.bag_path}",
        f"output_path:={result_path}",
        f"ros_domain_id:={args.ros_domain_id}",
    ]

    case_dir.mkdir(parents=True, exist_ok=True)
    return_code: int | None = None
    timeout = False
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=args.timeout_s,
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        timeout = True

    evaluator = _load_result(result_path)
    if evaluator is not None and evaluator.get("status") in {
        "passed",
        "failed",
    }:
        status = (
            "passed"
            if evaluator.get("status") == "passed"
            else "expected_backend_failure"
        )
        reason = (
            "evaluator passed"
            if status == "passed"
            else "evaluator reported an expected backend/scene failure"
        )
    else:
        status = "infrastructure_failure"
        if timeout:
            reason = f"launch exceeded timeout {args.timeout_s}s"
        elif return_code is None:
            reason = "launch timed out before a return code was observed"
        else:
            reason = (
                f"launch exited {return_code} without a valid evaluator result"
            )

    cell_result = {
        **metadata,
        "finished_at": _utc_now(),
        "launch_return_code": return_code,
        "status": status,
        "reason": reason,
        "evaluator_status": (
            evaluator.get("status") if evaluator is not None else None
        ),
    }
    _write_json(case_dir / "cell_result.json", cell_result)
    return cell_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factory-bag",
        type=Path,
        required=True,
        help="Feature-rich Factory rosbag2 directory",
    )
    parser.add_argument(
        "--groundplane-bag",
        type=Path,
        required=True,
        help="Feature-poor GroundPlane rosbag2 directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--point-densities",
        nargs="+",
        default=["1.0", "0.75", "0.5", "0.25"],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("liosam", "fastlio2"),
        default=["liosam", "fastlio2"],
    )
    parser.add_argument("--standard-repeats", type=int, default=1)
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument("--boundary-repeats", type=int, default=3)
    parser.add_argument(
        "--boundary-density",
        type=float,
        default=0.5,
        help="First degradation boundary to repeat three times",
    )
    parser.add_argument("--bag-rate", default="1.0")
    parser.add_argument("--ros-domain-id", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells with an existing valid cell_result.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.factory_bag = args.factory_bag.expanduser().resolve()
    args.groundplane_bag = args.groundplane_bag.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.standard_repeats < 1 or args.baseline_repeats < 1 or args.boundary_repeats < 1:
        raise SystemExit("all repeat counts must be positive")
    if not 0.0 < args.boundary_density <= 1.0:
        raise SystemExit("boundary density must be in (0, 1]")
    _preflight(args)
    cells = _build_cells(args)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_at": _utc_now(),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(args.output_root),
        "source_bags": {
            "factory": str(args.factory_bag),
            "groundplane": str(args.groundplane_bag),
        },
        "backends": args.backends,
        "point_densities": args.point_densities,
        "repeat_policy": {
            "standard": args.standard_repeats,
            "density_100_percent": args.baseline_repeats,
            "first_boundary_density": args.boundary_density,
            "first_boundary_repeats": args.boundary_repeats,
        },
        "cells": [],
        "status": "running",
    }
    summary_path = args.output_root / "summary.json"
    _write_json(summary_path, summary)

    for index, cell in enumerate(cells, start=1):
        case_result_path = args.output_root / cell.case_name / "cell_result.json"
        if args.resume:
            existing = _load_result(case_result_path)
            if existing is not None and existing.get("status") in {
                "passed",
                "expected_backend_failure",
            }:
                print(
                    f"[{index}/{len(cells)}] SKIP {cell.case_name} "
                    f"({existing['status']})",
                    flush=True,
                )
                summary["cells"].append(existing)
                continue

        print(
            f"[{index}/{len(cells)}] RUN {cell.case_name}",
            flush=True,
        )
        result = _run_cell(cell, args)
        summary["cells"].append(result)
        print(
            f"[{index}/{len(cells)}] {result['status']}: {result['reason']}",
            flush=True,
        )
        if result["status"] == "infrastructure_failure":
            summary["status"] = "blocked_infrastructure_failure"
            summary["finished_at"] = _utc_now()
            _write_json(summary_path, summary)
            return 2
        _write_json(summary_path, summary)

    summary["status"] = "completed"
    summary["finished_at"] = _utc_now()
    summary["counts"] = {
        status: sum(cell["status"] == status for cell in summary["cells"])
        for status in ("passed", "expected_backend_failure", "infrastructure_failure")
    }
    _write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as error:
        print(f"pilot preflight failed: {error}", file=sys.stderr)
        raise SystemExit(2)
