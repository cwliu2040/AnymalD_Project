#!/usr/bin/env python3
"""Backfill offline/map/run records for already accepted publication cells."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def infer_identity(run_dir: Path) -> dict[str, Any]:
    parts = run_dir.parts
    backend_indices = [index for index, value in enumerate(parts) if value in {"fastlio2", "liosam"}]
    if not backend_indices:
        raise ValueError(f"cannot infer backend from {run_dir}")
    index = backend_indices[-1]
    try:
        block = parts[index + 3]
        arm = parts[index + 4]
        if not block.startswith("block_") or not arm.startswith("arm_"):
            raise ValueError
        return {
            "backend": parts[index], "profile": parts[index + 1], "condition": parts[index + 2],
            "block": int(block.removeprefix("block_")), "arm": arm.removeprefix("arm_"),
        }
    except (IndexError, ValueError) as exc:
        raise ValueError(f"run path does not follow publication layout: {run_dir}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--dataset-role", choices=("formal", "excluded_smoke", "excluded_pilot"), required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dirs: list[Path] = []
    for value in args.input_root:
        root = value.expanduser().resolve()
        if not root.is_relative_to(PROJECT_ROOT):
            raise ValueError("input roots must remain inside the project")
        run_dirs.extend(sorted(path.parent.parent for path in root.glob("**/raw_bag/metadata.yaml")))
    identities: set[tuple[Any, ...]] = set()
    results = []
    for run_dir in run_dirs:
        identity = infer_identity(run_dir)
        key = tuple(identity[field] for field in ("backend", "profile", "condition", "block", "arm"))
        if key in identities:
            raise ValueError(f"duplicate cell identity across input roots: {key}")
        identities.add(key)
        commands = (
            (
                sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_slam_confidence_publication_run.py"),
                "--bag", str(run_dir / "raw_bag"), "--policy-diagnostics", str(run_dir / "policy_diagnostics.json"),
                "--arm", identity["arm"], "--output", str(run_dir / "offline_usability.json"),
            ),
            (
                sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_slam_map_consistency_run.py"),
                "--bag", str(run_dir / "raw_bag"), "--output", str(run_dir / "map_consistency.json"),
            ),
            (
                sys.executable, str(PROJECT_ROOT / "scripts/validation/build_slam_confidence_publication_run_record.py"),
                "--run-dir", str(run_dir), "--backend", identity["backend"],
                "--profile", identity["profile"], "--condition", identity["condition"],
                "--block", str(identity["block"]), "--arm", identity["arm"],
                "--dataset-role", args.dataset_role,
            ),
        )
        returncodes = [subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode for command in commands]
        results.append({"identity": identity, "returncodes": returncodes, "passed": returncodes == [0, 0, 0]})
    report = {"cell_count": len(results), "passed_cell_count": sum(value["passed"] for value in results), "cells": results}
    print(json.dumps(report, indent=2))
    return 0 if results and all(value["passed"] for value in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
