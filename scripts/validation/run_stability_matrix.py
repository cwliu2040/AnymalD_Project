#!/usr/bin/env python3
"""Run or resume a versioned Factory stability qualification matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS2_WORKSPACE = PROJECT_ROOT / "deployment" / "ros2_ws"
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "stability_matrix.yaml"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "logs" / "stability_benchmarks" / "qualification_v1"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--phase",
    choices=("pure_translation", "turning"),
    required=True,
)
parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
parser.add_argument(
    "--factory-friction",
    type=float,
    default=1.0,
    help="Runtime-only Factory friction passed to every benchmark run.",
)
parser.add_argument(
    "--enhanced-determinism",
    choices=("true", "false"),
    default="true",
)
parser.add_argument(
    "--rerun",
    action="store_true",
    help="Run cases again even when their existing gate passed.",
)
parser.add_argument(
    "--profile",
    action="append",
    dest="selected_profiles",
    help="Run only this configured profile; may be specified repeatedly.",
)
parser.add_argument(
    "--repetitions",
    type=int,
    help="Override the configured repetition count for a screening run.",
)
parser.add_argument(
    "--fail-fast",
    action="store_true",
    help="Stop after the first failed run.",
)
args = parser.parse_args()


def _project_path(path: Path, *, must_exist: bool) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        parser.error(f"path must remain inside {PROJECT_ROOT}: {resolved}")
    if must_exist and not resolved.is_file():
        parser.error(f"file does not exist: {resolved}")
    return resolved


def _load_existing_gate(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return gate if gate.get("gate", {}).get("passed") else None


def main() -> None:
    if not math.isfinite(args.factory_friction) or args.factory_friction <= 0.0:
        parser.error("--factory-friction must be finite and positive")
    matrix_path = _project_path(args.matrix, must_exist=True)
    output_root = _project_path(args.output_root, must_exist=False)
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != 1:
        raise ValueError("stability matrix schema_version must be 1")
    repetitions = (
        args.repetitions
        if args.repetitions is not None
        else int(matrix["repetitions"])
    )
    phase = matrix["phases"][args.phase]
    configured_profiles = tuple(
        str(value) for value in phase["profiles"]
    )
    profiles = (
        tuple(args.selected_profiles)
        if args.selected_profiles
        else configured_profiles
    )
    unknown_profiles = sorted(set(profiles) - set(configured_profiles))
    if unknown_profiles:
        parser.error(
            f"profiles are not configured for {args.phase}: "
            f"{unknown_profiles}"
        )
    launch_file = str(phase["launch_file"])
    if repetitions <= 0 or not profiles:
        raise ValueError("stability matrix must contain repetitions and profiles")

    environment = os.environ.copy()
    environment["ROS_LOG_DIR"] = str(PROJECT_ROOT / "logs" / "ros")
    source_root = str(PROJECT_ROOT / "source" / "anymal_locomotion")
    environment["PYTHONPATH"] = (
        f"{source_root}:{environment.get('PYTHONPATH', '')}"
    )
    results = []
    for profile in profiles:
        for repetition in range(1, repetitions + 1):
            run_dir = (
                output_root
                / args.phase
                / profile
                / f"run_{repetition:02d}"
            )
            gate_path = run_dir / "gate.json"
            existing = None if args.rerun else _load_existing_gate(gate_path)
            if existing is not None:
                print(f"SKIP passed {profile} run {repetition:02d}", flush=True)
                results.append(
                    {
                        "profile": profile,
                        "repetition": repetition,
                        "passed": True,
                        "skipped": True,
                        "gate_path": str(gate_path),
                    }
                )
                continue

            print(f"RUN {profile} repetition {repetition:02d}", flush=True)
            launch = subprocess.run(
                (
                    "ros2",
                    "launch",
                    "anymal_locomotion_ros2",
                    launch_file,
                    f"profile:={profile}",
                    f"output_dir:={run_dir}",
                    f"factory_friction:={args.factory_friction}",
                    f"enhanced_determinism:={args.enhanced_determinism}",
                ),
                cwd=ROS2_WORKSPACE,
                env=environment,
                check=False,
            )
            evaluation_returncode = None
            if (
                (run_dir / "locomotion_diagnostics.json").is_file()
                and (run_dir / "driver.json").is_file()
            ):
                evaluation = subprocess.run(
                    (
                        sys.executable,
                        str(
                            PROJECT_ROOT
                            / "scripts"
                            / "validation"
                            / "evaluate_stability_trace.py"
                        ),
                        "--trace",
                        str(run_dir / "locomotion_diagnostics.json"),
                        "--driver",
                        str(run_dir / "driver.json"),
                        "--output",
                        str(gate_path),
                    ),
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=False,
                )
                evaluation_returncode = evaluation.returncode
            passed = (
                launch.returncode == 0
                and evaluation_returncode == 0
                and _load_existing_gate(gate_path) is not None
            )
            results.append(
                {
                    "profile": profile,
                    "repetition": repetition,
                    "passed": passed,
                    "skipped": False,
                    "launch_returncode": launch.returncode,
                    "evaluation_returncode": evaluation_returncode,
                    "gate_path": str(gate_path),
                }
            )
            if args.fail_fast and not passed:
                break
        if args.fail_fast and results and not results[-1]["passed"]:
            break

    failed = [result for result in results if not result["passed"]]
    summary = {
        "schema_version": 1,
        "phase": args.phase,
        "matrix_path": str(matrix_path),
        "output_root": str(output_root),
        "factory_friction": args.factory_friction,
        "enhanced_determinism": args.enhanced_determinism == "true",
        "expected_run_count": len(profiles) * repetitions,
        "completed_run_count": len(results),
        "passed_run_count": len(results) - len(failed),
        "failed_run_count": len(failed),
        "passed": not failed and len(results) == len(profiles) * repetitions,
        "runs": results,
    }
    summary_path = output_root / args.phase / "matrix_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in (
        "phase",
        "expected_run_count",
        "passed_run_count",
        "failed_run_count",
        "passed",
    )}, ensure_ascii=False, indent=2))
    print(f"Matrix summary written to: {summary_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
