#!/usr/bin/env python3
"""Run formal model1450 against the validated LIO-SAM policy-state path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS2_WORKSPACE = PROJECT_ROOT / "deployment" / "ros2_ws"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "liosam_policy_state_quality.yaml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "logs"
    / "slam_policy_state_quality"
    / "model1450_liosam_predictor_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return value


def validate_config(config: dict[str, Any]) -> dict[str, str]:
    if config.get("schema_version") != 1:
        raise ValueError("policy-state config schema_version must be 1")
    runtime = config["runtime"]
    if runtime["mapping_confidence_authority_topic"] != "/slam/odom":
        raise ValueError("mapping confidence authority must remain /slam/odom")
    if runtime["policy_odometry_topic"] != "/slam/policy_odom":
        raise ValueError("LIO-SAM policy state must use /slam/policy_odom")
    if runtime["runtime_ground_truth_input"] is not False:
        raise ValueError("runtime ground truth must remain forbidden")
    if runtime["native_deskew_required"] is not True:
        raise ValueError("qualification requires native deskew")
    qualification = config["qualification"]
    policy_path = (PROJECT_ROOT / qualification["policy_path"]).resolve()
    metadata_path = (PROJECT_ROOT / qualification["metadata_path"]).resolve()
    for path in (policy_path, metadata_path):
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
            raise ValueError(f"qualification artifact is invalid: {path}")
    if _sha256(policy_path) != qualification["policy_sha256"]:
        raise ValueError("formal policy hash does not match config")
    if _sha256(metadata_path) != qualification["metadata_sha256"]:
        raise ValueError("formal policy metadata hash does not match config")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("observation", {}).get("dimension") != 48:
        raise ValueError("control policy metadata is not explicitly 48-D")
    return {
        "policy_path": str(policy_path),
        "metadata_path": str(metadata_path),
        "policy_sha256": _sha256(policy_path),
        "metadata_sha256": _sha256(metadata_path),
    }


def launch_command(
    *,
    profile: str,
    run_dir: Path,
    artifacts: dict[str, str],
    config: dict[str, Any],
) -> tuple[str, ...]:
    runtime = config["runtime"]
    qualification = config["qualification"]
    return (
        "ros2",
        "launch",
        "anymal_locomotion_ros2",
        "fastlio2_locomotion_benchmark.launch.py",
        "slam_backend:=liosam",
        f"policy_odometry_topic:={runtime['policy_odometry_topic']}",
        "expected_confidence_backend:=liosam",
        "expected_calibration_id:=native-v1-edc098b0bd98",
        "enable_confidence:=true",
        f"profile:={profile}",
        f"output_dir:={run_dir}",
        f"simulation_steps:={int(qualification['simulation_steps'])}",
        f"policy_path:={artifacts['policy_path']}",
        f"metadata_path:={artifacts['metadata_path']}",
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not config_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("config must remain inside the repository")
    if not output_root.is_relative_to(PROJECT_ROOT):
        raise ValueError("output root must remain inside the repository")
    config = _load_mapping(config_path)
    artifacts = validate_config(config)
    qualification = config["qualification"]
    configured_profiles = tuple(qualification["required_profiles"])
    profiles = tuple(args.profiles) if args.profiles else configured_profiles
    if unknown := sorted(set(profiles) - set(configured_profiles)):
        raise ValueError(f"unconfigured profiles: {unknown}")
    repetitions = (
        int(args.repetitions)
        if args.repetitions is not None
        else int(qualification["repetitions_per_profile"])
    )
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    environment = os.environ.copy()
    environment["ROS_LOG_DIR"] = str(PROJECT_ROOT / "logs" / "ros")
    source_path = str(PROJECT_ROOT / "source" / "anymal_locomotion")
    environment["PYTHONPATH"] = (
        f"{source_path}:{environment.get('PYTHONPATH', '')}"
    )
    cells: list[dict[str, Any]] = []
    stop = False
    for profile in profiles:
        for repetition in range(1, repetitions + 1):
            run_dir = output_root / profile / f"run_{repetition:02d}"
            cell_path = run_dir / "cell.json"
            if cell_path.is_file() and not args.rerun:
                existing = _load_json(cell_path)
                print(
                    f"SKIP existing {profile}/run_{repetition:02d} "
                    f"passed={bool(existing.get('passed'))}",
                    flush=True,
                )
                cells.append(existing)
                continue
            print(f"RUN {profile}/run_{repetition:02d}", flush=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            launch = subprocess.run(
                launch_command(
                    profile=profile,
                    run_dir=run_dir,
                    artifacts=artifacts,
                    config=config,
                ),
                cwd=ROS2_WORKSPACE,
                env=environment,
                check=False,
            )
            trace_path = run_dir / "locomotion_diagnostics.json"
            driver_path = run_dir / "driver.json"
            policy_path = run_dir / "policy_diagnostics.json"
            stability_path = run_dir / "stability_gate.json"
            quality_path = run_dir / "policy_state_gate.json"
            action_sensitivity_path = run_dir / "action_sensitivity_gate.json"
            stability_returncode = None
            quality_returncode = None
            action_sensitivity_returncode = None
            if trace_path.is_file() and driver_path.is_file():
                stability_returncode = subprocess.run(
                    (
                        sys.executable,
                        str(PROJECT_ROOT / "scripts/validation/evaluate_stability_trace.py"),
                        "--trace",
                        str(trace_path),
                        "--driver",
                        str(driver_path),
                        "--output",
                        str(stability_path),
                    ),
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=False,
                ).returncode
            if policy_path.is_file() and trace_path.is_file():
                quality_returncode = subprocess.run(
                    (
                        sys.executable,
                        str(PROJECT_ROOT / "scripts/validation/evaluate_slam_policy_state_quality.py"),
                        "--policy-diagnostics",
                        str(policy_path),
                        "--locomotion-diagnostics",
                        str(trace_path),
                        "--config",
                        str(config_path),
                        "--output",
                        str(quality_path),
                    ),
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=False,
                ).returncode
                action_sensitivity_returncode = subprocess.run(
                    (
                        sys.executable,
                        str(
                            PROJECT_ROOT
                            / "scripts/validation/evaluate_policy_state_action_sensitivity.py"
                        ),
                        "--policy-diagnostics",
                        str(policy_path),
                        "--locomotion-diagnostics",
                        str(trace_path),
                        "--config",
                        str(config_path),
                        "--output",
                        str(action_sensitivity_path),
                    ),
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=False,
                ).returncode
            stability = _load_json(stability_path) if stability_path.is_file() else {}
            quality = _load_json(quality_path) if quality_path.is_file() else {}
            action_sensitivity = (
                _load_json(action_sensitivity_path)
                if action_sensitivity_path.is_file()
                else {}
            )
            stability_passed = bool(stability.get("gate", {}).get("passed"))
            quality_passed = bool(quality.get("passed"))
            action_sensitivity_passed = bool(
                action_sensitivity.get("passed")
            )
            passed = (
                launch.returncode == 0
                and stability_returncode == 0
                and quality_returncode == 0
                and action_sensitivity_returncode == 0
                and stability_passed
                and quality_passed
                and action_sensitivity_passed
            )
            cell = {
                "schema_version": 1,
                "kind": "liosam_policy_state_qualification_cell",
                "profile": profile,
                "repetition": repetition,
                "passed": passed,
                "launch_returncode": launch.returncode,
                "stability_passed": stability_passed,
                "policy_state_quality_passed": quality_passed,
                "action_sensitivity_passed": action_sensitivity_passed,
                "formal_policy_sha256": artifacts["policy_sha256"],
                "runtime_ground_truth_used": False,
                "paths": {
                    "stability_gate": str(stability_path.relative_to(PROJECT_ROOT)),
                    "policy_state_gate": str(quality_path.relative_to(PROJECT_ROOT)),
                    "action_sensitivity_gate": str(
                        action_sensitivity_path.relative_to(PROJECT_ROOT)
                    ),
                },
            }
            _write_json(cell_path, cell)
            cells.append(cell)
            if not passed and args.fail_fast:
                stop = True
                break
        if stop:
            break

    expected = len(profiles) * repetitions
    passed_count = sum(bool(cell.get("passed")) for cell in cells)
    summary = {
        "schema_version": 1,
        "kind": "liosam_policy_state_qualification",
        "contract_id": config["contract_id"],
        "passed": len(cells) == expected and passed_count == expected,
        "complete": len(cells) == expected,
        "expected_cell_count": expected,
        "executed_cell_count": len(cells),
        "passed_cell_count": passed_count,
        "formal_policy_sha256": artifacts["policy_sha256"],
        "config_sha256": _sha256(config_path),
        "runtime_ground_truth_used": False,
        "cells": cells,
    }
    _write_json(output_root / "qualification_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
