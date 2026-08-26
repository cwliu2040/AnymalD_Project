#!/usr/bin/env python3
"""Plan or execute the fail-closed model1450 touchdown baseline envelope."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS2_WORKSPACE = PROJECT_ROOT / "deployment/ros2_ws"
PROTOCOL_PATH = PROJECT_ROOT / "configs/slam_low_level_touchdown_headroom_v1.yaml"
RELEASE_PATH = PROJECT_ROOT / "configs/slam_low_level_touchdown_headroom_release_v1.yaml"
VALIDATOR_PATH = PROJECT_ROOT / "scripts/validation/validate_slam_low_level_touchdown_headroom_protocol.py"
ISAAC_PYTHON = Path(os.environ.get("ISAACLAB_ROOT", "/home/ros/IsaacLab")) / "_isaac_sim/python.sh"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _validator_module():
    spec = importlib.util.spec_from_file_location("touchdown_protocol", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load touchdown protocol validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project_file(relative: str, expected: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
        raise ValueError(f"release artifact is missing: {relative}")
    if _sha256(path) != expected:
        raise ValueError(f"release artifact hash mismatch: {relative}")
    return path


def validate_release(release: dict[str, Any], protocol_path: Path) -> None:
    if release.get("status") != "baseline_envelope_execution_authorized_only":
        raise ValueError("release status does not authorize the baseline envelope")
    if release["protocol"]["sha256"] != _sha256(protocol_path):
        raise ValueError("release protocol SHA-256 mismatch")
    artifacts = (
        release["base_policy"], release["velocity_estimator"], release["stability_gate"]
    )
    pairs = (
        (artifacts[0]["path"], artifacts[0]["sha256"]),
        (artifacts[0]["metadata_path"], artifacts[0]["metadata_sha256"]),
        (artifacts[1]["metadata_path"], artifacts[1]["metadata_sha256"]),
        (artifacts[1]["onnx_path"], artifacts[1]["onnx_sha256"]),
        (artifacts[2]["path"], artifacts[2]["sha256"]),
    )
    for relative, expected in pairs:
        _project_file(str(relative), str(expected))
    boundaries = release.get("boundaries", {})
    if boundaries.get("live_execution_authorized") is not True:
        raise ValueError("baseline execution is not authorized")
    if boundaries.get("authorized_stages") != ["baseline_envelope"]:
        raise ValueError("release must authorize only baseline_envelope")
    required_false = (
        "actual_slam_effect_claim_allowed", "touchdown_intervention_enabled",
        "teacher_training_authorized", "adaptation_training_authorized",
        "ppo_training_authorized", "ros_policy_wiring_authorized",
        "default_switch_authorized", "physical_robot_authorized",
    )
    if any(boundaries.get(key) is not False for key in required_false):
        raise ValueError("release broadens authorization beyond the baseline envelope")


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=PROJECT_ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def capture_execution_baseline() -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    if branch != "exp/slam-fastlio2":
        raise ValueError("execution is restricted to exp/slam-fastlio2")
    changed = set(_git("diff", "--name-only", "--").splitlines())
    changed.update(_git("diff", "--cached", "--name-only", "--").splitlines())
    changed.update(_git("ls-files", "--others", "--exclude-standard").splitlines())
    excluded = ("build/", "install/", "log/", "logs/", "outputs/")
    owned = sorted(
        value for value in changed
        if value and value != "lidar_type" and not value.startswith(excluded)
    )
    files = []
    for relative in owned:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
            raise ValueError(f"dirty project path is not a regular file: {relative}")
        files.append({
            "path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size,
        })
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": branch,
        "project_owned_dirty_count": len(files),
        "dirty_files": files,
        "reproduction_rule": "git_commit_plus_exact_dirty_file_sha256",
    }


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _environment(domain_id: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "ROS_DOMAIN_ID": str(domain_id),
        "ROS_LOCALHOST_ONLY": "1",
        "ROS_LOG_DIR": str(PROJECT_ROOT / "logs/ros"),
    })
    source = str(PROJECT_ROOT / "source/anymal_locomotion")
    environment["PYTHONPATH"] = f"{source}:{environment.get('PYTHONPATH', '')}".rstrip(":")
    return environment


def _run_logged(
    name: str, command: tuple[str, ...], run_dir: Path, environment: dict[str, str],
) -> int:
    with (run_dir / f"{name}.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, env=environment, check=False,
            stdout=stream, stderr=subprocess.STDOUT,
        )
    return int(result.returncode)


def execute_cell(
    row: dict[str, Any], release: dict[str, Any], stage_root: Path, domain_id: int,
    release_path: Path,
) -> dict[str, Any]:
    runtime = release["baseline_runtime"]
    runtime_profile = runtime["profile_mapping"][row["profile"]]
    run_dir = stage_root / row["profile"] / f"block_{row['block_id']}" / "zero"
    run_dir.mkdir(parents=True, exist_ok=False)
    env = _environment(domain_id)
    command = (
        "ros2", "launch", "anymal_locomotion_ros2",
        "fastlio2_locomotion_benchmark.launch.py",
        "slam_backend:=fastlio2", f"ros_domain_id:={domain_id}",
        "enable_confidence:=false", "enable_velocity_estimator:=true",
        "policy_inference_trigger:=estimator_joint_state",
        "policy_odometry_topic:=/locomotion/estimated_odom",
        f"velocity_estimator_metadata_path:={PROJECT_ROOT / release['velocity_estimator']['metadata_path']}",
        f"profile:={runtime_profile}", f"simulation_seed:={row['simulation_seed']}",
        f"simulation_steps:={int(runtime['simulation_steps'])}",
        f"point_density:={float(runtime['point_density'])}",
        f"point_density_profile:={runtime['point_density_profile']}",
        f"policy_path:={PROJECT_ROOT / release['base_policy']['path']}",
        f"metadata_path:={PROJECT_ROOT / release['base_policy']['metadata_path']}",
        f"output_dir:={run_dir}", "record_bag:=false",
    )
    with (run_dir / "launch.log").open("w", encoding="utf-8") as stream:
        launch = subprocess.run(
            command, cwd=ROS2_WORKSPACE, env=env, check=False,
            stdout=stream, stderr=subprocess.STDOUT,
        )
    steps: list[dict[str, Any]] = []
    required_launch_outputs = (
        run_dir / "locomotion_diagnostics.json",
        run_dir / "policy_diagnostics.json",
        run_dir / "driver.json",
    )
    if launch.returncode == 0 and all(path.is_file() for path in required_launch_outputs):
        commands = (
            ("baseline_trace", (
                str(ISAAC_PYTHON),
                str(PROJECT_ROOT / "scripts/validation/validate_slam_low_level_baseline_trace.py"),
                "--diagnostics", str(run_dir / "policy_diagnostics.json"),
                "--release", str(release_path),
                "--output", str(run_dir / "baseline_trace_validation.json"),
            )),
            ("stability", (
                sys.executable,
                str(PROJECT_ROOT / "scripts/validation/evaluate_stability_trace.py"),
                "--trace", str(run_dir / "locomotion_diagnostics.json"),
                "--driver", str(run_dir / "driver.json"),
                "--config", str(PROJECT_ROOT / release["stability_gate"]["path"]),
                "--output", str(run_dir / "stability_gate.json"),
            )),
        )
        for name, postprocess in commands:
            returncode = _run_logged(name, postprocess, run_dir, env)
            steps.append({"name": name, "returncode": returncode})
            if returncode != 0:
                break
        if len(steps) == 2 and all(step["returncode"] == 0 for step in steps):
            returncode = _run_logged("baseline_record", (
                sys.executable,
                str(PROJECT_ROOT / "scripts/validation/build_slam_low_level_baseline_run_record.py"),
                "--run-dir", str(run_dir), "--profile", str(row["profile"]),
                "--runtime-profile", str(runtime_profile), "--block", str(row["block_id"]),
            ), run_dir, env)
            steps.append({"name": "baseline_record", "returncode": returncode})
    record_path = run_dir / "baseline_envelope_run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    reasons = []
    if launch.returncode != 0:
        reasons.append("launch_failed")
    if not all(path.is_file() for path in required_launch_outputs):
        reasons.append("required_launch_output_missing")
    if any(step["returncode"] != 0 for step in steps):
        reasons.append("postprocess_failed")
    if record.get("gate", {}).get("passed") is not True:
        reasons.append("baseline_integrity_or_safety_gate_failed")
    result = {
        **row, "runtime_profile": runtime_profile, "run_dir": str(run_dir),
        "launch_returncode": int(launch.returncode), "postprocess": steps,
        "passed": not reasons, "stop_required": bool(reasons),
        "stop_reasons": sorted(set(reasons)),
    }
    _write(run_dir / "cell.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs/slam_low_level_touchdown_headroom_v1",
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--release", type=Path, default=RELEASE_PATH)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    release_path = args.release.expanduser().resolve()
    if not output_root.is_relative_to(PROJECT_ROOT):
        raise ValueError("output root must remain inside the project")
    if any(
        not path.is_relative_to(PROJECT_ROOT) or not path.is_file()
        for path in (protocol_path, release_path)
    ):
        raise ValueError("protocol and release must be existing project-local files")
    protocol, release = _load_yaml(protocol_path), _load_yaml(release_path)
    validation = _validator_module().validate_protocol(protocol)
    if not validation["passed"]:
        raise ValueError(validation["failures"])
    validate_release(release, protocol_path)
    schedule = validation["baseline_schedule"]
    plan = {
        "stage": "baseline_envelope", "selected_count": len(schedule),
        "profiles": release["baseline_runtime"]["profile_mapping"],
        "blocks": protocol["baseline_envelope"]["block_ids"],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    stage_root = output_root / "baseline_envelope"
    if stage_root.exists():
        raise ValueError("baseline output already exists; refusing to overwrite or mix evidence")
    snapshot = capture_execution_baseline()
    manifest = {
        "schema_version": 1,
        "kind": "slam_low_level_touchdown_baseline_matrix",
        "dataset_role": "anti_collapse_calibration_only",
        "stage": "baseline_envelope",
        "protocol_path": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": _sha256(protocol_path),
        "release_path": str(release_path.relative_to(PROJECT_ROOT)),
        "release_sha256": _sha256(release_path),
        "selected_count": len(schedule), "schedule": schedule,
        "worktree_snapshot": snapshot,
    }
    _write(stage_root / "run_manifest.json", manifest)
    results = []
    for row in schedule:
        result = execute_cell(row, release, stage_root, args.domain_id, release_path)
        results.append(result)
        if result["stop_required"]:
            break
    collection_passed = len(results) == len(schedule) and all(row["passed"] for row in results)
    envelope_path = stage_root / "model1450_anti_collapse_envelope.json"
    envelope_returncode = None
    if collection_passed:
        envelope_returncode = _run_logged("build_envelope", (
            sys.executable,
            str(PROJECT_ROOT / "scripts/validation/build_slam_low_level_model1450_envelope.py"),
            "--records-root", str(stage_root), "--protocol", str(protocol_path),
            "--output", str(envelope_path),
        ), stage_root, _environment(args.domain_id))
        collection_passed = envelope_returncode == 0 and envelope_path.is_file()
    summary = {
        **manifest, "executed_count": len(results),
        "collection_passed": collection_passed,
        "stopped_early": len(results) < len(schedule),
        "envelope_returncode": envelope_returncode,
        "envelope_path": str(envelope_path) if envelope_path.is_file() else None,
        "results": results,
    }
    _write(stage_root / "matrix_summary.json", summary)
    print(json.dumps({
        "stage": "baseline_envelope", "selected_count": len(schedule),
        "executed_count": len(results), "collection_passed": collection_passed,
        "stopped_early": summary["stopped_early"],
    }, indent=2))
    return 0 if collection_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
