#!/usr/bin/env python3
"""Plan or execute the staged real-backend fixed-speed-scale causal pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS2_WORKSPACE = PROJECT_ROOT / "deployment/ros2_ws"
PROTOCOL_PATH = PROJECT_ROOT / "configs/slam_speed_scale_causal_pilot_v1.yaml"
RELEASE_PATH = PROJECT_ROOT / "configs/slam_speed_scale_causal_release_v1.yaml"
VALIDATOR_PATH = PROJECT_ROOT / "scripts/validation/validate_slam_speed_scale_causal_protocol.py"
ANALYZER_PATH = PROJECT_ROOT / "scripts/validation/analyze_slam_speed_scale_causal_pilot.py"
ISAAC_PYTHON = Path(os.environ.get("ISAACLAB_ROOT", "/home/ros/IsaacLab")) / "_isaac_sim/python.sh"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_release(release: dict, protocol_path: Path) -> None:
    if release["protocol"]["sha256"] != _sha256(protocol_path):
        raise ValueError("release protocol SHA-256 mismatch")
    paths = [
        (release["base_policy"]["path"], release["base_policy"]["sha256"]),
        (release["export_manifest"]["path"], release["export_manifest"]["sha256"]),
        (release["common_runtime"]["velocity_estimator_metadata_path"], release["common_runtime"]["velocity_estimator_metadata_sha256"]),
    ]
    for arm in release["arms"].values():
        paths.extend((
            (arm["policy_path"], arm["policy_sha256"]),
            (arm["torchscript_path"], arm["torchscript_sha256"]),
            (arm["metadata_path"], arm["metadata_sha256"]),
            (arm["parity_path"], arm["parity_sha256"]),
        ))
    for relative, expected in paths:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"release artifact mismatch: {relative}")
    for name, arm in release["arms"].items():
        if float(arm["assigned_scale"]) <= 0.0 or float(arm["assigned_scale"]) > 1.0:
            raise ValueError(f"invalid assigned scale: {name}")
        parity = json.loads((PROJECT_ROOT / arm["parity_path"]).read_text(encoding="utf-8"))
        if parity.get("passed") is not True:
            raise ValueError(f"release parity did not pass: {name}")


def require_execution_authorization(release: dict, stage: str) -> None:
    boundaries = release.get("boundaries", {})
    if boundaries.get("live_execution_authorized") is not True:
        raise ValueError("live execution is not authorized in the release config")
    if stage not in boundaries.get("authorized_stages", []):
        raise ValueError(f"stage is not explicitly authorized: {stage}")


def capture_execution_baseline() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ("git", *args), cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    branch = git("branch", "--show-current")
    if branch != "exp/slam-fastlio2":
        raise ValueError("execution is restricted to exp/slam-fastlio2")
    changed = set(git("diff", "--name-only", "--").splitlines())
    changed.update(git("diff", "--cached", "--name-only", "--").splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    excluded = ("build/", "install/", "log/", "logs/", "outputs/")
    files = []
    for relative in sorted(
        value for value in changed
        if value and value != "lidar_type" and not value.startswith(excluded)
    ):
        path = (PROJECT_ROOT / relative).resolve()
        if path.is_file():
            files.append({"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    return {
        "git_commit": git("rev-parse", "HEAD"), "git_branch": branch,
        "project_owned_dirty": bool(files), "dirty_files": files,
        "reproduction_rule": "git_commit_plus_exact_dirty_file_sha256",
    }


def require_stage_prerequisite(
    output_root: Path, protocol: dict, stage: str, release: dict | None = None,
) -> None:
    if stage != "causal_pilot":
        return
    locked = (release or {}).get("prerequisites", {}).get("wiring_smoke")
    root = (
        (PROJECT_ROOT / locked["output_path"]).resolve()
        if isinstance(locked, dict) else output_root / "wiring_smoke"
    )
    manifest = root / "run_manifest.json"
    if not manifest.is_file():
        raise ValueError("causal pilot requires completed wiring smoke")
    if isinstance(locked, dict):
        decision_path = root / "decision.json"
        if _sha256(manifest) != locked["manifest_sha256"]:
            raise ValueError("hash-locked wiring manifest changed")
        if not decision_path.is_file() or _sha256(decision_path) != locked["decision_sha256"]:
            raise ValueError("hash-locked wiring decision changed")
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_value.get("release_sha256") != locked["executed_release_sha256"]:
            raise ValueError("wiring manifest executed-release mismatch")
    analyzer = _module("speed_analysis", ANALYZER_PATH)
    decision = analyzer.analyze_records(analyzer.load_records(root), protocol, "wiring_smoke")
    required = locked.get("required_decision", "WIRING_PASS") if isinstance(locked, dict) else "WIRING_PASS"
    if decision["decision"]["status"] != required:
        raise ValueError("causal pilot requires WIRING_PASS")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _environment(domain_id: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "ROS_DOMAIN_ID": str(domain_id), "ROS_LOCALHOST_ONLY": "1",
        "ROS_LOG_DIR": str(PROJECT_ROOT / "logs/ros"),
    })
    source = str(PROJECT_ROOT / "source/anymal_locomotion")
    environment["PYTHONPATH"] = f"{source}:{environment.get('PYTHONPATH', '')}".rstrip(":")
    return environment


def execute_cell(
    row: dict, protocol: dict, release: dict, stage_root: Path, domain_id: int,
    release_path: Path = RELEASE_PATH,
) -> dict:
    common = protocol["common"]
    arm = release["arms"][row["arm"]]
    run_dir = stage_root / row["backend"] / row["profile"] / f"block_{row['block_id']}" / row["arm"]
    run_dir.mkdir(parents=True, exist_ok=True)
    timeline = common["point_density_timeline_s"]
    command = (
        "ros2", "launch", "anymal_locomotion_ros2", "fastlio2_locomotion_benchmark.launch.py",
        f"slam_backend:={row['backend']}", f"expected_confidence_backend:={row['backend']}",
        f"ros_domain_id:={domain_id}",
        f"expected_calibration_id:={release['common_runtime']['confidence'][row['backend']]}",
        "enable_confidence:=true", "enable_velocity_estimator:=true",
        "policy_inference_trigger:=estimator_joint_state", "policy_odometry_topic:=/locomotion/estimated_odom",
        f"velocity_estimator_metadata_path:={PROJECT_ROOT / release['common_runtime']['velocity_estimator_metadata_path']}",
        f"profile:={row['profile']}", f"simulation_seed:={row['simulation_seed']}",
        f"simulation_steps:={int(common['simulation_steps'])}", "point_density:=1.0",
        f"point_density_profile:={common['point_density_profile']}",
        f"point_density_min:={common['point_density_min']}",
        f"point_density_healthy_s:={timeline['healthy']}",
        f"point_density_ramp_down_s:={timeline['ramp_down']}",
        f"point_density_hold_s:={timeline['low_support_hold']}",
        f"point_density_ramp_up_s:={timeline['recovery']}",
        f"policy_path:={PROJECT_ROOT / arm['policy_path']}",
        f"metadata_path:={PROJECT_ROOT / arm['metadata_path']}",
        f"output_dir:={run_dir}", "confidence_loss_is_outcome:=true", "record_bag:=true",
    )
    env = _environment(domain_id)
    with (run_dir / "launch.log").open("w", encoding="utf-8") as stream:
        launch = subprocess.run(command, cwd=ROS2_WORKSPACE, env=env, check=False, stdout=stream, stderr=subprocess.STDOUT)
    steps = []
    def run(name: str, cmd: tuple[str, ...], environment=None) -> int:
        with (run_dir / f"{name}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=environment or env, check=False, stdout=stream, stderr=subprocess.STDOUT)
        steps.append({"name": name, "returncode": result.returncode})
        return result.returncode
    diagnostics = run_dir / "policy_diagnostics.json"
    bag = run_dir / "raw_bag"
    if diagnostics.is_file():
        run("speed_scale_trace", (
            str(ISAAC_PYTHON), str(PROJECT_ROOT / "scripts/validation/validate_slam_speed_scale_trace.py"),
            "--diagnostics", str(diagnostics), "--release", str(release_path),
            "--arm", row["arm"], "--output", str(run_dir / "speed_scale_trace_validation.json"),
        ))
    if (run_dir / "locomotion_diagnostics.json").is_file() and (run_dir / "driver.json").is_file():
        run("stability", (
            sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_stability_trace.py"),
            "--trace", str(run_dir / "locomotion_diagnostics.json"), "--driver", str(run_dir / "driver.json"),
            "--config", str(PROJECT_ROOT / "configs/stability_diagnostics_confidence_aware.yaml"),
            "--output", str(run_dir / "stability_gate.json"),
        ))
    if (bag / "metadata.yaml").is_file() and diagnostics.is_file():
        run("offline_usability", (
            sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_slam_confidence_publication_run.py"),
            "--bag", str(bag), "--policy-diagnostics", str(diagnostics), "--arm", "B",
            "--output", str(run_dir / "offline_usability.json"),
        ))
        run("map_consistency", (
            sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_slam_map_consistency_run.py"),
            "--bag", str(bag), "--output", str(run_dir / "map_consistency.json"),
        ))
        estimator_env = env.copy()
        estimator_env["PYTHONPATH"] = f"{PROJECT_ROOT / 'deployment/python_vendor'}:{env.get('PYTHONPATH', '')}".rstrip(":")
        run("estimator_replay", (
            sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_velocity_estimator_replay.py"),
            "--bag", str(bag), "--estimator-metadata", str(PROJECT_ROOT / release["common_runtime"]["velocity_estimator_metadata_path"]),
            "--policy-metadata", str(PROJECT_ROOT / arm["metadata_path"]), "--sync-tolerance-s", "0.025",
            "--output", str(run_dir / "velocity_estimator_replay.json"),
        ), estimator_env)
    required = tuple(run_dir / name for name in (
        "speed_scale_trace_validation.json", "stability_gate.json", "offline_usability.json",
        "map_consistency.json", "velocity_estimator_replay.json",
    ))
    if launch.returncode == 0 and all(path.is_file() for path in required):
        run("run_record", (
            sys.executable, str(PROJECT_ROOT / "scripts/validation/build_slam_speed_scale_run_record.py"),
            "--run-dir", str(run_dir), "--backend", row["backend"], "--profile", row["profile"],
            "--block", str(row["block_id"]), "--arm", row["arm"], "--stage", row["stage"],
        ))
    record_path = run_dir / "speed_scale_run_record.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    reasons = []
    if not record.get("gate", {}).get("passed", False):
        reasons.append("data_integrity_or_wiring_failure")
    trace = record.get("treatment_trace", {})
    if trace.get("passed") is not True or not math.isfinite(float(trace.get("maximum_formula_error", math.nan))):
        reasons.append("assigned_or_realized_scale_violation")
    required_metrics = (
        "moving_speed_mps", "normalized_progress",
        "valid_requested_usable_next_horizon_failure_fraction",
        "tracking_restricted_mean_survival_time_s",
    )
    metrics = record.get("metrics", {})
    if any(
        metrics.get(name) is None or not math.isfinite(float(metrics[name]))
        for name in required_metrics
    ):
        reasons.append("missing_or_nonfinite_metric")
    result = {
        **row, "run_dir": str(run_dir), "launch_returncode": launch.returncode,
        "postprocess": steps, "passed": not reasons, "stop_required": bool(reasons),
        "stop_reasons": sorted(set(reasons)),
        "simulation_safety_event": bool(record.get("metrics", {}).get("fall") or record.get("metrics", {}).get("base_contact")),
    }
    _write(run_dir / "cell.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("wiring_smoke", "causal_pilot"), default="wiring_smoke")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/slam_speed_scale_causal_pilot_v1")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--release", type=Path, default=RELEASE_PATH)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output, protocol_path, release_path = (p.expanduser().resolve() for p in (args.output_root, args.protocol, args.release))
    if any(not p.is_relative_to(PROJECT_ROOT) for p in (output, protocol_path, release_path)):
        raise ValueError("all paths must remain inside the project")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    release = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    validator = _module("speed_protocol", VALIDATOR_PATH)
    validation = validator.validate_protocol(protocol)
    if not validation["passed"]:
        raise ValueError(validation["failures"])
    validate_release(release, protocol_path)
    schedule = validation["schedules"][args.stage]
    if not args.execute:
        print(json.dumps({
            "stage": args.stage,
            "selected_count": len(schedule),
            "live_execution_authorized": release.get("boundaries", {}).get(
                "live_execution_authorized", False
            ),
            "stage_authorized": args.stage in release.get("boundaries", {}).get(
                "authorized_stages", []
            ),
        }, indent=2))
        return 0
    require_execution_authorization(release, args.stage)
    baseline = capture_execution_baseline()
    require_stage_prerequisite(output, protocol, args.stage, release)
    stage_root = output / args.stage
    if stage_root.exists() and any(stage_root.iterdir()):
        raise ValueError("stage output must be fresh")
    manifest = {
        "schema_version": 1, "kind": "slam_speed_scale_causal_matrix",
        "dataset_role": "excluded_causal_development", "stage": args.stage,
        "git_commit": baseline["git_commit"], "worktree_snapshot": baseline,
        "protocol_path": str(protocol_path.relative_to(PROJECT_ROOT)), "protocol_sha256": _sha256(protocol_path),
        "release_path": str(release_path.relative_to(PROJECT_ROOT)), "release_sha256": _sha256(release_path),
        "selected_count": len(schedule), "schedule": schedule,
    }
    _write(stage_root / "run_manifest.json", manifest)
    results = []
    for row in schedule:
        result = execute_cell(row, protocol, release, stage_root, args.domain_id, release_path)
        results.append(result)
        if result["stop_required"]:
            break
    analyzer = _module("speed_analysis", ANALYZER_PATH)
    decision = analyzer.analyze_records(analyzer.load_records(stage_root), protocol, args.stage)
    _write(stage_root / "decision.json", decision)
    (stage_root / "decision.md").write_text(analyzer._markdown(decision), encoding="utf-8")
    _write(stage_root / "matrix_summary.json", {**manifest, "results": results, "decision": decision["decision"]})
    print(json.dumps({"executed_count": len(results), "decision": decision["decision"]}, indent=2))
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
