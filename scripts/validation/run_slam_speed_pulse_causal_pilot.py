#!/usr/bin/env python3
"""Plan or execute the staged matched-prefix command-pulse causal pilot."""

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
PROTOCOL_PATH = PROJECT_ROOT / "configs/slam_speed_pulse_causal_pilot_v2.yaml"
RELEASE_PATH = PROJECT_ROOT / "configs/slam_speed_pulse_causal_release_v2.yaml"
VALIDATOR_PATH = PROJECT_ROOT / "scripts/validation/validate_slam_speed_pulse_causal_protocol.py"
ANALYZER_PATH = PROJECT_ROOT / "scripts/validation/analyze_slam_speed_pulse_causal_pilot.py"
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
    paths = (
        (release["policy"]["path"], release["policy"]["sha256"]),
        (release["policy"]["metadata_path"], release["policy"]["metadata_sha256"]),
        (release["common_runtime"]["velocity_estimator_metadata_path"], release["common_runtime"]["velocity_estimator_metadata_sha256"]),
    )
    for relative, expected in paths:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"release artifact mismatch: {relative}")


def require_execution_authorization(release: dict, stage: str) -> None:
    boundaries = release.get("boundaries", {})
    if boundaries.get("live_execution_authorized") is not True:
        raise ValueError("live execution is not authorized")
    if stage not in boundaries.get("authorized_stages", []):
        raise ValueError(f"stage is not explicitly authorized: {stage}")


def capture_execution_baseline() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(("git", *args), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    branch = git("branch", "--show-current")
    if branch != "exp/slam-fastlio2":
        raise ValueError("execution is restricted to exp/slam-fastlio2")
    changed = set(git("diff", "--name-only", "--").splitlines())
    changed.update(git("diff", "--cached", "--name-only", "--").splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    excluded = ("build/", "install/", "log/", "logs/", "outputs/")
    files = []
    for relative in sorted(value for value in changed if value and value != "lidar_type" and not value.startswith(excluded)):
        path = (PROJECT_ROOT / relative).resolve()
        if path.is_file():
            files.append({"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    return {"git_commit": git("rev-parse", "HEAD"), "git_branch": branch, "project_owned_dirty": bool(files), "dirty_files": files, "reproduction_rule": "git_commit_plus_exact_dirty_file_sha256"}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _environment(domain_id: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"ROS_DOMAIN_ID": str(domain_id), "ROS_LOCALHOST_ONLY": "1", "ROS_LOG_DIR": str(PROJECT_ROOT / "logs/ros")})
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'source/anymal_locomotion'}:{env.get('PYTHONPATH', '')}".rstrip(":")
    return env


def execute_cell(row: dict, protocol: dict, release: dict, stage_root: Path, domain_id: int, protocol_path: Path, release_path: Path) -> dict:
    common, treatment = protocol["common"], protocol["treatment"]
    run_dir = stage_root / row["backend"] / row["profile"] / f"block_{row['block_id']}" / row["arm"]
    run_dir.mkdir(parents=True, exist_ok=True)
    timeline = common["point_density_timeline_s"]
    assigned_scales = row.get("assigned_scales", [row.get("assigned_scale", 1.0)] * 3)
    command = (
        "ros2", "launch", "anymal_locomotion_ros2", "fastlio2_locomotion_benchmark.launch.py",
        f"slam_backend:={row['backend']}", f"expected_confidence_backend:={row['backend']}", f"ros_domain_id:={domain_id}",
        f"expected_calibration_id:={release['common_runtime']['confidence'][row['backend']]}",
        "enable_confidence:=true", "enable_velocity_estimator:=true", "policy_inference_trigger:=estimator_joint_state",
        "policy_odometry_topic:=/locomotion/estimated_odom",
        f"velocity_estimator_metadata_path:={PROJECT_ROOT / release['common_runtime']['velocity_estimator_metadata_path']}",
        f"profile:={row['profile']}", f"simulation_seed:={row['simulation_seed']}", f"simulation_steps:={int(common['simulation_steps'])}",
        "point_density:=1.0", f"point_density_profile:={common['point_density_profile']}", f"point_density_min:={common['point_density_min']}",
        f"point_density_healthy_s:={timeline['healthy']}", f"point_density_ramp_down_s:={timeline['ramp_down']}",
        f"point_density_hold_s:={timeline['low_support_hold']}", f"point_density_ramp_up_s:={timeline['recovery']}",
        "command_scale_pulse_enabled:=true", f"command_scale_pulse_start_s:={treatment['pulse_start_relative_to_profile_s']}",
        f"command_scale_pulse_duration_s:={treatment['pulse_duration_s']}",
        f"command_scale_pulse_scale_x:={assigned_scales[0]}",
        f"command_scale_pulse_scale_y:={assigned_scales[1]}",
        f"command_scale_pulse_scale_z:={assigned_scales[2]}",
        f"policy_path:={PROJECT_ROOT / release['policy']['path']}", f"metadata_path:={PROJECT_ROOT / release['policy']['metadata_path']}",
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
    diagnostics, driver, bag = run_dir / "policy_diagnostics.json", run_dir / "driver.json", run_dir / "raw_bag"
    if diagnostics.is_file() and driver.is_file():
        run("speed_pulse_trace", (str(ISAAC_PYTHON), str(PROJECT_ROOT / "scripts/validation/validate_slam_speed_pulse_trace.py"), "--diagnostics", str(diagnostics), "--driver", str(driver), "--protocol", str(protocol_path), "--release", str(release_path), "--arm", row["arm"], "--output", str(run_dir / "speed_pulse_trace_validation.json")))
    if (run_dir / "locomotion_diagnostics.json").is_file() and driver.is_file():
        run("stability", (sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_stability_trace.py"), "--trace", str(run_dir / "locomotion_diagnostics.json"), "--driver", str(driver), "--config", str(PROJECT_ROOT / "configs/stability_diagnostics_confidence_aware.yaml"), "--output", str(run_dir / "stability_gate.json")))
    if (bag / "metadata.yaml").is_file() and diagnostics.is_file():
        run("offline_usability", (sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_slam_confidence_publication_run.py"), "--bag", str(bag), "--policy-diagnostics", str(diagnostics), "--arm", "B", "--output", str(run_dir / "offline_usability.json")))
        run("map_consistency", (sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_slam_map_consistency_run.py"), "--bag", str(bag), "--output", str(run_dir / "map_consistency.json")))
        estimator_env = env.copy()
        estimator_env["PYTHONPATH"] = f"{PROJECT_ROOT / 'deployment/python_vendor'}:{env.get('PYTHONPATH', '')}".rstrip(":")
        run("estimator_replay", (sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_velocity_estimator_replay.py"), "--bag", str(bag), "--estimator-metadata", str(PROJECT_ROOT / release["common_runtime"]["velocity_estimator_metadata_path"]), "--policy-metadata", str(PROJECT_ROOT / release["policy"]["metadata_path"]), "--sync-tolerance-s", "0.025", "--output", str(run_dir / "velocity_estimator_replay.json")), estimator_env)
    required = tuple(run_dir / name for name in ("speed_pulse_trace_validation.json", "stability_gate.json", "offline_usability.json", "map_consistency.json", "velocity_estimator_replay.json"))
    if launch.returncode == 0 and all(path.is_file() for path in required):
        run("run_record", (sys.executable, str(PROJECT_ROOT / "scripts/validation/build_slam_speed_pulse_run_record.py"), "--run-dir", str(run_dir), "--protocol", str(protocol_path), "--backend", row["backend"], "--profile", row["profile"], "--block", str(row["block_id"]), "--arm", row["arm"], "--stage", row["stage"]))
    record_path = run_dir / "speed_pulse_run_record.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    reasons = []
    if not record.get("gate", {}).get("passed", False): reasons.append("data_integrity_or_wiring_failure")
    trace = record.get("pulse_trace", {})
    if not trace.get("passed") or not math.isfinite(float(trace.get("maximum_action_error", math.nan))): reasons.append("pulse_trace_failure")
    result = {**row, "run_dir": str(run_dir), "launch_returncode": launch.returncode, "postprocess": steps, "passed": not reasons, "stop_required": bool(reasons), "stop_reasons": reasons, "simulation_safety_event": bool(record.get("metrics", {}).get("fall") or record.get("metrics", {}).get("base_contact"))}
    _write(run_dir / "cell.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("wiring_smoke", "causal_pilot"), default="wiring_smoke")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/slam_speed_pulse_causal_pilot_v2")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--release", type=Path, default=RELEASE_PATH)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output, protocol_path, release_path = args.output_root.resolve(), args.protocol.resolve(), args.release.resolve()
    protocol, release = yaml.safe_load(protocol_path.read_text()), yaml.safe_load(release_path.read_text())
    validator = _module("pulse_protocol", VALIDATOR_PATH)
    analyzer_path = PROJECT_ROOT / protocol.get(
        "analysis_script", "scripts/validation/analyze_slam_speed_pulse_causal_pilot.py"
    )
    analyzer = _module("pulse_analysis", analyzer_path)
    validation = validator.validate_protocol(protocol)
    if not validation["passed"]: raise ValueError(validation["failures"])
    validate_release(release, protocol_path)
    schedule = validation["schedules"][args.stage]
    if not args.execute:
        print(json.dumps({"stage": args.stage, "selected_count": len(schedule), "live_execution_authorized": release["boundaries"]["live_execution_authorized"], "stage_authorized": args.stage in release["boundaries"]["authorized_stages"]}, indent=2))
        return 0
    require_execution_authorization(release, args.stage)
    baseline = capture_execution_baseline()
    if args.stage == "causal_pilot":
        prior = analyzer.analyze_records(
            analyzer.load_records(output / "wiring_smoke"), protocol, "wiring_smoke"
        )
        if prior["decision"]["status"] != "WIRING_PASS": raise ValueError("causal pilot requires WIRING_PASS")
    stage_root = output / args.stage
    if stage_root.exists() and any(stage_root.iterdir()): raise ValueError("stage output must be fresh")
    manifest = {"schema_version": 1, "kind": "slam_speed_pulse_causal_matrix", "stage": args.stage, "git_commit": baseline["git_commit"], "worktree_snapshot": baseline, "protocol_sha256": _sha256(protocol_path), "release_sha256": _sha256(release_path), "selected_count": len(schedule), "schedule": schedule}
    _write(stage_root / "run_manifest.json", manifest)
    results = []
    for row in schedule:
        result = execute_cell(row, protocol, release, stage_root, args.domain_id, protocol_path, release_path)
        results.append(result)
        if result["stop_required"]: break
        records = analyzer.load_records(stage_root)
        group = [r for r in records if (r["identity"]["backend"], r["identity"]["profile"], r["identity"]["block_id"]) == (row["backend"], row["profile"], row["block_id"])]
        if len(group) == 4 and protocol["experimental_design"]["exact_prestate_matching_required"]:
            matched, _ = analyzer._pre_match({r["identity"]["arm"]: r for r in group}, protocol)
            if not matched:
                result["stop_required"] = True; result["passed"] = False; result["stop_reasons"].append("pre_pulse_matching_failure"); break
    decision = analyzer.analyze_records(analyzer.load_records(stage_root), protocol, args.stage)
    _write(stage_root / "decision.json", decision)
    _write(stage_root / "matrix_summary.json", {**manifest, "results": results, "decision": decision["decision"]})
    print(json.dumps({"executed_count": len(results), "decision": decision["decision"]}, indent=2))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
