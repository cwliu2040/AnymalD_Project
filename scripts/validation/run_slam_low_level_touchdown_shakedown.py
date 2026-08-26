#!/usr/bin/env python3
"""Plan or execute the fail-closed block597 touchdown wiring smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
ROS_WS = ROOT / "deployment/ros2_ws"
PROTOCOL = ROOT / "configs/slam_low_level_touchdown_headroom_v1.yaml"
RELEASE = ROOT / "configs/slam_low_level_touchdown_shakedown_release_v1.yaml"
VALIDATOR = ROOT / "scripts/validation/validate_slam_low_level_touchdown_headroom_protocol.py"
ANALYZER = ROOT / "scripts/validation/analyze_slam_low_level_touchdown_headroom.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = _module("touchdown_protocol", VALIDATOR)
A = _module("touchdown_analysis", ANALYZER)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE).stdout.strip()


def _snapshot() -> dict:
    changed = set(_git("diff", "--name-only", "--").splitlines())
    changed.update(_git("ls-files", "--others", "--exclude-standard").splitlines())
    owned = sorted(path for path in changed if path and path != "lidar_type" and
                   not path.startswith(("build/", "install/", "log/", "logs/", "outputs/")))
    return {"git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"),
            "dirty_files": [{"path": value, "sha256": _sha(ROOT / value)} for value in owned]}


def validate_release(release: dict, protocol_path: Path) -> None:
    if release.get("status") not in {
        "block597_execution_authorized", "block597_wiring_pass_execution_closed",
    }:
        raise ValueError("unsupported block597 release status")
    if release["protocol"]["sha256"] != _sha(protocol_path):
        raise ValueError("protocol hash mismatch")
    for section in ("base_policy", "velocity_estimator", "phase_tracker", "kinematics",
                    "baseline_envelope", "stability_gate"):
        value = release[section]
        pairs = [("path", "sha256")]
        if section == "base_policy":
            pairs.append(("metadata_path", "metadata_sha256"))
        if section == "velocity_estimator":
            pairs = [("metadata_path", "metadata_sha256")]
        for path_key, hash_key in pairs:
            path = (ROOT / value[path_key]).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or _sha(path) != value[hash_key]:
                raise ValueError(f"release artifact mismatch: {value[path_key]}")
    boundary = release["boundaries"]
    authorized = release.get("status") == "block597_execution_authorized"
    if boundary.get("live_execution_authorized") is not authorized:
        raise ValueError("release status and live authorization disagree")
    if boundary.get("authorized_stages") != (["wiring_smoke"] if authorized else []):
        raise ValueError("only wiring_smoke may be authorized")
    if boundary.get("authorized_blocks") != [597] or boundary.get("pilot_or_confirmation_authorized") is not False:
        raise ValueError("release broadened beyond block597")


def _environment(domain: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"ROS_DOMAIN_ID": str(domain), "ROS_LOCALHOST_ONLY": "1", "ROS_LOG_DIR": str(ROOT / "logs/ros")})
    env["PYTHONPATH"] = f"{ROOT / 'source/anymal_locomotion'}:{env.get('PYTHONPATH', '')}".rstrip(":")
    return env


def execute_cell(row: dict, release: dict, stage_root: Path, domain: int) -> dict:
    runtime_profile = release["runtime"]["profile_mapping"][row["profile"]]
    run_dir = stage_root / row["backend"] / row["profile"] / f"block_{row['block_id']}" / row["arm"]
    run_dir.mkdir(parents=True, exist_ok=False)
    env = _environment(domain)
    command = (
        "ros2", "launch", "anymal_locomotion_ros2", "fastlio2_locomotion_benchmark.launch.py",
        f"slam_backend:={row['backend']}", f"expected_confidence_backend:={row['backend']}",
        f"expected_calibration_id:={release['runtime']['confidence_calibration'][row['backend']]}",
        f"ros_domain_id:={domain}", "enable_confidence:=true", "enable_velocity_estimator:=true",
        "policy_inference_trigger:=estimator_joint_state", "policy_odometry_topic:=/locomotion/estimated_odom",
        f"velocity_estimator_metadata_path:={ROOT / release['velocity_estimator']['metadata_path']}",
        f"profile:={runtime_profile}", f"simulation_seed:={row['simulation_seed']}",
        f"simulation_steps:={release['runtime']['simulation_steps']}", "point_density:=1.0",
        "point_density_profile:=constant", f"policy_path:={ROOT / release['base_policy']['path']}",
        f"metadata_path:={ROOT / release['base_policy']['metadata_path']}", f"output_dir:={run_dir}",
        "confidence_loss_is_outcome:=true", "record_bag:=true",
        "enable_touchdown_residual_experiment:=true", f"touchdown_residual_arm:={row['arm']}",
        f"touchdown_phase_artifact_path:={ROOT / release['phase_tracker']['path']}",
        f"touchdown_phase_artifact_sha256:={release['phase_tracker']['sha256']}",
    )
    with (run_dir / "launch.log").open("w", encoding="utf-8") as stream:
        launch = subprocess.run(command, cwd=ROS_WS, env=env, check=False,
                                stdout=stream, stderr=subprocess.STDOUT)
    steps = []
    def run(name: str, command: tuple[str, ...]) -> int:
        with (run_dir / f"{name}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, check=False,
                                    stdout=stream, stderr=subprocess.STDOUT)
        steps.append({"name": name, "returncode": result.returncode})
        return result.returncode
    bag = run_dir / "raw_bag"
    if (run_dir / "locomotion_diagnostics.json").is_file() and (run_dir / "driver.json").is_file():
        run("stability", (sys.executable, str(ROOT / "scripts/validation/evaluate_stability_trace.py"),
            "--trace", str(run_dir / "locomotion_diagnostics.json"), "--driver", str(run_dir / "driver.json"),
            "--config", str(ROOT / release["stability_gate"]["path"]), "--output", str(run_dir / "stability_gate.json")))
    if (bag / "metadata.yaml").is_file() and (run_dir / "policy_diagnostics.json").is_file():
        run("offline_usability", (sys.executable,
            str(ROOT / "scripts/validation/evaluate_slam_confidence_publication_run.py"),
            "--bag", str(bag), "--policy-diagnostics", str(run_dir / "policy_diagnostics.json"),
            "--arm", "A", "--output", str(run_dir / "offline_usability.json")))
    required = [run_dir / name for name in ("policy_diagnostics.json", "locomotion_diagnostics.json",
                "driver.json", "stability_gate.json", "offline_usability.json")]
    if launch.returncode == 0 and all(path.is_file() for path in required) and all(step["returncode"] == 0 for step in steps):
        run("run_record", (sys.executable, str(ROOT / "scripts/validation/build_slam_low_level_touchdown_run_record.py"),
            "--run-dir", str(run_dir), "--backend", row["backend"], "--profile", row["profile"],
            "--block", str(row["block_id"]), "--arm", row["arm"], "--stage", row["stage"]))
    record_path = run_dir / "touchdown_headroom_run_record.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    reasons = []
    if launch.returncode != 0: reasons.append("launch_failed")
    if not all(path.is_file() for path in required): reasons.append("required_output_missing")
    if any(step["returncode"] != 0 for step in steps): reasons.append("postprocess_failed")
    if record.get("gate", {}).get("passed") is not True: reasons.append("wiring_or_safety_gate_failed")
    result = {**row, "runtime_profile": runtime_profile, "run_dir": str(run_dir),
              "launch_returncode": launch.returncode, "postprocess": steps,
              "passed": not reasons, "stop_required": bool(reasons), "stop_reasons": reasons}
    _write(run_dir / "cell.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/slam_low_level_touchdown_headroom_v1")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    protocol_path, release_path = args.protocol.resolve(), args.release.resolve()
    protocol, release = _load(protocol_path), _load(release_path)
    validation = V.validate_protocol(protocol)
    if not validation["passed"]: raise ValueError(validation["failures"])
    validate_release(release, protocol_path)
    schedule = V.build_stage_schedule(protocol, "wiring_smoke")
    plan = {"stage": "wiring_smoke", "selected_count": len(schedule), "schedule": schedule,
            "execution_authorized": release["boundaries"]["live_execution_authorized"]}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return 0
    if release.get("status") != "block597_execution_authorized":
        raise ValueError("block597 execution is closed")
    stage_root = args.output_root.resolve() / "wiring_smoke"
    if stage_root.exists(): raise ValueError("wiring_smoke output already exists")
    manifest = {"schema_version": 1, "kind": "touchdown_block597_wiring_smoke", **plan,
                "protocol_sha256": _sha(protocol_path), "release_sha256": _sha(release_path),
                "worktree_snapshot": _snapshot()}
    _write(stage_root / "run_manifest.json", manifest)
    results = []
    for row in schedule:
        result = execute_cell(row, release, stage_root, args.domain_id)
        results.append(result)
        if result["stop_required"]: break
    decision = A.analyze_records(A.load_records(stage_root), protocol, "wiring_smoke")
    _write(stage_root / "decision.json", decision)
    passed = len(results) == len(schedule) and all(row["passed"] for row in results) and decision["decision"]["status"] == "WIRING_PASS"
    _write(stage_root / "matrix_summary.json", {**manifest, "executed_count": len(results),
           "collection_passed": passed, "stopped_early": len(results) < len(schedule), "decision": decision, "results": results})
    print(json.dumps({"selected_count": len(schedule), "executed_count": len(results),
                      "collection_passed": passed, "decision": decision["decision"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
