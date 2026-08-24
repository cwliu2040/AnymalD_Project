#!/usr/bin/env python3
"""Plan or execute the staged real-backend action-risk intervention pilot."""

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
PROTOCOL_PATH = PROJECT_ROOT / "configs/slam_action_risk_intervention_pilot.yaml"
RELEASE_PATH = PROJECT_ROOT / "configs/slam_action_risk_intervention_release_v1.yaml"
VALIDATOR_PATH = PROJECT_ROOT / "scripts/validation/validate_slam_action_risk_intervention_protocol.py"
ANALYZER_PATH = PROJECT_ROOT / "scripts/validation/analyze_slam_action_risk_intervention_pilot.py"
ISAAC_PYTHON = Path(os.environ.get("ISAACLAB_ROOT", "/home/ros/IsaacLab")) / "_isaac_sim/python.sh"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _validator_module():
    spec = importlib.util.spec_from_file_location("intervention_protocol", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _analyzer_module():
    spec = importlib.util.spec_from_file_location("intervention_analysis", ANALYZER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_release(release: dict, protocol_path: Path) -> None:
    if release["protocol"]["sha256"] != _sha256(protocol_path):
        raise ValueError("release protocol SHA-256 mismatch")
    paths = [
        (release["base_policy"]["path"], release["base_policy"]["sha256"]),
        (
            release["common_runtime"]["velocity_estimator_metadata_path"],
            release["common_runtime"]["velocity_estimator_metadata_sha256"],
        ),
    ]
    for arm in release["arms"].values():
        paths.extend((
            (arm["policy_path"], arm["policy_sha256"]),
            (arm["metadata_path"], arm["metadata_sha256"]),
            (arm["parity_path"], arm["parity_sha256"]),
        ))
    for relative, expected in paths:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"release artifact mismatch: {relative}")
    for arm in release["arms"].values():
        parity = json.loads((PROJECT_ROOT / arm["parity_path"]).read_text(encoding="utf-8"))
        if parity.get("passed") is not True:
            raise ValueError(f"release parity did not pass: {arm['parity_path']}")


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=PROJECT_ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def capture_execution_baseline() -> dict:
    branch = _git("branch", "--show-current")
    if branch != "exp/slam-fastlio2":
        raise ValueError("execution is restricted to exp/slam-fastlio2")
    changed = set(_git("diff", "--name-only", "--").splitlines())
    changed.update(_git("diff", "--cached", "--name-only", "--").splitlines())
    untracked = set(
        _git("ls-files", "--others", "--exclude-standard").splitlines()
    )
    excluded = ("build/", "install/", "log/", "logs/", "outputs/")
    owned = sorted(
        path for path in changed | untracked
        if path and path != "lidar_type" and not path.startswith(excluded)
    )
    files = []
    for relative in owned:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"dirty path escapes project: {relative}")
        if path.is_file():
            files.append({
                "path": relative, "exists": True,
                "sha256": _sha256(path), "size_bytes": path.stat().st_size,
            })
        elif not path.exists():
            files.append({
                "path": relative, "exists": False,
                "sha256": None, "size_bytes": None,
            })
        else:
            raise ValueError(f"dirty project path is not a regular file: {relative}")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": branch,
        "project_owned_dirty": bool(files),
        "dirty_files": files,
        "reproduction_rule": "git_commit_plus_exact_dirty_file_sha256",
    }


def require_execution_authorization(release: dict, stage: str) -> None:
    boundaries = release.get("boundaries", {})
    authorized = boundaries.get("authorized_stages", [])
    if boundaries.get("live_execution_authorized") is not True:
        raise ValueError("live execution is not authorized in the release config")
    if not isinstance(authorized, list) or stage not in authorized:
        raise ValueError(f"stage is not explicitly authorized: {stage}")


def require_stage_prerequisite(
    output_root: Path, protocol: dict, stage: str,
    protocol_path: Path = PROTOCOL_PATH,
    release: dict | None = None,
) -> None:
    prerequisite = {
        "pilot": ("wiring_smoke", "WIRING_PASS"),
        "expanded_only_after_pilot_inconclusive": ("pilot", "INCONCLUSIVE"),
    }.get(stage)
    if prerequisite is None:
        return
    prerequisite_stage, required_decision = prerequisite
    reused = (release or {}).get("prerequisites", {}).get(prerequisite_stage)
    if stage == "pilot" and isinstance(reused, dict):
        root = (PROJECT_ROOT / reused["output_path"]).resolve()
        reused_protocol_path = (PROJECT_ROOT / reused["protocol_path"]).resolve()
        manifest_path = root / "run_manifest.json"
        if any(not path.is_relative_to(PROJECT_ROOT) for path in (
            root, reused_protocol_path, manifest_path,
        )):
            raise ValueError("reused prerequisite escapes the project")
        if not root.is_dir() or not reused_protocol_path.is_file() or not manifest_path.is_file():
            raise ValueError("reused wiring prerequisite evidence is missing")
        if _sha256(reused_protocol_path) != reused["protocol_sha256"]:
            raise ValueError("reused wiring protocol SHA-256 mismatch")
        if _sha256(manifest_path) != reused["manifest_sha256"]:
            raise ValueError("reused wiring manifest SHA-256 mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_sha256") != reused["protocol_sha256"]:
            raise ValueError("reused wiring manifest protocol mismatch")
        analyzer = _analyzer_module()
        reused_protocol = _load_yaml(reused_protocol_path)
        decision = analyzer.analyze_records(
            analyzer.load_records(root), reused_protocol, prerequisite_stage,
        )["decision"]["status"]
        expected = reused.get("required_decision", required_decision)
        if decision != expected:
            raise ValueError(
                f"{stage} requires reused {prerequisite_stage}={expected}; got {decision}"
            )
        return
    root = output_root / prerequisite_stage
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing prerequisite manifest: {prerequisite_stage}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != prerequisite_stage:
        raise ValueError("prerequisite manifest stage mismatch")
    if manifest.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("prerequisite used a different protocol")
    analyzer = _analyzer_module()
    decision = analyzer.analyze_records(
        analyzer.load_records(root), protocol, prerequisite_stage,
    )["decision"]["status"]
    if decision != required_decision:
        raise ValueError(
            f"{stage} requires {prerequisite_stage}={required_decision}; got {decision}"
        )


def require_fresh_stage_output(stage_root: Path) -> None:
    execution_markers = (
        "run_manifest.json", "matrix_summary.json", "decision.json",
    )
    if any((stage_root / name).exists() for name in execution_markers):
        raise ValueError("stage output already contains execution artifacts")
    if stage_root.exists() and any(stage_root.glob("**/cell.json")):
        raise ValueError("stage output already contains cell artifacts")


def _repeated_simulation_safety_harm(
    results: list[dict], protocol: dict,
) -> bool:
    simulation = protocol.get("decision_gate", {}).get(
        "safety_stop", {},
    ).get("simulation")
    if not isinstance(simulation, dict):
        return False
    minimum = int(
        simulation["route_fail_minimum_distinct_blocks_same_backend_profile"]
    )
    by_pair: dict[tuple[str, str, int], dict[str, bool]] = {}
    for result in results:
        if not all(key in result for key in ("backend", "profile", "block_id", "arm")):
            continue
        key = (result["backend"], result["profile"], int(result["block_id"]))
        by_pair.setdefault(key, {})[result["arm"]] = bool(
            result.get("simulation_safety_event", False)
        )
    excess: dict[tuple[str, str], set[int]] = {}
    for (backend, profile, block), arms in by_pair.items():
        if set(arms) != {"smooth", "zero", "antismooth"}:
            continue
        if arms["smooth"] and not arms["zero"]:
            excess.setdefault((backend, profile), set()).add(block)
    return any(len(blocks) >= minimum for blocks in excess.values())


def execute_schedule(
    schedule: list[dict], executor, protocol: dict | None = None,
) -> list[dict]:
    results = []
    for row in schedule:
        result = executor(row)
        results.append(result)
        if result["stop_required"]:
            break
        if protocol is not None and _repeated_simulation_safety_harm(results, protocol):
            result["stop_required"] = True
            result["passed"] = False
            result.setdefault("stop_reasons", []).append(
                "repeated_paired_smooth_specific_safety_harm"
            )
            result["stop_reasons"] = sorted(set(result["stop_reasons"]))
            break
    return results


def cell_stop_reasons(record: dict, row: dict, protocol: dict) -> list[str]:
    reasons = []
    if not record.get("gate", {}).get("passed", False):
        reasons.append("data_integrity_or_wiring_failure")
    intervention = record.get("intervention", {})
    maximum_residual = intervention.get("maximum_realized_residual")
    if intervention.get("passed") is not True:
        reasons.append("intervention_trace_failure")
    if maximum_residual is None or not math.isfinite(float(maximum_residual)):
        reasons.append("nonfinite_intervention_residual")
    elif float(maximum_residual) > (
        float(protocol["intervention"]["raw_action_linf_limit"])
        + float(intervention.get("atol", 0.0))
    ):
        reasons.append("intervention_linf_violation")
    metrics = record.get("metrics", {})
    required_metrics = (
        "action_rate_rms_per_s", "while_stable_roll_pitch_rate_rms_radps",
        "tracking_restricted_mean_survival_time_s",
        "valid_requested_usable_next_horizon_failure_fraction", "moving_speed_mps",
    )
    if any(
        metrics.get(name) is None or not math.isfinite(float(metrics[name]))
        for name in required_metrics
    ):
        reasons.append("missing_or_nonfinite_metric")
    legacy_safety = protocol.get("decision_gate", {}).get(
        "safety_stop", {},
    ).get("any_smooth_arm_fall_or_base_contact") is True
    if legacy_safety and row["arm"] == "smooth" and (
        bool(metrics.get("fall")) or bool(metrics.get("base_contact"))
    ):
        reasons.append("smooth_arm_safety_failure")
    return sorted(set(reasons))


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
    row: dict, protocol: dict, release: dict, output_root: Path, domain_id: int,
    release_path: Path = RELEASE_PATH,
) -> dict:
    common = protocol["common"]
    arm = release["arms"][row["arm"]]
    run_dir = output_root / row["backend"] / row["profile"] / f"block_{row['block_id']}" / row["arm"]
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
        launch = subprocess.run(
            command, cwd=ROS2_WORKSPACE, env=env, check=False,
            stdout=stream, stderr=subprocess.STDOUT,
        )
    diagnostics = run_dir / "policy_diagnostics.json"
    bag = run_dir / "raw_bag"
    steps = []
    def run(name: str, cmd: tuple[str, ...], *, environment: dict[str, str] | None = None) -> int:
        with (run_dir / f"{name}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                cmd, cwd=PROJECT_ROOT, env=environment or env, check=False,
                stdout=stream, stderr=subprocess.STDOUT,
            )
        steps.append({"name": name, "returncode": result.returncode})
        return result.returncode
    if diagnostics.is_file():
        run("intervention_trace", (
            str(ISAAC_PYTHON), str(PROJECT_ROOT / "scripts/validation/validate_slam_action_intervention_trace.py"),
            "--diagnostics", str(diagnostics), "--release", str(release_path),
            "--arm", row["arm"], "--output", str(run_dir / "intervention_trace_validation.json"),
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
        deployment_vendor = str(PROJECT_ROOT / "deployment/python_vendor")
        estimator_env["PYTHONPATH"] = (
            f"{deployment_vendor}:{env.get('PYTHONPATH', '')}"
        ).rstrip(":")
        run("estimator_replay", (
            sys.executable, str(PROJECT_ROOT / "scripts/validation/evaluate_velocity_estimator_replay.py"),
            "--bag", str(bag), "--estimator-metadata",
            str(PROJECT_ROOT / release["common_runtime"]["velocity_estimator_metadata_path"]),
            "--policy-metadata", str(PROJECT_ROOT / arm["metadata_path"]),
            "--sync-tolerance-s", "0.025", "--output", str(run_dir / "velocity_estimator_replay.json"),
        ), environment=estimator_env)
    required = tuple(run_dir / name for name in (
        "intervention_trace_validation.json", "stability_gate.json", "offline_usability.json",
        "map_consistency.json", "velocity_estimator_replay.json",
    ))
    if launch.returncode == 0 and all(path.is_file() for path in required):
        run("run_record", (
            sys.executable, str(PROJECT_ROOT / "scripts/validation/build_slam_action_risk_intervention_run_record.py"),
            "--run-dir", str(run_dir), "--backend", row["backend"], "--profile", row["profile"],
            "--block", str(row["block_id"]), "--arm", row["arm"], "--stage", row["stage"],
        ))
    record_path = run_dir / "intervention_run_record.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    stop_reasons = cell_stop_reasons(record, row, protocol)
    result = {
        **row, "run_dir": str(run_dir), "launch_returncode": launch.returncode,
        "postprocess": steps, "passed": not stop_reasons,
        "stop_required": bool(stop_reasons), "stop_reasons": sorted(set(stop_reasons)),
        "simulation_safety_event": bool(
            record.get("metrics", {}).get("fall")
            or record.get("metrics", {}).get("base_contact")
        ),
    }
    _write(run_dir / "cell.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("wiring_smoke", "pilot", "expanded_only_after_pilot_inconclusive"), default="wiring_smoke")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/slam_action_risk_intervention_pilot_v1")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--release", type=Path, default=RELEASE_PATH)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("output root must remain inside the project")
    protocol_path = args.protocol.expanduser().resolve()
    release_path = args.release.expanduser().resolve()
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
    schedule = validation["schedules"][args.stage]
    head = _git("rev-parse", "HEAD")
    manifest = {
        "schema_version": 2, "kind": "slam_action_risk_intervention_matrix",
        "dataset_role": "excluded_causal_development", "stage": args.stage,
        "git_commit": head, "protocol_path": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": _sha256(protocol_path),
        "release_path": str(release_path.relative_to(PROJECT_ROOT)),
        "release_sha256": _sha256(release_path), "selected_count": len(schedule),
        "schedule": schedule,
    }
    if not args.execute:
        print(json.dumps({"stage": args.stage, "selected_count": len(schedule)}, indent=2))
        return 0
    require_execution_authorization(release, args.stage)
    baseline = capture_execution_baseline()
    manifest["git_commit"] = baseline["git_commit"]
    manifest["worktree_snapshot"] = baseline
    require_stage_prerequisite(
        output, protocol, args.stage, protocol_path, release,
    )
    stage_root = output / args.stage
    require_fresh_stage_output(stage_root)
    _write(stage_root / "run_manifest.json", manifest)
    results = execute_schedule(
        schedule,
        lambda row: execute_cell(
            row, protocol, release, stage_root, args.domain_id, release_path,
        ),
        protocol,
    )
    analyzer = _analyzer_module()
    decision = analyzer.analyze_records(analyzer.load_records(stage_root), protocol, args.stage)
    _write(stage_root / "decision.json", {
        "schema_version": 1,
        "kind": "slam_action_risk_intervention_decision",
        "dataset_role": "excluded_causal_development",
        **decision,
    })
    (stage_root / "decision.md").write_text(
        analyzer._markdown(decision), encoding="utf-8"
    )
    summary = {
        **manifest,
        "collection_passed": all(row["passed"] for row in results),
        "decision": decision["decision"],
        "results": results,
    }
    _write(stage_root / "matrix_summary.json", summary)
    print(json.dumps({
        "stage": args.stage,
        "selected_count": len(schedule),
        "executed_count": len(results),
        "collection_passed": summary["collection_passed"],
        "decision": summary["decision"],
    }, indent=2))
    return 0 if summary["collection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
