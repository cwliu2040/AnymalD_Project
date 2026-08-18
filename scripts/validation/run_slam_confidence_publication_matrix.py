#!/usr/bin/env python3
"""Plan or execute the preregistered A/B/C/D publication matrix safely."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROS_PACKAGE_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "deployment/ros2_ws/src/anymal_locomotion_ros2"
)
sys.path.insert(0, str(ROS_PACKAGE_SOURCE))
from anymal_locomotion_ros2.lio_benchmark_core import get_motion_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS2_WORKSPACE = PROJECT_ROOT / "deployment" / "ros2_ws"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/slam_confidence_publication_protocol.yaml"
DEFAULT_RELEASE = PROJECT_ROOT / "configs/slam_confidence_sim_release_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "logs/slam_confidence_publication_v1"
ARM_IDS = ("A", "B", "C", "D")
REQUIRED_BAG_TOPICS = (
    "/clock", "/odom", "/imu/data", "/lidar/points_raw", "/cmd_vel",
    "/joint_states", "/foot_contacts", "/locomotion/estimated_odom",
    "/slam/odom", "/slam_confidence",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(value: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(value).expanduser()
    resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path escapes project: {resolved}")
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def route_contract(profile_name: str) -> dict[str, Any]:
    profile = get_motion_profile(profile_name)
    serialized = asdict(profile)
    serialized["duration_s"] = profile.duration_s
    encoded = json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    return {
        "profile": serialized,
        "profile_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_path": str(
            ROS_PACKAGE_SOURCE
            / "anymal_locomotion_ros2/lio_benchmark_core.py"
        ),
        "source_sha256": _sha256(
            ROS_PACKAGE_SOURCE
            / "anymal_locomotion_ros2/lio_benchmark_core.py"
        ),
    }


def validate_raw_bag(bag_dir: Path) -> dict[str, Any]:
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.is_file():
        return {"passed": False, "failures": ["bag metadata is missing"]}
    metadata = _load_yaml(metadata_path)["rosbag2_bagfile_information"]
    counts = {
        item["topic_metadata"]["name"]: int(item["message_count"])
        for item in metadata.get("topics_with_message_count", [])
    }
    failures = [
        f"required topic missing or empty: {topic}"
        for topic in REQUIRED_BAG_TOPICS
        if counts.get(topic, 0) <= 0
    ]
    files = [metadata_path]
    for relative in metadata.get("relative_file_paths", []):
        path = (bag_dir / str(relative)).resolve()
        if not path.is_relative_to(bag_dir.resolve()) or not path.is_file():
            failures.append(f"invalid bag data file: {relative}")
        else:
            files.append(path)
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return {
        "passed": not failures,
        "failures": failures,
        "message_count": int(metadata.get("message_count", 0)),
        "duration_ns": int(metadata.get("duration", {}).get("nanoseconds", 0)),
        "topic_message_counts": counts,
        "fingerprint_sha256": digest.hexdigest(),
    }


def validate_artifacts(release: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Resolve and hash-lock every executable policy arm plus estimator15."""
    resolved: dict[str, dict[str, str]] = {}
    for arm_id in ARM_IDS:
        arm = release["arms"][arm_id]
        policy = _project_path(arm["policy_path"])
        metadata = _project_path(arm["metadata_path"])
        policy_sha = _sha256(policy)
        metadata_sha = _sha256(metadata)
        if policy_sha != str(arm["policy_sha256"]):
            raise ValueError(f"arm {arm_id} policy SHA-256 mismatch")
        if metadata_sha != str(arm["metadata_sha256"]):
            raise ValueError(f"arm {arm_id} metadata SHA-256 mismatch")
        document = _load_yaml(metadata)
        expected_dimension = int(arm["observation_dimension"])
        if document.get("observation", {}).get("dimension") != expected_dimension:
            raise ValueError(f"arm {arm_id} observation dimension mismatch")
        resolved[arm_id] = {
            "policy_path": str(policy),
            "policy_sha256": policy_sha,
            "metadata_path": str(metadata),
            "metadata_sha256": metadata_sha,
            "observation_dimension": str(expected_dimension),
        }
    common = release["common_runtime"]
    sync_tolerance_s = float(common["velocity_estimator_sync_tolerance_s"])
    if sync_tolerance_s <= 0.0:
        raise ValueError("estimator synchronization tolerance must be positive")
    estimator = _project_path(common["velocity_estimator_metadata_path"])
    if _sha256(estimator) != str(common["velocity_estimator_metadata_sha256"]):
        raise ValueError("estimator15 metadata SHA-256 mismatch")
    resolved["common"] = {
        "velocity_estimator_metadata_path": str(estimator),
        "velocity_estimator_metadata_sha256": _sha256(estimator),
        "velocity_estimator_sync_tolerance_s": str(sync_tolerance_s),
        "stability_config_path": str(_project_path(common["stability_config_path"])),
        "model48_agent_config_path": str(_project_path(common["model48_agent_config_path"])),
        "model48_checkpoint_path": str(_project_path(release["arms"]["C"]["checkpoint_path"])),
    }
    return resolved


def stability_artifact_data_valid(document: dict[str, Any]) -> bool:
    """Separate trace integrity from an observed safety failure."""
    summary = document.get("summary", {})
    hard = summary.get("hard_failures", {})
    return bool(
        int(summary.get("sample_count", 0)) >= 2
        and int(hard.get("non_finite_sample_count", 1)) == 0
        and isinstance(document.get("gate", {}).get("failures", []), list)
    )


def balanced_arm_order(block_index: int, stratum_index: int) -> tuple[str, ...]:
    """Deterministic balanced Latin-square order, reversed on odd strata."""
    base = ARM_IDS if stratum_index % 2 == 0 else tuple(reversed(ARM_IDS))
    shift = block_index % len(base)
    return base[shift:] + base[:shift]


def build_schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = protocol["live_matrix"]
    rows: list[dict[str, Any]] = []
    stratum = 0
    for backend in matrix["backends"]:
        for profile in matrix["profiles"]:
            for condition in matrix["perception_conditions"]:
                for block_index, block_id in enumerate(matrix["paired_block_ids"]):
                    for order, arm in enumerate(balanced_arm_order(block_index, stratum)):
                        rows.append(
                            {
                                "backend": str(backend),
                                "profile": str(profile),
                                "condition": str(condition),
                                "block_id": int(block_id),
                                "simulation_seed": int(block_id),
                                "arm": arm,
                                "arm_order": order,
                            }
                        )
                stratum += 1
    if len(rows) != int(matrix["expected_run_count"]):
        raise ValueError("generated schedule does not match expected_run_count")
    return rows


def build_pilot_schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = protocol["sample_size_pilot_matrix"]
    formal_blocks = set(protocol["live_matrix"]["paired_block_ids"])
    pilot_blocks = set(matrix["paired_block_ids"])
    if formal_blocks & pilot_blocks:
        raise ValueError("sample-size pilot blocks must be disjoint from formal blocks")
    rows: list[dict[str, Any]] = []
    stratum = 0
    for backend in matrix["backends"]:
        for profile in matrix["profiles"]:
            for condition in matrix["perception_conditions"]:
                for block_index, block_id in enumerate(matrix["paired_block_ids"]):
                    for order, arm in enumerate(balanced_arm_order(block_index, stratum)):
                        rows.append(
                            {
                                "backend": str(backend), "profile": str(profile),
                                "condition": str(condition), "block_id": int(block_id),
                                "simulation_seed": int(block_id), "arm": arm, "arm_order": order,
                            }
                        )
                stratum += 1
    if len(rows) != int(matrix["expected_run_count"]):
        raise ValueError("generated pilot schedule does not match expected_run_count")
    return rows


def build_challenge_calibration_schedule(
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    matrix = protocol["challenge_calibration_matrix"]
    rows: list[dict[str, Any]] = []
    stratum = 0
    for backend in matrix["backends"]:
        for profile in matrix["profiles"]:
            for support in matrix["support_fraction_candidates"]:
                support_value = float(support)
                support_id = f"{support_value:.3f}".replace(".", "p")
                arms = tuple(str(value) for value in matrix["policy_arms"])
                if stratum % 2:
                    arms = tuple(reversed(arms))
                for block_id in matrix["paired_block_ids"]:
                    for order, arm in enumerate(arms):
                        rows.append({
                            "backend": str(backend),
                            "profile": str(profile),
                            "condition": f"gradual_support_{support_id}",
                            "minimum_support_fraction": support_value,
                            "block_id": int(block_id),
                            "simulation_seed": int(block_id),
                            "arm": arm,
                            "arm_order": order,
                        })
                stratum += 1
    if len(rows) != int(matrix["expected_run_count"]):
        raise ValueError(
            "generated challenge calibration schedule does not match expected_run_count"
        )
    return rows


def _selected(value: str, selections: list[str] | None) -> bool:
    return not selections or value in selections


def select_schedule(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if _selected(row["backend"], args.backend)
        and _selected(row["profile"], args.profile)
        and _selected(row["condition"], args.condition)
        and _selected(row["arm"], args.arm)
        and (not args.block or row["block_id"] in args.block)
    ]
    if not selected:
        raise ValueError("selection contains no publication cells")
    return selected


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=PROJECT_ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()


def project_owned_untracked(paths: list[str]) -> list[str]:
    runtime_roots = ("build/", "install/", "log/", "logs/", "outputs/")
    runtime_files = {"lidar_type"}
    return sorted(
        path for path in paths
        if path not in runtime_files and not path.startswith(runtime_roots)
    )


def validate_clean_execution_baseline() -> str:
    if _git_output("branch", "--show-current") != "exp/slam-fastlio2":
        raise ValueError("publication execution is restricted to exp/slam-fastlio2")
    if _git_output("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("publication execution requires a clean tracked worktree")
    untracked_text = _git_output("ls-files", "--others", "--exclude-standard")
    untracked = project_owned_untracked(untracked_text.splitlines() if untracked_text else [])
    if untracked:
        raise ValueError(f"publication execution has project-owned untracked files: {untracked[:5]}")
    return _git_output("rev-parse", "HEAD")


def validate_formal_authorization(protocol: dict[str, Any], release: dict[str, Any]) -> str:
    if not protocol.get("formal_collection_authorized", False):
        raise ValueError("formal collection is not authorized by the frozen protocol")
    head = validate_clean_execution_baseline()
    return head


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _condition_arguments(
    protocol: dict[str, Any], row: dict[str, Any]
) -> tuple[str, str]:
    if "minimum_support_fraction" in row:
        return "gradual_v2", str(float(row["minimum_support_fraction"]))
    condition = str(row["condition"])
    if condition == "native":
        return "constant", "1.0"
    configured = protocol["live_matrix"]["perception_conditions"][condition]
    return "gradual_v2", str(configured["minimum_support_fraction"])


def execute_cell(
    row: dict[str, Any], protocol: dict[str, Any], release: dict[str, Any],
    artifacts: dict[str, dict[str, str]], output_root: Path, dataset_role: str,
    domain_id: int,
) -> dict[str, Any]:
    arm_id = row["arm"]
    backend = row["backend"]
    calibration = protocol["frozen_artifacts"]["confidence"][backend]["calibration_id"]
    route = route_contract(row["profile"])
    density_profile, density_min = _condition_arguments(protocol, row)
    run_dir = output_root / backend / row["profile"] / row["condition"] / f"block_{row['block_id']}" / f"arm_{arm_id}"
    command = (
        "ros2", "launch", "anymal_locomotion_ros2", "fastlio2_locomotion_benchmark.launch.py",
        f"slam_backend:={backend}", f"expected_confidence_backend:={backend}",
        f"ros_domain_id:={domain_id}",
        f"expected_calibration_id:={calibration}", "enable_confidence:=true",
        "enable_velocity_estimator:=true", "policy_inference_trigger:=estimator_joint_state",
        "policy_odometry_topic:=/locomotion/estimated_odom",
        f"velocity_estimator_metadata_path:={artifacts['common']['velocity_estimator_metadata_path']}",
        "velocity_estimator_sync_tolerance_s:="
        f"{release['common_runtime']['velocity_estimator_sync_tolerance_s']}",
        f"profile:={row['profile']}", f"simulation_seed:={row['simulation_seed']}",
        f"simulation_steps:={int(protocol['live_matrix']['simulation_steps'])}",
        "point_density:=1.0", f"point_density_profile:={density_profile}",
        f"point_density_min:={density_min}", f"policy_path:={artifacts[arm_id]['policy_path']}",
        f"confidence_loss_is_outcome:={'true' if row['condition'] != 'native' else 'false'}",
        f"metadata_path:={artifacts[arm_id]['metadata_path']}", f"output_dir:={run_dir}",
    )
    environment = os.environ.copy()
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    environment["ROS_LOCALHOST_ONLY"] = "1"
    environment["ROS_LOG_DIR"] = str(PROJECT_ROOT / "logs/ros")
    project_source = str(PROJECT_ROOT / "source/anymal_locomotion")
    existing_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        f"{project_source}:{existing_python_path}"
        if existing_python_path else project_source
    )
    command = (*command, "record_bag:=true")
    completed = subprocess.run(command, cwd=ROS2_WORKSPACE, env=environment, check=False)
    driver_path = run_dir / "driver.json"
    diagnostics_path = run_dir / "policy_diagnostics.json"
    driver = json.loads(driver_path.read_text(encoding="utf-8")) if driver_path.is_file() else {}
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.is_file() else {}
    records = diagnostics.get("records", [])
    expected_dimension = int(artifacts[arm_id]["observation_dimension"])
    diagnostics_ok = bool(records) and all(
        len(record.get("observation", [])) == expected_dimension for record in records
    )
    trace_path = run_dir / "locomotion_diagnostics.json"
    stability_path = run_dir / "stability_gate.json"
    stability_returncode = None
    if trace_path.is_file() and driver_path.is_file():
        stability = subprocess.run(
            (
                sys.executable,
                str(PROJECT_ROOT / "scripts/validation/evaluate_stability_trace.py"),
                "--trace", str(trace_path), "--driver", str(driver_path),
                "--config", artifacts["common"]["stability_config_path"],
                "--output", str(stability_path),
            ),
            cwd=PROJECT_ROOT, env=environment, check=False,
        )
        stability_returncode = stability.returncode
    stability_document = (
        json.loads(stability_path.read_text(encoding="utf-8"))
        if stability_path.is_file() else {}
    )
    stability_gate = stability_document.get("gate", {})
    stability_data_valid = stability_artifact_data_valid(stability_document)
    mechanism_returncode = None
    mechanism_gate: dict[str, Any] = {"passed": True, "required": False}
    mechanism_path = run_dir / "mechanism_sidecar.json"
    if arm_id in {"B", "C", "D"} and diagnostics_path.is_file():
        isaaclab_root = Path(os.environ.get("ISAACLAB_ROOT", "/home/ros/IsaacLab"))
        mechanism = subprocess.run(
            (
                str(isaaclab_root / "_isaac_sim/python.sh"),
                str(PROJECT_ROOT / "scripts/validation/build_slam_confidence_mechanism_sidecar.py"),
                "--diagnostics", str(diagnostics_path),
                "--checkpoint", artifacts["common"]["model48_checkpoint_path"],
                "--agent-config", artifacts["common"]["model48_agent_config_path"],
                "--executed-arm", arm_id, "--action-atol", "1e-5",
                "--output", str(mechanism_path),
            ),
            cwd=PROJECT_ROOT, env=environment, check=False,
        )
        mechanism_returncode = mechanism.returncode
        mechanism_gate = (
            json.loads(mechanism_path.read_text(encoding="utf-8")).get("gate", {})
            if mechanism_path.is_file() else {"passed": False, "required": True}
        )
        mechanism_gate["required"] = True
    bag_gate = validate_raw_bag(run_dir / "raw_bag")
    estimator_replay_path = run_dir / "velocity_estimator_replay.json"
    estimator_replay_returncode = None
    estimator_replay_gate: dict[str, Any] = {"passed": False}
    if bag_gate.get("passed"):
        estimator_environment = environment.copy()
        deployment_vendor = str(PROJECT_ROOT / "deployment/python_vendor")
        estimator_environment["PYTHONPATH"] = (
            f"{deployment_vendor}:{environment['PYTHONPATH']}"
        )
        estimator_replay = subprocess.run(
            (
                sys.executable,
                str(PROJECT_ROOT / "scripts/validation/evaluate_velocity_estimator_replay.py"),
                "--bag", str(run_dir / "raw_bag"),
                "--estimator-metadata", artifacts["common"]["velocity_estimator_metadata_path"],
                "--policy-metadata", artifacts[arm_id]["metadata_path"],
                "--sync-tolerance-s",
                str(release["common_runtime"]["velocity_estimator_sync_tolerance_s"]),
                "--output", str(estimator_replay_path),
            ),
            cwd=PROJECT_ROOT, env=estimator_environment, check=False,
        )
        estimator_replay_returncode = estimator_replay.returncode
        if estimator_replay_path.is_file():
            estimator_replay_gate = json.loads(
                estimator_replay_path.read_text(encoding="utf-8")
            ).get("gate", {})
    offline_path = run_dir / "offline_usability.json"
    offline_returncode = None
    offline_gate: dict[str, Any] = {"passed": False}
    if bag_gate.get("passed") and diagnostics_path.is_file():
        offline = subprocess.run(
            (
                sys.executable,
                str(PROJECT_ROOT / "scripts/validation/evaluate_slam_confidence_publication_run.py"),
                "--bag", str(run_dir / "raw_bag"),
                "--policy-diagnostics", str(diagnostics_path),
                "--arm", arm_id, "--output", str(offline_path),
            ),
            cwd=PROJECT_ROOT, env=environment, check=False,
        )
        offline_returncode = offline.returncode
        if offline_path.is_file():
            offline_gate = json.loads(offline_path.read_text(encoding="utf-8")).get("gate", {})
    map_path = run_dir / "map_consistency.json"
    map_returncode = None
    map_gate: dict[str, Any] = {"passed": False}
    map_outcome: dict[str, Any] = {"map_registration_valid": False}
    if bag_gate.get("passed"):
        map_evaluation = subprocess.run(
            (
                sys.executable,
                str(PROJECT_ROOT / "scripts/validation/evaluate_slam_map_consistency_run.py"),
                "--bag", str(run_dir / "raw_bag"), "--output", str(map_path),
            ),
            cwd=PROJECT_ROOT, env=environment, check=False,
        )
        map_returncode = map_evaluation.returncode
        if map_path.is_file():
            map_document = json.loads(map_path.read_text(encoding="utf-8"))
            map_gate = map_document.get("gate", {})
            map_outcome = map_document.get(
                "outcome", {"map_registration_valid": True}
            )
    run_record_path = run_dir / "publication_run_record.json"
    run_record_returncode = None
    run_record_gate: dict[str, Any] = {"passed": False}
    if offline_gate.get("passed") and map_gate.get("passed"):
        run_record = subprocess.run(
            (
                sys.executable,
                str(PROJECT_ROOT / "scripts/validation/build_slam_confidence_publication_run_record.py"),
                "--run-dir", str(run_dir), "--backend", str(row["backend"]),
                "--profile", str(row["profile"]), "--condition", str(row["condition"]),
                "--block", str(row["block_id"]), "--arm", arm_id,
                "--dataset-role", dataset_role,
            ),
            cwd=PROJECT_ROOT, env=environment, check=False,
        )
        run_record_returncode = run_record.returncode
        if run_record_path.is_file():
            run_record_gate = json.loads(run_record_path.read_text(encoding="utf-8")).get("gate", {})
    collection_valid = bool(
        completed.returncode == 0 and driver.get("passed") and diagnostics_ok
        and stability_returncode in (0, 1) and stability_data_valid
        and (mechanism_returncode in (None, 0)) and mechanism_gate.get("passed")
        and bag_gate.get("passed")
        and estimator_replay_returncode == 0 and estimator_replay_gate.get("passed")
        and offline_returncode == 0 and offline_gate.get("passed")
        and map_returncode == 0 and map_gate.get("passed")
        and run_record_returncode == 0 and run_record_gate.get("passed")
    )
    cell = {
        "schema_version": 1, "dataset_role": dataset_role,
        "ros_domain_id": domain_id, "route_contract": route,
        **row, "artifacts": {"arm": artifacts[arm_id], "common": artifacts["common"]},
        "command": list(command), "launch_returncode": completed.returncode,
        "driver_passed": bool(driver.get("passed")), "diagnostics_passed": diagnostics_ok,
        "stability_returncode": stability_returncode, "stability_gate": stability_gate,
        "stability_data_integrity_passed": stability_data_valid,
        "stability_outcome_passed": bool(stability_gate.get("passed")),
        "mechanism_returncode": mechanism_returncode, "mechanism_gate": mechanism_gate,
        "raw_bag_gate": bag_gate,
        "velocity_estimator_replay_returncode": estimator_replay_returncode,
        "velocity_estimator_replay_gate": estimator_replay_gate,
        "offline_usability_returncode": offline_returncode,
        "offline_usability_gate": offline_gate,
        "map_consistency_returncode": map_returncode,
        "map_consistency_gate": map_gate,
        "map_outcome": map_outcome,
        "publication_run_record_returncode": run_record_returncode,
        "publication_run_record_gate": run_record_gate,
        "collection_valid": collection_valid,
        "policy_or_slam_failure_retained_as_outcome": bool(
            not stability_gate.get("passed")
            or not map_outcome.get("map_registration_valid", False)
        ),
        "passed": collection_valid,
    }
    _write_json(run_dir / "cell.json", cell)
    return cell


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", action="append")
    parser.add_argument("--profile", action="append")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--arm", action="append", choices=ARM_IDS)
    parser.add_argument("--block", action="append", type=int)
    parser.add_argument("--execute", action="store_true")
    role = parser.add_mutually_exclusive_group()
    role.add_argument("--formal", action="store_true")
    role.add_argument("--pilot", action="store_true")
    role.add_argument("--challenge-calibration", action="store_true")
    parser.add_argument("--domain-id", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol_path = _project_path(args.protocol)
    release_path = _project_path(args.release)
    output_root = _project_path(args.output_root, must_exist=False)
    protocol, release = _load_yaml(protocol_path), _load_yaml(release_path)
    artifacts = validate_artifacts(release)
    if args.challenge_calibration:
        schedule = build_challenge_calibration_schedule(protocol)
    else:
        schedule = build_pilot_schedule(protocol) if args.pilot else build_schedule(protocol)
    selected = select_schedule(schedule, args)
    if not 0 <= args.domain_id <= 232:
        raise ValueError("ROS domain ID must be in [0, 232]")
    if args.formal and not output_root.is_relative_to(
        PROJECT_ROOT / "outputs/slam_confidence_publication_v1"
    ):
        raise ValueError("formal output must be under outputs/slam_confidence_publication_v1")
    if args.pilot and output_root == DEFAULT_OUTPUT.resolve():
        raise ValueError("pilot collection requires an explicit, dedicated output root")
    if args.challenge_calibration and output_root == DEFAULT_OUTPUT.resolve():
        raise ValueError(
            "challenge calibration requires an explicit, dedicated output root"
        )
    dataset_role = (
        "formal" if args.formal else
        "excluded_pilot" if args.pilot else
        "excluded_calibration" if args.challenge_calibration else
        "excluded_smoke"
    )
    if args.formal:
        head = validate_formal_authorization(protocol, release)
    elif (args.pilot or args.challenge_calibration) and args.execute:
        head = validate_clean_execution_baseline()
    else:
        head = _git_output("rev-parse", "HEAD")
    manifest = {
        "schema_version": 1, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(protocol_path), "release_sha256": _sha256(release_path),
        "git_commit": head, "dataset_role": dataset_role,
        "formal": args.formal, "pilot": args.pilot,
        "challenge_calibration": args.challenge_calibration,
        "ros_domain_id": args.domain_id,
        "full_schedule_count": len(schedule),
        "selected_count": len(selected),
        "route_contracts": {
            profile: route_contract(profile)
            for profile in sorted({row["profile"] for row in selected})
        },
        "schedule": selected,
    }
    _write_json(output_root / "run_manifest.json", manifest)
    if not args.execute:
        print(json.dumps({key: manifest[key] for key in ("dataset_role", "full_schedule_count", "selected_count")}, indent=2))
        return 0
    results = [
        execute_cell(
            row, protocol, release, artifacts, output_root, dataset_role,
            args.domain_id,
        )
        for row in selected
    ]
    summary = {**manifest, "passed": all(row["passed"] for row in results), "results": results}
    _write_json(output_root / "matrix_summary.json", summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
