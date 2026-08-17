#!/usr/bin/env python3
"""Replay every accepted live publication bag through both native backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS2_WORKSPACE = PROJECT_ROOT / "deployment/ros2_ws"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/slam_confidence_publication_protocol.yaml"


def bag_fingerprint(bag: Path) -> str:
    files = sorted(path for path in bag.iterdir() if path.is_file())
    if not files or not (bag / "metadata.yaml").is_file():
        raise ValueError(f"invalid rosbag2 directory: {bag}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_schedule(records: list[tuple[Path, dict[str, Any]]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    backends = protocol["replay_matrix"]["backends"]
    repetitions = int(protocol["replay_matrix"]["repetitions_per_backend_bag"])
    rows = []
    seen: set[tuple[Any, ...]] = set()
    for record_path, record in records:
        identity = record["identity"]
        source_key = tuple(
            identity[field] for field in ("backend", "profile", "condition", "paired_block_id", "arm")
        )
        if source_key in seen:
            raise ValueError(f"duplicate live source identity: {source_key}")
        seen.add(source_key)
        bag = record_path.parent / "raw_bag"
        fingerprint = bag_fingerprint(bag)
        for replay_backend in backends:
            for repetition in range(1, repetitions + 1):
                rows.append(
                    {
                        "source_identity": identity,
                        "source_record": str(record_path),
                        "source_bag": str(bag),
                        "source_bag_fingerprint_sha256": fingerprint,
                        "replay_backend": replay_backend,
                        "repetition": repetition,
                        "closed_loop_gait_causality_allowed": False,
                    }
                )
    return rows


def formal_source_identity_complete(
    records: list[tuple[Path, dict[str, Any]]], protocol: dict[str, Any],
) -> bool:
    matrix = protocol["live_matrix"]
    expected = {
        (backend, profile, condition, int(block), arm)
        for backend in matrix["backends"] for profile in matrix["profiles"]
        for condition in matrix["perception_conditions"]
        for block in matrix["paired_block_ids"] for arm in matrix["policy_arms"]
    }
    observed = [
        (
            record["identity"]["backend"], record["identity"]["profile"],
            record["identity"]["condition"], int(record["identity"]["paired_block_id"]),
            record["identity"]["arm"],
        )
        for _, record in records
    ]
    return len(observed) == len(set(observed)) and set(observed) == expected


def compare_repetitions(
    first: dict[str, Any], second: dict[str, Any], *, atol: float, rtol: float,
) -> dict[str, Any]:
    counts_match = first.get("counts") == second.get("counts")
    first_metrics = first.get("trajectory", {})
    second_metrics = second.get("trajectory", {})
    keys_match = set(first_metrics) == set(second_metrics)
    differences = {}
    finite_values = True
    within_preferred_tolerance = True
    if keys_match:
        for key in sorted(first_metrics):
            left, right = first_metrics[key], second_metrics[key]
            if isinstance(left, list) or isinstance(right, list):
                left_array = [float(value) for value in left]
                right_array = [float(value) for value in right]
                finite = len(left_array) == len(right_array) and all(
                    math.isfinite(value) for value in (*left_array, *right_array)
                )
                absolute = max(
                    (abs(a - b) for a, b in zip(left_array, right_array, strict=True)),
                    default=0.0,
                ) if finite else None
                preferred = bool(finite and absolute is not None and absolute <= atol + rtol * max(
                    [abs(value) for value in (*left_array, *right_array)] or [0.0]
                ))
                differences[key] = {
                    "finite": finite, "maximum_absolute_difference": absolute,
                    "within_preferred_tolerance": preferred,
                }
            elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
                absolute = abs(float(left) - float(right))
                allowed = atol + rtol * max(abs(float(left)), abs(float(right)))
                finite = math.isfinite(float(left)) and math.isfinite(float(right))
                preferred = finite and absolute <= allowed
                differences[key] = {
                    "absolute_difference": absolute, "preferred_allowed": allowed,
                    "finite": finite, "within_preferred_tolerance": preferred,
                }
            else:
                finite = left == right
                preferred = finite
                differences[key] = {"exact_equal": finite}
            finite_values = finite_values and finite
            within_preferred_tolerance = within_preferred_tolerance and preferred
    return {
        "counts_match_exactly": counts_match,
        "trajectory_keys_match": keys_match,
        "trajectory_values_finite": finite_values,
        "within_preferred_diagnostic_tolerance": within_preferred_tolerance,
        "numeric_differences_are_exclusion_gate": False,
        "trajectory_differences": differences,
        "passed": counts_match and keys_match and finite_values,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--allow-excluded", action="store_true")
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--source-backend", action="append")
    parser.add_argument("--profile", action="append")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--block", type=int, action="append")
    parser.add_argument("--arm", action="append")
    parser.add_argument("--replay-backend", action="append")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.is_relative_to(PROJECT_ROOT) or not output_root.is_relative_to(PROJECT_ROOT):
        raise ValueError("replay paths must remain inside the project")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    record_paths = sorted(input_root.glob("**/publication_run_record.json"))
    all_records = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in record_paths]
    if not all_records:
        raise ValueError("no accepted live run records found")
    if not all(record.get("gate", {}).get("passed") for _, record in all_records):
        raise ValueError("replay source contains a failed live run record")
    roles = {record["dataset_role"] for _, record in all_records}
    if roles != {"formal"} and not args.allow_excluded:
        raise ValueError("excluded sources require --allow-excluded")
    if roles == {"formal"}:
        formal_root = (PROJECT_ROOT / "outputs/slam_confidence_publication_v1").resolve()
        if not protocol.get("formal_collection_authorized", False):
            raise ValueError("formal replay is locked until formal collection is authorized")
        if not input_root.is_relative_to(formal_root) or not output_root.is_relative_to(formal_root):
            raise ValueError("formal replay input and output must remain under the frozen outputs root")
        if not formal_source_identity_complete(all_records, protocol):
            raise ValueError("formal replay sources do not exactly match the frozen live schedule")
        tracked_status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=no"),
            cwd=PROJECT_ROOT, check=True, text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
        if tracked_status:
            raise ValueError("formal replay requires a clean tracked worktree")
    def selected(record: dict[str, Any]) -> bool:
        identity = record["identity"]
        return (
            (not args.source_backend or identity["backend"] in args.source_backend)
            and (not args.profile or identity["profile"] in args.profile)
            and (not args.condition or identity["condition"] in args.condition)
            and (not args.block or int(identity["paired_block_id"]) in args.block)
            and (not args.arm or identity["arm"] in args.arm)
        )
    records = [(path, record) for path, record in all_records if selected(record)]
    if not records:
        raise ValueError("replay selection contains no live run records")
    schedule = build_schedule(records, protocol)
    if args.replay_backend:
        schedule = [row for row in schedule if row["replay_backend"] in args.replay_backend]
        if not schedule:
            raise ValueError("replay backend selection contains no cells")
    results = []
    prior_manifest_path = output_root / "replay_manifest.json"
    prior_results: dict[str, dict[str, Any]] = {}
    if args.reuse_existing and prior_manifest_path.is_file():
        prior = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        prior_results = {value["output_path"]: value for value in prior.get("results", [])}
    for row in schedule:
        identity = row["source_identity"]
        run_dir = output_root.joinpath(
            identity["backend"], identity["profile"], identity["condition"],
            f"block_{identity['paired_block_id']}", f"arm_{identity['arm']}",
            row["replay_backend"], f"repetition_{row['repetition']:02d}",
        )
        result_path = run_dir / "replay.json"
        prior_result = prior_results.get(str(result_path))
        reusable = bool(
            prior_result
            and prior_result.get("source_bag_fingerprint_sha256") == row["source_bag_fingerprint_sha256"]
            and prior_result.get("replay_backend") == row["replay_backend"]
            and prior_result.get("repetition") == row["repetition"]
            and result_path.is_file()
        )
        if args.execute and not reusable:
            run_dir.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["ROS_DOMAIN_ID"] = str(args.domain_id)
            environment["ROS_LOCALHOST_ONLY"] = "1"
            environment["ROS_LOG_DIR"] = str(PROJECT_ROOT / "logs/ros")
            completed = subprocess.run(
                (
                    "ros2", "launch", "anymal_locomotion_ros2", "slam_backend_native_replay.launch.py",
                    f"bag_path:={row['source_bag']}", f"slam_backend:={row['replay_backend']}",
                    "deskew_mode:=native", f"output_path:={result_path}",
                    f"ros_domain_id:={args.domain_id}",
                    "trajectory_thresholds_are_outcomes:=true",
                ),
                cwd=ROS2_WORKSPACE, env=environment, check=False,
            )
            launch_returncode = completed.returncode
        else:
            launch_returncode = prior_result.get("launch_returncode") if reusable else None
        replay = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        results.append(
            {
                **row, "output_path": str(result_path), "launch_returncode": launch_returncode,
                "reused_existing": reusable,
                "status": replay.get("status"),
                "passed": replay.get("status") == "passed" and not replay.get("failures"),
            }
        )
    reproducibility = []
    config = protocol["replay_matrix"]["reproducibility"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for result in results:
        identity = result["source_identity"]
        key = tuple(identity[field] for field in ("backend", "profile", "condition", "paired_block_id", "arm")) + (result["replay_backend"],)
        grouped.setdefault(key, []).append(result)
    if args.execute:
        for key, pair in grouped.items():
            if len(pair) != 2 or not all(value["passed"] for value in pair):
                reproducibility.append({"key": key, "passed": False, "reason": "missing_or_failed_repetition"})
                continue
            payloads = [json.loads(Path(value["output_path"]).read_text(encoding="utf-8")) for value in sorted(pair, key=lambda value: value["repetition"])]
            reproducibility.append(
                {
                    "key": key,
                    **compare_repetitions(
                        payloads[0], payloads[1],
                        atol=float(config["trajectory_metric_absolute_tolerance"]),
                        rtol=float(config["trajectory_metric_relative_tolerance"]),
                    ),
                }
            )
    selection_used = any(
        value for value in (
            args.source_backend, args.profile, args.condition, args.block, args.arm, args.replay_backend,
        )
    )
    formal_replay_complete = bool(
        roles == {"formal"}
        and not selection_used
        and len(all_records) == int(protocol["live_matrix"]["expected_run_count"])
        and len(schedule) == len(all_records) * len(protocol["replay_matrix"]["backends"])
        * int(protocol["replay_matrix"]["repetitions_per_backend_bag"])
    )
    execution_passed = bool(
        args.execute and all(value["passed"] for value in results)
        and all(value["passed"] for value in reproducibility)
    )
    manifest = {
        "schema_version": 1,
        "kind": "slam_confidence_publication_replay_matrix",
        "dataset_roles": sorted(roles),
        "closed_loop_gait_causality_allowed": False,
        "available_source_live_run_count": len(all_records),
        "selected_source_live_run_count": len(records),
        "expected_replay_count": len(schedule),
        "results": results,
        "reproducibility": reproducibility,
        "formal_replay_complete": formal_replay_complete and execution_passed,
        "passed": execution_passed,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "replay_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("dataset_roles", "available_source_live_run_count", "selected_source_live_run_count", "expected_replay_count", "closed_loop_gait_causality_allowed", "formal_replay_complete", "passed")}, indent=2))
    return 0 if (not args.execute or manifest["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
