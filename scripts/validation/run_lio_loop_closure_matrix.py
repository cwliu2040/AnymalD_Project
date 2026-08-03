#!/usr/bin/env python3
"""Capture and replay a deterministic LIO-SAM loop-closure matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS2_WORKSPACE = PROJECT_ROOT / "deployment" / "ros2_ws"
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "lio_loop_closure_matrix.yaml"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "lio_sam_loop_closure" / "qualification_v1"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
parser.add_argument(
    "--reuse-capture-root",
    type=Path,
    help=(
        "Reuse capture bags and capture reports from another matrix output "
        "root; only replay stages are written to --output-root."
    ),
)
parser.add_argument(
    "--policy-path",
    type=Path,
    help="Project-local ONNX policy used while capturing each source bag.",
)
parser.add_argument(
    "--metadata-path",
    type=Path,
    help="Project-local policy metadata used while capturing each source bag.",
)
parser.add_argument(
    "--rerun",
    action="store_true",
    help="Rerun passed stages; existing rosbag directories are never removed.",
)
args = parser.parse_args()


def _project_path(path: Path, *, file_required: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        parser.error(f"path must remain inside {PROJECT_ROOT}: {resolved}")
    if file_required and not resolved.is_file():
        parser.error(f"file does not exist: {resolved}")
    return resolved


def _passed_report(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return report if report.get("status") == "passed" else None


def _run(command: tuple[str, ...], environment: dict[str, str]) -> int:
    return subprocess.run(
        command,
        cwd=ROS2_WORKSPACE,
        env=environment,
        check=False,
    ).returncode


def main() -> None:
    matrix_path = _project_path(args.matrix, file_required=True)
    output_root = _project_path(args.output_root)
    capture_root = (
        _project_path(args.reuse_capture_root)
        if args.reuse_capture_root is not None
        else None
    )
    if capture_root is not None and not capture_root.is_dir():
        parser.error(f"capture root does not exist: {capture_root}")
    policy_path = (
        _project_path(args.policy_path, file_required=True)
        if args.policy_path is not None
        else None
    )
    metadata_path = (
        _project_path(args.metadata_path, file_required=True)
        if args.metadata_path is not None
        else None
    )
    if (policy_path is None) != (metadata_path is None):
        parser.error("--policy-path and --metadata-path must be provided together")
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != 1:
        raise ValueError("loop matrix schema_version must be 1")
    repetitions = int(matrix["repetitions"])
    profiles = tuple(matrix["profiles"])
    search = matrix["loop_search"]
    if repetitions <= 0 or not profiles:
        raise ValueError("loop matrix must contain repetitions and profiles")
    for profile in profiles:
        if profile["enabled_expectation"] not in {"required", "forbidden"}:
            raise ValueError("enabled_expectation must be required or forbidden")

    environment = os.environ.copy()
    environment["ROS_LOG_DIR"] = str(PROJECT_ROOT / "logs" / "ros")
    source_root = str(PROJECT_ROOT / "source" / "anymal_locomotion")
    environment["PYTHONPATH"] = (
        f"{source_root}:{environment.get('PYTHONPATH', '')}"
    )
    common_loop_args = (
        f"loop_search_radius:={float(search['radius_m'])}",
        "loop_search_time_diff:="
        f"{float(search['minimum_time_difference_s'])}",
        f"loop_search_keyframes:={int(search['neighboring_keyframes'])}",
        "loop_fitness_score:="
        f"{float(search['maximum_icp_fitness_score'])}",
    )
    capture_policy_args = (
        (
            f"policy_path:={policy_path}",
            f"metadata_path:={metadata_path}",
        )
        if policy_path is not None and metadata_path is not None
        else ()
    )

    results: list[dict] = []
    for profile in profiles:
        profile_name = str(profile["name"])
        enabled_expectation = str(profile["enabled_expectation"])
        for repetition in range(1, repetitions + 1):
            run_dir = (
                output_root / profile_name / f"run_{repetition:02d}"
            )
            capture_run_dir = (
                capture_root / profile_name / f"run_{repetition:02d}"
                if capture_root is not None
                else run_dir
            )
            capture_report_path = capture_run_dir / "capture" / "metrics.json"
            bag_path = capture_run_dir / "capture" / "bag"
            capture_passed = _passed_report(capture_report_path) is not None
            bag_ready = (bag_path / "metadata.yaml").is_file()
            capture_returncode = None
            if capture_root is not None:
                if not (capture_passed and bag_ready):
                    raise RuntimeError(
                        "reused capture is incomplete; expected a passed "
                        f"capture report and metadata.yaml under {capture_run_dir}"
                    )
                print(
                    f"REUSE CAPTURE {profile_name} run {repetition:02d}",
                    flush=True,
                )
            elif args.rerun or not (capture_passed and bag_ready):
                if bag_path.exists():
                    raise RuntimeError(
                        "refusing to overwrite an existing rosbag; preserve or "
                        f"move it before rerunning: {bag_path}"
                    )
                print(
                    f"CAPTURE {profile_name} run {repetition:02d}",
                    flush=True,
                )
                capture_returncode = _run(
                    (
                        "ros2",
                        "launch",
                        "anymal_locomotion_ros2",
                        "lio_benchmark.launch.py",
                        f"profile:={profile_name}",
                        f"output_dir:={run_dir / 'capture'}",
                        "record_bag:=true",
                        "loop_closure_enable:=false",
                        "loop_closure_expectation:=disabled",
                        *capture_policy_args,
                        *common_loop_args,
                    ),
                    environment,
                )
                capture_passed = (
                    capture_returncode == 0
                    and _passed_report(capture_report_path) is not None
                    and (bag_path / "metadata.yaml").is_file()
                )
            else:
                print(
                    f"SKIP capture {profile_name} run {repetition:02d}",
                    flush=True,
                )

            run_result = {
                "profile": profile_name,
                "repetition": repetition,
                "bag_path": str(bag_path),
                "capture_passed": capture_passed,
                "capture_returncode": capture_returncode,
                "replays": {},
            }
            if not capture_passed:
                results.append(run_result)
                continue

            for mode, enabled, expectation in (
                ("disabled", "false", "forbidden"),
                ("enabled", "true", enabled_expectation),
            ):
                report_path = run_dir / f"replay_{mode}.json"
                replay_report = None if args.rerun else _passed_report(report_path)
                replay_returncode = None
                if replay_report is None:
                    print(
                        f"REPLAY {mode} {profile_name} "
                        f"run {repetition:02d}",
                        flush=True,
                    )
                    replay_returncode = _run(
                        (
                            "ros2",
                            "launch",
                            "anymal_locomotion_ros2",
                            "lio_replay_benchmark.launch.py",
                            f"bag_path:={bag_path}",
                            f"output_path:={report_path}",
                            f"loop_closure_enable:={enabled}",
                            f"loop_closure_expectation:={expectation}",
                            *common_loop_args,
                        ),
                        environment,
                    )
                    replay_report = _passed_report(report_path)
                else:
                    print(
                        f"SKIP replay {mode} {profile_name} "
                        f"run {repetition:02d}",
                        flush=True,
                    )
                run_result["replays"][mode] = {
                    "passed": (
                        replay_returncode in {None, 0}
                        and replay_report is not None
                    ),
                    "returncode": replay_returncode,
                    "report_path": str(report_path),
                    "same_sensor_bag": str(bag_path),
                    "expectation": expectation,
                }
            results.append(run_result)

    expected_runs = len(profiles) * repetitions
    passed_runs = sum(
        result["capture_passed"]
        and all(
            replay["passed"]
            for replay in result["replays"].values()
        )
        and len(result["replays"]) == 2
        for result in results
    )
    summary = {
        "schema_version": 1,
        "matrix_path": str(matrix_path),
        "output_root": str(output_root),
        "policy_path": str(policy_path) if policy_path is not None else None,
        "metadata_path": (
            str(metadata_path) if metadata_path is not None else None
        ),
        "expected_run_count": expected_runs,
        "passed_run_count": passed_runs,
        "failed_run_count": expected_runs - passed_runs,
        "passed": passed_runs == expected_runs,
        "runs": results,
    }
    summary_path = output_root / "matrix_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "expected_run_count": expected_runs,
                "passed_run_count": passed_runs,
                "failed_run_count": expected_runs - passed_runs,
                "passed": summary["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Matrix summary written to: {summary_path}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
