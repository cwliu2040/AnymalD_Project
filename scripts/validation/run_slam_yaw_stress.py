#!/usr/bin/env python3
# flake8: noqa: E402
"""Capture and replay the native-deskew yaw-stress pilot/formal matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS2_WORKSPACE = PROJECT_ROOT / "deployment" / "ros2_ws"
PACKAGE_PYTHON = ROS2_WORKSPACE / "src" / "anymal_locomotion_ros2"
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "slam_yaw_stress_matrix.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "slam_yaw_stress" / "pilot_v1"

import sys

sys.path.insert(0, str(PACKAGE_PYTHON))

from anymal_locomotion_ros2.yaw_stress_core import (  # noqa: E402
    build_blind_review_assignments,
    build_replay_cells,
    counterbalanced_schedule,
    file_sha256,
    validate_bag_metadata,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=PROJECT_ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _provenance(matrix_path: Path) -> dict[str, Any]:
    policy = (
        PROJECT_ROOT
        / "exported/anymal_d_locomotion_v1/recovery_v0.4.0/policy.onnx"
    )
    configs = (
        matrix_path,
        ROS2_WORKSPACE
        / "src/anymal_locomotion_ros2/config/lio_sam_params.yaml",
        ROS2_WORKSPACE
        / "src/anymal_locomotion_ros2/config/fastlio2_anymal_ouster32.yaml",
        policy,
    )
    return {
        "created_at": _utc_now(),
        "project_root": str(PROJECT_ROOT),
        "git": {
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "dirty_paths": _git("status", "--short").splitlines(),
        },
        "sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in configs
            if path.is_file()
        },
        "contracts": {
            "deskew_mode": "native",
            "loop_closure_enable": False,
            "uniform_point_density": True,
            "fastlio_laser_map_enabled": False,
            "full_map_export_qualified": False,
        },
    }


def _run(command: list[str], environment: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        return subprocess.run(
            command,
            cwd=ROS2_WORKSPACE,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def _validate_bag(bag_path: Path) -> list[str]:
    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.is_file():
        return [f"missing rosbag metadata: {metadata_path}"]
    document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return ["rosbag metadata is not a mapping"]
    return validate_bag_metadata(document)


def _capture(
    profile: str,
    seed: int,
    output_root: Path,
    environment: dict[str, str],
) -> Path:
    capture_dir = output_root / "captures" / profile / f"seed_{seed}"
    bag_path = capture_dir / "bag"
    failures = _validate_bag(bag_path) if bag_path.exists() else ["not captured"]
    if failures:
        if bag_path.exists():
            raise RuntimeError(
                f"refusing to overwrite invalid existing bag {bag_path}: {failures}"
            )
        returncode = _run(
            [
                "ros2", "launch", "anymal_locomotion_ros2",
                "lio_benchmark.launch.py", f"profile:={profile}",
                f"seed:={seed}", f"output_dir:={capture_dir}",
                "record_bag:=true", "use_motion_deskew:=false",
                "feature_cloud_info_topic:=/lio_sam/deskew/cloud_info",
                "motion_deskew_apply_translation:=false",
                "motion_deskew_replace_upstream_rotation:=false",
                "motion_deskew_required:=false",
                "loop_closure_enable:=false",
                "loop_closure_expectation:=disabled",
            ],
            environment,
            capture_dir / "capture.log",
        )
        failures = _validate_bag(bag_path)
        benchmark_report_path = capture_dir / "metrics.json"
        benchmark_report = None
        if benchmark_report_path.is_file():
            try:
                benchmark_report = json.loads(
                    benchmark_report_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                benchmark_report = None
        motion_driver_invalid = bool(
            isinstance(benchmark_report, dict)
            and benchmark_report.get("status") == "invalid"
        )
        _write_json(
            capture_dir / "source_validation.json",
            {
                "schema_version": 1,
                "status": "passed" if not failures else "failed",
                "launch_returncode": returncode,
                "failures": failures,
                "motion_driver_invalid": motion_driver_invalid,
                "benchmark_status": benchmark_report.get("status")
                if isinstance(benchmark_report, dict) else None,
                "profile": profile,
                "seed": seed,
                "bag_path": str(bag_path),
            },
        )
        if failures:
            raise RuntimeError(f"capture contract failed: {failures}")
    return bag_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("plan", "capture", "replay"), default="plan")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--schedule-seed", type=int, default=20260804)
    parser.add_argument(
        "--limit-captures",
        type=int,
        help="Run only the first N scheduled source bags for a non-qualification smoke.",
    )
    parser.add_argument(
        "--allow-dirty-smoke",
        action="store_true",
        help="Allow capture/replay from a dirty tree for local smoke only.",
    )
    parser.add_argument(
        "--rerun-replays",
        action="store_true",
        help="Rerun evaluator cells even when a valid report already exists.",
    )
    parser.add_argument(
        "--render-visuals",
        action="store_true",
        help="Render anonymous fixed-view MP4s during replay.",
    )
    args = parser.parse_args()

    matrix_path = args.matrix.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for path in (matrix_path, output_root):
        if not path.is_relative_to(PROJECT_ROOT):
            parser.error(f"path must remain inside {PROJECT_ROOT}: {path}")
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    mode = matrix["formal"] if args.formal else matrix["pilot"]
    densities = mode.get(
        "uniform_point_densities", matrix["uniform_point_densities"]
    )
    cells = build_replay_cells(
        seeds=mode["seeds"], backends=matrix["backends"],
        point_densities=densities,
    )
    schedule = counterbalanced_schedule(cells, schedule_seed=args.schedule_seed)
    if args.limit_captures is not None:
        if args.limit_captures < 1:
            parser.error("--limit-captures must be positive")
        selected_keys: list[tuple[str, int]] = []
        for cell in schedule:
            key = (cell.profile, cell.seed)
            if key not in selected_keys:
                selected_keys.append(key)
            if len(selected_keys) == args.limit_captures:
                break
        schedule = [
            cell
            for cell in schedule
            if (cell.profile, cell.seed) in selected_keys
        ]
    blind_by_case = {
        str(item["case_name"]): str(item["blind_id"])
        for item in build_blind_review_assignments(
            (cell.case_name for cell in schedule),
            review_seed=args.schedule_seed,
        )
    }
    plan = {
        "schema_version": 1,
        "mode": "formal" if args.formal else "pilot",
        "qualification_eligible": not args.allow_dirty_smoke
        and args.limit_captures is None,
        "matrix_path": str(matrix_path),
        "schedule_seed": args.schedule_seed,
        "capture_count": len({(cell.profile, cell.seed) for cell in schedule}),
        "replay_count": len(schedule),
        "provenance": _provenance(matrix_path),
        "schedule": [cell.__dict__ | {"case_name": cell.case_name} for cell in schedule],
    }
    _write_json(output_root / "experiment_plan.json", plan)
    if args.stage == "plan":
        print(json.dumps({"capture_count": plan["capture_count"], "replay_count": len(schedule)}))
        return 0
    if plan["provenance"]["git"]["dirty_paths"] and not args.allow_dirty_smoke:
        raise RuntimeError(
            "capture/replay requires a clean Git baseline; use "
            "--allow-dirty-smoke only for non-qualification smoke"
        )

    environment = os.environ.copy()
    environment["ROS_LOG_DIR"] = str(PROJECT_ROOT / "logs" / "ros")
    environment["PYTHONPATH"] = f"{PACKAGE_PYTHON}:{environment.get('PYTHONPATH', '')}"
    captures: dict[tuple[str, int], Path] = {}
    for cell in schedule:
        key = (cell.profile, cell.seed)
        if key not in captures:
            captures[key] = _capture(*key, output_root, environment)
    if args.stage == "capture":
        return 0

    results = []
    for index, cell in enumerate(schedule, 1):
        case_dir = output_root / "replays" / cell.case_name
        report = case_dir / "replay.json"
        video_path = (
            output_root
            / "visual_review"
            / "videos"
            / f"{blind_by_case[cell.case_name]}.mp4"
        )
        reusable_visual = not args.render_visuals or video_path.is_file()
        if report.is_file() and not args.rerun_replays and reusable_visual:
            try:
                existing = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if isinstance(existing, dict) and existing.get("status") in {
                "passed", "failed"
            }:
                results.append(
                    {
                        "index": index,
                        "case_name": cell.case_name,
                        "returncode": None,
                        "result_exists": True,
                        "evaluator_status": existing["status"],
                        "resumed": True,
                    }
                )
                continue
        returncode = _run(
            [
                "ros2", "launch", "anymal_locomotion_ros2",
                "slam_backend_native_replay.launch.py",
                f"slam_backend:={cell.backend}", "deskew_mode:=native",
                "yaw_stress_mode:=true",
                "enable_effect_diagnostics:=true",
                f"point_density:={cell.point_density}",
                f"bag_path:={captures[(cell.profile, cell.seed)]}",
                f"output_path:={report}",
                *(
                    [
                        "visual_video_path:="
                        + str(
                            video_path
                        )
                    ]
                    if args.render_visuals else []
                ),
            ],
            environment,
            case_dir / "launch.log",
        )
        valid_report = report.is_file()
        evaluator_status = None
        if valid_report:
            try:
                evaluator_status = json.loads(
                    report.read_text(encoding="utf-8")
                ).get("status")
            except (OSError, json.JSONDecodeError):
                valid_report = False
        result = {
            "index": index, "case_name": cell.case_name,
            "returncode": returncode, "result_exists": valid_report,
            "evaluator_status": evaluator_status,
            "resumed": False,
        }
        results.append(result)
        _write_json(case_dir / "cell_result.json", result)
        if not valid_report:
            _write_json(
                output_root / "run_summary.json",
                {"status": "infrastructure_failure", "results": results},
            )
            raise RuntimeError(f"replay produced no evaluator report: {cell.case_name}")
    _write_json(output_root / "run_summary.json", {"status": "complete", "results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
