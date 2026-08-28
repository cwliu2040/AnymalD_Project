#!/usr/bin/env python3
"""Run and reduce the frozen post-training J0/J1/J2 motion audit."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "anymal_locomotion"
sys.path.insert(0, str(SOURCE_ROOT))

from anymal_locomotion.joint_training_contract import (  # noqa: E402
    evaluate_body_lidar_mechanism,
    evaluate_paired_noninferiority,
)


TASKS = {
    "J0": "Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0",
    "J1": "Isaac-Velocity-Flat-Anymal-D-Locomotion-JointTraining-J1-v0",
    "J2": "Isaac-Velocity-Flat-Anymal-D-Locomotion-JointTraining-J2-v0",
}
CAUSAL_TASKS = {
    "J0": "Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0",
    "J1": "Isaac-Velocity-Flat-Anymal-D-Locomotion-CausalJointTraining-J1-v0",
    "J2": "Isaac-Velocity-Flat-Anymal-D-Locomotion-CausalJointTraining-J2-v0",
}
CONSTRAINED_BARRIER_TASKS = {
    "J0": "Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0",
    "J1": "Isaac-Velocity-Flat-Anymal-D-Locomotion-ConstrainedBarrier-J1-v0",
    "J2": "Isaac-Velocity-Flat-Anymal-D-Locomotion-ConstrainedBarrier-J2-v0",
}


def _project_path(value: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(value)
    resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path must remain inside repository: {resolved}")
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _checkpoints(protocol: dict[str, Any]) -> dict[tuple[str, int], Path]:
    source = _project_path(
        protocol["initialization_and_anchoring"]["source_checkpoint"], must_exist=True
    )
    result = {("J0", int(seed)): source for seed in protocol["post_training_motion_audit"]["seeds"]}
    checkpoint_root = Path(
        protocol["training_execution"].get(
            "checkpoint_root", "logs/rsl_rl/anymal_d_locomotion_joint_training_v1"
        )
    )
    checkpoint_name = str(
        protocol["training_execution"].get("checkpoint_name", "model_299.pt")
    )
    for row in protocol["training_execution"]["runs"]:
        arm = str(row["arm"])
        seed = int(row["seed"])
        checkpoint = _project_path(
            checkpoint_root / str(row["run"]) / checkpoint_name,
            must_exist=True,
        )
        result[(arm, seed)] = checkpoint
    expected = {
        (arm, int(seed))
        for arm in ("J0", "J1", "J2")
        for seed in protocol["post_training_motion_audit"]["seeds"]
    }
    if set(result) != expected:
        raise ValueError("motion-audit checkpoint matrix is incomplete")
    return result


def _run_one(
    *,
    python: Path,
    arm: str,
    seed: int,
    profile: str,
    command: dict[str, float],
    checkpoint: Path,
    cfg: dict[str, Any],
    output: Path,
    tasks: dict[str, str] = TASKS,
) -> None:
    command_line = [
        str(python),
        str(PROJECT_ROOT / "scripts/rsl_rl/evaluate.py"),
        "--task", tasks[arm],
        "--checkpoint", str(checkpoint),
        "--num_envs", str(cfg["num_environments"]),
        "--seed", str(seed),
        "--vx", str(command["vx_mps"]),
        "--vy", str(command["vy_mps"]),
        "--wz", str(command["wz_radps"]),
        "--steps", str(cfg["steps"]),
        "--warmup_steps", str(cfg["warmup_steps"]),
        "--motion-audit",
        "--output", str(output),
        "--headless",
    ]
    child_env = os.environ.copy()
    existing_python_path = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = str(SOURCE_ROOT) + (
        os.pathsep + existing_python_path if existing_python_path else ""
    )
    completed = subprocess.run(
        command_line,
        cwd=PROJECT_ROOT,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        tail = completed.stdout[-6000:]
        raise RuntimeError(
            f"motion audit failed for {arm}/{seed}/{profile}:\n{tail}"
        )
    if not output.is_file():
        raise RuntimeError(f"motion audit produced no report: {output}")


def _ratio(candidate: float, comparator: float) -> float | None:
    return candidate / comparator if abs(comparator) > 1.0e-9 else None


def _compare(
    candidate: dict[str, Any],
    comparator: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    candidate_motion = candidate["motion_audit"]
    comparator_motion = comparator["motion_audit"]
    candidate_extra = candidate_motion["additional_metrics"]
    comparator_extra = comparator_motion["additional_metrics"]
    command = candidate["command"]
    frozen = protocol["future_evaluation"]
    matched_cfg = frozen["matched_motion"]

    linear_active = math.hypot(command["vx_mps"], command["vy_mps"]) >= 0.25
    yaw_active = abs(command["wz_radps"]) >= 0.25
    linear_progress_ratio = (
        _ratio(
            candidate_extra["positive_linear_progress_m"],
            comparator_extra["positive_linear_progress_m"],
        )
        if linear_active
        else None
    )
    yaw_progress_ratio = (
        _ratio(
            candidate_extra["positive_yaw_progress_rad"],
            comparator_extra["positive_yaw_progress_rad"],
        )
        if yaw_active
        else None
    )
    speed_difference = (
        abs(
            candidate_extra["mean_planar_speed_mps"]
            - comparator_extra["mean_planar_speed_mps"]
        )
        if linear_active
        else 0.0
    )
    yaw_difference = (
        abs(
            candidate_extra["mean_absolute_yaw_rate_radps"]
            - comparator_extra["mean_absolute_yaw_rate_radps"]
        )
        if yaw_active
        else 0.0
    )
    candidate_stopped = candidate_motion["gait_safety_metrics"]["stopped_fraction"]
    comparator_stopped = comparator_motion["gait_safety_metrics"]["stopped_fraction"]
    matched_checks = {
        "moving_linear_speed": speed_difference
        <= matched_cfg["maximum_abs_moving_linear_speed_difference_mps"],
        "realized_yaw_rate": yaw_difference
        <= matched_cfg["maximum_abs_realized_yaw_rate_difference_radps"],
        "linear_progress": linear_progress_ratio is None
        or linear_progress_ratio >= matched_cfg["minimum_linear_progress_ratio"],
        "yaw_progress": yaw_progress_ratio is None
        or yaw_progress_ratio >= matched_cfg["minimum_yaw_progress_ratio"],
        "stopped_fraction": candidate_stopped - comparator_stopped
        <= matched_cfg["maximum_stopped_fraction_excess"],
        "candidate_absolute_stopped_fraction": candidate_stopped
        <= matched_cfg["maximum_candidate_stopped_fraction"],
    }
    anti_collapse = evaluate_paired_noninferiority(
        candidate_motion["gait_safety_metrics"],
        comparator_motion["gait_safety_metrics"],
        frozen["paired_noninferiority_metrics"],
    )
    direction = frozen["directional_gate"]
    candidate_body = {"available": True, **candidate_motion["body_lidar_metrics"]}
    comparator_body = {"available": True, **comparator_motion["body_lidar_metrics"]}
    body_lidar = evaluate_body_lidar_mechanism(
        candidate_body,
        comparator_body,
        minimum_improvement_fraction=direction["minimum_improvement_fraction"],
        maximum_regression_fraction=direction["maximum_regression_fraction"],
    )
    safety = {
        "candidate_survival": candidate["stability"]["survival_rate"] == 1.0,
        "candidate_termination": candidate["stability"]["termination_events"] == 0,
        "no_event_excess": candidate["stability"]["termination_events"]
        <= comparator["stability"]["termination_events"],
    }
    finite = _finite_tree(candidate_motion) and _finite_tree(comparator_motion)
    passed = bool(
        finite
        and all(matched_checks.values())
        and anti_collapse["passed"]
        and body_lidar["passed"]
        and all(safety.values())
    )
    continuation_safety_passed = bool(
        finite
        and all(matched_checks.values())
        and anti_collapse["passed"]
        and all(safety.values())
    )
    return {
        "passed": passed,
        "finite": finite,
        "matched_motion": {
            "passed": all(matched_checks.values()),
            "checks": matched_checks,
            "moving_linear_speed_difference_mps": speed_difference,
            "realized_yaw_rate_difference_radps": yaw_difference,
            "linear_progress_ratio": linear_progress_ratio,
            "yaw_progress_ratio": yaw_progress_ratio,
            "stopped_fraction_excess": candidate_stopped - comparator_stopped,
        },
        "anti_collapse": anti_collapse,
        "body_lidar": body_lidar,
        "safety": safety,
        "continuation_safety_passed": continuation_safety_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/slam_confidence_joint_training_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/slam_confidence_joint_training_v1/post_training_motion_audit"),
    )
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=Path("/home/ros/IsaacLab/_isaac_sim/python.sh"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol_path = _project_path(args.protocol, must_exist=True)
    output_dir = _project_path(args.output_dir)
    python = args.isaac_python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_id"] == "slam-confidence-constrained-barrier-v4":
        tasks = CONSTRAINED_BARRIER_TASKS
    elif protocol["protocol_id"] in {
            "slam-confidence-causal-joint-training-v2",
            "slam-confidence-constrained-joint-training-v3",
        }:
        tasks = CAUSAL_TASKS
    else:
        tasks = TASKS
    cfg = protocol["post_training_motion_audit"]
    if cfg["status"] != "authorized_not_started":
        raise RuntimeError("motion audit is not in its one-shot authorized state")
    if not protocol["training_execution"]["completed"]:
        raise RuntimeError("motion audit requires completed fixed-budget training")
    if protocol["execution_gates"]["ppo_training_authorized"]:
        raise RuntimeError("PPO gate must remain closed during evaluation")
    checkpoints = _checkpoints(protocol)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_paths: dict[tuple[str, int, str], Path] = {}
    total = len(TASKS) * len(cfg["seeds"]) * len(cfg["profiles"])
    completed_count = 0
    for seed_value in cfg["seeds"]:
        seed = int(seed_value)
        for profile, command in cfg["profiles"].items():
            for arm in ("J0", "J1", "J2"):
                output = output_dir / f"{arm}_seed{seed}_{profile}.json"
                report_paths[(arm, seed, profile)] = output
                if not (args.resume and output.is_file()):
                    _run_one(
                        python=python,
                        arm=arm,
                        seed=seed,
                        profile=profile,
                        command=command,
                        checkpoint=checkpoints[(arm, seed)],
                        cfg=cfg,
                        output=output,
                        tasks=tasks,
                    )
                completed_count += 1
                print(f"[motion-audit] {completed_count}/{total} {arm} seed={seed} profile={profile}", flush=True)

    comparisons = []
    for seed_value in cfg["seeds"]:
        seed = int(seed_value)
        for profile in cfg["profiles"]:
            reports = {
                arm: json.loads(report_paths[(arm, seed, profile)].read_text(encoding="utf-8"))
                for arm in ("J0", "J1", "J2")
            }
            for candidate_arm, comparator_arm in (("J1", "J0"), ("J2", "J1")):
                comparisons.append(
                    {
                        "comparison": f"{candidate_arm}-{comparator_arm}",
                        "seed": seed,
                        "profile": profile,
                        **_compare(reports[candidate_arm], reports[comparator_arm], protocol),
                    }
                )

    required = int(cfg["minimum_passing_seeds_per_profile"])
    profile_support = []
    for comparison in ("J1-J0", "J2-J1"):
        for profile in cfg["profiles"]:
            rows = [
                row
                for row in comparisons
                if row["comparison"] == comparison and row["profile"] == profile
            ]
            passing = sum(bool(row["passed"]) for row in rows)
            profile_support.append(
                {
                    "comparison": comparison,
                    "profile": profile,
                    "passing_seeds": passing,
                    "required_passing_seeds": required,
                    "supported": passing >= required,
                }
            )
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "kind": "post_training_fixed_command_motion_audit",
        "not_slam_efficacy": True,
        "runs_completed": total,
        "comparisons": comparisons,
        "profile_support": profile_support,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"runs_completed": total, "profile_support": profile_support}, indent=2))
    print(f"[motion-audit] summary={summary_path}")


if __name__ == "__main__":
    main()
