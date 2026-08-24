#!/usr/bin/env python3
"""Validate and render the plan-only speed-scale causal pilot schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/slam_speed_scale_causal_pilot_v1.yaml"
STAGES = ("wiring_smoke", "causal_pilot")
BACKENDS = ("fastlio2", "liosam")
ARMS = ("scale_100", "scale_075", "scale_050", "scale_025")
SCALES = (1.0, 0.75, 0.5, 0.25)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_schedule(protocol: dict[str, Any], stage_name: str) -> list[dict[str, Any]]:
    if stage_name not in STAGES:
        raise ValueError(f"unknown stage: {stage_name}")
    stage = protocol["stages"][stage_name]
    rows = []
    for backend_index, backend in enumerate(stage["backends"]):
        for profile_index, profile in enumerate(stage["profiles"]):
            for block_index, block_id in enumerate(stage["block_ids"]):
                arms = list(stage["arms"])
                shift = (backend_index + profile_index + block_index) % len(arms)
                arms = arms[shift:] + arms[:shift]
                for order, arm in enumerate(arms):
                    rows.append({
                        "stage": stage_name,
                        "backend": backend,
                        "profile": profile,
                        "block_id": int(block_id),
                        "simulation_seed": int(block_id),
                        "arm": arm,
                        "assigned_scale": float(protocol["treatment"]["arms"][arm]),
                        "arm_order": order,
                    })
    if len(rows) != int(stage["expected_run_count"]):
        raise ValueError(f"{stage_name} schedule count mismatch")
    return rows


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    failures = []
    boundaries = protocol.get("boundaries", {})
    for key in (
        "actual_slam_backend_required", "ros2_bridge_required", "model1450_frozen",
    ):
        if boundaries.get(key) is not True:
            failures.append(f"boundary must be true: {key}")
    for key in (
        "synthetic_confidence_schedule_allowed", "ground_truth_runtime_input",
        "future_label_runtime_input", "backend_id_runtime_model_input",
        "confidence_estimator_refit_allowed", "policy_training_allowed",
        "ppo_training_allowed", "physical_robot_use_authorized",
        "live_execution_authorized",
    ):
        if boundaries.get(key) is not False:
            failures.append(f"boundary must be false: {key}")
    treatment = protocol.get("treatment", {})
    if tuple(treatment.get("arms", {})) != ARMS:
        failures.append("speed arms must retain frozen descending order")
    elif tuple(float(value) for value in treatment["arms"].values()) != SCALES:
        failures.append("speed scales must be exactly 1.0/0.75/0.5/0.25")
    if treatment.get("formula_invalid_or_stale") != "effective_command_xyz = exact_zero":
        failures.append("invalid/stale command must be exact zero")
    if treatment.get("realized_speed_is_intended_mediator_not_a_confound") is not True:
        failures.append("speed must be declared as the intended mediator")
    if float(protocol.get("outcomes", {}).get("horizon_s", 0.0)) != 0.5:
        failures.append("risk horizon must remain 0.5 s")
    gates = protocol.get("implementation_gates", {})
    for key in ("fixed_scale_runtime_wiring_complete", "artifact_hash_lock_complete"):
        if gates.get(key) is not True:
            failures.append(f"phase3b implementation gate must be true: {key}")
    for key in (
        "wiring_smoke_authorized", "causal_pilot_authorized",
        "action_conditioned_model_training_authorized",
    ):
        if gates.get(key) is not False:
            failures.append(f"execution/training gate must remain false: {key}")

    forbidden = set()
    for bounds in protocol.get("disjointness", {}).get("forbidden_block_ranges", []):
        if len(bounds) != 2 or int(bounds[0]) > int(bounds[1]):
            failures.append("invalid forbidden block range")
            continue
        forbidden.update(range(int(bounds[0]), int(bounds[1]) + 1))
    schedules = {}
    used = set()
    for stage_name in STAGES:
        stage = protocol.get("stages", {}).get(stage_name, {})
        if tuple(stage.get("backends", [])) != BACKENDS:
            failures.append(f"{stage_name} must report FAST and LIO separately")
        if tuple(stage.get("arms", [])) != ARMS:
            failures.append(f"{stage_name} arm order mismatch")
        blocks = {int(value) for value in stage.get("block_ids", [])}
        if blocks & forbidden:
            failures.append(f"{stage_name} reuses a forbidden block")
        if blocks & used:
            failures.append(f"{stage_name} reuses another speed stage block")
        used |= blocks
        try:
            schedules[stage_name] = build_schedule(protocol, stage_name)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(str(exc))
            schedules[stage_name] = []
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "stage_counts": {name: len(rows) for name, rows in schedules.items()},
        "schedules": schedules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.protocol.expanduser().resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
        raise ValueError("protocol must be an existing project-local file")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "kind": "slam_speed_scale_causal_protocol_validation",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": _sha256(path),
        **validate_protocol(protocol),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if not output.is_relative_to(PROJECT_ROOT):
            raise ValueError("output must remain inside the project")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
