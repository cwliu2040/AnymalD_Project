#!/usr/bin/env python3
"""Validate and render the frozen action-risk intervention pilot schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/slam_action_risk_intervention_pilot.yaml"
STAGE_NAMES = ("wiring_smoke", "pilot", "expanded_only_after_pilot_inconclusive")
EXPECTED_BACKENDS = ("fastlio2", "liosam")
EXPECTED_ARMS = ("smooth", "zero", "antismooth")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT) or not resolved.is_file():
        raise ValueError("protocol must be an existing project-local file")
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("protocol root must be a mapping")
    return value


def build_stage_schedule(protocol: dict[str, Any], stage_name: str) -> list[dict[str, Any]]:
    if stage_name not in STAGE_NAMES:
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
                        "backend": str(backend),
                        "profile": str(profile),
                        "block_id": int(block_id),
                        "simulation_seed": int(block_id),
                        "arm": str(arm),
                        "arm_order": order,
                    })
    if len(rows) != int(stage["expected_run_count"]):
        raise ValueError(f"{stage_name} schedule count mismatch")
    return rows


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    failures = []
    boundaries = protocol.get("boundaries", {})
    required_true = (
        "actual_slam_backend_required", "ros2_bridge_required",
        "model1450_frozen",
    )
    required_false = (
        "synthetic_confidence_schedule_allowed", "ground_truth_runtime_input",
        "future_label_runtime_input", "intervention_id_runtime_policy_input",
        "backend_id_runtime_policy_input", "policy_training_allowed",
        "ppo_training_allowed",
    )
    for key in required_true:
        if boundaries.get(key) is not True:
            failures.append(f"boundary must be true: {key}")
    for key in required_false:
        if boundaries.get(key) is not False:
            failures.append(f"boundary must be false: {key}")
    intervention = protocol.get("intervention", {})
    arms = intervention.get("arms", {})
    if tuple(arms) != EXPECTED_ARMS:
        failures.append("intervention arms must be ordered smooth/zero/antismooth")
    coefficients = {name: float(value.get("alpha", 999.0)) for name, value in arms.items()}
    if coefficients.get("zero") != 0.0:
        failures.append("zero arm alpha must be exact zero")
    if coefficients.get("smooth", 0.0) <= 0.0:
        failures.append("smooth alpha must be positive")
    if coefficients.get("antismooth") != -coefficients.get("smooth", 1.0):
        failures.append("antismooth alpha must be the signed smooth counterpart")
    if float(intervention.get("raw_action_linf_limit", 0.0)) <= 0.0:
        failures.append("raw action residual limit must be positive")
    if not intervention.get("apply_only_while_tracking_valid", False):
        failures.append("intervention must be disabled when tracking is invalid")
    common = protocol.get("common", {})
    timeline = common.get("point_density_timeline_s", {})
    active = common.get("active_healthy_window_s", [])
    if len(active) != 2 or float(active[0]) < float(common.get("command_warmup_s", 0.0)):
        failures.append("active healthy window must begin after command warmup")
    if len(active) == 2 and float(active[1]) > float(timeline.get("healthy", 0.0)):
        failures.append("active healthy window must end before degradation")
    if float(protocol.get("outcomes", {}).get("horizon_s", 0.0)) != 0.5:
        failures.append("localization hazard horizon must remain 0.5 s")
    gate = protocol.get("decision_gate", {})
    if int(gate.get("minimum_ordered_backend_profile_strata", 0)) != 3:
        failures.append("pilot must require three of four ordered backend/profile strata")
    if float(gate.get("no_action_rate_separation_absolute_tolerance_per_s", 0.0)) <= 0.0:
        failures.append("action-rate separation tolerance must be positive")
    forbidden = set(int(value) for value in protocol.get("disjointness", {}).get("forbidden_block_ids", []))
    schedules = {}
    used = set()
    for stage_name in STAGE_NAMES:
        stage = protocol.get("stages", {}).get(stage_name, {})
        if tuple(stage.get("backends", [])) != EXPECTED_BACKENDS:
            failures.append(f"{stage_name} must include FAST and LIO separately")
        if tuple(stage.get("arms", [])) != EXPECTED_ARMS:
            failures.append(f"{stage_name} arm order mismatch")
        blocks = set(int(value) for value in stage.get("block_ids", []))
        if blocks & forbidden:
            failures.append(f"{stage_name} reuses a forbidden block")
        if blocks & used:
            failures.append(f"{stage_name} reuses another intervention stage block")
        used |= blocks
        try:
            schedules[stage_name] = build_stage_schedule(protocol, stage_name)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(str(exc))
            schedules[stage_name] = []
    return {
        "passed": not failures,
        "failures": failures,
        "stage_counts": {name: len(rows) for name, rows in schedules.items()},
        "schedules": schedules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.expanduser().resolve()
    protocol = load_protocol(protocol_path)
    report = {
        "schema_version": 1,
        "kind": "slam_action_risk_intervention_protocol_validation",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": _sha256(protocol_path),
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
