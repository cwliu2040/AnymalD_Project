#!/usr/bin/env python3
"""Validate and render the matched-prefix command-pulse pilot schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/slam_speed_pulse_causal_pilot_v2.yaml"
STAGES = ("wiring_smoke", "causal_pilot")
BACKENDS = ("fastlio2", "liosam")
ARMS = ("scale_100", "scale_075", "scale_050", "scale_025")
SCALES = (1.0, 0.75, 0.5, 0.25)
COMPONENT_ARMS = ("control", "uniform_075", "preserve_yaw", "preserve_translation")
COMPONENT_SCALES = (
    (1.0, 1.0, 1.0),
    (0.75, 0.75, 0.75),
    (0.75, 0.75, 1.0),
    (1.0, 1.0, 0.75),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_schedule(protocol: dict[str, Any], stage_name: str) -> list[dict[str, Any]]:
    stage = protocol["stages"][stage_name]
    rows = []
    for backend_index, backend in enumerate(stage["backends"]):
        for profile_index, profile in enumerate(stage["profiles"]):
            for block_index, block in enumerate(stage["block_ids"]):
                arms = list(stage["arms"])
                shift = (backend_index + profile_index + block_index) % len(arms)
                for order, arm in enumerate(arms[shift:] + arms[:shift]):
                    assigned = protocol["treatment"]["arms"][arm]
                    row = {
                        "stage": stage_name, "backend": backend, "profile": profile,
                        "block_id": int(block), "simulation_seed": int(block),
                        "arm": arm, "arm_order": order,
                    }
                    if isinstance(assigned, list):
                        row["assigned_scales"] = [float(value) for value in assigned]
                    else:
                        row["assigned_scale"] = float(assigned)
                    rows.append(row)
    if len(rows) != int(stage["expected_run_count"]):
        raise ValueError(f"{stage_name} schedule count mismatch")
    return rows


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    failures = []
    boundaries = protocol.get("boundaries", {})
    for key in ("actual_slam_backend_required", "ros2_bridge_required", "model1450_plus_arm_b_frozen", "whole_episode_scale_forbidden"):
        if boundaries.get(key) is not True:
            failures.append(f"boundary must be true: {key}")
    for key in ("synthetic_confidence_schedule_allowed", "ground_truth_runtime_input", "future_label_runtime_input", "policy_training_allowed", "ppo_training_allowed", "physical_robot_use_authorized", "live_execution_authorized"):
        if boundaries.get(key) is not False:
            failures.append(f"boundary must be false: {key}")
    treatment = protocol.get("treatment", {})
    arm_names = tuple(treatment.get("arms", {}))
    if arm_names == ARMS:
        if tuple(float(v) for v in treatment["arms"].values()) != SCALES:
            failures.append("scalar pulse arms must be exactly 1.0/0.75/0.5/0.25")
    elif arm_names == COMPONENT_ARMS:
        vectors = tuple(tuple(float(x) for x in v) for v in treatment["arms"].values())
        if vectors != COMPONENT_SCALES:
            failures.append("component pulse arms do not match the frozen XYZ vectors")
    else:
        failures.append("unsupported pulse arm set")
    if treatment.get("untreated_prefix_required") is not True or treatment.get("post_pulse_recovery_required") is not True:
        failures.append("untreated prefix and recovery are mandatory")
    design = protocol.get("experimental_design", {})
    if design.get("type") != "randomized_balanced_repeated_run":
        failures.append("effect pilot must use randomized balanced repeated runs")
    if design.get("exact_prestate_matching_required") is not False:
        failures.append("independent runs must not claim exact prestate matching")
    if design.get("post_treatment_outcome_based_exclusion_forbidden") is not True:
        failures.append("post-treatment outcome exclusion must remain forbidden")
    start = float(treatment.get("pulse_start_relative_to_profile_s", -1))
    duration = float(treatment.get("pulse_duration_s", -1))
    horizon = float(treatment.get("horizon_s", -1))
    if start < float(protocol.get("common", {}).get("command_warmup_s", 1e9)):
        failures.append("pulse must start after command warmup")
    if duration <= 0.0 or duration < horizon or horizon != 0.5:
        failures.append("pulse must cover the frozen 0.5 s horizon")
    if float(protocol.get("common", {}).get("point_density_timeline_s", {}).get("healthy", -1.0)) != start:
        failures.append("pulse must start at sensor-degradation onset")
    if tuple(protocol.get("trace_contract", {}).get("required_phases", [])) != ("untreated_prefix", "pulse", "recovered"):
        failures.append("trace must verify prefix, pulse, and recovery")
    trace_contract = protocol.get("trace_contract", {})
    if float(trace_contract.get("pre_pulse_sample_max_age_s", -1.0)) != 0.12:
        failures.append("pre-pulse maximum sample age must remain 0.12 s")
    if float(trace_contract.get("pre_pulse_max_pairwise_time_skew_s", -1.0)) != 0.04:
        failures.append("pre-pulse maximum pairwise time skew must remain 0.04 s")
    linf = trace_contract.get("pre_pulse_max_pairwise_linf", {})
    rms = trace_contract.get("pre_pulse_max_pairwise_rms", {})
    if float(linf.get("joint_velocity", -1.0)) != 3.25:
        failures.append("joint-velocity L-infinity hard bound must remain 3.25 rad/s")
    if float(rms.get("joint_velocity", -1.0)) != 1.25:
        failures.append("joint-velocity RMS matching bound must remain 1.25 rad/s")
    calibration = trace_contract.get("composite_matching_calibration", {})
    if calibration.get("dataset_role") != "excluded_wiring_development_pre_treatment_only":
        failures.append("composite matching must use excluded pre-treatment development data")
    if calibration.get("source_attempts") != [2, 3, 4] or calibration.get("post_pulse_outcomes_used") is not False:
        failures.append("composite matching calibration provenance must remain frozen")
    if calibration.get("backends") != ["fastlio2", "liosam"]:
        failures.append("composite calibration must cover both SLAM backends")
    if int(calibration.get("pairwise_comparisons", 0)) != 24:
        failures.append("composite calibration must retain 24 pre-treatment comparisons")
    if float(linf.get("base_angular_velocity", -1.0)) != 0.25:
        failures.append("base-angular-velocity matching bound must remain 0.25 rad/s")
    if float(linf.get("previous_action", -1.0)) != 0.45:
        failures.append("previous-action matching bound must remain 0.45")
    if calibration.get("thresholds_frozen_before_attempt5") is not True:
        failures.append("composite thresholds must be frozen before attempt5")
    decision_gate = protocol.get("decision_gate", {})
    if arm_names == COMPONENT_ARMS:
        comparison = decision_gate.get("component_specific_comparison", {})
        if comparison.get("reference_arm") != "uniform_075":
            failures.append("component arms must compare against uniform_075")
        if tuple(comparison.get("candidate_arms", [])) != COMPONENT_ARMS[2:]:
            failures.append("component candidate arms must be frozen")
        if comparison.get("primary_improvement_backend") != "liosam":
            failures.append("LIO-SAM must remain the component improvement backend")
        if tuple(comparison.get("prespecified_profiles", [])) != (
            "curve_1_5_right_1_0", "curve_1_5_left_1_0",
        ):
            failures.append("component comparison profiles must be frozen curves")
        if int(comparison.get("minimum_supported_candidates", 0)) != 1:
            failures.append("at least one component candidate must be required")
        if comparison.get("fastlio2_role") != "noninferiority_safeguard":
            failures.append("FAST-LIO2 must remain a component safeguard")
    else:
        headroom = decision_gate.get("ceiling_aware_headroom", {})
        if headroom.get("primary_improvement_backend") != "liosam":
            failures.append("LIO-SAM must remain the prespecified improvement backend")
        if tuple(headroom.get("prespecified_improvement_profiles", [])) != (
            "curve_1_5_right_1_0", "lateral_right_1_5",
        ):
            failures.append("LIO-SAM improvement profiles must be frozen before collection")
        if int(headroom.get("minimum_supported_primary_backend_strata", 0)) != 1:
            failures.append("exactly one or more prespecified LIO-SAM strata must show headroom")
        if headroom.get("fastlio2_role") != "noninferiority_or_ceiling_safeguard":
            failures.append("FAST-LIO2 must remain a safeguard rather than forced superiority target")
    gates = protocol.get("implementation_gates", {})
    if gates.get("pulse_core_complete") is not True or gates.get("benchmark_driver_wiring_complete") is not True:
        failures.append("pulse core and driver wiring must be complete")
    for key in ("trace_and_analysis_complete", "artifact_hash_lock_complete"):
        if gates.get(key) is not True:
            failures.append(f"completed implementation gate must be true: {key}")
    if any(gates.get(key) is not False for key in ("wiring_smoke_authorized", "causal_pilot_authorized", "action_conditioned_model_training_authorized")):
        failures.append("execution/training gates must remain false")
    forbidden = set()
    for low, high in protocol.get("disjointness", {}).get("forbidden_block_ranges", []):
        forbidden.update(range(int(low), int(high) + 1))
    schedules = {}
    used = set()
    for stage_name in STAGES:
        stage = protocol["stages"][stage_name]
        if tuple(stage["backends"]) != BACKENDS or tuple(stage["arms"]) != arm_names:
            failures.append(f"{stage_name} backend/arm contract mismatch")
        blocks = {int(v) for v in stage["block_ids"]}
        if blocks & forbidden or blocks & used:
            failures.append(f"{stage_name} block IDs are not fresh")
        used |= blocks
        schedules[stage_name] = build_schedule(protocol, stage_name)
    expected_blocks = protocol.get("frozen_fresh_blocks", {})
    if expected_blocks:
        if protocol["stages"]["wiring_smoke"]["block_ids"] != expected_blocks.get("wiring_smoke"):
            failures.append("wiring smoke blocks do not match frozen fresh blocks")
        if protocol["stages"]["causal_pilot"]["block_ids"] != expected_blocks.get("causal_pilot"):
            failures.append("causal pilot blocks do not match frozen fresh blocks")
    return {"passed": not failures, "failures": sorted(set(failures)), "stage_counts": {k: len(v) for k, v in schedules.items()}, "schedules": schedules}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    path = args.protocol.expanduser().resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
        raise ValueError("protocol must be an existing project-local file")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    report = {"schema_version": 1, "kind": "slam_speed_pulse_protocol_validation", "protocol_sha256": _sha256(path), **validate_protocol(protocol)}
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
