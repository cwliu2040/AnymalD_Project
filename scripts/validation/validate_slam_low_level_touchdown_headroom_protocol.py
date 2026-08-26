#!/usr/bin/env python3
"""Validate the command-preserving touchdown-headroom protocol and schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/slam_low_level_touchdown_headroom_v1.yaml"
STAGE_NAMES = ("wiring_smoke", "pilot", "confirmation")
EXPECTED_BACKENDS = ("fastlio2", "liosam")
EXPECTED_ARMS = ("zero", "touchdown_soft_low", "touchdown_soft")
EXPECTED_PROFILES = (
    "mixed_curve_bounded", "pure_yaw_bounded", "lateral_translation_bounded",
)
FORBIDDEN_RUNTIME_INPUTS = {
    "ground_truth", "future_label", "backend_id", "intervention_id",
    "profile", "route_phase", "hardcoded_turn_timer", "contact_truth",
}


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


def build_baseline_schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = protocol["baseline_envelope"]
    rows = [
        {
            "stage": "baseline_envelope",
            "dataset_role": str(baseline["dataset_role"]),
            "profile": str(profile),
            "block_id": int(block_id),
            "simulation_seed": int(block_id),
            "arm": "zero",
        }
        for profile in baseline["profiles"]
        for block_id in baseline["block_ids"]
    ]
    if len(rows) != int(baseline["expected_run_count"]):
        raise ValueError("baseline envelope schedule count mismatch")
    return rows


def build_stage_schedule(protocol: dict[str, Any], stage_name: str) -> list[dict[str, Any]]:
    if stage_name not in STAGE_NAMES:
        raise ValueError(f"unknown stage: {stage_name}")
    stage = protocol["stages"][stage_name]
    rows: list[dict[str, Any]] = []
    for backend_index, backend in enumerate(stage["backends"]):
        for profile_index, profile in enumerate(stage["profiles"]):
            for block_index, block_id in enumerate(stage["block_ids"]):
                arms = list(stage["arms"])
                shift = (backend_index + profile_index + block_index) % len(arms)
                ordered = arms[shift:] + arms[:shift]
                for order, arm in enumerate(ordered):
                    rows.append({
                        "stage": stage_name,
                        "dataset_role": str(stage["dataset_role"]),
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
    failures: list[str] = []
    boundaries = protocol.get("boundaries", {})
    required_true = (
        "model1450_frozen",
        "original_requested_command_bit_exact_for_all_arms",
        "actual_slam_backends_required_for_effect_stages",
    )
    required_false = (
        "command_scaling_allowed",
        "previous_action_smoothing_family_reused",
        "fixed_joint_offset_family_allowed",
        "ground_truth_intervention_input_allowed",
        "contact_truth_intervention_input_allowed",
        "future_label_intervention_input_allowed",
        "backend_id_intervention_input_allowed",
        "profile_or_route_phase_intervention_input_allowed",
        "live_execution_authorized",
        "teacher_training_authorized",
        "adaptation_training_authorized",
        "ppo_training_authorized",
        "ros_policy_wiring_authorized",
        "physical_robot_authorized",
    )
    for key in required_true:
        if boundaries.get(key) is not True:
            failures.append(f"boundary must be true: {key}")
    for key in required_false:
        if boundaries.get(key) is not False:
            failures.append(f"boundary must be false: {key}")
    if boundaries.get("authorized_stages") != []:
        failures.append("no execution stage may be authorized")
    if float(boundaries.get("residual_linf_limit", 0.0)) != 0.05:
        failures.append("residual L-inf boundary must remain 0.05")

    mechanism = protocol.get("mechanism", {})
    if mechanism.get("family") == "previous_action_smoothing_direction":
        failures.append("retired previous-action smoothing family cannot be reused")
    phase_source = mechanism.get("phase_source", {})
    deployable = set(str(value) for value in phase_source.get("deployable_inputs_only", []))
    if deployable != {
        "joint_position_history", "joint_velocity_history", "previous_action_history",
    }:
        failures.append("phase estimator inputs must remain deployable joint/action history only")
    if deployable & FORBIDDEN_RUNTIME_INPUTS:
        failures.append("phase estimator contains a forbidden runtime input")
    if phase_source.get("simulator_contact_truth_role") != (
        "offline_phase_estimator_label_and_evaluation_only"
    ):
        failures.append("contact truth must remain offline-only")
    kinematics = mechanism.get("deployable_kinematics", {})
    if kinematics.get("canonical_joint_mapping_required") is not True:
        failures.append("deployable kinematics must preserve canonical joint mapping")
    if kinematics.get("foot_position_jacobian_source") != (
        "joint_position_history_and_frozen_robot_model"
    ):
        failures.append("foot Jacobian must derive from deployable joint history")
    if kinematics.get("foot_vertical_velocity_source") != (
        "joint_position_velocity_history_and_frozen_robot_model"
    ):
        failures.append("foot velocity must derive from deployable joint history")
    if kinematics.get("simulator_foot_velocity_runtime_input_allowed") is not False:
        failures.append("simulator foot velocity cannot enter the intervention runtime")
    solve = mechanism.get("solve", {})
    if solve.get("method") != "minimum_norm_damped_foot_xyz_jacobian_solve":
        failures.append("mechanism must use the frozen foot-Jacobian solve")
    if solve.get("target_horizontal_foot_displacement_m") != [0.0, 0.0]:
        failures.append("mechanism must target exact-zero first-order horizontal foot change")
    if float(solve.get("residual_linf_limit", 0.0)) != 0.05:
        failures.append("mechanism residual limit must remain 0.05")
    arms = mechanism.get("arms", {})
    if tuple(arms) != EXPECTED_ARMS:
        failures.append("arms must be ordered zero/low/primary touchdown shaping")
    fractions = {
        name: float(value.get("attenuation_fraction", -1.0)) for name, value in arms.items()
    }
    if fractions.get("zero") != 0.0:
        failures.append("zero arm must have exact-zero attenuation")
    if not 0.0 < fractions.get("touchdown_soft_low", -1.0) < fractions.get(
        "touchdown_soft", -1.0
    ) <= 1.0:
        failures.append("touchdown attenuation fractions must form zero < low < primary <= 1")

    coverage = protocol.get("command_coverage", {})
    if tuple(coverage.get("profiles", [])) != EXPECTED_PROFILES:
        failures.append("command coverage must retain mixed, pure-yaw, and lateral profiles")
    if not all(coverage.get(key) is True for key in (
        "free_rotation_required", "mixed_translation_yaw_required",
        "lateral_translation_required", "bounded_random_command_static_property_test_required",
        "hardcoded_turn_timer_forbidden_for_phase_estimation",
    )):
        failures.append("command-coverage requirements are incomplete")

    baseline = protocol.get("baseline_envelope", {})
    if baseline.get("dataset_role") != "anti_collapse_calibration_only":
        failures.append("baseline data must remain calibration-only")
    if baseline.get("slam_effect_claim_allowed") is not False:
        failures.append("baseline data cannot support a SLAM effect claim")
    if baseline.get("freeze_before_wiring_smoke") is not True:
        failures.append("anti-collapse envelope must freeze before effect stages")
    if baseline.get("bounds_rule", {}).get("candidate_results_may_change_bounds") is not False:
        failures.append("candidate outcomes cannot change the baseline envelope")
    bounds_rule = baseline.get("bounds_rule", {})
    if float(bounds_rule.get("within_run_lower_quantile", -1.0)) != 0.01:
        failures.append("within-run lower envelope quantile must remain 0.01")
    if float(bounds_rule.get("within_run_upper_quantile", -1.0)) != 0.99:
        failures.append("within-run upper envelope quantile must remain 0.99")
    if bounds_rule.get("profile_lower_bound") != "minimum_within_run_lower_quantile":
        failures.append("profile lower envelope must be hierarchical across complete runs")
    if bounds_rule.get("profile_upper_bound") != "maximum_within_run_upper_quantile":
        failures.append("profile upper envelope must be hierarchical across complete runs")
    if bounds_rule.get("each_complete_run_weighted_once") is not True:
        failures.append("baseline envelope must weight complete runs once")
    try:
        baseline_schedule = build_baseline_schedule(protocol)
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(str(exc))
        baseline_schedule = []

    disjointness = protocol.get("disjointness", {})
    reserved = set(int(value) for value in disjointness.get("reserved_block_ids", []))
    if reserved != {581, 582, 583, 584}:
        failures.append("blocks 581 through 584 must remain explicitly reserved")
    minimum = int(disjointness.get("minimum_new_block_id", 0))
    schedules: dict[str, list[dict[str, Any]]] = {}
    used = {int(row["block_id"]) for row in baseline_schedule}
    if any(block < minimum or block in reserved for block in used):
        failures.append("baseline uses a retired or reserved block")
    for stage_name in STAGE_NAMES:
        stage = protocol.get("stages", {}).get(stage_name, {})
        if tuple(stage.get("backends", [])) != EXPECTED_BACKENDS:
            failures.append(f"{stage_name} must report FAST and LIO separately")
        if tuple(stage.get("arms", [])) != EXPECTED_ARMS:
            failures.append(f"{stage_name} arm order mismatch")
        if stage_name != "wiring_smoke" and tuple(stage.get("profiles", [])) != EXPECTED_PROFILES:
            failures.append(f"{stage_name} must retain all prespecified command profiles")
        blocks = set(int(value) for value in stage.get("block_ids", []))
        if any(block < minimum or block in reserved for block in blocks):
            failures.append(f"{stage_name} uses a retired or reserved block")
        if blocks & used:
            failures.append(f"{stage_name} reuses a baseline or effect-stage block")
        used |= blocks
        try:
            schedules[stage_name] = build_stage_schedule(protocol, stage_name)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(str(exc))
            schedules[stage_name] = []

    outcomes = protocol.get("outcomes", {})
    if float(outcomes.get("future_slam_horizon_s", 0.0)) != 0.5:
        failures.append("future SLAM horizon must remain 0.5 s")
    if outcomes.get("realized_speed_role") != (
        "whole_matched_group_equivalence_gate_not_posthoc_run_filter"
    ):
        failures.append("realized speed cannot be used as a post-hoc run filter")
    if outcomes.get("pooled_backend_rescue_forbidden") is not True:
        failures.append("pooled backend rescue must remain forbidden")

    gate = protocol.get("decision_gate", {})
    motion = gate.get("matched_motion_all_required", {})
    if float(motion.get("maximum_absolute_moving_linear_speed_difference_mps", 9.0)) != 0.05:
        failures.append("matched linear-speed guard must remain 0.05 m/s")
    if float(motion.get("minimum_progress_ratio", 0.0)) != 0.98:
        failures.append("matched progress ratio must remain 0.98")
    if gate.get("confirmation_pass_next_step") != (
        "privileged_teacher_then_history_adaptation_module"
    ):
        failures.append("confirmation may unlock only teacher then adaptation training")
    prerequisites = protocol.get("stage_prerequisites", {})
    if "baseline_envelope_artifact_frozen" not in prerequisites.get("wiring_smoke", []):
        failures.append("wiring smoke must require a frozen baseline envelope")
    if "phase_estimator_artifact_frozen" not in prerequisites.get("wiring_smoke", []):
        failures.append("wiring smoke must require a frozen phase estimator")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "baseline_count": len(baseline_schedule),
        "stage_counts": {name: len(rows) for name, rows in schedules.items()},
        "baseline_schedule": baseline_schedule,
        "schedules": schedules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.expanduser().resolve()
    report = {
        "schema_version": 1,
        "kind": "slam_low_level_touchdown_headroom_protocol_validation",
        "protocol_id": load_protocol(protocol_path).get("protocol_id"),
        "protocol_sha256": _sha256(protocol_path),
        **validate_protocol(load_protocol(protocol_path)),
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
