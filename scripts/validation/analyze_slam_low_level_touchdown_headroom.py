#!/usr/bin/env python3
"""Apply the frozen matched-speed touchdown-headroom decision gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "configs/slam_low_level_touchdown_headroom_v1.yaml"
VALIDATOR_PATH = (
    PROJECT_ROOT / "scripts/validation/validate_slam_low_level_touchdown_headroom_protocol.py"
)
ARMS = ("zero", "touchdown_soft_low", "touchdown_soft")
REQUIRED_METRICS = (
    "moving_linear_speed_mps",
    "realized_yaw_rate_radps",
    "normalized_progress",
    "stopped_fraction",
    "touchdown_vertical_speed_abs_mps",
    "touchdown_contact_impulse_ns",
    "touchdown_roll_pitch_rate_impulse_rad",
    "lidar_scan_time_rotation_rad",
    "valid_requested_usable_next_horizon_failure_fraction",
    "tracking_restricted_mean_survival_time_s",
)
ADDITIONAL_MECHANISM_METRICS = (
    "touchdown_contact_impulse_ns",
    "touchdown_roll_pitch_rate_impulse_rad",
    "lidar_scan_time_rotation_rad",
)
COLLAPSE_CHECKS = (
    "stop_envelope_pass", "shuffle_envelope_pass",
    "posture_envelope_pass", "limiter_equivalence_pass",
)
TRACE_CHECKS = (
    "requested_command_exact_across_arms",
    "model1450_observation_command_exact_original",
    "invalid_or_stale_exact_hard_stop",
    "residual_linf_bound",
    "residual_only_in_eligible_late_swing",
    "horizontal_foot_correction_first_order",
)


def _validator_module():
    spec = importlib.util.spec_from_file_location("touchdown_protocol", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load touchdown protocol validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(record: dict[str, Any]) -> tuple[str, str, int, str]:
    identity = record["identity"]
    return (
        str(identity["backend"]), str(identity["profile"]),
        int(identity["block_id"]), str(identity["arm"]),
    )


def _expected(protocol: dict[str, Any], stage: str) -> set[tuple[str, str, int, str]]:
    return {
        (row["backend"], row["profile"], row["block_id"], row["arm"])
        for row in _validator_module().build_stage_schedule(protocol, stage)
    }


def _finite_metric(record: dict[str, Any], name: str) -> float:
    value = float(record.get("metrics", {}).get(name, math.nan))
    if not math.isfinite(value):
        raise ValueError(f"metric must be finite: {name}")
    return value


def _mean(records: Iterable[dict[str, Any]], metric: str) -> float:
    values = [_finite_metric(record, metric) for record in records]
    if not values:
        raise ValueError(f"cannot average empty metric: {metric}")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def load_records(
    root: Path, pattern: str = "touchdown_headroom_run_record.json",
) -> list[dict[str, Any]]:
    if Path(pattern).name != pattern:
        raise ValueError("record pattern must be a basename")
    paths = sorted(root.glob(f"**/{pattern}"))
    values = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("run records must be mappings")
    return values


def analyze_records(
    records: list[dict[str, Any]], protocol: dict[str, Any], stage: str,
) -> dict[str, Any]:
    if stage not in {"wiring_smoke", "pilot", "confirmation"}:
        raise ValueError("stage must be wiring_smoke, pilot, or confirmation")
    expected = _expected(protocol, stage)
    identities = [_identity(record) for record in records]
    observed = set(identities)
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    failures: list[str] = []
    if duplicates:
        failures.append("duplicate_run_identities")
    if expected - observed:
        failures.append("missing_run_identities")
    if observed - expected:
        failures.append("unexpected_run_identities")

    stage_role = str(protocol["stages"][stage]["dataset_role"])
    by_block: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        backend, profile, block, arm = _identity(record)
        if arm not in ARMS:
            failures.append("unsupported_arm")
            continue
        if record.get("dataset_role") != stage_role:
            failures.append("incorrect_dataset_role")
        if record.get("identity", {}).get("stage") != stage:
            failures.append("stage_mismatch")
        if record.get("gate", {}).get("passed") is not True:
            failures.append("run_data_integrity_failure")
        checks = record.get("trace", {}).get("checks", {})
        for check in TRACE_CHECKS:
            if checks.get(check) is not True:
                failures.append(f"trace_check_failure:{check}")
        if arm == "zero" and checks.get("zero_arm_bit_exact_model1450") is not True:
            failures.append("zero_arm_not_exact_model1450")
        maximum_residual = float(record.get("trace", {}).get("maximum_residual_linf", math.nan))
        if not math.isfinite(maximum_residual):
            failures.append("nonfinite_residual_linf")
        elif maximum_residual > float(protocol["boundaries"]["residual_linf_limit"]) + 1.0e-7:
            failures.append("residual_linf_violation")
        for metric in REQUIRED_METRICS:
            try:
                _finite_metric(record, metric)
            except (TypeError, ValueError):
                failures.append(f"missing_or_nonfinite_metric:{metric}")
        collapse = record.get("anti_collapse", {})
        for check in COLLAPSE_CHECKS:
            if not isinstance(collapse.get(check), bool):
                failures.append(f"missing_anti_collapse_result:{check}")
        by_block[(backend, profile, block)][arm] = record

    for arms in by_block.values():
        if tuple(sorted(arms)) != tuple(sorted(ARMS)):
            failures.append("incomplete_matched_arms")
    failures = sorted(set(failures))
    integrity_passed = not failures and observed == expected
    inventory = {
        "expected_run_count": len(expected),
        "observed_run_count": len(records),
        "missing_identities": sorted(expected - observed),
        "unexpected_identities": sorted(observed - expected),
        "duplicate_identities": duplicates,
    }
    if stage == "wiring_smoke":
        passed = integrity_passed
        return {
            "stage": stage,
            "inventory": inventory,
            "integrity": {"passed": passed, "failures": failures},
            "decision": {
                "status": "WIRING_PASS" if passed else "WIRING_FAIL",
                "claim_allowed": False,
                "next_step": (
                    "pilot_requires_separate_authorization" if passed else "repair_wiring"
                ),
            },
        }
    if not integrity_passed:
        return {
            "stage": stage,
            "inventory": inventory,
            "integrity": {"passed": False, "failures": failures},
            "decision": {
                "status": "FAIL",
                "claim_allowed": False,
                "next_step": "repair_or_stop_without_effect_claim",
            },
        }

    gate = protocol["decision_gate"]
    motion_gate = gate["matched_motion_all_required"]
    minimum_blocks = int(gate["minimum_distinct_improved_blocks_per_supported_stratum"])
    safety_minimum = int(
        gate["safety"]["repeated_treatment_specific_excess_distinct_blocks_same_stratum"]
    )
    block_reports: list[dict[str, Any]] = []
    safety_excess: dict[str, list[int]] = defaultdict(list)
    for (backend, profile, block), arms in sorted(by_block.items()):
        zero = arms["zero"]
        low = arms["touchdown_soft_low"]
        primary = arms["touchdown_soft"]
        differences = {
            metric: _finite_metric(primary, metric) - _finite_metric(zero, metric)
            for metric in REQUIRED_METRICS
        }
        progress_denominator = _finite_metric(zero, "normalized_progress")
        progress_ratio = (
            _finite_metric(primary, "normalized_progress") / progress_denominator
            if progress_denominator > 0.0 else 0.0
        )
        matched_motion = bool(
            abs(differences["moving_linear_speed_mps"])
            <= float(motion_gate["maximum_absolute_moving_linear_speed_difference_mps"])
            and abs(differences["realized_yaw_rate_radps"])
            <= float(motion_gate["maximum_absolute_realized_yaw_rate_difference_radps"])
            and progress_ratio >= float(motion_gate["minimum_progress_ratio"])
            and differences["stopped_fraction"]
            <= float(motion_gate["maximum_stopped_fraction_excess"])
        )
        touchdown_improved = differences["touchdown_vertical_speed_abs_mps"] <= 0.0
        additional_improved = sum(
            int(differences[metric] <= 0.0) for metric in ADDITIONAL_MECHANISM_METRICS
        )
        low_dose_consistent = bool(
            _finite_metric(primary, "touchdown_vertical_speed_abs_mps")
            <= _finite_metric(low, "touchdown_vertical_speed_abs_mps")
            <= _finite_metric(zero, "touchdown_vertical_speed_abs_mps")
        )
        mechanism_passed = bool(
            touchdown_improved
            and additional_improved
            >= int(gate["mechanism_all_required"][
                "minimum_additional_improved_body_or_lidar_endpoints"
            ])
            and low_dose_consistent
        )
        slam_passed = bool(
            differences["valid_requested_usable_next_horizon_failure_fraction"] <= 0.0
            and differences["tracking_restricted_mean_survival_time_s"] >= 0.0
        )
        anti_collapse_passed = all(
            primary.get("anti_collapse", {}).get(check) is True
            for check in COLLAPSE_CHECKS
        )
        zero_event = bool(zero.get("metrics", {}).get("fall")) or bool(
            zero.get("metrics", {}).get("base_contact")
        )
        primary_event = bool(primary.get("metrics", {}).get("fall")) or bool(
            primary.get("metrics", {}).get("base_contact")
        )
        if primary_event and not zero_event:
            safety_excess[f"{backend}/{profile}"].append(block)
        supported = bool(
            matched_motion and mechanism_passed and slam_passed
            and anti_collapse_passed and not primary_event
        )
        block_reports.append({
            "backend": backend,
            "profile": profile,
            "block_id": block,
            "matched_motion": matched_motion,
            "mechanism_passed": mechanism_passed,
            "slam_direction_passed": slam_passed,
            "anti_collapse_passed": anti_collapse_passed,
            "low_dose_consistent": low_dose_consistent,
            "additional_improved_mechanism_endpoints": additional_improved,
            "progress_ratio": progress_ratio,
            "primary_minus_zero": differences,
            "treatment_specific_safety_excess": primary_event and not zero_event,
            "supported": supported,
        })

    stratum_reports = []
    for backend in ("fastlio2", "liosam"):
        for profile in protocol["stages"][stage]["profiles"]:
            rows = [
                row for row in block_reports
                if row["backend"] == backend and row["profile"] == profile
            ]
            improved_blocks = [int(row["block_id"]) for row in rows if row["supported"]]
            primary_records = [
                by_block[(backend, profile, int(row["block_id"]))]["touchdown_soft"]
                for row in rows
            ]
            zero_records = [
                by_block[(backend, profile, int(row["block_id"]))]["zero"]
                for row in rows
            ]
            mean_differences = {
                metric: _mean(primary_records, metric) - _mean(zero_records, metric)
                for metric in REQUIRED_METRICS
            }
            supported = bool(
                len(improved_blocks) >= minimum_blocks
                and mean_differences[
                    "valid_requested_usable_next_horizon_failure_fraction"
                ] <= 0.0
                and mean_differences["tracking_restricted_mean_survival_time_s"] >= 0.0
            )
            stratum_reports.append({
                "backend": backend,
                "profile": profile,
                "supported": supported,
                "improved_blocks": improved_blocks,
                "required_improved_blocks": minimum_blocks,
                "mean_primary_minus_zero": mean_differences,
            })

    repeated_safety_harm = any(
        len(set(blocks)) >= safety_minimum for blocks in safety_excess.values()
    )
    any_treatment_specific_safety_excess = any(safety_excess.values())
    any_treatment_anti_collapse_failure = any(
        record["identity"]["arm"] != "zero"
        and not all(
            record.get("anti_collapse", {}).get(check) is True
            for check in COLLAPSE_CHECKS
        )
        for record in records
    )
    supported_strata = [
        f"{row['backend']}/{row['profile']}" for row in stratum_reports if row["supported"]
    ]
    enough_strata = len(supported_strata) >= int(
        gate["minimum_supported_backend_profile_strata"]
    )
    if repeated_safety_harm:
        status = "FAIL"
    elif any_treatment_specific_safety_excess:
        status = str(gate["safety"]["single_treatment_specific_excess_disposition"])
    elif any_treatment_anti_collapse_failure:
        status = "INCONCLUSIVE"
    elif enough_strata:
        status = "PILOT_PASS" if stage == "pilot" else "HEADROOM_PASS"
    else:
        status = str(gate["otherwise"])
    return {
        "stage": stage,
        "inventory": inventory,
        "integrity": {"passed": True, "failures": []},
        "block_reports": block_reports,
        "strata": stratum_reports,
        "simulation_safety": {
            "treatment_specific_excess_blocks": {
                key: sorted(set(value)) for key, value in sorted(safety_excess.items())
            },
            "repeated_treatment_specific_harm": repeated_safety_harm,
            "any_treatment_specific_excess": any_treatment_specific_safety_excess,
        },
        "anti_collapse": {
            "all_treatment_records_passed": not any_treatment_anti_collapse_failure,
        },
        "decision": {
            "status": status,
            "claim_allowed": status == "HEADROOM_PASS",
            "supported_strata": supported_strata,
            "pooled_backend_rescue_used": False,
            "next_step": (
                gate["pilot_pass_next_step"] if status == "PILOT_PASS"
                else gate["confirmation_pass_next_step"] if status == "HEADROOM_PASS"
                else gate["fail_next_step"] if status == "FAIL"
                else "review_without_parameter_tuning_or_claim"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--stage", choices=("wiring_smoke", "pilot", "confirmation"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    report = analyze_records(load_records(args.records_root), protocol, args.stage)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if not output.is_relative_to(PROJECT_ROOT):
            raise ValueError("output must remain inside the project")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["decision"]["status"] not in {"FAIL", "WIRING_FAIL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
