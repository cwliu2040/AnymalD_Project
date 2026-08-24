#!/usr/bin/env python3
"""Apply the frozen futility gate to action-risk intervention run records."""

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
PROTOCOL_PATH = PROJECT_ROOT / "configs/slam_action_risk_intervention_pilot.yaml"
VALIDATOR_PATH = PROJECT_ROOT / "scripts/validation/validate_slam_action_risk_intervention_protocol.py"
ARMS = ("smooth", "zero", "antismooth")
METRICS = (
    "action_rate_rms_per_s",
    "while_stable_roll_pitch_rate_rms_radps",
    "tracking_restricted_mean_survival_time_s",
    "valid_requested_usable_next_horizon_failure_fraction",
    "moving_speed_mps",
)


def _validator_module():
    spec = importlib.util.spec_from_file_location("intervention_protocol", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load intervention protocol validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(record: dict[str, Any]) -> tuple[str, str, int, str]:
    value = record["identity"]
    return (
        str(value["backend"]), str(value["profile"]),
        int(value["block_id"]), str(value["arm"]),
    )


def _expected_identities(protocol: dict[str, Any], stage: str) -> set[tuple[str, str, int, str]]:
    rows = _validator_module().build_stage_schedule(protocol, stage)
    return {
        (row["backend"], row["profile"], row["block_id"], row["arm"])
        for row in rows
    }


def load_records(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("**/intervention_run_record.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("run-record roots must be mappings")
    return records


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("metric values must be nonempty and finite")
    return float(np.mean(array))


def _simulation_safety_analysis(
    by_pair: dict[tuple[str, str, int], dict[str, dict[str, Any]]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    simulation = protocol.get("decision_gate", {}).get(
        "safety_stop", {},
    ).get("simulation")
    smooth_event_any = any(
        bool(arms.get("smooth", {}).get("metrics", {}).get("fall"))
        or bool(arms.get("smooth", {}).get("metrics", {}).get("base_contact"))
        for arms in by_pair.values()
    )
    if not isinstance(simulation, dict):
        return {
            "semantics": "v1_any_smooth_event",
            "smooth_event_any": smooth_event_any,
            "paired_excess_by_stratum": {},
            "minimum_repeated_blocks": 1,
            "route_fail": smooth_event_any,
            "pass_condition": not smooth_event_any,
        }

    minimum = int(
        simulation["route_fail_minimum_distinct_blocks_same_backend_profile"]
    )
    excess_by_stratum: dict[str, list[int]] = defaultdict(list)
    for (backend, profile, block), arms in by_pair.items():
        if "smooth" not in arms or "zero" not in arms:
            continue
        smooth = arms["smooth"]["metrics"]
        zero = arms["zero"]["metrics"]
        smooth_event = bool(smooth.get("fall")) or bool(smooth.get("base_contact"))
        zero_event = bool(zero.get("fall")) or bool(zero.get("base_contact"))
        if smooth_event and not zero_event:
            excess_by_stratum[f"{backend}/{profile}"].append(block)
    rendered = {
        key: sorted(set(blocks)) for key, blocks in sorted(excess_by_stratum.items())
    }
    route_fail = any(len(blocks) >= minimum for blocks in rendered.values())
    return {
        "semantics": "v2_repeated_paired_smooth_specific_excess",
        "smooth_event_any": smooth_event_any,
        "paired_excess_by_stratum": rendered,
        "minimum_repeated_blocks": minimum,
        "route_fail": route_fail,
        "pass_condition": not any(rendered.values()),
    }


def analyze_records(
    records: list[dict[str, Any]], protocol: dict[str, Any], stage: str,
) -> dict[str, Any]:
    expected = _expected_identities(protocol, stage)
    identities = [_identity(record) for record in records]
    identity_set = set(identities)
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    missing = sorted(expected - identity_set)
    unexpected = sorted(identity_set - expected)
    failures: list[str] = []
    if duplicates:
        failures.append("duplicate_run_identities")
    if unexpected:
        failures.append("unexpected_run_identities")

    limit = float(protocol["intervention"]["raw_action_linf_limit"])
    by_pair: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        backend, profile, block, arm = _identity(record)
        if arm not in ARMS:
            failures.append("unsupported_arm")
            continue
        if record.get("dataset_role") != "excluded_causal_development":
            failures.append("incorrect_dataset_role")
        if record.get("identity", {}).get("stage") != stage:
            failures.append("stage_mismatch")
        intervention = record.get("intervention", {})
        if not record.get("gate", {}).get("passed", False):
            failures.append("data_integrity_gate_failure")
        if not intervention.get("passed", False):
            failures.append("intervention_trace_failure")
        maximum_residual = intervention.get("maximum_realized_residual")
        residual_atol = float(intervention.get("atol", 0.0))
        if maximum_residual is None or not math.isfinite(float(maximum_residual)):
            failures.append("nonfinite_intervention_residual")
        elif float(maximum_residual) > limit + residual_atol:
            failures.append("intervention_linf_violation")
        metrics = record.get("metrics", {})
        for metric in METRICS:
            value = metrics.get(metric)
            if value is None or not math.isfinite(float(value)):
                failures.append(f"missing_or_nonfinite_metric:{metric}")
        by_pair[(backend, profile, block)][arm] = record

    incomplete_pairs = False
    for arms in by_pair.values():
        if tuple(sorted(arms)) != tuple(sorted(ARMS)):
            failures.append("incomplete_paired_arms")
            incomplete_pairs = True
    safety = _simulation_safety_analysis(by_pair, protocol)
    safety_terminated_early = bool(
        stage != "wiring_smoke"
        and missing
        and safety["semantics"] == "v2_repeated_paired_smooth_specific_excess"
        and safety["route_fail"]
        and not duplicates
        and not unexpected
        and not incomplete_pairs
        and not failures
    )
    if missing and not safety_terminated_early:
        failures.append("missing_run_identities")
    failures = sorted(set(failures))
    integrity_passed = not failures and (
        identity_set == expected or safety_terminated_early
    )

    exact_zero = bool(records) and all(
        record["intervention"]["checks"].get("zero_exact_arm_B", False)
        for record in records if record["identity"]["arm"] == "zero"
    )
    invalid_fallback = bool(records) and all(
        record["intervention"]["checks"].get("invalid_exact_arm_B", False)
        for record in records
    )
    nonzero_arms_realized = bool(records) and all(
        record["intervention"]["checks"].get("nonzero_arm_realized", False)
        for record in records if record["identity"]["arm"] != "zero"
    )

    report: dict[str, Any] = {
        "stage": stage,
        "inventory": {
            "expected_run_count": len(expected),
            "observed_run_count": len(records),
            "missing_identities": missing,
            "unexpected_identities": unexpected,
            "duplicate_identities": duplicates,
            "safety_terminated_early": safety_terminated_early,
        },
        "integrity": {
            "passed": integrity_passed,
            "failures": failures,
            "zero_arm_exact_arm_B": exact_zero,
            "invalid_stale_exact_arm_B": invalid_fallback,
            "nonzero_arms_realized": nonzero_arms_realized,
        },
    }

    if stage == "wiring_smoke":
        passed = integrity_passed and exact_zero and invalid_fallback and nonzero_arms_realized
        report["decision"] = {
            "status": "WIRING_PASS" if passed else "WIRING_FAIL",
            "claim_allowed": False,
            "next_step": "pilot_requires_separate_authorization" if passed else "repair_wiring",
        }
        return report

    if safety_terminated_early:
        report["simulation_safety"] = safety
        report["decision"] = {
            "status": "FAIL",
            "claim_allowed": False,
            "next_step": protocol["decision_gate"]["fail_next_step"],
        }
        return report

    if not integrity_passed:
        report["decision"] = {
            "status": "FAIL",
            "claim_allowed": False,
            "next_step": protocol["decision_gate"]["fail_next_step"],
        }
        return report

    strata: dict[str, Any] = {}
    ordered_action = 0
    ordered_body = 0
    all_action_differences = []
    speed_guard = True
    for backend in protocol["stages"][stage]["backends"]:
        for profile in protocol["stages"][stage]["profiles"]:
            key = f"{backend}/{profile}"
            pairs = [arms for (b, p, _), arms in by_pair.items() if b == backend and p == profile]
            arm_means = {
                arm: {
                    metric: _mean(pair[arm]["metrics"][metric] for pair in pairs)
                    for metric in METRICS
                }
                for arm in ARMS
            }
            action_order = (
                arm_means["smooth"]["action_rate_rms_per_s"]
                < arm_means["zero"]["action_rate_rms_per_s"]
                < arm_means["antismooth"]["action_rate_rms_per_s"]
            )
            body_order = (
                arm_means["smooth"]["while_stable_roll_pitch_rate_rms_radps"]
                < arm_means["zero"]["while_stable_roll_pitch_rate_rms_radps"]
                < arm_means["antismooth"]["while_stable_roll_pitch_rate_rms_radps"]
            )
            ordered_action += int(action_order)
            ordered_body += int(body_order)
            for treatment in ("smooth", "antismooth"):
                all_action_differences.append(
                    arm_means[treatment]["action_rate_rms_per_s"]
                    - arm_means["zero"]["action_rate_rms_per_s"]
                )
            speed_difference = (
                arm_means["smooth"]["moving_speed_mps"]
                - arm_means["zero"]["moving_speed_mps"]
            )
            speed_guard = speed_guard and abs(speed_difference) <= float(
                protocol["decision_gate"]["pass_all_required"]
                ["maximum_absolute_moving_speed_difference_vs_zero_mps"]
            )
            strata[key] = {
                "arm_means": arm_means,
                "action_rate_ordered": action_order,
                "body_rate_ordered": body_order,
                "smooth_minus_zero_moving_speed_mps": speed_difference,
            }

    backend_effects = {}
    backend_direction_pass = True
    harm = False
    for backend in protocol["stages"][stage]["backends"]:
        pairs = [arms for (b, _, _), arms in by_pair.items() if b == backend]
        differences = {
            metric: _mean(
                pair["smooth"]["metrics"][metric] - pair["zero"]["metrics"][metric]
                for pair in pairs
            )
            for metric in METRICS
        }
        survival_ok = differences["tracking_restricted_mean_survival_time_s"] >= 0.0
        hazard_ok = differences[
            "valid_requested_usable_next_horizon_failure_fraction"
        ] <= 0.0
        body_worse = differences["while_stable_roll_pitch_rate_rms_radps"] > 0.0
        survival_worse = differences["tracking_restricted_mean_survival_time_s"] < 0.0
        harm = harm or (body_worse and survival_worse)
        backend_direction_pass = backend_direction_pass and survival_ok and hazard_ok
        backend_effects[backend] = {
            "smooth_minus_zero": differences,
            "survival_nonnegative": survival_ok,
            "hazard_nonpositive": hazard_ok,
            "body_rate_and_survival_both_worse": body_worse and survival_worse,
        }

    threshold = int(protocol["decision_gate"]["minimum_ordered_backend_profile_strata"])
    separation_tolerance = float(
        protocol["decision_gate"]["no_action_rate_separation_absolute_tolerance_per_s"]
    )
    no_action_separation = bool(all_action_differences) and all(
        abs(value) <= separation_tolerance for value in all_action_differences
    )
    pass_conditions = {
        "integrity": integrity_passed,
        "zero_and_invalid_fallback": exact_zero and invalid_fallback,
        "nonzero_interventions_realized": nonzero_arms_realized,
        "action_rate_ordered_strata": ordered_action >= threshold,
        "body_rate_ordered_strata": ordered_body >= threshold,
        "backend_survival_and_hazard_direction": backend_direction_pass,
        "moving_speed_guard": speed_guard,
        "smooth_arm_safety": safety["pass_condition"],
    }
    if not integrity_passed or safety["route_fail"] or harm or no_action_separation:
        status = "FAIL"
        next_step = protocol["decision_gate"]["fail_next_step"]
    elif all(pass_conditions.values()):
        status = "PASS"
        next_step = protocol["decision_gate"]["pass_next_step"]
    else:
        status = "INCONCLUSIVE"
        next_step = protocol["decision_gate"]["inconclusive_next_step"]
    report.update({
        "strata": strata,
        "backend_effects": backend_effects,
        "gate": {
            "ordered_action_rate_strata": ordered_action,
            "ordered_body_rate_strata": ordered_body,
            "required_ordered_strata": threshold,
            "no_action_rate_separation": no_action_separation,
            "smooth_arm_safety_event_any": safety["smooth_event_any"],
            "smooth_arm_safety_failure": safety["route_fail"],
            "simulation_safety": safety,
            "body_rate_and_survival_harm": harm,
            "pass_conditions": pass_conditions,
        },
        "decision": {"status": status, "claim_allowed": False, "next_step": next_step},
    })
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Action-risk intervention decision", "",
        f"Stage: `{report['stage']}`", "",
        f"Decision: **{report['decision']['status']}**", "",
        f"Next step: `{report['decision']['next_step']}`", "",
        "This is an excluded causal-development futility gate, not publication efficacy evidence.", "",
        "## Inventory", "",
        f"- Expected runs: {report['inventory']['expected_run_count']}",
        f"- Observed runs: {report['inventory']['observed_run_count']}",
        f"- Integrity passed: {report['integrity']['passed']}",
    ]
    if "gate" in report:
        lines += [
            "", "## Gate", "",
            f"- Ordered action-rate strata: {report['gate']['ordered_action_rate_strata']}/{report['gate']['required_ordered_strata']}",
            f"- Ordered body-rate strata: {report['gate']['ordered_body_rate_strata']}/{report['gate']['required_ordered_strata']}",
            f"- Any smooth-arm safety event: {report['gate']['smooth_arm_safety_event_any']}",
            f"- Repeated paired smooth-specific safety harm: {report['gate']['simulation_safety']['route_fail']}",
            f"- Body-rate plus survival harm: {report['gate']['body_rate_and_survival_harm']}",
            f"- No action-rate separation: {report['gate']['no_action_rate_separation']}",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("wiring_smoke", "pilot", "expanded_only_after_pilot_inconclusive"),
        required=True,
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    paths = [args.input_root, args.protocol, args.output]
    resolved = [path.expanduser().resolve() for path in paths]
    if any(not path.is_relative_to(PROJECT_ROOT) for path in resolved):
        raise ValueError("all paths must remain inside the project")
    root, protocol_path, output = resolved
    if not root.is_dir() or not protocol_path.is_file():
        raise FileNotFoundError("input root or protocol is missing")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "kind": "slam_action_risk_intervention_decision",
        "dataset_role": "excluded_causal_development",
        **analyze_records(load_records(root), protocol, args.stage),
    }
    markdown = (args.markdown_output or output.with_suffix(".md")).expanduser().resolve()
    if not markdown.is_relative_to(PROJECT_ROOT):
        raise ValueError("markdown output must remain inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "inventory": report["inventory"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
