#!/usr/bin/env python3
"""Analyze matched-prefix command-pulse causal pilot run records."""

from __future__ import annotations

from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = PROJECT_ROOT / "scripts/validation/validate_slam_speed_pulse_causal_protocol.py"
ARMS = ("scale_100", "scale_075", "scale_050", "scale_025")
HAZARD = "pulse_window_intention_to_treat_failure_fraction"


def _risk(record: dict[str, Any]) -> float:
    metrics = record.get("metrics", {})
    value = metrics.get(HAZARD)
    if value is None:
        conditional = metrics.get(
            "pulse_window_valid_requested_usable_next_horizon_failure_fraction"
        )
        value = 1.0 if conditional is None else conditional
    return float(value)


def _validator():
    spec = importlib.util.spec_from_file_location("pulse_protocol", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_records(root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(root.glob("**/speed_pulse_run_record.json"))]


def _pre_match(arms: dict[str, dict], protocol: dict) -> tuple[bool, dict]:
    observations = {
        arm: np.asarray(record["pulse_trace"]["pre_pulse"]["observation"], dtype=np.float64)
        for arm, record in arms.items()
    }
    details = {}
    passed = True
    sample_times = [float(arms[arm]["pulse_trace"]["pre_pulse"]["elapsed_s"]) for arm in ARMS]
    time_skew = max(sample_times) - min(sample_times)
    time_limit = float(protocol["trace_contract"]["pre_pulse_max_pairwise_time_skew_s"])
    details["sample_time_alignment"] = {
        "maximum_pairwise_skew_s": time_skew,
        "limit_s": time_limit,
        "passed": time_skew <= time_limit,
    }
    passed = passed and time_skew <= time_limit
    for name, bounds in protocol["trace_contract"]["pre_pulse_matching_offsets"].items():
        low, high = map(int, bounds)
        values = [observations[arm][low:high] for arm in ARMS]
        differences = [np.abs(a - b) for i, a in enumerate(values) for b in values[i + 1:]]
        maximum = max(float(np.max(value)) for value in differences)
        limit = float(protocol["trace_contract"]["pre_pulse_max_pairwise_linf"][name])
        term_passed = maximum <= limit
        detail = {"maximum_pairwise_linf": maximum, "linf_limit": limit}
        if name == "joint_velocity":
            maximum_rms = max(float(np.sqrt(np.mean(np.square(value)))) for value in differences)
            rms_limit = float(protocol["trace_contract"]["pre_pulse_max_pairwise_rms"][name])
            term_passed = term_passed and maximum_rms <= rms_limit
            detail.update({"maximum_pairwise_rms": maximum_rms, "rms_limit": rms_limit})
        detail["passed"] = term_passed
        details[name] = detail
        passed = passed and term_passed
    return passed, details


def analyze_records(records: list[dict[str, Any]], protocol: dict, stage: str) -> dict:
    schedule = _validator().build_schedule(protocol, stage)
    expected = {(r["backend"], r["profile"], r["block_id"], r["arm"]) for r in schedule}
    identities = [(r["identity"]["backend"], r["identity"]["profile"], int(r["identity"]["block_id"]), r["identity"]["arm"]) for r in records]
    observed = set(identities)
    failures = []
    if len(identities) != len(observed): failures.append("duplicate_run_identities")
    if expected - observed: failures.append("missing_run_identities")
    if observed - expected: failures.append("unexpected_run_identities")
    by_pair: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for record in records:
        identity, metrics = record.get("identity", {}), record.get("metrics", {})
        arm = identity.get("arm")
        if record.get("dataset_role") != "excluded_causal_development" or identity.get("stage") != stage or arm not in ARMS:
            failures.append("record_contract_failure")
            continue
        if not record.get("gate", {}).get("passed") or not record.get("pulse_trace", {}).get("passed"):
            failures.append("wiring_or_trace_failure")
        if not math.isfinite(_risk(record)):
            failures.append(f"missing_or_nonfinite:{HAZARD}")
        value = metrics.get("pulse_window_moving_speed_mps")
        if value is None or not math.isfinite(float(value)):
            failures.append("missing_or_nonfinite:pulse_window_moving_speed_mps")
        by_pair[(identity["backend"], identity["profile"], int(identity["block_id"]))][arm] = record
    matching = {}
    for key, arms in by_pair.items():
        if set(arms) != set(ARMS):
            failures.append("incomplete_matched_arms")
            continue
        passed, detail = _pre_match(arms, protocol)
        matching["/".join(map(str, key))] = {"passed": passed, "terms": detail}
        if not passed and protocol["experimental_design"]["exact_prestate_matching_required"]:
            failures.append("pre_pulse_matching_failure")
    failures = sorted(set(failures))
    integrity = not failures and observed == expected
    report: dict[str, Any] = {
        "stage": stage,
        "inventory": {"expected_run_count": len(expected), "observed_run_count": len(records)},
        "integrity": {"passed": integrity, "failures": failures},
        "pre_pulse_matching": matching,
    }
    if stage == "wiring_smoke" or not integrity:
        passed = stage == "wiring_smoke" and integrity
        report["decision"] = {"status": "WIRING_PASS" if passed else "WIRING_FAIL" if stage == "wiring_smoke" else "FAIL", "claim_allowed": False, "next_step": "causal_pilot_requires_separate_authorization" if passed else "repair_or_redesign_wiring"}
        return report
    headroom = protocol["decision_gate"]["ceiling_aware_headroom"]
    primary_backend = str(headroom["primary_improvement_backend"])
    primary_profiles = tuple(headroom["prespecified_improvement_profiles"])
    minimum_benefit = float(
        headroom["supported_stratum_all_required"]
        ["minimum_mean_full_minus_reduced_hazard"]
    )
    ceiling_threshold = float(headroom["full_scale_hazard_ceiling_threshold"])
    fast_margin = float(
        headroom["fastlio2_maximum_mean_reduced_minus_full_hazard"]
    )
    strata, supported_primary = {}, 0
    fast_safeguards = []
    repeated_harm = False
    for backend in protocol["stages"][stage]["backends"]:
        for profile in protocol["stages"][stage]["profiles"]:
            pairs = [arms for (b, p, _), arms in by_pair.items() if b == backend and p == profile]
            means = {arm: float(np.mean([_risk(pair[arm]) for pair in pairs])) for arm in ARMS}
            ascending = [means[a] for a in reversed(ARMS)]
            ordered = all(a <= b for a, b in zip(ascending, ascending[1:]))
            ceiling = all(
                _risk(pair["scale_100"]) <= ceiling_threshold
                for pair in pairs
            )
            reduced = {}
            supported_arm = False
            for arm in ARMS[1:]:
                block_differences = [
                    _risk(pair[arm]) - _risk(pair["scale_100"])
                    for pair in pairs
                ]
                benefit = -float(np.mean(block_differences))
                arm_supported = bool(
                    sum(value <= 0.0 for value in block_differences)
                    >= int(headroom["supported_stratum_all_required"]["one_reduced_scale_minimum_nonpositive_blocks"])
                    and benefit >= minimum_benefit
                )
                supported_arm = supported_arm or arm_supported
                reduced[arm] = {
                    "reduced_minus_full_by_block": block_differences,
                    "mean_full_minus_reduced_hazard": benefit,
                    "repeated_nonpositive_and_meaningful": arm_supported,
                }
            supported = bool(not ceiling and ordered and supported_arm)
            if backend == primary_backend and profile in primary_profiles:
                supported_primary += int(supported)
            mean_reduced_minus_full = float(np.mean([
                means[arm] - means["scale_100"] for arm in ARMS[1:]
            ]))
            if backend == "fastlio2":
                fast_safeguards.append(
                    bool(ceiling or mean_reduced_minus_full <= fast_margin)
                )
            for arm in ARMS[1:]:
                excess = sum(bool(pair[arm]["metrics"].get("fall") or pair[arm]["metrics"].get("base_contact")) and not bool(pair["scale_100"]["metrics"].get("fall") or pair["scale_100"]["metrics"].get("base_contact")) for pair in pairs)
                repeated_harm = repeated_harm or excess >= 2
            strata[f"{backend}/{profile}"] = {
                "mean_hazard": means,
                "full_scale_ceiling": ceiling,
                "hazard_nondecreasing_with_scale": ordered,
                "reduced_scale_evidence": reduced,
                "supported_headroom": supported,
                "mean_reduced_minus_full_hazard": mean_reduced_minus_full,
            }
    backend_effects = {}
    for backend in protocol["stages"][stage]["backends"]:
        pairs = [arms for (b, _, _), arms in by_pair.items() if b == backend]
        difference = float(np.mean([np.mean([_risk(pair[a]) for a in ARMS[1:]]) - _risk(pair["scale_100"]) for pair in pairs]))
        backend_effects[backend] = {"mean_reduced_minus_full_hazard": difference, "nonpositive": difference <= 0.0}
    conditions = {
        "complete_integrity_valid_records": integrity,
        "randomized_balanced_schedule_complete": observed == expected,
        "pre_pulse_covariates_reported_without_exclusion": len(matching) == len(by_pair),
        "liosam_supported_headroom_strata_meets_minimum": supported_primary >= int(headroom["minimum_supported_primary_backend_strata"]),
        "fastlio2_noninferiority_or_ceiling_safeguard": bool(fast_safeguards) and all(fast_safeguards),
        "no_repeated_reduced_scale_specific_safety_harm": not repeated_harm,
    }
    passed = all(conditions.values())
    report.update({"strata": strata, "backend_effects": backend_effects, "gate": {"conditions": conditions, "supported_primary_backend_strata": supported_primary, "required_supported_primary_backend_strata": int(headroom["minimum_supported_primary_backend_strata"]), "fastlio2_safeguards": fast_safeguards}, "decision": {"status": "PASS" if passed else "FAIL", "claim_allowed": False, "next_step": protocol["decision_gate"]["pass_next_step" if passed else "otherwise_next_step"]}})
    return report
