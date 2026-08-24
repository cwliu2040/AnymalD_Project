#!/usr/bin/env python3
"""Analyze whether component-specific pulses improve on uniform slowdown."""

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
ARMS = ("control", "uniform_075", "preserve_yaw", "preserve_translation")
HAZARD = "pulse_window_intention_to_treat_failure_fraction"
SPEED = "pulse_window_moving_speed_mps"


def _validator():
    spec = importlib.util.spec_from_file_location("component_pulse_protocol", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_records(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("**/speed_pulse_run_record.json"))
    ]


def _risk(record: dict[str, Any]) -> float:
    metrics = record.get("metrics", {})
    value = metrics.get(HAZARD)
    if value is None:
        conditional = metrics.get(
            "pulse_window_valid_requested_usable_next_horizon_failure_fraction"
        )
        value = 1.0 if conditional is None else conditional
    return float(value)


def _prestate_diagnostic(arms: dict[str, dict], protocol: dict) -> dict:
    observations = {
        arm: np.asarray(record["pulse_trace"]["pre_pulse"]["observation"], dtype=np.float64)
        for arm, record in arms.items()
    }
    sample_times = [
        float(arms[arm]["pulse_trace"]["pre_pulse"]["elapsed_s"]) for arm in ARMS
    ]
    report: dict[str, Any] = {
        "maximum_pairwise_time_skew_s": max(sample_times) - min(sample_times),
        "used_for_exclusion": False,
    }
    for name, bounds in protocol["trace_contract"]["pre_pulse_matching_offsets"].items():
        low, high = map(int, bounds)
        values = [observations[arm][low:high] for arm in ARMS]
        differences = [
            np.abs(left - right)
            for index, left in enumerate(values)
            for right in values[index + 1:]
        ]
        report[name] = {
            "maximum_pairwise_linf": max(float(np.max(value)) for value in differences)
        }
    return report


def _safety(record: dict[str, Any]) -> bool:
    metrics = record.get("metrics", {})
    return bool(metrics.get("fall") or metrics.get("base_contact"))


def analyze_records(records: list[dict[str, Any]], protocol: dict, stage: str) -> dict:
    schedule = _validator().build_schedule(protocol, stage)
    expected = {
        (row["backend"], row["profile"], row["block_id"], row["arm"])
        for row in schedule
    }
    identities = [
        (
            record.get("identity", {}).get("backend"),
            record.get("identity", {}).get("profile"),
            int(record.get("identity", {}).get("block_id", -1)),
            record.get("identity", {}).get("arm"),
        )
        for record in records
    ]
    observed = set(identities)
    failures: list[str] = []
    if len(identities) != len(observed):
        failures.append("duplicate_run_identities")
    if expected - observed:
        failures.append("missing_run_identities")
    if observed - expected:
        failures.append("unexpected_run_identities")

    grouped: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for record in records:
        identity = record.get("identity", {})
        arm = identity.get("arm")
        if (
            record.get("dataset_role") != protocol["dataset_role"]
            or identity.get("stage") != stage
            or arm not in ARMS
        ):
            failures.append("record_contract_failure")
            continue
        if not record.get("gate", {}).get("passed"):
            failures.append("data_integrity_failure")
        if not record.get("pulse_trace", {}).get("passed"):
            failures.append("pulse_trace_failure")
        if not math.isfinite(_risk(record)):
            failures.append(f"missing_or_nonfinite:{HAZARD}")
        speed = record.get("metrics", {}).get(SPEED)
        if speed is None or not math.isfinite(float(speed)):
            failures.append(f"missing_or_nonfinite:{SPEED}")
        grouped[(identity["backend"], identity["profile"], int(identity["block_id"]))][arm] = record

    prestate = {}
    for key, arms in grouped.items():
        if set(arms) != set(ARMS):
            failures.append("incomplete_component_arms")
            continue
        prestate["/".join(map(str, key))] = _prestate_diagnostic(arms, protocol)

    failures = sorted(set(failures))
    integrity = not failures and observed == expected
    report: dict[str, Any] = {
        "stage": stage,
        "inventory": {"expected_run_count": len(expected), "observed_run_count": len(records)},
        "integrity": {"passed": integrity, "failures": failures},
        "pre_pulse_covariates": prestate,
    }
    if stage == "wiring_smoke" or not integrity:
        passed = stage == "wiring_smoke" and integrity
        report["decision"] = {
            "status": "WIRING_PASS" if passed else "WIRING_FAIL" if stage == "wiring_smoke" else "FAIL",
            "claim_allowed": False,
            "next_step": "causal_pilot_requires_separate_authorization" if passed else "repair_or_redesign_wiring",
        }
        return report

    gate = protocol["decision_gate"]["component_specific_comparison"]
    reference = gate["reference_arm"]
    candidates = tuple(gate["candidate_arms"])
    primary_backend = gate["primary_improvement_backend"]
    profiles = tuple(gate["prespecified_profiles"])
    minimum_nonworse = int(gate["minimum_nonworse_blocks_out_of_4"])
    minimum_benefit = float(gate["minimum_mean_uniform_minus_candidate_hazard"])
    minimum_speed_delta = float(gate["minimum_mean_candidate_minus_uniform_speed_mps"])
    fast_margin = float(gate["fastlio2_maximum_mean_candidate_minus_control_hazard"])

    strata: dict[str, Any] = {}
    supported_candidates: set[str] = set()
    repeated_harm = False
    for backend in protocol["stages"][stage]["backends"]:
        for profile in protocol["stages"][stage]["profiles"]:
            blocks = [arms for (b, p, _), arms in grouped.items() if b == backend and p == profile]
            arm_means = {
                arm: {
                    "hazard": float(np.mean([_risk(block[arm]) for block in blocks])),
                    "moving_speed_mps": float(np.mean([float(block[arm]["metrics"][SPEED]) for block in blocks])),
                }
                for arm in ARMS
            }
            comparisons = {}
            for candidate in candidates:
                risk_differences = [
                    _risk(block[candidate]) - _risk(block[reference]) for block in blocks
                ]
                speed_differences = [
                    float(block[candidate]["metrics"][SPEED])
                    - float(block[reference]["metrics"][SPEED])
                    for block in blocks
                ]
                excess_safety = sum(
                    _safety(block[candidate]) and not _safety(block[reference])
                    for block in blocks
                )
                repeated_harm = repeated_harm or excess_safety >= 2
                supported = bool(
                    backend == primary_backend
                    and profile in profiles
                    and sum(value <= 0.0 for value in risk_differences) >= minimum_nonworse
                    and -float(np.mean(risk_differences)) >= minimum_benefit
                    and float(np.mean(speed_differences)) >= minimum_speed_delta
                    and excess_safety < 2
                )
                if supported:
                    supported_candidates.add(candidate)
                comparisons[candidate] = {
                    "candidate_minus_uniform_hazard_by_block": risk_differences,
                    "mean_uniform_minus_candidate_hazard": -float(np.mean(risk_differences)),
                    "mean_candidate_minus_uniform_speed_mps": float(np.mean(speed_differences)),
                    "candidate_specific_excess_safety_blocks": excess_safety,
                    "supported_primary_stratum": supported,
                }
            strata[f"{backend}/{profile}"] = {
                "arm_means": arm_means,
                "component_vs_uniform": comparisons,
            }

    fast_safeguards = {}
    for candidate in candidates:
        values = []
        for profile in profiles:
            blocks = [arms for (b, p, _), arms in grouped.items() if b == "fastlio2" and p == profile]
            values.extend(_risk(block[candidate]) - _risk(block["control"]) for block in blocks)
        fast_safeguards[candidate] = {
            "mean_candidate_minus_control_hazard": float(np.mean(values)),
            "passed": bool(values and float(np.mean(values)) <= fast_margin),
        }
    qualified = sorted(
        candidate for candidate in supported_candidates if fast_safeguards[candidate]["passed"]
    )
    conditions = {
        "complete_integrity_valid_records": integrity,
        "randomized_balanced_schedule_complete": observed == expected,
        "pre_pulse_covariates_reported_without_exclusion": len(prestate) == len(grouped),
        "at_least_one_component_candidate_beats_uniform_on_liosam": len(supported_candidates) >= int(gate["minimum_supported_candidates"]),
        "same_candidate_passes_fastlio2_safeguard": bool(qualified),
        "no_repeated_candidate_specific_safety_harm": not repeated_harm,
    }
    passed = all(conditions.values())
    report.update({
        "strata": strata,
        "gate": {
            "conditions": conditions,
            "supported_liosam_candidates": sorted(supported_candidates),
            "qualified_candidates": qualified,
            "fastlio2_safeguards": fast_safeguards,
        },
        "decision": {
            "status": "PASS" if passed else "FAIL",
            "claim_allowed": False,
            "next_step": protocol["decision_gate"]["pass_next_step" if passed else "otherwise_next_step"],
        },
    })
    return report
