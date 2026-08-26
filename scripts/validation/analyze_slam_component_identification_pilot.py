#!/usr/bin/env python3
"""Validate fresh two-level component-identification run records."""

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
ARMS = (
    "control", "uniform_075",
    "reduce_translation_075", "reduce_translation_0875",
    "reduce_yaw_075", "reduce_yaw_0875",
)
HAZARD = "pulse_window_intention_to_treat_failure_fraction"
SPEED = "pulse_window_moving_speed_mps"


def _validator():
    spec = importlib.util.spec_from_file_location("identification_protocol", VALIDATOR_PATH)
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
    return float(record["metrics"][HAZARD])


def _safety(record: dict[str, Any]) -> bool:
    metrics = record["metrics"]
    return bool(metrics.get("fall") or metrics.get("base_contact"))


def analyze_records(records: list[dict[str, Any]], protocol: dict, stage: str) -> dict:
    schedule = _validator().build_schedule(protocol, stage)
    expected = {
        (row["backend"], row["profile"], row["block_id"], row["arm"])
        for row in schedule
    }
    observed_list = []
    failures = []
    grouped: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for record in records:
        identity = record.get("identity", {})
        key = (
            identity.get("backend"), identity.get("profile"),
            int(identity.get("block_id", -1)), identity.get("arm"),
        )
        observed_list.append(key)
        hazard = record.get("metrics", {}).get(HAZARD)
        speed = record.get("metrics", {}).get(SPEED)
        if (
            record.get("dataset_role") != protocol["dataset_role"]
            or identity.get("stage") != stage
            or identity.get("arm") not in ARMS
        ):
            failures.append("record_contract_failure")
            continue
        if not record.get("gate", {}).get("passed"):
            failures.append("data_integrity_failure")
        if not record.get("pulse_trace", {}).get("passed"):
            failures.append("pulse_trace_failure")
        if hazard is None or not math.isfinite(float(hazard)):
            failures.append(f"missing_or_nonfinite:{HAZARD}")
        if speed is None or not math.isfinite(float(speed)):
            failures.append(f"missing_or_nonfinite:{SPEED}")
        grouped[key[:3]][key[3]] = record
    observed = set(observed_list)
    if len(observed_list) != len(observed):
        failures.append("duplicate_run_identities")
    if expected - observed:
        failures.append("missing_run_identities")
    if observed - expected:
        failures.append("unexpected_run_identities")

    prestate = {}
    for key, arms in grouped.items():
        if set(arms) != set(ARMS):
            failures.append("incomplete_identification_arms")
            continue
        times = [float(arms[arm]["pulse_trace"]["pre_pulse"]["elapsed_s"]) for arm in ARMS]
        confidence = [float(arms[arm]["pulse_trace"]["pre_pulse"]["observation"][48]) for arm in ARMS]
        prestate["/".join(map(str, key))] = {
            "maximum_pairwise_time_skew_s": max(times) - min(times),
            "confidence_probability_range": [min(confidence), max(confidence)],
            "used_for_exclusion": False,
        }
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
            "next_step": "identification_requires_separate_authorization" if passed else "repair_or_redesign_wiring",
        }
        return report

    identification = protocol["decision_gate"]["component_identification"]
    strata = {}
    repeated_harm = False
    variation_passes = []
    arm_counts_pass = True
    translation_scales, yaw_scales = set(), set()
    for backend in protocol["stages"][stage]["backends"]:
        for profile in protocol["stages"][stage]["profiles"]:
            block_groups = [arms for (b, p, _), arms in grouped.items() if b == backend and p == profile]
            values = [_risk(arms[arm]) for arms in block_groups for arm in ARMS]
            distinct = len({round(value, 12) for value in values})
            variation = distinct >= int(identification["minimum_distinct_targets_per_stratum"])
            variation_passes.append(variation)
            arm_counts = {arm: sum(arm in arms for arms in block_groups) for arm in ARMS}
            arm_counts_pass = arm_counts_pass and all(count == len(block_groups) for count in arm_counts.values())
            arm_means = {
                arm: {
                    "hazard": float(np.mean([_risk(arms[arm]) for arms in block_groups])),
                    "moving_speed_mps": float(np.mean([float(arms[arm]["metrics"][SPEED]) for arms in block_groups])),
                }
                for arm in ARMS
            }
            safety_excess = {}
            for arm in ARMS[1:]:
                excess = sum(_safety(arms[arm]) and not _safety(arms["control"]) for arms in block_groups)
                safety_excess[arm] = excess
                repeated_harm = repeated_harm or excess >= 2
            strata[f"{backend}/{profile}"] = {
                "run_count": len(values), "distinct_target_count": distinct,
                "target_variation_passed": variation, "arm_counts": arm_counts,
                "arm_means": arm_means, "arm_specific_safety_excess_blocks": safety_excess,
            }
    for record in records:
        scales = record["pulse_trace"]["assigned_scales"]
        translation_scales.add(round(float(np.mean(scales[:2])), 6))
        yaw_scales.add(round(float(scales[2]), 6))
    conditions = {
        "complete_source_integrity": integrity,
        "balanced_arm_counts": arm_counts_pass,
        "translation_scale_levels_present": translation_scales == set(map(float, identification["required_translation_scales"])),
        "yaw_scale_levels_present": yaw_scales == set(map(float, identification["required_yaw_scales"])),
        "target_variation_in_every_backend_profile": all(variation_passes),
        "no_repeated_arm_specific_safety_harm": not repeated_harm,
        "reserved_validation_blocks_untouched": not any(int(record["identity"]["block_id"]) in set(identification["reserved_validation_blocks"]) for record in records),
    }
    passed = all(conditions.values())
    report.update({
        "strata": strata,
        "gate": {
            "conditions": conditions,
            "translation_scale_levels": sorted(translation_scales),
            "yaw_scale_levels": sorted(yaw_scales),
        },
        "decision": {
            "status": "PASS" if passed else "FAIL",
            "claim_allowed": False,
            "next_step": protocol["decision_gate"]["pass_next_step" if passed else "otherwise_next_step"],
        },
    })
    return report
