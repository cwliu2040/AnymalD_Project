#!/usr/bin/env python3
"""Apply the frozen data-development gate to fixed-speed-scale run records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "configs/slam_speed_scale_causal_pilot_v1.yaml"
VALIDATOR_PATH = PROJECT_ROOT / "scripts/validation/validate_slam_speed_scale_causal_protocol.py"
ARMS = ("scale_100", "scale_075", "scale_050", "scale_025")
METRICS = (
    "moving_speed_mps", "normalized_progress", "normalized_progress_per_elapsed_second",
    "valid_requested_usable_next_horizon_failure_fraction",
    "tracking_restricted_mean_survival_time_s",
    "stance_weighted_foot_slip_rms_mps", "while_stable_roll_pitch_rate_rms_radps",
)


def _validator_module():
    spec = importlib.util.spec_from_file_location("speed_protocol", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_records(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("**/speed_scale_run_record.json"))
    ]


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("metric values must be nonempty and finite")
    return float(np.mean(array))


def analyze_records(records: list[dict[str, Any]], protocol: dict, stage: str) -> dict:
    schedule = _validator_module().build_schedule(protocol, stage)
    expected = {
        (r["backend"], r["profile"], r["block_id"], r["arm"]) for r in schedule
    }
    identities = [
        (r["identity"]["backend"], r["identity"]["profile"],
         int(r["identity"]["block_id"]), r["identity"]["arm"])
        for r in records
    ]
    observed = set(identities)
    failures: list[str] = []
    if len(identities) != len(observed):
        failures.append("duplicate_run_identities")
    if expected - observed:
        failures.append("missing_run_identities")
    if observed - expected:
        failures.append("unexpected_run_identities")
    by_pair: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for record in records:
        identity = record.get("identity", {})
        arm = identity.get("arm")
        if record.get("dataset_role") != "excluded_causal_development":
            failures.append("incorrect_dataset_role")
        if identity.get("stage") != stage or arm not in ARMS:
            failures.append("identity_contract_failure")
            continue
        if not record.get("gate", {}).get("passed", False):
            failures.append("data_integrity_gate_failure")
        trace = record.get("treatment_trace", {})
        if not trace.get("passed", False):
            failures.append("assigned_or_realized_scale_violation")
        expected_scale = float(protocol["treatment"]["arms"][arm])
        if float(trace.get("assigned_command_scale", -1)) != expected_scale:
            failures.append("assigned_scale_mismatch")
        metrics = record.get("metrics", {})
        for metric in METRICS:
            value = metrics.get(metric)
            if value is None or not math.isfinite(float(value)):
                failures.append(f"missing_or_nonfinite_metric:{metric}")
        by_pair[(identity["backend"], identity["profile"], int(identity["block_id"]))][arm] = record
    for arms in by_pair.values():
        if set(arms) != set(ARMS):
            failures.append("incomplete_matched_arms")
    failures = sorted(set(failures))
    integrity = not failures and observed == expected
    report: dict[str, Any] = {
        "stage": stage,
        "inventory": {"expected_run_count": len(expected), "observed_run_count": len(records)},
        "integrity": {"passed": integrity, "failures": failures},
    }
    if stage == "wiring_smoke" or not integrity:
        passed = stage == "wiring_smoke" and integrity
        report["decision"] = {
            "status": "WIRING_PASS" if passed else "WIRING_FAIL" if stage == "wiring_smoke" else "FAIL",
            "claim_allowed": False,
            "next_step": "causal_pilot_requires_separate_authorization" if passed else "repair_wiring",
        }
        return report

    strata = {}
    speed_monotonic = 0
    hazard_monotonic = 0
    repeated_harm = False
    for backend in protocol["stages"][stage]["backends"]:
        for profile in protocol["stages"][stage]["profiles"]:
            pairs = [arms for (b, p, _), arms in by_pair.items() if b == backend and p == profile]
            means = {
                arm: {metric: _mean(pair[arm]["metrics"][metric] for pair in pairs) for metric in METRICS}
                for arm in ARMS
            }
            speeds = [means[arm]["moving_speed_mps"] for arm in reversed(ARMS)]
            hazards = [means[arm]["valid_requested_usable_next_horizon_failure_fraction"] for arm in reversed(ARMS)]
            speed_ok = all(a <= b for a, b in zip(speeds, speeds[1:]))
            hazard_ok = all(a <= b for a, b in zip(hazards, hazards[1:]))
            speed_monotonic += int(speed_ok)
            hazard_monotonic += int(hazard_ok)
            safety_by_arm = {
                arm: [bool(pair[arm]["metrics"].get("fall") or pair[arm]["metrics"].get("base_contact")) for pair in pairs]
                for arm in ARMS
            }
            for arm in ARMS[1:]:
                excess_blocks = sum(treatment and not control for treatment, control in zip(safety_by_arm[arm], safety_by_arm["scale_100"]))
                repeated_harm = repeated_harm or excess_blocks >= 2
            strata[f"{backend}/{profile}"] = {
                "arm_means": means,
                "realized_speed_nondecreasing_with_scale": speed_ok,
                "hazard_nondecreasing_with_scale": hazard_ok,
            }
    backend_hazard = {}
    backend_ok = True
    for backend in protocol["stages"][stage]["backends"]:
        pairs = [arms for (b, _, _), arms in by_pair.items() if b == backend]
        difference = _mean(
            np.mean([
                pair[arm]["metrics"]["valid_requested_usable_next_horizon_failure_fraction"]
                for arm in ARMS[1:]
            ]) - pair["scale_100"]["metrics"]["valid_requested_usable_next_horizon_failure_fraction"]
            for pair in pairs
        )
        ok = difference <= 0.0
        backend_ok = backend_ok and ok
        backend_hazard[backend] = {"mean_reduced_minus_full_hazard": difference, "nonpositive": ok}
    conditions = {
        "complete_integrity_valid_records": integrity,
        "realized_speed_nondecreasing_in_at_least_3_of_4_strata": speed_monotonic >= 3,
        "hazard_nondecreasing_in_at_least_3_of_4_strata": hazard_monotonic >= 3,
        "each_backend_reduced_minus_full_hazard_nonpositive": backend_ok,
        "no_repeated_reduced_scale_specific_safety_harm": not repeated_harm,
    }
    passed = all(conditions.values())
    report.update({
        "strata": strata,
        "backend_hazard": backend_hazard,
        "gate": {"conditions": conditions, "speed_monotonic_strata": speed_monotonic, "hazard_monotonic_strata": hazard_monotonic},
        "decision": {
            "status": "PASS" if passed else "FAIL",
            "claim_allowed": False,
            "next_step": protocol["decision_gate"]["pass_next_step" if passed else "otherwise_next_step"],
        },
    })
    return report


def _markdown(report: dict) -> str:
    return (
        "# Speed-scale causal pilot decision\n\n"
        f"Stage: `{report['stage']}`\n\n"
        f"Decision: **{report['decision']['status']}**\n\n"
        f"Next step: `{report['decision']['next_step']}`\n\n"
        "Excluded causal-development evidence; no efficacy claim is allowed.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--stage", choices=("wiring_smoke", "causal_pilot"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, protocol_path, output = (p.expanduser().resolve() for p in (args.root, args.protocol, args.output))
    if any(not p.is_relative_to(PROJECT_ROOT) for p in (root, protocol_path, output)):
        raise ValueError("all paths must remain inside the project")
    report = analyze_records(load_records(root), yaml.safe_load(protocol_path.read_text()), args.stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))
    return 0 if report["decision"]["status"] in ("WIRING_PASS", "PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
