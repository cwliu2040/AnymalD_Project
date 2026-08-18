#!/usr/bin/env python3
"""Select a frozen marginal-support challenge from excluded calibration runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_calibration(
    records: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    rule = matrix["selection_rule"]
    target_low, target_high = map(float, rule["tracking_event_fraction_target"])
    boundary_tolerance = float(rule["event_time_must_not_be_locked_to_ramp_boundary_s"])
    minimum_span = float(rule["minimum_event_time_span_across_support_candidates_s"])
    reaction_window = float(rule["minimum_policy_reaction_window_after_degradation_onset_s"])
    timeline = matrix["timeline_s"]
    healthy = float(timeline["healthy"])
    ramp_end = healthy + float(timeline["ramp_down"])
    hold_end = ramp_end + float(timeline["low_support_hold"])
    horizon = hold_end + float(timeline["recovery"])
    boundaries = (healthy, ramp_end, hold_end, horizon)

    schedule_by_identity = {
        (
            row["backend"], row["profile"], row["condition"],
            int(row["block_id"]), row["arm"],
        ): row
        for row in schedule
    }
    enriched: list[dict[str, Any]] = []
    identities: list[tuple[Any, ...]] = []
    for record in records:
        identity = record["identity"]
        key = (
            identity["backend"], identity["profile"], identity["condition"],
            int(identity["paired_block_id"]), identity["arm"],
        )
        identities.append(key)
        row = schedule_by_identity.get(key)
        if row is None:
            continue
        metrics = record["metrics"]
        event_time = float(metrics["tracking_restricted_mean_survival_time_s"])
        enriched.append({
            "backend": key[0], "profile": key[1], "arm": key[4],
            "support": float(row["minimum_support_fraction"]),
            "event_observed": bool(metrics["tracking_event_observed"]),
            "event_time_s": event_time,
            "event_fraction": min(event_time, horizon) / horizon,
            "gate_passed": bool(record.get("gate", {}).get("passed")),
        })

    expected_count = int(matrix["expected_run_count"])
    integrity_checks = {
        "excluded_calibration_role_only": {
            record.get("dataset_role") for record in records
        } == {"excluded_calibration"},
        "expected_record_count": len(records) == expected_count,
        "unique_identities": len(identities) == len(set(identities)),
        "exact_schedule_coverage": len(enriched) == len(schedule) == expected_count,
        "all_data_integrity_gates_passed": all(row["gate_passed"] for row in enriched),
    }

    sweeps: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    candidates: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        sweeps[(row["backend"], row["profile"])].append(row)
        candidates[row["support"]].append(row)

    sweep_checks: dict[str, dict[str, Any]] = {}
    sweep_passed = True
    expected_supports = {float(value) for value in matrix["support_fraction_candidates"]}
    for (backend, profile), rows in sorted(sweeps.items()):
        restricted = [min(row["event_time_s"], horizon) for row in rows]
        checks = {
            "complete_support_sweep": {row["support"] for row in rows} == expected_supports,
            "event_and_censoring_bracketed": (
                any(row["event_observed"] for row in rows)
                and any(not row["event_observed"] for row in rows)
            ),
            "minimum_event_time_span_met": max(restricted) - min(restricted) >= minimum_span,
        }
        passed = all(checks.values())
        sweep_passed = sweep_passed and passed
        sweep_checks[f"{backend}/{profile}"] = {
            "checks": checks,
            "restricted_event_time_span_s": max(restricted) - min(restricted),
            "passed": passed,
        }

    support_results: list[dict[str, Any]] = []
    for support, rows in sorted(candidates.items()):
        observed = all(row["event_observed"] for row in rows)
        fractions = [row["event_fraction"] for row in rows]
        event_times = [row["event_time_s"] for row in rows]
        checks = {
            "all_backends_profiles_arms_present": len(rows) == (
                len(matrix["backends"]) * len(matrix["profiles"])
                * len(matrix["policy_arms"]) * len(matrix["paired_block_ids"])
            ),
            "all_tracking_events_observed": observed,
            "tracking_event_fraction_in_target": observed and all(
                target_low <= value <= target_high for value in fractions
            ),
            "minimum_policy_reaction_window_met": observed and all(
                value - healthy >= reaction_window for value in event_times
            ),
            "not_locked_to_phase_boundary": observed and all(
                min(abs(value - boundary) for boundary in boundaries) > boundary_tolerance
                for value in event_times
            ),
        }
        support_results.append({
            "minimum_support_fraction": support,
            "checks": checks,
            "event_time_range_s": [min(event_times), max(event_times)],
            "event_fraction_range": [min(fractions), max(fractions)],
            "passed": all(checks.values()),
        })

    qualified = [
        row["minimum_support_fraction"] for row in support_results if row["passed"]
    ]
    # A marginal challenge uses the least degradation that still produces an
    # event in every backend.  Selecting a lower support would make the test
    # unnecessarily severe and could hide rather than identify policy value.
    selected = max(qualified) if qualified and sweep_passed else None
    passed = all(integrity_checks.values()) and selected is not None
    return {
        "schema_version": 1,
        "kind": "slam_confidence_challenge_selection",
        "dataset_role": "excluded_calibration",
        "integrity": {"checks": integrity_checks, "passed": all(integrity_checks.values())},
        "timeline_s": {
            "healthy": healthy, "ramp_end": ramp_end,
            "hold_end": hold_end, "challenge_horizon": horizon,
        },
        "sweep_checks": sweep_checks,
        "support_candidates": support_results,
        "selected_minimum_support_fraction": selected,
        "formal_condition_freeze_allowed": passed,
        "passed": passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/slam_confidence_publication_protocol.yaml")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.input_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_relative_to(PROJECT_ROOT) or not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("calibration paths must remain inside the project")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    summary = json.loads((root / "matrix_summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("**/publication_run_record.json"))
    ]
    result = evaluate_calibration(
        records, summary["schedule"], protocol["challenge_calibration_matrix"]
    )
    result["git_commit"] = summary["git_commit"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "selected_minimum_support_fraction": result["selected_minimum_support_fraction"],
    }, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
