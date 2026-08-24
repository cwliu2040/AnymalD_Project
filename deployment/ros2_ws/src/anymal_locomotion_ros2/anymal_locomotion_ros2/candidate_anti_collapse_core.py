"""Ceiling-aware qualification for a candidate relative to B and old C."""

from __future__ import annotations

import math
from typing import Any, Iterable


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def evaluate_candidate_strata(
    strata: Iterable[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    rows = list(strata)
    if not rows:
        raise ValueError("candidate evaluation requires strata")
    ceiling = config["ceiling_aware_evaluation"]
    collapse = config["anti_collapse"]
    fast_results = []
    lio_results = []
    failures = []
    for row in rows:
        backend = str(row["backend"])
        risk_difference = _finite(row["candidate_minus_b_risk"], "risk difference")
        progress_ratio = _finite(row["candidate_over_b_progress_ratio"], "progress ratio")
        stop_excess = _finite(row["candidate_minus_b_stopped_fraction"], "stop excess")
        safety_excess = int(row["candidate_minus_b_safety_events"])
        result = {
            "backend": backend,
            "profile": str(row["profile"]),
            "risk_difference": risk_difference,
            "progress_ratio": progress_ratio,
            "stop_excess": stop_excess,
            "safety_excess": safety_excess,
        }
        if backend == "fastlio2":
            passed = bool(
                risk_difference <= float(ceiling["fastlio2"]["risk_difference_candidate_minus_b_maximum"])
                and progress_ratio >= float(ceiling["fastlio2"]["progress_ratio_candidate_over_b_minimum"])
                and stop_excess <= 0.0 and safety_excess <= 0
            )
            result["safeguard_passed"] = passed
            fast_results.append(result)
            if not passed:
                failures.append("fastlio2_noninferiority_or_ceiling_failure")
        elif backend == "liosam":
            b_equivalence = _finite(row["equivalence_fraction_vs_b"], "B equivalence")
            old_c_equivalence = _finite(row["equivalence_fraction_vs_old_c"], "old-C equivalence")
            improved = bool(
                risk_difference <= float(ceiling["liosam"]["risk_difference_candidate_minus_b_maximum"])
                and progress_ratio >= float(ceiling["liosam"]["progress_ratio_candidate_over_b_minimum"])
                and stop_excess <= 0.0 and safety_excess <= 0
            )
            noncollapsed = bool(
                b_equivalence <= float(collapse["maximum_equivalence_fraction_vs_arm_b"])
                and old_c_equivalence <= float(collapse["maximum_equivalence_fraction_vs_old_c"])
            )
            result.update({
                "equivalence_fraction_vs_b": b_equivalence,
                "equivalence_fraction_vs_old_c": old_c_equivalence,
                "improved": improved,
                "noncollapsed": noncollapsed,
            })
            lio_results.append(result)
        else:
            failures.append("unsupported_backend")
    required = int(ceiling["liosam"]["minimum_improved_degraded_strata"])
    improved_lio = sum(int(row["improved"] and row["noncollapsed"]) for row in lio_results)
    collapsed = bool(lio_results) and all(not row["noncollapsed"] for row in lio_results)
    if not fast_results:
        failures.append("missing_fastlio2_safeguard")
    if not lio_results:
        failures.append("missing_liosam_improvement_strata")
    if improved_lio < required:
        failures.append("insufficient_liosam_improved_noncollapsed_strata")
    failures = sorted(set(failures))
    if collapsed:
        status = str(collapse["collapsed_status"])
    elif failures:
        status = "FAIL"
    else:
        status = "PASS"
    return {
        "status": status,
        "passed": status == "PASS",
        "failures": failures,
        "fastlio2": fast_results,
        "liosam": lio_results,
        "improved_noncollapsed_liosam_strata": improved_lio,
        "required_improved_liosam_strata": required,
    }
