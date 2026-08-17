from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/run_slam_confidence_publication_replay.py"
SPEC = importlib.util.spec_from_file_location("publication_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_replay_reproducibility_accepts_tiny_float_drift() -> None:
    first = {"counts": {"odom": 10}, "trajectory": {"ate": 0.1, "samples": 10, "vector": [1, 2]}}
    second = {"counts": {"odom": 10}, "trajectory": {"ate": 0.100001, "samples": 10, "vector": [1, 2]}}
    result = MODULE.compare_repetitions(first, second, atol=1e-6, rtol=1e-4)
    assert result["passed"]


def test_replay_reproducibility_rejects_counts_but_quantifies_metric_drift() -> None:
    first = {"counts": {"odom": 10}, "trajectory": {"ate": 0.1}}
    second = {"counts": {"odom": 11}, "trajectory": {"ate": 0.2}}
    result = MODULE.compare_repetitions(first, second, atol=1e-6, rtol=1e-4)
    assert not result["passed"]
    assert not result["counts_match_exactly"]
    assert not result["within_preferred_diagnostic_tolerance"]
    assert not result["numeric_differences_are_exclusion_gate"]


def test_finite_metric_drift_is_outcome_not_exclusion() -> None:
    first = {"counts": {"odom": 10}, "trajectory": {"ate": 0.1}}
    second = {"counts": {"odom": 10}, "trajectory": {"ate": 0.2}}
    result = MODULE.compare_repetitions(first, second, atol=1e-6, rtol=1e-4)
    assert result["passed"]
    assert not result["within_preferred_diagnostic_tolerance"]


def test_formal_source_identity_requires_exact_schedule() -> None:
    protocol = {
        "live_matrix": {
            "backends": ["fastlio2"], "profiles": ["route"],
            "perception_conditions": {"native": {}}, "paired_block_ids": [43],
            "policy_arms": ["A", "B"],
        }
    }
    def record(arm: str) -> tuple[Path, dict]:
        return Path("record.json"), {
            "identity": {
                "backend": "fastlio2", "profile": "route", "condition": "native",
                "paired_block_id": 43, "arm": arm,
            }
        }
    assert MODULE.formal_source_identity_complete([record("A"), record("B")], protocol)
    assert not MODULE.formal_source_identity_complete([record("A"), record("A")], protocol)
