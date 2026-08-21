from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/analyze_slam_confidence_publication.py"
SPEC = importlib.util.spec_from_file_location("publication_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _protocol() -> dict:
    return {
        "live_matrix": {
            "backends": ["fastlio2"], "profiles": ["route"],
            "perception_conditions": {"native": {}}, "paired_block_ids": [43],
            "policy_arms": ["A", "B"], "expected_run_count": 2,
        }
    }


def _record(arm: str) -> dict:
    return {
        "identity": {
            "backend": "fastlio2", "profile": "route", "condition": "native",
            "paired_block_id": 43, "arm": arm,
        },
        "gate": {"passed": True},
    }


def test_formal_completeness_requires_exact_unique_schedule() -> None:
    result = MODULE.validate_formal_completeness([_record("A"), _record("B")], _protocol(), {"formal"})
    assert result["passed"]
    duplicate = MODULE.validate_formal_completeness([_record("A"), _record("A")], _protocol(), {"formal"})
    assert not duplicate["passed"]
    assert not duplicate["checks"]["unique_cell_identity"]
    assert not duplicate["checks"]["exact_frozen_schedule"]


def test_excluded_role_can_never_be_formally_complete() -> None:
    result = MODULE.validate_formal_completeness(
        [_record("A"), _record("B")], _protocol(), {"excluded_smoke"},
    )
    assert not result["passed"]
    assert not result["checks"]["formal_role_only"]


def test_formal_completeness_uses_only_frozen_formal_conditions() -> None:
    protocol = _protocol()
    protocol["live_matrix"]["perception_conditions"] = {
        "native": {}, "gradual_support_loss": {},
    }
    protocol["live_matrix"]["formal_conditions"] = ["gradual_support_loss"]
    records = [_record("A"), _record("B")]
    for record in records:
        record["identity"]["condition"] = "gradual_support_loss"

    result = MODULE.validate_formal_completeness(records, protocol, {"formal"})

    assert result["passed"]
    assert result["checks"]["exact_frozen_schedule"]
