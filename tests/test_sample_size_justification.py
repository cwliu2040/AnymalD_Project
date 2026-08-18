from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/justify_slam_confidence_sample_size.py"
SPEC = importlib.util.spec_from_file_location("sample_size_justification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _records(protocol: dict) -> list[dict]:
    matrix = protocol["sample_size_pilot_matrix"]
    return [
        {
            "dataset_role": "excluded_pilot",
            "identity": {
                "backend": backend, "profile": profile, "condition": condition,
                "paired_block_id": block, "arm": arm,
            },
            "gate": {"passed": True},
        }
        for backend in matrix["backends"] for profile in matrix["profiles"]
        for condition in matrix["perception_conditions"]
        for block in matrix["paired_block_ids"] for arm in matrix["policy_arms"]
    ]


def test_frozen_pilot_records_are_exact_and_disjoint() -> None:
    protocol = yaml.safe_load((ROOT / "configs/slam_confidence_publication_protocol.yaml").read_text())
    result = MODULE.validate_pilot_records(_records(protocol), protocol)
    assert result["passed"]
    assert all(result["checks"].values())


def test_smoke_role_cannot_justify_sample_size() -> None:
    protocol = yaml.safe_load((ROOT / "configs/slam_confidence_publication_protocol.yaml").read_text())
    records = _records(protocol)
    records[0]["dataset_role"] = "excluded_smoke"
    result = MODULE.validate_pilot_records(records, protocol)
    assert not result["passed"]
    assert not result["checks"]["excluded_pilot_role_only"]


def test_efficiency_planning_uses_bounded_progress_difference() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_confidence_publication_protocol.yaml").read_text()
    )
    planning = protocol["statistics"]["sample_size_planning"]
    assert planning["precision_halfwidth_targets"][
        "normalized_progress_difference"
    ] == 0.05
    assert planning[
        "efficiency_noninferiority_normalized_progress_difference"
    ] == -0.10
    source = SCRIPT.read_text()
    assert 'metrics["normalized_progress_difference"]' in source
    assert 'metric="normalized_progress"' not in source  # positional helper call
    assert "log_efficiency_ratio" not in source
