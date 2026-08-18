from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/select_slam_confidence_challenge.py"


def _module():
    spec = importlib.util.spec_from_file_location("challenge_selection", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(backend: str, support: float, arm: str, event_time: float, observed: bool):
    condition = f"gradual_support_{support:.3f}".replace(".", "p")
    return {
        "dataset_role": "excluded_calibration",
        "identity": {
            "backend": backend, "profile": "curve", "condition": condition,
            "paired_block_id": 243, "arm": arm,
        },
        "gate": {"passed": True},
        "metrics": {
            "tracking_event_observed": observed,
            "tracking_restricted_mean_survival_time_s": event_time,
        },
    }


def test_selection_requires_bracket_and_selects_highest_degradation() -> None:
    module = _module()
    supports = [0.35, 0.45, 0.55]
    matrix = {
        "expected_run_count": 12, "backends": ["fastlio2", "liosam"],
        "profiles": ["curve"], "policy_arms": ["C", "D"],
        "paired_block_ids": [243], "support_fraction_candidates": supports,
        "timeline_s": {"healthy": 3.0, "ramp_down": 6.0, "low_support_hold": 4.0, "recovery": 3.0},
        "selection_rule": {
            "tracking_event_fraction_target": [0.20, 0.80],
            "event_time_must_not_be_locked_to_ramp_boundary_s": 0.20,
            "minimum_event_time_span_across_support_candidates_s": 1.0,
            "minimum_policy_reaction_window_after_degradation_onset_s": 2.0,
        },
    }
    event_times = {0.35: (6.0, True), 0.45: (8.0, True), 0.55: (17.0, False)}
    records, schedule = [], []
    for backend in matrix["backends"]:
        for support in supports:
            for arm in matrix["policy_arms"]:
                event_time, observed = event_times[support]
                record = _record(backend, support, arm, event_time, observed)
                records.append(record)
                schedule.append({
                    **record["identity"], "block_id": 243,
                    "minimum_support_fraction": support,
                })
    result = module.evaluate_calibration(records, schedule, matrix)
    assert result["passed"] is True
    assert result["selected_minimum_support_fraction"] == 0.35


def test_selection_rejects_ramp_locked_sweep() -> None:
    module = _module()
    supports = [0.35, 0.45]
    matrix = {
        "expected_run_count": 8, "backends": ["fastlio2", "liosam"],
        "profiles": ["curve"], "policy_arms": ["C", "D"],
        "paired_block_ids": [243], "support_fraction_candidates": supports,
        "timeline_s": {"healthy": 3.0, "ramp_down": 6.0, "low_support_hold": 4.0, "recovery": 3.0},
        "selection_rule": {
            "tracking_event_fraction_target": [0.20, 0.80],
            "event_time_must_not_be_locked_to_ramp_boundary_s": 0.20,
            "minimum_event_time_span_across_support_candidates_s": 1.0,
            "minimum_policy_reaction_window_after_degradation_onset_s": 2.0,
        },
    }
    records, schedule = [], []
    for backend in matrix["backends"]:
        for support in supports:
            for arm in matrix["policy_arms"]:
                record = _record(backend, support, arm, 3.5, True)
                records.append(record)
                schedule.append({
                    **record["identity"], "block_id": 243,
                    "minimum_support_fraction": support,
                })
    result = module.evaluate_calibration(records, schedule, matrix)
    assert result["passed"] is False
    assert result["selected_minimum_support_fraction"] is None
