from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publication_diagnostics",
    ROOT / "scripts/validation/diagnose_slam_confidence_publication.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _records() -> list[dict]:
    records = []
    for block in range(10):
        for arm, speed, outcome in (("C", 1.0 + 0.01 * block, 2.0 + 0.02 * block), ("D", 0.9, 1.0)):
            records.append({
                "identity": {
                    "backend": "fastlio2", "profile": "p", "condition": "gradual_support_loss",
                    "paired_block_id": block, "arm": arm,
                },
                "metrics": {"moving_speed_mps": speed, "outcome": outcome},
            })
    return records


def test_matched_adjustment_reports_overlap_and_post_treatment_boundary() -> None:
    result = MODULE.matched_adjustment(
        _records(), treatment="C", control="D", outcome="outcome",
        mediator="moving_speed_mps", threshold=0.2, resamples=100,
    )
    assert result["pair_count"] == 10
    assert not result["zero_within_observed_mediator_difference_range"]
    assert result["post_treatment_exploratory_only"]


def test_phase_boundaries_are_half_open() -> None:
    protocol = {
        "live_matrix": {"perception_conditions": {"gradual_support_loss": {"timeline_s": {
            "healthy": 3.0, "ramp_down": 6.0, "low_support_hold": 4.0, "recovery": 3.0,
        }}}}
    }
    windows = MODULE._phase_windows(protocol)
    assert MODULE._phase_name(2.999, windows) == "healthy"
    assert MODULE._phase_name(3.0, windows) == "ramp_down"
    assert MODULE._phase_name(16.0, windows) == "after_recovery"


def test_phase_contrasts_mark_metric_without_numeric_pairs_unavailable() -> None:
    records = _records()
    for block in range(10):
        for arm in ("A", "B"):
            records.append({
                "identity": {
                    "backend": "fastlio2", "profile": "p", "condition": "gradual_support_loss",
                    "paired_block_id": block, "arm": arm,
                },
                "metrics": {"moving_speed_mps": None, "outcome": 1.0},
            })
    for record in records:
        record["identity"]["phase"] = "after_recovery"
        record["metrics"]["moving_speed_mps"] = None
        record["metrics"]["stance_weighted_foot_slip_rms_mps"] = record["metrics"]["outcome"]
        record["metrics"]["while_stable_roll_pitch_rate_rms_radps"] = record["metrics"]["outcome"]
    result = MODULE.phase_contrasts(records, resamples=100)
    assert result["after_recovery"]["learned_gait"]["moving_speed_mps"]["status"] == (
        "unavailable_no_numeric_pairs"
    )


def test_gait_association_reports_speed_adjusted_partial_correlation() -> None:
    rows = []
    for block in range(10):
        speed = 0.5 + 0.05 * block
        rows.append({
            "identity": {"backend": "fastlio2", "profile": "p", "paired_block_id": block},
            "gait": {field: speed + 0.01 * index for index, field in enumerate(MODULE.GAIT_FIELDS)},
            "metrics": {
                "moving_speed_mps": speed,
                "stance_weighted_foot_slip_rms_mps": speed + 0.01 * (block % 2),
                "while_stable_roll_pitch_rate_rms_radps": speed + 0.02 * (block % 3),
                "tracking_restricted_mean_survival_time_s": 5.0 - speed + 0.01 * (block % 2),
                "normalized_progress": speed + 0.03 * (block % 4),
            },
        })
    result = MODULE.gait_associations(rows, resamples=100)
    value = result["stance_width"]["normalized_progress"]
    assert "partial_moving_speed_pearson_r" in value
    assert value["partial_moving_speed_cluster_bootstrap_95pct"] is not None
