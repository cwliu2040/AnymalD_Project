from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stability_safety_failure_is_not_data_integrity_failure() -> None:
    runner = _load(
        "publication_matrix_failure_disposition",
        ROOT / "scripts/validation/run_slam_confidence_publication_matrix.py",
    )
    document = {
        "summary": {
            "sample_count": 100,
            "hard_failures": {
                "terminated_count": 1,
                "non_finite_sample_count": 0,
            },
        },
        "gate": {"passed": False, "failures": ["terminated_count=1"]},
    }
    assert runner.stability_artifact_data_valid(document)
    document["summary"]["hard_failures"]["non_finite_sample_count"] = 1
    assert not runner.stability_artifact_data_valid(document)


def test_invalid_icp_is_retained_as_binary_map_outcome() -> None:
    evaluator = _load(
        "map_failure_disposition",
        ROOT / "scripts/validation/evaluate_slam_map_consistency_run.py",
    )
    reference = np.zeros((600, 3), dtype=np.float64)
    estimated = np.full((600, 3), 1000.0, dtype=np.float64)
    metrics, outcome = evaluator.score_map_outcome(
        estimated, reference, maximum_points=5000
    )
    assert not outcome["map_registration_valid"]
    assert outcome["map_registration_failure_reason"]
    assert set(metrics) == set(evaluator.MAP_METRIC_NAMES)
    assert all(value is None for value in metrics.values())


def test_protocol_retains_scheduled_failures_without_survivorship_filter() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/slam_confidence_publication_protocol.yaml").read_text()
    )
    assert protocol["statistics"]["experimental_unit"] == "scheduled_live_run"
    assert protocol["statistics"][
        "scheduled_policy_or_slam_failures_are_retained_as_outcomes"
    ]
    assert protocol["run_disposition"]["policy_or_slam_failure"].startswith(
        "retain_as_outcome"
    )


def test_estimator_replay_uses_policy_reset_ack_boundary() -> None:
    source = (
        ROOT / "scripts/validation/evaluate_velocity_estimator_replay.py"
    ).read_text()
    assert 'reset_ack_topic = "/simulation/episode_reset_ack"' in source
    assert '"episode_boundary": "policy_episode_reset_ack"' in source
    assert source.count("synchronizer.reset()") >= 2


def test_run_record_and_backfill_preserve_failed_attempts() -> None:
    record_source = (
        ROOT / "scripts/validation/build_slam_confidence_publication_run_record.py"
    ).read_text()
    backfill_source = (
        ROOT / "scripts/validation/backfill_slam_confidence_publication_metrics.py"
    ).read_text()
    assert '"experimental_unit": "scheduled_live_run"' in record_source
    assert '"policy_or_slam_failure_retained_as_outcome": True' in record_source
    assert '"source_cell_preserved": True' in backfill_source
    assert "publication_backfill.json" in backfill_source
