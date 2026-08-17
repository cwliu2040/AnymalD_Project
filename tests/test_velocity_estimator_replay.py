from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/evaluate_velocity_estimator_replay.py"
SPEC = importlib.util.spec_from_file_location("velocity_estimator_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_estimator_replay_parity_uses_exact_stamps() -> None:
    replay = {stamp: np.asarray([stamp, 0.0, 0.0], dtype=np.float32) for stamp in range(100)}
    recorded = {stamp: value.copy() for stamp, value in replay.items()}
    result = MODULE.compare_estimates(replay, recorded, atol=1e-5)
    assert result["passed"]
    assert result["exact_stamp_match_count"] == 100
    assert result["maximum_absolute_velocity_error_mps"] == 0.0


def test_estimator_replay_parity_fails_missing_or_wrong_values() -> None:
    replay = {stamp: np.zeros(3, dtype=np.float32) for stamp in range(100)}
    missing = {stamp: np.zeros(3, dtype=np.float32) for stamp in range(90)}
    assert not MODULE.compare_estimates(replay, missing, atol=1e-5)["passed"]
    wrong = {stamp: np.ones(3, dtype=np.float32) for stamp in range(100)}
    assert not MODULE.compare_estimates(replay, wrong, atol=1e-5)["passed"]


def test_estimator_replay_uses_callback_order_independent_stamp_bundles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "EstimatorInputSynchronizer" in source
    assert "source_stamp_nearest_after_per_topic_watermark" in source
    assert '"ground_truth_used_by_runtime_or_replay": False' in source


def test_dual_backend_stamp_sync_qualification_is_exact() -> None:
    evidence = json.loads(
        (
            ROOT
            / "docs/validation/slam_confidence_estimator15_stamp_sync_qualification.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["passed"]
    assert len(evidence["cells"]) == 8
    assert {
        (cell["backend"], cell["arm"]) for cell in evidence["cells"]
    } == {
        (backend, arm)
        for backend in ("fastlio2", "liosam")
        for arm in ("A", "B", "C", "D")
    }
    for cell in evidence["cells"]:
        assert cell["passed"]
        assert cell["exact_stamp_matches"] == cell["replay_outputs"]
        assert cell["maximum_absolute_velocity_error_mps"] == 0.0
    assert evidence["aggregate"]["exact_stamp_match_count"] == 5486
    assert evidence["aggregate"]["matched_replay_fraction"] == 1.0
