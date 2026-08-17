from __future__ import annotations

import importlib.util
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
