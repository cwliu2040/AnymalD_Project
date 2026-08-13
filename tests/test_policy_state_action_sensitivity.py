"""Tests for policy action-difference aggregation."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/evaluate_policy_state_action_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("action_sensitivity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_action_difference_metrics() -> None:
    values = np.vstack((np.zeros(12), np.full(12, 0.2)))
    metrics = MODULE.action_difference_metrics(values)
    assert metrics["sample_count"] == 2
    assert metrics["mean_absolute"] == pytest.approx(0.1)
    assert metrics["absolute_max"] == pytest.approx(0.2)


def test_action_difference_metrics_fail_closed_without_samples() -> None:
    metrics = MODULE.action_difference_metrics(np.empty((0, 12)))
    assert metrics["sample_count"] == 0
    assert math.isnan(metrics["mean_absolute"])
