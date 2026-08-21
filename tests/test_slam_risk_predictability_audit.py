from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slam_risk_predictability_audit",
    ROOT / "scripts/validation/audit_slam_risk_predictability.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_history_summary_is_causal_and_has_expected_layout() -> None:
    times = np.asarray([0.0, 0.5, 1.0])
    values = np.asarray([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])
    summary = MODULE.history_summary(times, values)
    assert summary.shape == (12,)
    np.testing.assert_allclose(summary[:2], [2.0, 2.0])
    np.testing.assert_allclose(summary[-2:], [2.0, 0.0])
    changed_future = np.vstack((values, [100.0, 100.0]))
    np.testing.assert_allclose(
        summary,
        MODULE.history_summary(times, changed_future[:3]),
    )


def test_summary_columns_select_each_statistic_block() -> None:
    columns = MODULE.summary_columns((48, 49, 50))
    assert columns.tolist() == [
        offset + block * 51 for block in range(6) for offset in (48, 49, 50)
    ]


def test_recovery_target_requires_full_horizon_and_detects_recovery() -> None:
    stamps = [0, 500_000_000, 1_000_000_000, 1_500_000_000]
    values = [False, False, True, None]
    assert MODULE.recovery_target(
        clock_ns=0, label_stamps=stamps, label_values=values,
    )
    assert MODULE.recovery_target(
        clock_ns=500_000_000, label_stamps=stamps, label_values=values,
    ) is None


def test_equal_run_weights_give_each_run_equal_total_mass() -> None:
    runs = np.asarray(["a", "a", "a", "b"])
    weights = MODULE.equal_run_weights(runs)
    assert np.isclose(np.sum(weights[runs == "a"]), np.sum(weights[runs == "b"]))
    assert np.isclose(np.mean(weights), 1.0)


def _evaluation(difference: float, lower: float, upper: float, ece: float = 0.0):
    comparison = {
        "mean_difference": difference,
        "cluster_bootstrap_95pct": {"lower": lower, "upper": upper},
        "log_loss_difference": difference,
        "ece_difference": ece,
    }
    return {
        scheme: {
            backend: {
                endpoint: {
                    "status": "measured",
                    "comparisons_to_slam_state_only": {
                        "motion_without_estimator_velocity": comparison,
                    },
                }
                for endpoint in MODULE.ENDPOINTS
            }
            for backend in MODULE.BACKENDS
        }
        for scheme in ("block_5fold", "leave_one_arm_out")
    }


def test_decision_rules_are_frozen() -> None:
    assert MODULE.classify_audit(_evaluation(-0.02, -0.03, -0.01))["status"] == "PASS"
    assert MODULE.classify_audit(_evaluation(-0.01, -0.03, 0.01))["status"] == "INCONCLUSIVE"
    assert MODULE.classify_audit(_evaluation(0.02, 0.01, 0.03))["status"] == "FAIL"
