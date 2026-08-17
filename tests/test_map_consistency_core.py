from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from anymal_locomotion_ros2.map_consistency_core import (
    SurfacePose,
    build_observed_surface,
    evaluate_map_consistency,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/validate_map_consistency_synthetic.py"


def _synthetic_module():
    spec = importlib.util.spec_from_file_location("map_synthetic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_recovers_single_se2_alignment() -> None:
    module = _synthetic_module()
    reference, healthy, _ = module.synthetic_maps()
    result = evaluate_map_consistency(healthy, reference)
    assert result.reference_distance_p95_m <= 0.03
    assert result.duplicate_surface_fraction <= 0.01


def test_metric_detects_split_surface_copy() -> None:
    module = _synthetic_module()
    reference, _, split = module.synthetic_maps()
    result = evaluate_map_consistency(split, reference)
    assert result.reference_distance_p95_m >= 0.15
    assert result.duplicate_surface_fraction >= 0.10


def test_synthetic_publication_gate_passes() -> None:
    result = _synthetic_module().validate_synthetic()
    assert result["passed"]
    assert all(result["checks"].values())


def test_metric_rejects_nonfinite_or_too_small_inputs() -> None:
    reference = np.zeros((20, 3))
    for invalid in (np.zeros((19, 3)), np.full((20, 3), np.nan)):
        try:
            evaluate_map_consistency(invalid, reference)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid point map was accepted")


def test_observed_surface_uses_interpolated_pose_and_lidar_offset() -> None:
    local = np.column_stack((np.linspace(0.0, 1.0, 20), np.zeros(20), np.zeros(20)))
    poses = [
        SurfacePose(0, 0.0, 0.0, 0.0, 0.0),
        SurfacePose(10, 2.0, 0.0, 0.0, 0.0),
    ]
    surface = build_observed_surface([(5, local)], poses, points_per_scan=20)
    np.testing.assert_allclose(surface[:, 0], local[:, 0] + 1.20)
    np.testing.assert_allclose(surface[:, 1], local[:, 1])
    np.testing.assert_allclose(surface[:, 2], local[:, 2] + 0.35)
