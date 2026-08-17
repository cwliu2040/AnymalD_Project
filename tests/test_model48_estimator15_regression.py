from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/validate_model48_estimator15_regression.py"
SPEC = importlib.util.spec_from_file_location("model48_regression", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_curated_five_profile_regression_passes() -> None:
    release = yaml.safe_load((ROOT / "configs/slam_confidence_sim_release_v1.yaml").read_text())
    result = MODULE.validate_regression(release)
    assert result["passed"]
    assert result["aggregate"]["total_environment_count"] == 2560
    assert result["aggregate"]["total_distinct_hard_terminated_env_count"] == 6
    assert result["aggregate"]["pooled_distinct_hard_terminated_env_fraction"] == 6 / 2560
