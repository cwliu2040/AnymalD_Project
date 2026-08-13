from pathlib import Path
import importlib.util

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_matrix_validator():
    path = ROOT / "scripts/validation/validate_slam_confidence_behavior_matrix.py"
    spec = importlib.util.spec_from_file_location("behavior_matrix_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_behavior_gate_has_four_disjoint_ordered_windows() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/slam_confidence_behavior_gate.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["seed"] == 43
    assert config["num_envs"] == 512
    assert config["steps"] == 1000
    assert config["command"] == {"vx": 1.5, "vy": 0.0, "wz": 0.0}
    windows = config["windows"]
    assert list(windows) == [
        "healthy",
        "degraded_late",
        "invalid_settled",
        "recovery_settled",
    ]
    intervals = [tuple(window["phase"]) for window in windows.values()]
    assert all(0.0 <= start < end <= 1.0 for start, end in intervals)
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))


def test_behavior_gate_covers_each_required_behavior() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/slam_confidence_behavior_gate.yaml").read_text(
            encoding="utf-8"
        )
    )
    gates = config["gates"]
    assert "healthy_linear_rmse_max_mps" in gates
    assert "degraded_speed_ratio_to_healthy_max" in gates
    assert "invalid_mean_planar_speed_max_mps" in gates
    assert "invalid_speed_above_0_50_fraction_max" in gates
    assert "recovery_linear_rmse_max_mps" in gates
    assert "hard_termination_fraction_max" in gates


def test_holdout_profiles_are_distinct_and_add_yaw_gates() -> None:
    expected = {
        "lateral_left": (44, {"vx": 0.0, "vy": 1.5, "wz": 0.0}),
        "lateral_right": (45, {"vx": 0.0, "vy": -1.5, "wz": 0.0}),
        "reverse": (46, {"vx": -1.5, "vy": 0.0, "wz": 0.0}),
        "combined": (47, {"vx": 1.5, "vy": 0.0, "wz": 1.0}),
    }
    for name, (seed, command) in expected.items():
        config = yaml.safe_load(
            (
                ROOT
                / f"configs/slam_confidence_behavior_holdout_{name}.yaml"
            ).read_text(encoding="utf-8")
        )
        assert config["seed"] == seed
        assert config["command"] == command
        assert config["num_envs"] == 512
        assert config["steps"] == 1000
        assert set(config["windows"]) == {
            "healthy",
            "degraded_late",
            "invalid_settled",
            "recovery_settled",
        }
        gates = config["gates"]
        assert "healthy_yaw_mae_max_radps" in gates
        assert "degraded_yaw_mae_max_radps" in gates
        assert "recovery_yaw_mae_max_radps" in gates


def test_play_evaluator_applies_optional_yaw_gates() -> None:
    source = (ROOT / "scripts/rsl_rl/play.py").read_text(encoding="utf-8")
    for check_name in (
        "healthy_yaw_tracking",
        "degraded_yaw_tracking",
        "recovery_yaw_tracking",
    ):
        assert check_name in source
    assert "hard_terminated_envs" in source
    assert '"hard_terminated_env_fraction"' in source


def test_holdout_safety_overlay_limits_terminated_environments() -> None:
    overlay = yaml.safe_load(
        (
            ROOT
            / "configs/slam_confidence_behavior_holdout_safety_overlay.yaml"
        ).read_text(encoding="utf-8")
    )
    assert overlay["application"] == "each_profile"
    assert overlay["hard_termination_env_fraction_max"] == 0.01
    assert len(overlay["required_profiles"]) == 5


def test_behavior_matrix_validator_applies_overlay_to_each_profile() -> None:
    validator = _load_matrix_validator()
    required = [f"profile-{index}" for index in range(5)]
    overlay = {
        "schema_version": 1,
        "application": "each_profile",
        "required_profiles": required,
        "hard_termination_env_fraction_max": 0.01,
    }
    reports = [
        {
            "checkpoint": "/project/model.pt",
            "num_envs": 512,
            "behavior_gate_profile": {"profile_id": profile_id},
            "behavior_gate": {
                "passed": True,
                "derived": {
                    "hard_terminated_env_count": 5,
                    "hard_terminated_env_fraction": 5 / 512,
                },
            },
        }
        for profile_id in required
    ]
    result = validator.validate_reports(overlay, reports)
    assert result["passed"] is True
    reports[-1]["behavior_gate"]["derived"] = {
        "hard_terminated_env_count": 6,
        "hard_terminated_env_fraction": 6 / 512,
    }
    result = validator.validate_reports(overlay, reports)
    assert result["passed"] is False
