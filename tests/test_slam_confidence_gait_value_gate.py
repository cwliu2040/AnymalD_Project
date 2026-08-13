from pathlib import Path
import importlib.util

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    path = ROOT / "scripts/validation/validate_slam_confidence_gait_value.py"
    spec = importlib.util.spec_from_file_location("gait_value_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(multiplier: float = 1.0) -> dict:
    metrics = {
        "linear_rmse": 0.20 * multiplier,
        "action_rate_rms": 0.10 * multiplier,
        "body_tilt_rms": 0.10 * multiplier,
        "body_roll_pitch_rate_rms_radps": 0.10 * multiplier,
        "mean_absolute_vertical_speed_mps": 0.10 * multiplier,
        "mean_stance_foot_slip_speed_mps": 0.10 * multiplier,
    }
    return {
        "steps": 1000,
        "num_envs": 512,
        "behavior_gate_profile": {
            "path": "/project/config.yaml",
            "sha256": "abc",
            "profile_id": "matched",
            "seed": 43,
            "command": {"vx": 1.5, "vy": 0.0, "wz": 0.0},
            "cycle_s": 10.0,
        },
        "velocity_estimator": {"metadata": "exported/estimator/metadata.json"},
        "termination_summary": {"hard_terminated_env_fraction": 0.0},
        "behavior_gate": {"passed": True},
        "windows": {
            name: dict(metrics)
            for name in (
                "healthy",
                "degraded_late",
                "invalid_settled",
                "recovery_settled",
            )
        },
    }


def test_gait_value_gate_requires_real_transition_improvement() -> None:
    validator = _load_validator()
    gate = yaml.safe_load(
        (ROOT / "configs/slam_confidence_gait_value_gate.yaml").read_text(
            encoding="utf-8"
        )
    )
    baseline = _report()
    candidate = _report(0.94)
    result = validator.compare_reports(gate, baseline, candidate)
    assert result["passed"] is True
    candidate = _report(1.0)
    result = validator.compare_reports(gate, baseline, candidate)
    assert result["checks"]["degraded_invalid_gait_value_added"] is False
    assert result["passed"] is False


def test_gait_value_gate_rejects_unmatched_estimator() -> None:
    validator = _load_validator()
    gate = yaml.safe_load(
        (ROOT / "configs/slam_confidence_gait_value_gate.yaml").read_text(
            encoding="utf-8"
        )
    )
    baseline = _report()
    candidate = _report(0.90)
    candidate["velocity_estimator"]["metadata"] = "different.json"
    try:
        validator.compare_reports(gate, baseline, candidate)
    except ValueError as error:
        assert "same proprioceptive velocity estimator" in str(error)
    else:
        raise AssertionError("unmatched estimator should be rejected")


def test_play_reports_all_frozen_gait_metrics() -> None:
    source = (ROOT / "scripts/rsl_rl/play.py").read_text(encoding="utf-8")
    gate = yaml.safe_load(
        (ROOT / "configs/slam_confidence_gait_value_gate.yaml").read_text(
            encoding="utf-8"
        )
    )
    for metric in gate["gait_metrics"]:
        assert f'"{metric}"' in source
    assert "--confidence_supervisor_baseline" in source
