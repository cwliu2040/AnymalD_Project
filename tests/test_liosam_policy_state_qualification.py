"""Static and unit checks for the LIO-SAM policy-state qualification runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "liosam_policy_state_quality.yaml"
RUNNER = ROOT / "scripts/validation/run_liosam_policy_state_qualification.py"


def _module():
    spec = importlib.util.spec_from_file_location("policy_state_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_freezes_formal_policy_and_architecture_boundaries(tmp_path: Path) -> None:
    module = _module()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    artifacts = module.validate_config(config)
    command = module.launch_command(
        profile="forward_1_5",
        run_dir=tmp_path,
        artifacts=artifacts,
        config=config,
    )
    assert "slam_backend:=liosam" in command
    assert "policy_odometry_topic:=/slam/policy_odom" in command
    assert "enable_confidence:=true" in command
    assert any(item.endswith("recovery_v0.4.0/policy.onnx") for item in command)
    assert config["runtime"]["mapping_confidence_authority_topic"] == "/slam/odom"
    assert config["runtime"]["runtime_ground_truth_input"] is False


def test_qualification_requires_multiple_profiles_and_repetitions() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    qualification = config["qualification"]
    assert qualification["control_policy"] == "Recovery v0.4.0 model1450"
    assert qualification["repetitions_per_profile"] == 3
    assert set(qualification["required_profiles"]) >= {
        "stationary",
        "forward_1_5",
        "lateral_1_5",
        "lateral_right_1_5",
        "backward_1_0",
        "curve_1_5_left_1_0",
    }
