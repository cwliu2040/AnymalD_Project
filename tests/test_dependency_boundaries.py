"""Static dependency and artifact-boundary tests."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PACKAGE = PROJECT_ROOT / "source" / "anymal_locomotion"
EVALUATION_SCRIPT = PROJECT_ROOT / "scripts" / "rsl_rl" / "evaluate.py"


def test_training_package_does_not_import_rclpy() -> None:
    violations: list[str] = []
    for path in TRAINING_PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name == "rclpy" for alias in node.names):
                violations.append(str(path))
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "rclpy":
                violations.append(str(path))
    assert violations == []


def test_artifact_configuration_is_project_local() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "artifacts.yaml").read_text(encoding="utf-8"))
    root = Path(config["project_root"]).resolve()
    assert root == PROJECT_ROOT
    for key in ("logs_root", "checkpoints_root", "exports_root"):
        path = Path(config[key]).resolve()
        assert root in path.parents
        assert not str(path).startswith("/home/ros/IsaacLab/logs")


def test_evaluation_uses_public_fixed_command_configuration() -> None:
    source = EVALUATION_SCRIPT.read_text(encoding="utf-8")
    assert "command_cfg.ranges.lin_vel_x" in source
    assert "command_cfg.ranges.lin_vel_y" in source
    assert "command_cfg.ranges.ang_vel_z" in source
    assert "command_manager._" not in source
    assert "rclpy" not in source
    assert 'LOG_ROOT / "evaluation"' in source
