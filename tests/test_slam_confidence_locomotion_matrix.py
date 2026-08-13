from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/slam_confidence_locomotion_matrix.yaml"
RUNNER = ROOT / "scripts/validation/run_slam_confidence_locomotion_matrix.py"
LAUNCH = (
    ROOT
    / "deployment/ros2_ws/src/anymal_locomotion_ros2/launch"
    / "fastlio2_locomotion_benchmark.launch.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("confidence_matrix", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_freezes_candidate_native_backends_and_holdouts() -> None:
    matrix = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert matrix["schema_version"] == 1
    assert matrix["repetitions"] == 1
    assert matrix["candidate"]["checkpoint_sha256"] == (
        "73d5870de4acc5c79b0d0fa30f15cd70ed5c9539f9b7ec9a4d35d30d6377fb16"
    )
    assert matrix["candidate"]["policy_sha256"] == (
        "511383d9a6b0d4d6c26588667e195f9d66b7e703db991af59fcc8aad4aeb84df"
    )
    assert matrix["backends"] == {
        "fastlio2": {
            "expected_calibration_id": "native-v1-1e6cf8347be1",
            "deskew_mode": "native",
        },
        "liosam": {
            "expected_calibration_id": "native-v1-edc098b0bd98",
            "deskew_mode": "native",
        },
    }
    assert matrix["profiles"] == [
        "forward_1_5",
        "lateral_1_5",
        "lateral_right_1_5",
        "backward_1_0",
        "curve_1_5_left_1_0",
    ]


def test_matrix_candidate_hashes_and_parity_are_self_consistent() -> None:
    module = _load_runner()
    matrix = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    artifacts = module.validate_candidate_artifacts(matrix)
    assert artifacts["checkpoint_sha256"] == matrix["candidate"][
        "checkpoint_sha256"
    ]
    assert artifacts["policy_sha256"] == matrix["candidate"]["policy_sha256"]


def test_policy_diagnostics_gate_requires_real_51d_confidence_consumption() -> None:
    module = _load_runner()
    gate = {
        "observation_dimension": 51,
        "minimum_records": 1,
        "require_tracking_valid_observation": True,
        "require_non_fail_closed_observation": True,
    }
    record = {
        "observation": [0.0] * 48 + [0.9, 1.0, 0.1],
        "watchdog_timed_out": False,
    }
    passed = module.validate_policy_diagnostics(
        {"schema_version": 2, "records": [record]}, gate
    )
    assert passed["passed"]
    fail_closed = module.validate_policy_diagnostics(
        {
            "schema_version": 2,
            "records": [
                {**record, "observation": [0.0] * 48 + [0.0, 0.0, 1.0]}
            ],
        },
        gate,
    )
    assert not fail_closed["passed"]


def test_live_launch_selects_native_backend_and_exact_identity() -> None:
    source = LAUNCH.read_text(encoding="utf-8")
    assert 'choices=["fastlio2", "liosam"]' in source
    assert '"use_motion_deskew": "false"' in source
    assert '"odometry_topic": policy_odometry_topic' in source
    assert 'default_value="/slam/odom"' in source
    assert '"LIO-SAM uses /slam/policy_odom while /slam/odom remains "' in source
    assert '"expected_slam_confidence_backend"' in source
    assert '"expected_slam_confidence_calibration_id"' in source
    assert "the locomotion policy does not consume it" not in source
