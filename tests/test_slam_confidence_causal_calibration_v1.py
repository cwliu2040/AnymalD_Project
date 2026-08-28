from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/calibrate_slam_confidence_causal_joint_training_v2.py"
PROTOCOL = ROOT / "configs/slam_confidence_causal_calibration_v1.yaml"
V2 = ROOT / "configs/slam_confidence_causal_joint_training_v2.yaml"
REPORT = ROOT / "outputs/slam_confidence_causal_joint_training_v2/calibration_v1/summary.json"


def _module():
    spec = importlib.util.spec_from_file_location("causal_calibration", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scan_translation_proxy_is_zero_for_constant_velocity_and_positive_for_acceleration() -> None:
    module = _module()
    samples = [
        {
            "time_s": 0.02 * index,
            "actual_linear_velocity_body_mps": [0.1 * index, 0.0, 0.0],
            "command": [1.0, 0.0, 0.0],
        }
        for index in range(1, 8)
    ]
    assert module.scan_translation_rms(samples, 0.10) > 0.0
    for sample in samples:
        sample["actual_linear_velocity_body_mps"] = [1.0, 0.0, 0.0]
    assert module.scan_translation_rms(samples, 0.10) == 0.0


def test_nonnegative_fit_never_uses_negative_motion_coefficients() -> None:
    module = _module()
    x = np.asarray([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    coefficient, _ = module._nonnegative_fit(x, np.asarray([1.0, 2.0, -1.0, -2.0]))
    assert coefficient[0] > 0.0
    assert coefficient[1] == 0.0


def test_failed_calibration_stays_frozen_while_separate_exploratory_ppo_is_explicit() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    v2 = yaml.safe_load(V2.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert protocol["status"] == "retrospective_actual_backend_calibration_failed_execution_closed"
    assert protocol["execution"]["retrospective_analysis_authorized"] is False
    assert protocol["execution"]["fresh_simulation_collection_authorized"] is False
    assert protocol["execution"]["ppo_training_authorized"] is False
    assert v2["execution_gates"]["calibration_data_collection_authorized"] is False
    assert v2["execution_gates"]["ppo_training_authorized"] is False
    assert v2["calibration_gate"]["exploratory_user_override_after_motion_only_calibration_fail"] is True
    assert report["decision"] == "CALIBRATION_FAIL"
    assert report["proxy_parameters_frozen"] is False
    assert report["ppo_may_start"] is False
