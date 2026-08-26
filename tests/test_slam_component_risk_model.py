from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/validation/fit_slam_component_risk_model.py"
SPEC = importlib.util.spec_from_file_location("component_risk_model", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frozen_config_is_offline_only_and_excludes_identity_leakage() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/slam_component_risk_model_v1.yaml").read_text(encoding="utf-8")
    )
    contract = config["runtime_feature_contract"]
    assert contract["backend_id_allowed"] is False
    assert contract["profile_id_allowed"] is False
    assert contract["ground_truth_allowed"] is False
    assert contract["future_label_allowed"] is False
    assert config["boundaries"]["offline_model_fit_authorized"] is True
    assert all(
        config["boundaries"][key] is False
        for key in (
            "ros_policy_wiring_authorized", "live_execution_authorized",
            "ppo_training_authorized", "default_switch_authorized",
            "physical_robot_authorized",
        )
    )


def test_state_feature_contract_reduces_observation_without_future_data() -> None:
    observation = np.arange(51, dtype=np.float64)
    features = MODULE.state_features(observation)
    assert features.shape == (len(MODULE.STATE_FEATURE_NAMES),)
    assert features[0:3].tolist() == [48.0, 49.0, 50.0]
    assert features[3:6].tolist() == [9.0, 10.0, 11.0]
    assert np.isclose(features[-2], np.sqrt(np.mean(np.square(observation[24:36]))))


def test_component_features_keep_translation_and_yaw_distinct() -> None:
    state = np.zeros(len(MODULE.STATE_FEATURE_NAMES), dtype=np.float64)
    state[0] = 0.6
    state[5] = 1.0
    preserve_yaw, names = MODULE.action_features(
        "component_action", state, np.asarray([0.75, 0.75, 1.0])
    )
    preserve_translation, _ = MODULE.action_features(
        "component_action", state, np.asarray([1.0, 1.0, 0.75])
    )
    assert names[-9:-7] == ("candidate_translation_scale", "candidate_yaw_scale")
    assert preserve_yaw[-9:-7].tolist() == [0.75, 1.0]
    assert preserve_translation[-9:-7].tolist() == [1.0, 0.75]
    assert not np.allclose(preserve_yaw, preserve_translation)


def test_ridge_fit_is_deterministic_and_probability_clipped() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = np.asarray([0.0, 0.25, 0.75, 1.0])
    first = MODULE.fit_ridge(x, y, 1.0)
    second = MODULE.fit_ridge(x, y, 1.0)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    prediction = MODULE.predict_ridge(first, np.asarray([[-100.0], [100.0]]))
    assert prediction.tolist() == [0.0, 1.0]
