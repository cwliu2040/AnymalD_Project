from __future__ import annotations

from pathlib import Path

import torch
import yaml

from anymal_locomotion.causal_slam_dynamics_core import (
    causal_slam_transition,
    delayed_localization_advantage,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/slam_confidence_causal_joint_training_v2.yaml"
CONSTRAINED_PROTOCOL = ROOT / "configs/slam_confidence_constrained_joint_training_v3.yaml"
JOINT_MDP = ROOT / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion/velocity/config/anymal_d/mdp/joint_training.py"
FLAT_ENV = ROOT / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py"
TRAIN = ROOT / "scripts/rsl_rl/train.py"
PREFLIGHT = ROOT / "scripts/validation/preflight_slam_confidence_causal_joint_training_v2.py"


PARAMS = {
    "dt_s": 0.02,
    "translation_scale_m": 0.02,
    "rotation_scale_rad": 0.10,
    "degradation_rate_hz": 0.80,
    "recovery_rate_hz": 0.40,
    "invalid_enter_confidence": 0.20,
    "valid_exit_confidence": 0.40,
    "maximum_age_s": 1.0,
}


def test_same_state_different_scan_motion_changes_future_localization() -> None:
    initial = torch.tensor([[0.60, 1.0, 0.0], [0.60, 1.0, 0.0]])
    translation = torch.tensor([0.0, 0.08])
    rotation = torch.tensor([0.0, 0.40])
    support = torch.tensor([0.8, 0.8])
    state = initial
    for _ in range(25):
        state, _ = causal_slam_transition(
            state, translation, rotation, support, **PARAMS
        )
    assert state[0, 0] > initial[0, 0]
    assert state[1, 0] < initial[1, 0]
    assert state[0, 0] > state[1, 0]
    advantage = delayed_localization_advantage(
        state,
        initial,
        validity_bonus=0.50,
        normalized_age_penalty=0.25,
    )
    assert advantage[0] > 0.0
    assert advantage[1] < 0.0


def test_distortion_can_cause_validity_loss_and_age_growth() -> None:
    state = torch.tensor([[0.201, 1.0, 0.0]])
    state, diagnostics = causal_slam_transition(
        state,
        torch.tensor([0.10]),
        torch.tensor([0.50]),
        torch.tensor([0.7]),
        **PARAMS,
    )
    assert state[0, 1].item() == 0.0
    assert state[0, 2].item() > 0.0
    assert diagnostics["validity_lost"][0].item() == 1.0


def test_v2_wiring_removes_clock_from_actor_state_and_fails_closed() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    mdp_source = JOINT_MDP.read_text(encoding="utf-8")
    env_source = FLAT_ENV.read_text(encoding="utf-8")
    train_source = TRAIN.read_text(encoding="utf-8")
    preflight_source = PREFLIGHT.read_text(encoding="utf-8")
    assert protocol["runtime_actor_contract"]["localization_inputs"] == [
        "confidence", "tracking_valid", "normalized_age"
    ]
    assert protocol["causal_transition_contract"]["old_action_independent_clock_forbidden"] is True
    assert protocol["training_design"]["teacher_student_required_now"] is False
    assert protocol["nonlearning_preflight"]["execution_authorized"] is False
    assert protocol["execution_gates"]["nonlearning_isaac_wiring_preflight_authorized"] is False
    assert protocol["execution_gates"]["ppo_training_authorized"] is False
    assert protocol["training_budget"]["current_authorized_iterations_per_arm"] == 50
    assert protocol["training_execution"]["stage"] == "iteration_50"
    assert protocol["post_training_motion_audit"]["result"]["continuation_to_300_allowed"] is False
    assert protocol["nonlearning_preflight"]["result"]["ppo_constructed_or_run"] is False
    assert 'localization_mode == "causal"' in mdp_source
    assert "causal_slam_delayed_advantage" in env_source
    assert "_joint_training_tasks.update(_causal_joint_training_tasks)" in train_source
    assert "OnPolicyRunner" not in preflight_source
    assert "torch.optim" not in preflight_source
    assert "runner.learn" not in preflight_source


def test_v3_hard_action_constraint_result_is_fail_closed() -> None:
    protocol = yaml.safe_load(CONSTRAINED_PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["policy_constraint"]["deterministic_action_deviation_linf_limit"] == 0.05
    assert protocol["post_training_motion_audit"]["result"][
        "J2_minus_J1_all_profiles_continuation_safety_passed"
    ] is True
    assert protocol["post_training_motion_audit"]["result"]["continuation_to_300_allowed"] is False
    assert protocol["training_budget"]["execution_authorized"] is False
    assert protocol["execution_gates"]["ppo_training_authorized"] is False
