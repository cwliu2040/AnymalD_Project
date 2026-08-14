from __future__ import annotations

from pathlib import Path

import torch

from anymal_locomotion.policies.slam_confidence_residual import (
    FrozenBackboneAuxIntentGaitActor,
    FrozenBackboneIntentGaitActor,
)


ROOT = Path(__file__).resolve().parents[1]


def _actor() -> FrozenBackboneIntentGaitActor:
    torch.manual_seed(11)
    return FrozenBackboneIntentGaitActor(
        observation_dim=51,
        num_actions=12,
        backbone_hidden_dims=[128, 128, 128],
        gait_hidden_dims=[64, 64],
        activation="elu",
    )


def test_zero_start_and_trained_healthy_path_are_exact_model1450() -> None:
    actor = _actor()
    observation = torch.randn(8, 51)
    observation[:, 48:51] = torch.tensor([1.0, 1.0, 0.0])
    expected = actor.backbone(observation[:, :48])
    torch.testing.assert_close(actor(observation), expected, rtol=0.0, atol=0.0)

    with torch.no_grad():
        actor.gait_head[-1].bias.fill_(0.7)
    torch.testing.assert_close(actor(observation), expected, rtol=0.0, atol=0.0)


def test_intent_coordinate_reaches_exact_frozen_safe_command_action() -> None:
    actor = _actor()
    observation = torch.randn(8, 51)
    observation[:, 9:12] = torch.tensor([1.5, -0.4, 0.3])
    observation[:, 48:51] = torch.tensor([0.0, 0.0, 1.0])
    safe_observation = observation[:, :48].clone()
    safe_observation[:, 9:12] = 0.0
    expected = actor.backbone(safe_observation)

    with torch.no_grad():
        actor.gait_head[-1].bias[0] = 20.0
    torch.testing.assert_close(actor(observation), expected, rtol=1.0e-5, atol=1.0e-6)


def test_zero_intent_coordinate_has_nonzero_learning_gradient() -> None:
    actor = _actor()
    observation = torch.randn(8, 51)
    observation[:, 9:12] = torch.tensor([1.5, 0.0, 0.0])
    observation[:, 48:51] = torch.tensor([0.0, 0.0, 1.0])
    actor(observation).sum().backward()
    gradient = actor.gait_head[-1].bias.grad
    assert gradient is not None
    assert gradient[0].abs().item() > 0.0


def test_intent_logit_gain_changes_resolution_without_changing_zero_start() -> None:
    actor = _actor()
    amplified = FrozenBackboneIntentGaitActor(
        observation_dim=51,
        num_actions=12,
        backbone_hidden_dims=[128, 128, 128],
        gait_hidden_dims=[64, 64],
        activation="elu",
        intent_logit_gain=20.0,
    )
    amplified.load_state_dict(actor.state_dict())
    observation = torch.randn(4, 51)
    observation[:, 48:51] = torch.tensor([0.0, 0.0, 1.0])
    torch.testing.assert_close(
        actor(observation), amplified(observation), rtol=0.0, atol=0.0
    )
    with torch.no_grad():
        actor.gait_head[-1].bias[0] = 0.01
        amplified.gait_head[-1].bias[0] = 0.01
    base_delta = actor(observation) - actor.backbone(observation[:, :48])
    amplified_delta = amplified(observation) - amplified.backbone(observation[:, :48])
    assert torch.linalg.vector_norm(amplified_delta) > 10.0 * torch.linalg.vector_norm(
        base_delta
    )


def test_intent_gait_task_and_runner_are_separate_from_failed_c_v1() -> None:
    tasks = (
        ROOT
        / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion"
        / "velocity/config/anymal_d/__init__.py"
    ).read_text(encoding="utf-8")
    agent = (
        ROOT
        / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion"
        / "velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py"
    ).read_text(encoding="utf-8")
    assert "SlamConfidence-IntentGait-v0" in tasks
    assert "AnymalDLocomotionSlamConfidenceIntentGaitPPORunnerCfg" in tasks
    assert "SlamConfidence-IntentGaitGain20-v0" in tasks
    assert "SlamConfidenceIntentGaitActorCriticCfg" in agent


def test_aux_intent_is_separate_from_four_coordinate_gait_head() -> None:
    actor = FrozenBackboneAuxIntentGaitActor(
        observation_dim=51,
        num_actions=12,
        backbone_hidden_dims=[128, 128, 128],
        gait_hidden_dims=[64, 64],
        activation="elu",
    )
    assert actor.intent_head[-1].out_features == 1
    assert actor.gait_head[-1].out_features == 4
    observation = torch.randn(8, 51)
    observation[:, 48:51] = torch.tensor([1.0, 1.0, 0.0])
    expected = actor.backbone(observation[:, :48])
    with torch.no_grad():
        actor.intent_head[-1].bias.fill_(10.0)
        actor.gait_head[-1].bias.fill_(0.5)
    torch.testing.assert_close(actor(observation), expected, rtol=0.0, atol=0.0)


def test_aux_gait_coordinate_diagnostic_matches_bounded_head() -> None:
    actor = FrozenBackboneAuxIntentGaitActor(
        observation_dim=51,
        num_actions=12,
        backbone_hidden_dims=[128, 128, 128],
        gait_hidden_dims=[64, 64],
        activation="elu",
    )
    observation = torch.randn(8, 51)
    legacy_action = actor.backbone(observation[:, :48])
    coordinates = actor.gait_coordinates(observation, legacy_action)
    expected = actor.gait_parameter_limits * torch.tanh(
        actor.gait_head(torch.cat((observation, legacy_action), dim=-1))
    )
    torch.testing.assert_close(coordinates, expected)
    assert coordinates.shape == (8, 4)


def test_constrained_gait_never_amplifies_stride_or_action_delta() -> None:
    actor = FrozenBackboneAuxIntentGaitActor(
        observation_dim=51,
        num_actions=12,
        backbone_hidden_dims=[128, 128, 128],
        gait_hidden_dims=[64, 64],
        activation="elu",
        gait_parameter_limits=[0.6, 0.3, 0.2, 0.9],
        nonnegative_stride_smoothing=True,
    )
    observation = torch.randn(8, 51)
    legacy_action = actor.backbone(observation[:, :48])
    coordinates = actor.gait_coordinates(observation, legacy_action)
    assert torch.all(coordinates[:, 0] >= 0.0)
    assert torch.all(coordinates[:, 3] >= 0.0)
    coordinates.sum().backward()
    assert actor.gait_head[-1].bias.grad[3] > 0.0


def test_smoothing_gain_preserves_zero_start_and_increases_resolution() -> None:
    actor = FrozenBackboneAuxIntentGaitActor(
        observation_dim=51,
        num_actions=12,
        backbone_hidden_dims=[128, 128, 128],
        gait_hidden_dims=[64, 64],
        activation="elu",
        gait_parameter_limits=[0.6, 0.3, 0.2, 0.9],
        nonnegative_stride_smoothing=True,
        smoothing_logit_gain=10.0,
    )
    observation = torch.randn(8, 51)
    legacy_action = actor.backbone(observation[:, :48])
    assert torch.count_nonzero(actor.gait_coordinates(observation, legacy_action)) == 0
    with torch.no_grad():
        actor.gait_head[-1].bias[3] = 0.01
    smoothing = actor.gait_coordinates(observation, legacy_action)[:, 3]
    assert torch.all(smoothing > 0.08)


def test_phase_separated_stride_only_suppresses_valid_degradation() -> None:
    actor = FrozenBackboneAuxIntentGaitActor(
        observation_dim=51,
        num_actions=12,
        backbone_hidden_dims=[128, 128, 128],
        gait_hidden_dims=[64, 64],
        activation="elu",
        gait_parameter_limits=[0.6, 0.3, 0.2, 0.9],
        nonnegative_stride_smoothing=True,
        degraded_stride_min_scale=0.0,
        degraded_stride_confidence_low=0.2,
        degraded_stride_confidence_high=1.0,
        degraded_stride_age_ratio_max=0.45,
        degraded_stride_envelope_power=3.0,
    )
    observation = torch.randn(4, 51)
    observation[:, 48:51] = torch.tensor(
        [
            [0.2, 1.0, 0.30],  # valid degradation
            [0.5, 1.0, 0.30],  # valid recovery: older for its confidence
            [0.2, 0.0, 0.50],  # invalid stop
            [1.0, 1.0, 0.00],  # healthy
        ]
    )
    intent_action = actor.backbone(observation[:, :48])
    with torch.no_grad():
        actor.gait_head[-1].bias[0] = 0.5
    stride = actor.gait_coordinates(observation, intent_action)[:, 0]
    assert stride[0].item() == 0.0
    torch.testing.assert_close(stride[1:], stride[1].expand(3))
    assert stride[1].item() > 0.0


def test_invalid_phase_uses_safe_backbone_without_gait_delta() -> None:
    actor = FrozenBackboneAuxIntentGaitActor(
        observation_dim=51,
        num_actions=12,
        backbone_hidden_dims=[128, 128, 128],
        gait_hidden_dims=[64, 64],
        activation="elu",
        gait_parameter_limits=[0.6, 0.3, 0.2, 0.9],
        nonnegative_stride_smoothing=True,
        suppress_gait_when_tracking_invalid=True,
    )
    observation = torch.randn(8, 51)
    observation[:, 9:12] = torch.tensor([0.0, -1.5, 0.0])
    observation[:, 48:51] = torch.tensor([1.0, 0.0, 0.6])
    safe_observation = observation[:, :48].clone()
    safe_observation[:, 9:12] = 0.0
    with torch.no_grad():
        actor.intent_head[-1].bias.fill_(10.0)
        actor.gait_head[-1].bias.fill_(0.7)
    torch.testing.assert_close(
        actor(observation),
        actor.backbone(safe_observation),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_safe_scale_envelope_attenuates_but_keeps_valid_degraded_gait() -> None:
    actor = FrozenBackboneAuxIntentGaitActor(
        observation_dim=51,
        num_actions=12,
        backbone_hidden_dims=[128, 128, 128],
        gait_hidden_dims=[64, 64],
        activation="elu",
        suppress_gait_when_tracking_invalid=True,
        gait_delta_safe_scale_power=3.0,
    )
    observation = torch.randn(2, 51)
    observation[:, 9:12] = torch.tensor([0.0, -1.5, 0.0])
    observation[:, 48:51] = torch.tensor(
        [[0.6, 1.0, 0.1], [0.6, 0.0, 0.6]]
    )
    with torch.no_grad():
        actor.intent_head[-1].bias.fill_(10.0)
        actor.gait_head[-1].bias[1] = 0.7
    safe_observation = observation[:, :48].clone()
    safe_observation[:, 9:12] *= torch.tensor([0.5, 0.0]).unsqueeze(-1)
    safe_action = actor.backbone(safe_observation)
    output = actor(observation)
    assert torch.linalg.vector_norm(output[0] - safe_action[0]) > 0.0
    torch.testing.assert_close(output[1], safe_action[1], rtol=1.0e-5, atol=1.0e-6)


def test_aux_intent_blend_has_configurable_policy_coordinate_bound() -> None:
    actor = FrozenBackboneAuxIntentGaitActor(
        observation_dim=51,
        num_actions=12,
        backbone_hidden_dims=[128, 128, 128],
        gait_hidden_dims=[64, 64],
        activation="elu",
        intent_blend_max=0.8,
    )
    observation = torch.randn(4, 51)
    legacy_action = actor.backbone(observation[:, :48])
    with torch.no_grad():
        actor.intent_head[-1].bias.fill_(10.0)
    torch.testing.assert_close(
        actor.intent_blend(observation, legacy_action),
        torch.full((4, 1), 0.8),
    )


def test_aux_intent_algorithm_is_runtime_independent_and_registered() -> None:
    algorithm = (
        ROOT
        / "source/anymal_locomotion/anymal_locomotion/algorithms"
        / "slam_confidence_intent_ppo.py"
    ).read_text(encoding="utf-8")
    tasks = (
        ROOT
        / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion"
        / "velocity/config/anymal_d/__init__.py"
    ).read_text(encoding="utf-8")
    assert 'loss_dict["intent_auxiliary"]' in algorithm
    assert "actor.intent_head.parameters()" in algorithm
    assert "rclpy" not in algorithm
    assert "SlamConfidence-AuxIntentGait-v0" in tasks


def test_estimator_closed_loop_training_is_explicit_and_fail_closed() -> None:
    wrapper = (
        ROOT
        / "source/anymal_locomotion/anymal_locomotion"
        / "velocity_estimator_training.py"
    ).read_text(encoding="utf-8")
    train = (ROOT / "scripts/rsl_rl/train.py").read_text(encoding="utf-8")
    assert "VelocityEstimatorTrainingWrapper" in train
    assert "policy_observation_velocity_estimator" in train
    assert "actions[~self.ready] = 0.0" in wrapper
    assert "policy_observation[self.ready, 0:3]" in wrapper
    assert "rclpy" not in wrapper
    tasks = (
        ROOT
        / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion"
        / "velocity/config/anymal_d/__init__.py"
    ).read_text(encoding="utf-8")
    assert "SlamConfidence-EstimatorRobustGait-v0" in tasks
    assert "SlamConfidence-RecoverySafeGait-v0" in tasks
    assert "SlamConfidence-ConstrainedGait-v0" in tasks
    assert "SlamConfidence-AmplifiedSmoothing-v0" in tasks
    assert "SlamConfidence-PhaseSeparatedGait-v0" in tasks
    assert "SlamConfidence-DegradedSlip-v0" in tasks


def test_recovery_safe_gait_has_explicit_fall_and_recovery_costs() -> None:
    env_cfg = (
        ROOT
        / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion"
        / "velocity/config/anymal_d/flat_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "termination_penalty = RewTerm(func=isaac_mdp.is_terminated, weight=-200.0)" in env_cfg
    assert "confidence_recovery_ang_vel_xy_l2" in env_cfg
    assert "confidence_recovery_flat_orientation_l2" in env_cfg
    assert "confidence_degradation_feet_slide" in env_cfg
