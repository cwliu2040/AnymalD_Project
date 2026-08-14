from __future__ import annotations

from pathlib import Path

import torch

from anymal_locomotion.policies.slam_confidence_residual import (
    FrozenBackboneStructuredGaitActor,
)


ROOT = Path(__file__).resolve().parents[1]


def _actor() -> FrozenBackboneStructuredGaitActor:
    torch.manual_seed(7)
    return FrozenBackboneStructuredGaitActor(
        observation_dim=51,
        num_actions=12,
        backbone_hidden_dims=[128, 128, 128],
        gait_hidden_dims=[64, 64],
        activation="elu",
    )


def test_iteration_zero_and_trained_healthy_path_are_exact_backbone() -> None:
    actor = _actor()
    observation = torch.randn(16, 51)
    observation[:, 48:51] = torch.tensor([1.0, 1.0, 0.0])
    expected = actor.backbone(observation[:, :48])
    torch.testing.assert_close(actor(observation), expected, rtol=0.0, atol=0.0)

    with torch.no_grad():
        actor.gait_head[-1].bias.fill_(0.75)
    torch.testing.assert_close(actor(observation), expected, rtol=0.0, atol=0.0)


def test_head_emits_four_coordinates_not_an_arbitrary_joint_residual() -> None:
    actor = _actor()
    assert actor.gait_head[-1].out_features == 4
    assert actor.crouch_basis.shape == (12,)
    assert actor.stance_width_basis.shape == (12,)
    assert not any(parameter.requires_grad for parameter in actor.backbone.parameters())
    assert all(parameter.requires_grad for parameter in actor.gait_head.parameters())

    observation = torch.randn(4, 51)
    observation[:, 48:51] = torch.tensor([0.0, 0.0, 1.0])
    expected = actor.backbone(observation[:, :48])
    torch.testing.assert_close(actor(observation), expected, rtol=0.0, atol=0.0)

    with torch.no_grad():
        actor.gait_head[-1].bias[1] = 0.5
    delta = actor(observation) - expected
    coefficient = actor.gait_parameter_limits[1] * torch.tanh(torch.tensor(0.5))
    torch.testing.assert_close(
        delta,
        coefficient * actor.crouch_basis.expand_as(delta),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_structured_gait_task_runner_and_warm_start_are_registered() -> None:
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
    warm_start = (ROOT / "scripts/rsl_rl/train.py").read_text(encoding="utf-8")
    assert "SlamConfidence-StructuredGait-v0" in tasks
    assert "AnymalDLocomotionSlamConfidenceStructuredGaitPPORunnerCfg" in tasks
    assert "SlamConfidenceStructuredGaitActorCriticCfg" in agent
    assert 'name.startswith("actor.gait_head.")' in warm_start
    assert "frozen_backbone_structured_gait_warm_start" in warm_start
