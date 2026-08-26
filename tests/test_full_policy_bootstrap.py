from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="requires the Isaac Lab Python environment")

ROOT = Path(__file__).resolve().parents[1]
MODEL1450 = ROOT / "checkpoints/anymal_d_locomotion_v1/recovery_v0.4.0/model_1450.pt"

from anymal_locomotion.full_policy_bootstrap import (
    bootstrap_dense_actor_state,
    bootstrap_frozen_reference_actor_state,
    validate_fresh_optimizer_state,
    verify_dense_actor_output_parity,
)
from anymal_locomotion.joint_training_contract import FULL_POLICY_OBSERVATION_DIM
from anymal_locomotion.algorithms.behavior_anchored_ppo import BehaviorAnchoredPPO
from anymal_locomotion.policies.joint_training import AnchoredFullPolicyActorCritic
from anymal_locomotion.joint_training_motion_core import (
    angular_acceleration_l2,
    lidar_scan_rotation_distortion_l2,
    lidar_scan_translation_distortion_l2,
    linear_jerk_l2,
    localization_vulnerability,
)


class _DensePolicy(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 12),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.std = torch.nn.Parameter(torch.ones(12))


class _AnchoredDensePolicy(_DensePolicy):
    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim)
        self.reference_actor = torch.nn.Sequential(
            torch.nn.Linear(48, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 12),
        )
        self.reference_actor.requires_grad_(False)


def test_full_policy_bootstrap_preserves_actor_and_leaves_critic_fresh() -> None:
    torch.manual_seed(1450)
    source = _DensePolicy(48)
    target = _DensePolicy(FULL_POLICY_OBSERVATION_DIM)
    critic_before = {
        name: tensor.detach().clone()
        for name, tensor in target.state_dict().items()
        if name.startswith("critic.")
    }

    report = bootstrap_dense_actor_state(
        source.state_dict(),
        target.state_dict(),
        expected_target_input_dim=FULL_POLICY_OBSERVATION_DIM,
    )

    target_state = target.state_dict()
    assert report["target_actor_input_dimension"] == 1068
    assert report["new_input_columns"] == {"start": 48, "stop_exclusive": 1068}
    assert torch.equal(
        target_state["actor.0.weight"][:, :48],
        source.state_dict()["actor.0.weight"],
    )
    assert torch.count_nonzero(target_state["actor.0.weight"][:, 48:]).item() == 0
    for name, before in critic_before.items():
        assert torch.equal(target_state[name], before)


def test_full_policy_bootstrap_has_exact_model1450_output_parity() -> None:
    torch.manual_seed(27)
    source = _DensePolicy(48)
    target = _DensePolicy(FULL_POLICY_OBSERVATION_DIM)
    bootstrap_dense_actor_state(source.state_dict(), target.state_dict())
    legacy = torch.randn(64, 48)
    history = torch.randn(64, FULL_POLICY_OBSERVATION_DIM - 48)

    assert verify_dense_actor_output_parity(
        source.actor,
        target.actor,
        legacy,
        history,
        exact=True,
    )


def test_full_policy_bootstrap_rejects_architecture_drift() -> None:
    source = _DensePolicy(48)
    target = _DensePolicy(FULL_POLICY_OBSERVATION_DIM)
    wrong = copy.deepcopy(target.state_dict())
    wrong["actor.2.weight"] = torch.empty(64, 128)
    with pytest.raises(ValueError, match="shape mismatch"):
        bootstrap_dense_actor_state(source.state_dict(), wrong)

    with pytest.raises(ValueError, match="expected 1068"):
        bootstrap_dense_actor_state(
            source.state_dict(),
            _DensePolicy(51).state_dict(),
            expected_target_input_dim=1068,
        )


def test_actor_only_bootstrap_requires_fresh_optimizer() -> None:
    validate_fresh_optimizer_state({})
    with pytest.raises(RuntimeError, match="fresh optimizer"):
        validate_fresh_optimizer_state({0: {"step": torch.tensor(1)}})


def test_frozen_behavior_reference_is_exact_model1450_copy() -> None:
    torch.manual_seed(12)
    source = _DensePolicy(48)
    target = _AnchoredDensePolicy(FULL_POLICY_OBSERVATION_DIM)
    bootstrap_dense_actor_state(source.state_dict(), target.state_dict())
    copied = bootstrap_frozen_reference_actor_state(
        source.state_dict(), target.state_dict()
    )

    assert copied
    legacy = torch.randn(16, 48)
    with torch.no_grad():
        assert torch.equal(source.actor(legacy), target.reference_actor(legacy))
    assert not any(parameter.requires_grad for parameter in target.reference_actor.parameters())


def test_behavior_anchor_update_pulls_full_actor_toward_frozen_reference() -> None:
    torch.manual_seed(31)
    source = _DensePolicy(48)
    target = _AnchoredDensePolicy(FULL_POLICY_OBSERVATION_DIM)
    bootstrap_dense_actor_state(source.state_dict(), target.state_dict())
    bootstrap_frozen_reference_actor_state(source.state_dict(), target.state_dict())
    with torch.no_grad():
        target.actor[-1].bias.add_(0.25)
    observations = torch.randn(4, 8, FULL_POLICY_OBSERVATION_DIM)
    flat = observations.flatten(0, 1)
    with torch.no_grad():
        loss_before = torch.mean(
            torch.square(target.actor(flat) - target.reference_actor(flat[:, :48]))
        )
        reference_before = [value.clone() for value in target.reference_actor.parameters()]

    holder = SimpleNamespace(
        storage=SimpleNamespace(observations={"policy": observations}),
        policy=SimpleNamespace(
            actor=target.actor,
            reference_action=lambda value: target.reference_actor(value[..., :48]),
        ),
        optimizer=torch.optim.Adam(target.actor.parameters(), lr=1.0e-4),
        behavior_anchor_coef=1.0,
        behavior_anchor_num_epochs=1,
        behavior_anchor_num_mini_batches=4,
        max_grad_norm=1.0,
    )
    result = BehaviorAnchoredPPO._behavior_anchor_update(holder)
    with torch.no_grad():
        loss_after = torch.mean(
            torch.square(target.actor(flat) - target.reference_actor(flat[:, :48]))
        )

    assert result > 0.0
    assert loss_after < loss_before
    assert all(
        torch.equal(before, after)
        for before, after in zip(reference_before, target.reference_actor.parameters())
    )


def test_motion_objectives_preserve_constant_translation_and_yaw() -> None:
    localization = torch.tensor([[1.0, 1.0, 0.0], [0.2, 0.0, 1.0]])
    vulnerability = localization_vulnerability(localization, severity_gain=1.0)
    assert torch.equal(vulnerability, torch.tensor([1.0, 2.0]))
    zeros = torch.zeros(2, 3)
    constant_yaw = torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, -2.0]])

    assert torch.count_nonzero(angular_acceleration_l2(zeros, vulnerability)) == 0
    assert torch.count_nonzero(linear_jerk_l2(zeros, vulnerability)) == 0
    assert torch.count_nonzero(
        lidar_scan_translation_distortion_l2(zeros, vulnerability, 0.10)
    ) == 0
    assert torch.count_nonzero(
        lidar_scan_rotation_distortion_l2(constant_yaw, zeros, vulnerability, 0.10)
    ) == 0


def test_motion_objectives_penalize_roll_pitch_and_nonconstant_scan_motion() -> None:
    vulnerability = torch.tensor([1.0, 2.0])
    acceleration = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    angular_velocity = torch.tensor([[0.5, 0.0, 1.0], [0.5, 0.0, 1.0]])
    translation = lidar_scan_translation_distortion_l2(
        acceleration, vulnerability, 0.10
    )
    rotation = lidar_scan_rotation_distortion_l2(
        angular_velocity, acceleration, vulnerability, 0.10
    )
    assert translation[0] > 0.0 and rotation[0] > 0.0
    assert translation[1] == pytest.approx(2.0 * translation[0])
    assert rotation[1] == pytest.approx(2.0 * rotation[0])


def test_real_rsl_rl_behavior_anchored_ppo_completes_synthetic_update() -> None:
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict

    torch.manual_seed(1450)
    num_envs = 4
    rollout_steps = 4
    observation = TensorDict(
        {"policy": torch.randn(num_envs, FULL_POLICY_OBSERVATION_DIM)},
        batch_size=[num_envs],
    )
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = AnchoredFullPolicyActorCritic(
        observation,
        obs_groups,
        12,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    assert policy.std.requires_grad is False
    storage = RolloutStorage(
        "rl",
        num_envs,
        rollout_steps,
        observation,
        (12,),
        device="cpu",
    )
    algorithm = BehaviorAnchoredPPO(
        policy,
        storage,
        num_learning_epochs=1,
        num_mini_batches=2,
        learning_rate=1.0e-4,
        behavior_anchor_coef=0.25,
        behavior_anchor_num_epochs=1,
        behavior_anchor_num_mini_batches=2,
        schedule="fixed",
        device="cpu",
    )
    source_state = torch.load(
        MODEL1450, map_location="cpu", weights_only=False
    )["model_state_dict"]
    bootstrap_dense_actor_state(source_state, policy.state_dict())
    bootstrap_frozen_reference_actor_state(source_state, policy.state_dict())
    reference_before = [value.clone() for value in policy.reference_actor.parameters()]
    std_before = policy.std.clone()

    for _ in range(rollout_steps):
        algorithm.act(observation)
        next_observation = TensorDict(
            {"policy": torch.randn(num_envs, FULL_POLICY_OBSERVATION_DIM)},
            batch_size=[num_envs],
        )
        algorithm.process_env_step(
            next_observation,
            torch.randn(num_envs),
            torch.zeros(num_envs, dtype=torch.bool),
            {},
        )
        observation = next_observation
    algorithm.compute_returns(observation)
    losses = algorithm.update()

    assert "behavior_anchor" in losses
    assert all(math.isfinite(float(value)) for value in losses.values())
    assert storage.step == 0
    assert all(
        torch.equal(before, after)
        for before, after in zip(reference_before, policy.reference_actor.parameters())
    )
    assert torch.equal(std_before, policy.std)
