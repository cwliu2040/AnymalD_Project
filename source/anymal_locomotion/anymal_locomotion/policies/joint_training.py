"""Full locomotion policy with a frozen model1450 behavior reference."""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from tensordict import TensorDict

from anymal_locomotion.joint_training_contract import (
    FULL_POLICY_OBSERVATION_DIM,
    LEGACY_OBSERVATION_DIM,
)


class AnchoredFullPolicyActorCritic(ActorCritic):
    """Trainable 1068-D actor plus a frozen 48-D model1450 reference.

    The reference is not an action residual and does not constrain the policy
    architecture. It supplies only a behavior-anchoring target during PPO
    fine-tuning so the original walking skill cannot disappear silently.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        legacy_observation_dim: int = LEGACY_OBSERVATION_DIM,
        full_observation_dim: int = FULL_POLICY_OBSERVATION_DIM,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs.get("state_dependent_std", False):
            raise ValueError("joint-training policy requires state-independent action noise")
        if kwargs.get("actor_obs_normalization", False):
            raise ValueError("model1450 anchoring requires unnormalized actor observations")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        num_actor_obs = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        if legacy_observation_dim != LEGACY_OBSERVATION_DIM:
            raise ValueError("legacy observation dimension is frozen at 48")
        if full_observation_dim != FULL_POLICY_OBSERVATION_DIM or num_actor_obs != full_observation_dim:
            raise ValueError(
                f"joint-training actor requires {FULL_POLICY_OBSERVATION_DIM}-D input, received {num_actor_obs}"
            )
        self.legacy_observation_dim = legacy_observation_dim
        self.full_observation_dim = full_observation_dim
        self.reference_actor = MLP(
            legacy_observation_dim,
            num_actions,
            kwargs.get("actor_hidden_dims", (128, 128, 128)),
            kwargs.get("activation", "elu"),
        )
        self.reference_actor.requires_grad_(False)
        self.reference_actor.eval()
        if hasattr(self, "std"):
            self.std.requires_grad_(False)
        elif hasattr(self, "log_std"):
            self.log_std.requires_grad_(False)
        else:
            raise RuntimeError("joint-training policy lacks state-independent action noise")

    def train(self, mode: bool = True):
        result = super().train(mode)
        self.reference_actor.eval()
        return result

    def reference_action(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.full_observation_dim:
            raise ValueError("behavior reference received a non-contract observation")
        return self.reference_actor(observations[..., : self.legacy_observation_dim])
