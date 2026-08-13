"""Confidence-gated residual policy with a frozen 48-D locomotion backbone."""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from tensordict import TensorDict


class FrozenBackboneResidualActor(torch.nn.Module):
    """Preserve the formal actor and learn only a bounded safety residual.

    The legacy actor consumes observations 0..47.  A small adapter consumes the
    complete 51-D observation plus the legacy action, but its contribution is
    multiplied by confidence severity.  Healthy confidence therefore follows
    the legacy actor exactly even after the adapter has been trained.
    """

    __constants__ = [
        "in_features",
        "legacy_observation_dim",
        "confidence_offset",
        "command_offset",
        "command_dimension",
        "residual_action_limit",
    ]

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        backbone_hidden_dims: tuple[int, ...] | list[int],
        residual_hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        legacy_observation_dim: int = 48,
        confidence_offset: int = 48,
        command_offset: int = 9,
        command_dimension: int = 3,
        residual_action_limit: float = 2.0,
        use_action_skip: bool = False,
    ) -> None:
        super().__init__()
        if observation_dim != 51:
            raise ValueError(
                "confidence residual actor requires the 51-D observation "
                f"contract, received {observation_dim}"
            )
        if legacy_observation_dim != 48 or confidence_offset != 48:
            raise ValueError("legacy and confidence offsets must follow the 48+3 contract")
        if command_offset != 9 or command_dimension != 3:
            raise ValueError("velocity command must follow the offsets 9..11 contract")
        if residual_action_limit <= 0.0:
            raise ValueError("residual_action_limit must be positive")
        if not residual_hidden_dims:
            raise ValueError("residual_hidden_dims must not be empty")

        # ``in_features`` plus ``__getitem__`` retain compatibility with the
        # Isaac Lab ONNX exporter, which queries actor[0].in_features.
        self.in_features = observation_dim
        self.legacy_observation_dim = legacy_observation_dim
        self.confidence_offset = confidence_offset
        self.command_offset = command_offset
        self.command_dimension = command_dimension
        self.residual_action_limit = float(residual_action_limit)
        self.backbone = MLP(
            legacy_observation_dim,
            num_actions,
            backbone_hidden_dims,
            activation,
        )
        self.residual = MLP(
            observation_dim + num_actions,
            num_actions,
            residual_hidden_dims,
            activation,
        )
        final_layer = self.residual[-1]
        if not isinstance(final_layer, torch.nn.Linear):
            raise TypeError("residual adapter must end with a linear layer")
        torch.nn.init.zeros_(final_layer.weight)
        torch.nn.init.zeros_(final_layer.bias)
        if use_action_skip:
            self.action_skip = torch.nn.Linear(num_actions, num_actions)
            torch.nn.init.zeros_(self.action_skip.weight)
            torch.nn.init.zeros_(self.action_skip.bias)
        else:
            self.action_skip = torch.nn.Identity()

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def __getitem__(self, index: int) -> FrozenBackboneResidualActor:
        """Expose the observation width expected by Isaac Lab's ONNX exporter."""
        if index != 0:
            raise IndexError(index)
        return self

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy_action = self.backbone(
            observation[..., : self.legacy_observation_dim]
        )
        adapter_input = torch.cat((observation, legacy_action), dim=-1)
        residual_logits = self.residual(adapter_input) + self.action_skip(legacy_action)
        residual = self.residual_action_limit * torch.tanh(residual_logits)

        confidence = observation[..., self.confidence_offset]
        valid = observation[..., self.confidence_offset + 1]
        safe_scale = torch.clamp(valid, 0.0, 1.0) * torch.clamp(
            (confidence - 0.2) / 0.8,
            0.0,
            1.0,
        )
        severity = 1.0 - safe_scale
        return legacy_action + severity.unsqueeze(-1) * residual


class FrozenBackboneSafeCommandActor(torch.nn.Module):
    """Blend from the formal action to its exact confidence-scaled-command action.

    The per-action blend gain is initialized to exact zero, so iteration zero
    is identical to model1450 for every value of the three new observations.
    Once the teacher drives the gain to its configured limit, the actor blends
    toward the same frozen formal backbone evaluated with command offsets
    9..11 multiplied by the backend-neutral confidence safe scale.  There is
    no learned approximation of the joint-space correction.
    """

    __constants__ = [
        "in_features",
        "legacy_observation_dim",
        "confidence_offset",
        "command_offset",
        "command_dimension",
        "safe_command_gain_limit",
    ]

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        backbone_hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        legacy_observation_dim: int = 48,
        confidence_offset: int = 48,
        command_offset: int = 9,
        command_dimension: int = 3,
        safe_command_gain_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if observation_dim != 51:
            raise ValueError(
                "safe-command actor requires the 51-D observation contract, "
                f"received {observation_dim}"
            )
        if legacy_observation_dim != 48 or confidence_offset != 48:
            raise ValueError("legacy and confidence offsets must follow the 48+3 contract")
        if command_offset != 9 or command_dimension != 3:
            raise ValueError("velocity command must follow the offsets 9..11 contract")
        if not 0.0 < safe_command_gain_limit <= 1.0:
            raise ValueError("safe_command_gain_limit must be in (0, 1]")

        self.in_features = observation_dim
        self.legacy_observation_dim = legacy_observation_dim
        self.confidence_offset = confidence_offset
        self.command_offset = command_offset
        self.command_dimension = command_dimension
        self.safe_command_gain_limit = float(safe_command_gain_limit)
        self.backbone = MLP(
            legacy_observation_dim,
            num_actions,
            backbone_hidden_dims,
            activation,
        )
        self.safe_command_gain = torch.nn.Parameter(torch.zeros(num_actions))
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def __getitem__(self, index: int) -> FrozenBackboneSafeCommandActor:
        """Expose the observation width expected by Isaac Lab's ONNX exporter."""
        if index != 0:
            raise IndexError(index)
        return self

    def safe_scale(self, observation: torch.Tensor) -> torch.Tensor:
        confidence = observation[..., self.confidence_offset]
        valid = observation[..., self.confidence_offset + 1]
        return torch.clamp(valid, 0.0, 1.0) * torch.clamp(
            (confidence - 0.2) / 0.8,
            0.0,
            1.0,
        )

    def safe_action(self, observation: torch.Tensor) -> torch.Tensor:
        safe_observation = observation[..., : self.legacy_observation_dim].clone()
        command_slice = slice(
            self.command_offset,
            self.command_offset + self.command_dimension,
        )
        safe_observation[..., command_slice] *= self.safe_scale(observation).unsqueeze(-1)
        return self.backbone(safe_observation)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy_action = self.backbone(
            observation[..., : self.legacy_observation_dim]
        )
        safe_action = self.safe_action(observation)
        gain = torch.clamp(
            self.safe_command_gain,
            0.0,
            self.safe_command_gain_limit,
        )
        return legacy_action + gain * (safe_action - legacy_action)


class FrozenBackboneSafeGaitActor(FrozenBackboneSafeCommandActor):
    """Add a bounded low-confidence gait residual to the exact safe-command path.

    Both the safe-command gain and residual output start at exact zero, so the
    complete iteration-0 actor remains bit-exact with model1450.  The command
    path supplies an explicit, reachable stop target; PPO can use the residual
    only when confidence severity is non-zero to improve balance, foot slip,
    and transition smoothness without changing healthy locomotion.
    """

    __constants__ = ["residual_action_limit"]

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        backbone_hidden_dims: tuple[int, ...] | list[int],
        residual_hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        residual_action_limit: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            observation_dim=observation_dim,
            num_actions=num_actions,
            backbone_hidden_dims=backbone_hidden_dims,
            activation=activation,
            **kwargs,
        )
        if residual_action_limit <= 0.0:
            raise ValueError("residual_action_limit must be positive")
        if not residual_hidden_dims:
            raise ValueError("residual_hidden_dims must not be empty")
        self.residual_action_limit = float(residual_action_limit)
        self.residual = MLP(
            observation_dim + num_actions,
            num_actions,
            residual_hidden_dims,
            activation,
        )
        final_layer = self.residual[-1]
        if not isinstance(final_layer, torch.nn.Linear):
            raise TypeError("gait residual must end with a linear layer")
        torch.nn.init.zeros_(final_layer.weight)
        torch.nn.init.zeros_(final_layer.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy_action = self.backbone(
            observation[..., : self.legacy_observation_dim]
        )
        safe_action = self.safe_action(observation)
        gain = torch.clamp(
            self.safe_command_gain,
            0.0,
            self.safe_command_gain_limit,
        )
        safe_scale = self.safe_scale(observation)
        # The formal zero-command actor is already the strongest settled-stop
        # baseline. Reserve the residual for deceleration and recovery only.
        transition_gate = 4.0 * safe_scale * (1.0 - safe_scale)
        adapter_input = torch.cat((observation, safe_action), dim=-1)
        residual = self.residual_action_limit * torch.tanh(
            self.residual(adapter_input)
        )
        return (
            legacy_action
            + gain * (safe_action - legacy_action)
            + transition_gate.unsqueeze(-1) * residual
        )


class SlamConfidenceResidualActorCritic(ActorCritic):
    """RSL-RL actor-critic with a frozen model1450-compatible actor backbone."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        residual_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        residual_action_limit: float = 2.0,
        legacy_observation_dim: int = 48,
        confidence_offset: int = 48,
        command_offset: int = 9,
        command_dimension: int = 3,
        use_action_skip: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs.get("state_dependent_std", False):
            raise ValueError("confidence residual actor requires state-independent action noise")
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        num_actor_obs = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        actor_hidden_dims = kwargs.get("actor_hidden_dims", (128, 128, 128))
        activation = kwargs.get("activation", "elu")
        self.actor = FrozenBackboneResidualActor(
            observation_dim=num_actor_obs,
            num_actions=num_actions,
            backbone_hidden_dims=actor_hidden_dims,
            residual_hidden_dims=residual_hidden_dims,
            activation=activation,
            legacy_observation_dim=legacy_observation_dim,
            confidence_offset=confidence_offset,
            command_offset=command_offset,
            command_dimension=command_dimension,
            residual_action_limit=residual_action_limit,
            use_action_skip=use_action_skip,
        )
        if hasattr(self, "std"):
            self.std.requires_grad_(False)
        elif hasattr(self, "log_std"):
            self.log_std.requires_grad_(False)
        print(f"Confidence residual actor: {self.actor}")


class SlamConfidenceSafeCommandActorCritic(ActorCritic):
    """RSL-RL actor-critic with an exact frozen-backbone safe-command path."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        legacy_observation_dim: int = 48,
        confidence_offset: int = 48,
        command_offset: int = 9,
        command_dimension: int = 3,
        safe_command_gain_limit: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs.get("state_dependent_std", False):
            raise ValueError("safe-command actor requires state-independent action noise")
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        num_actor_obs = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        actor_hidden_dims = kwargs.get("actor_hidden_dims", (128, 128, 128))
        activation = kwargs.get("activation", "elu")
        self.actor = FrozenBackboneSafeCommandActor(
            observation_dim=num_actor_obs,
            num_actions=num_actions,
            backbone_hidden_dims=actor_hidden_dims,
            activation=activation,
            legacy_observation_dim=legacy_observation_dim,
            confidence_offset=confidence_offset,
            command_offset=command_offset,
            command_dimension=command_dimension,
            safe_command_gain_limit=safe_command_gain_limit,
        )
        if hasattr(self, "std"):
            self.std.requires_grad_(False)
        elif hasattr(self, "log_std"):
            self.log_std.requires_grad_(False)
        print(f"Confidence safe-command actor: {self.actor}")


class SlamConfidenceSafeGaitActorCritic(ActorCritic):
    """Fresh critic with a frozen model1450, safe command, and gait adapter."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        residual_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        residual_action_limit: float = 0.25,
        legacy_observation_dim: int = 48,
        confidence_offset: int = 48,
        command_offset: int = 9,
        command_dimension: int = 3,
        safe_command_gain_limit: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs.get("state_dependent_std", False):
            raise ValueError("safe-gait actor requires state-independent action noise")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        num_actor_obs = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        self.actor = FrozenBackboneSafeGaitActor(
            observation_dim=num_actor_obs,
            num_actions=num_actions,
            backbone_hidden_dims=kwargs.get("actor_hidden_dims", (128, 128, 128)),
            residual_hidden_dims=residual_hidden_dims,
            activation=kwargs.get("activation", "elu"),
            residual_action_limit=residual_action_limit,
            legacy_observation_dim=legacy_observation_dim,
            confidence_offset=confidence_offset,
            command_offset=command_offset,
            command_dimension=command_dimension,
            safe_command_gain_limit=safe_command_gain_limit,
        )
        if hasattr(self, "std"):
            self.std.requires_grad_(False)
        elif hasattr(self, "log_std"):
            self.log_std.requires_grad_(False)
        print(f"Confidence safe-gait actor: {self.actor}")
