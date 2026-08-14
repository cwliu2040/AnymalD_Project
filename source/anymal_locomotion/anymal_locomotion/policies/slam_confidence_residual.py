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


class FrozenBackboneGaitModeActor(torch.nn.Module):
    """Run model1450 on the externally governed legacy observation.

    State and dwell live in the environment/deployment command governor, not
    inside the neural network.  This keeps PPO minibatches and exported ONNX
    inference stateless while preserving the exact 51-D observation contract.
    """

    __constants__ = ["in_features", "legacy_observation_dim"]

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        backbone_hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        legacy_observation_dim: int = 48,
    ) -> None:
        super().__init__()
        if observation_dim != 51 or legacy_observation_dim != 48:
            raise ValueError("gait-mode actor requires the 48+3 observation contract")
        self.in_features = observation_dim
        self.legacy_observation_dim = legacy_observation_dim
        self.external_gait_mode_governor = True
        self.backbone = MLP(
            legacy_observation_dim,
            num_actions,
            backbone_hidden_dims,
            activation,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def __getitem__(self, index: int) -> FrozenBackboneGaitModeActor:
        if index != 0:
            raise IndexError(index)
        return self

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.backbone(observation[..., : self.legacy_observation_dim])


class FrozenBackboneStructuredGaitActor(torch.nn.Module):
    """Learn four interpretable gait coordinates around frozen model1450.

    The head cannot emit an arbitrary 12-D residual.  It controls stride
    attenuation, crouch, stance width, and action smoothing through fixed
    canonical joint-space bases. Positive stride modulation attenuates the
    formal action while a negative value can restore amplitude. Confidence
    severity gates the whole change,
    making the healthy path exactly model1450 for every trained head weight.
    """

    __constants__ = [
        "in_features",
        "legacy_observation_dim",
        "confidence_offset",
        "previous_action_offset",
        "gait_parameter_dim",
    ]

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        backbone_hidden_dims: tuple[int, ...] | list[int],
        gait_hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        legacy_observation_dim: int = 48,
        confidence_offset: int = 48,
        previous_action_offset: int = 36,
        gait_parameter_limits: tuple[float, ...] | list[float] = (0.6, 0.3, 0.2, 0.5),
    ) -> None:
        super().__init__()
        if observation_dim != 51 or legacy_observation_dim != 48:
            raise ValueError("structured gait actor requires the 48+3 observation contract")
        if num_actions != 12 or confidence_offset != 48 or previous_action_offset != 36:
            raise ValueError("structured gait actor requires canonical 12-D action offsets")
        if len(gait_parameter_limits) != 4 or any(
            limit <= 0.0 for limit in gait_parameter_limits
        ):
            raise ValueError("four positive gait parameter limits are required")
        if not gait_hidden_dims:
            raise ValueError("gait_hidden_dims must not be empty")

        self.in_features = observation_dim
        self.legacy_observation_dim = legacy_observation_dim
        self.confidence_offset = confidence_offset
        self.previous_action_offset = previous_action_offset
        self.gait_parameter_dim = 4
        self.backbone = MLP(
            legacy_observation_dim,
            num_actions,
            backbone_hidden_dims,
            activation,
        )
        self.gait_head = MLP(
            observation_dim + num_actions,
            self.gait_parameter_dim,
            gait_hidden_dims,
            activation,
        )
        final_layer = self.gait_head[-1]
        if not isinstance(final_layer, torch.nn.Linear):
            raise TypeError("structured gait head must end with a linear layer")
        torch.nn.init.zeros_(final_layer.weight)
        torch.nn.init.zeros_(final_layer.bias)

        self.register_buffer(
            "gait_parameter_limits",
            torch.tensor(gait_parameter_limits, dtype=torch.float32),
        )
        self.register_buffer(
            "crouch_basis",
            torch.tensor(
                [0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "stance_width_basis",
            torch.tensor(
                [1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=torch.float32,
            ),
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def __getitem__(self, index: int) -> FrozenBackboneStructuredGaitActor:
        if index != 0:
            raise IndexError(index)
        return self

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy_action = self.backbone(observation[..., : self.legacy_observation_dim])
        gait_coordinates = self.gait_parameter_limits * torch.tanh(
            self.gait_head(torch.cat((observation, legacy_action), dim=-1))
        )
        previous_action = observation[
            ...,
            self.previous_action_offset : self.previous_action_offset + 12,
        ]
        structured_delta = (
            gait_coordinates[..., 0:1] * (-legacy_action)
            + gait_coordinates[..., 1:2] * self.crouch_basis
            + gait_coordinates[..., 2:3] * self.stance_width_basis
            + gait_coordinates[..., 3:4] * (previous_action - legacy_action)
        )
        confidence = observation[..., self.confidence_offset]
        valid = observation[..., self.confidence_offset + 1]
        safe_scale = torch.clamp(valid, 0.0, 1.0) * torch.clamp(
            (confidence - 0.2) / 0.8,
            0.0,
            1.0,
        )
        severity = 1.0 - safe_scale
        return legacy_action + severity.unsqueeze(-1) * structured_delta


class FrozenBackboneIntentGaitActor(FrozenBackboneStructuredGaitActor):
    """Add a PPO-controlled locomotion-intent blend to the gait coordinates.

    Coordinate zero blends between the frozen raw-command action and the same
    frozen backbone evaluated with the backend-neutral safe command.  It starts
    at exact zero but has a non-zero boundary gradient.  The remaining four
    coordinates retain the structured balance-only action basis.
    """

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        backbone_hidden_dims: tuple[int, ...] | list[int],
        gait_hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        legacy_observation_dim: int = 48,
        confidence_offset: int = 48,
        previous_action_offset: int = 36,
        command_offset: int = 9,
        command_dimension: int = 3,
        intent_logit_gain: float = 1.0,
        gait_parameter_limits: tuple[float, ...] | list[float] = (
            1.0,
            0.6,
            0.3,
            0.2,
            0.5,
        ),
    ) -> None:
        if len(gait_parameter_limits) != 5:
            raise ValueError("intent gait actor requires five parameter limits")
        super().__init__(
            observation_dim=observation_dim,
            num_actions=num_actions,
            backbone_hidden_dims=backbone_hidden_dims,
            gait_hidden_dims=gait_hidden_dims,
            activation=activation,
            legacy_observation_dim=legacy_observation_dim,
            confidence_offset=confidence_offset,
            previous_action_offset=previous_action_offset,
            gait_parameter_limits=gait_parameter_limits[1:],
        )
        if command_offset != 9 or command_dimension != 3:
            raise ValueError("intent gait actor requires command offsets 9..11")
        if intent_logit_gain <= 0.0:
            raise ValueError("intent_logit_gain must be positive")
        self.gait_parameter_dim = 5
        self.command_offset = command_offset
        self.command_dimension = command_dimension
        self.intent_logit_gain = float(intent_logit_gain)
        self.policy_intent_blend = True
        self.gait_head = MLP(
            observation_dim + num_actions,
            self.gait_parameter_dim,
            gait_hidden_dims,
            activation,
        )
        final_layer = self.gait_head[-1]
        if not isinstance(final_layer, torch.nn.Linear):
            raise TypeError("intent gait head must end with a linear layer")
        torch.nn.init.zeros_(final_layer.weight)
        torch.nn.init.zeros_(final_layer.bias)
        self.gait_parameter_limits = torch.tensor(
            gait_parameter_limits,
            dtype=torch.float32,
            device=self.gait_parameter_limits.device,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy_observation = observation[..., : self.legacy_observation_dim]
        legacy_action = self.backbone(legacy_observation)
        coordinates = self.gait_parameter_limits * torch.tanh(
            self.gait_head(torch.cat((observation, legacy_action), dim=-1))
        )
        intent_blend = torch.clamp(
            self.intent_logit_gain * coordinates[..., 0:1],
            min=0.0,
            max=1.0,
        )

        confidence = observation[..., self.confidence_offset]
        valid = observation[..., self.confidence_offset + 1]
        safe_scale = torch.clamp(valid, 0.0, 1.0) * torch.clamp(
            (confidence - 0.2) / 0.8,
            0.0,
            1.0,
        )
        safe_observation = legacy_observation.clone()
        safe_observation[
            ...,
            self.command_offset : self.command_offset + self.command_dimension,
        ] *= safe_scale.unsqueeze(-1)
        safe_action = self.backbone(safe_observation)
        intent_action = legacy_action + intent_blend * (safe_action - legacy_action)

        previous_action = observation[
            ...,
            self.previous_action_offset : self.previous_action_offset + 12,
        ]
        gait_coordinates = coordinates[..., 1:]
        structured_delta = (
            gait_coordinates[..., 0:1] * (-intent_action)
            + gait_coordinates[..., 1:2] * self.crouch_basis
            + gait_coordinates[..., 2:3] * self.stance_width_basis
            + gait_coordinates[..., 3:4] * (previous_action - intent_action)
        )
        severity = 1.0 - safe_scale
        return intent_action + severity.unsqueeze(-1) * structured_delta


class FrozenBackboneAuxIntentGaitActor(FrozenBackboneStructuredGaitActor):
    """Separate one supervised intent coordinate from four PPO gait coordinates."""

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        backbone_hidden_dims: tuple[int, ...] | list[int],
        gait_hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        legacy_observation_dim: int = 48,
        confidence_offset: int = 48,
        previous_action_offset: int = 36,
        command_offset: int = 9,
        command_dimension: int = 3,
        gait_parameter_limits: tuple[float, ...] | list[float] = (0.6, 0.3, 0.2, 0.5),
        nonnegative_stride_smoothing: bool = False,
        smoothing_logit_gain: float = 1.0,
        degraded_stride_min_scale: float = 1.0,
        degraded_stride_confidence_low: float = 0.2,
        degraded_stride_confidence_high: float = 1.0,
        degraded_stride_age_ratio_max: float = 0.45,
        degraded_stride_envelope_power: float = 1.0,
        suppress_gait_when_tracking_invalid: bool = False,
        gait_delta_safe_scale_power: float = 0.0,
        intent_blend_max: float = 1.0,
    ) -> None:
        super().__init__(
            observation_dim=observation_dim,
            num_actions=num_actions,
            backbone_hidden_dims=backbone_hidden_dims,
            gait_hidden_dims=gait_hidden_dims,
            activation=activation,
            legacy_observation_dim=legacy_observation_dim,
            confidence_offset=confidence_offset,
            previous_action_offset=previous_action_offset,
            gait_parameter_limits=gait_parameter_limits,
        )
        if command_offset != 9 or command_dimension != 3:
            raise ValueError("aux intent actor requires command offsets 9..11")
        self.command_offset = command_offset
        self.command_dimension = command_dimension
        self.nonnegative_stride_smoothing = bool(nonnegative_stride_smoothing)
        if smoothing_logit_gain <= 0.0:
            raise ValueError("smoothing_logit_gain must be positive")
        self.smoothing_logit_gain = float(smoothing_logit_gain)
        if not 0.0 <= degraded_stride_min_scale <= 1.0:
            raise ValueError("degraded_stride_min_scale must be in [0, 1]")
        if degraded_stride_confidence_high <= degraded_stride_confidence_low:
            raise ValueError(
                "degraded_stride_confidence_high must exceed the low threshold"
            )
        self.degraded_stride_min_scale = float(degraded_stride_min_scale)
        self.degraded_stride_confidence_low = float(
            degraded_stride_confidence_low
        )
        self.degraded_stride_confidence_high = float(
            degraded_stride_confidence_high
        )
        if degraded_stride_age_ratio_max <= 0.0:
            raise ValueError("degraded_stride_age_ratio_max must be positive")
        self.degraded_stride_age_ratio_max = float(
            degraded_stride_age_ratio_max
        )
        if degraded_stride_envelope_power <= 0.0:
            raise ValueError("degraded_stride_envelope_power must be positive")
        self.degraded_stride_envelope_power = float(
            degraded_stride_envelope_power
        )
        self.suppress_gait_when_tracking_invalid = bool(
            suppress_gait_when_tracking_invalid
        )
        if gait_delta_safe_scale_power < 0.0:
            raise ValueError("gait_delta_safe_scale_power must be non-negative")
        self.gait_delta_safe_scale_power = float(gait_delta_safe_scale_power)
        if not 0.0 < intent_blend_max <= 1.0:
            raise ValueError("intent_blend_max must be in (0, 1]")
        self.intent_blend_max = float(intent_blend_max)
        self.policy_intent_blend = True
        self.intent_auxiliary = True
        self.intent_head = MLP(
            observation_dim + num_actions,
            1,
            gait_hidden_dims,
            activation,
        )
        final_layer = self.intent_head[-1]
        if not isinstance(final_layer, torch.nn.Linear):
            raise TypeError("intent head must end with a linear layer")
        torch.nn.init.zeros_(final_layer.weight)
        torch.nn.init.zeros_(final_layer.bias)

    def intent_blend(
        self,
        observation: torch.Tensor,
        legacy_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if legacy_action is None:
            legacy_action = self.backbone(
                observation[..., : self.legacy_observation_dim]
            )
        logits = self.intent_head(torch.cat((observation, legacy_action), dim=-1))
        return self.intent_blend_max * torch.clamp(
            torch.tanh(logits), min=0.0, max=1.0
        )

    def gait_coordinates(
        self,
        observation: torch.Tensor,
        intent_action: torch.Tensor,
    ) -> torch.Tensor:
        """Return the four bounded PPO gait coordinates for diagnostics/export."""
        gait_logits = self.gait_head(
            torch.cat((observation, intent_action), dim=-1)
        )
        coordinates = self.gait_parameter_limits * torch.tanh(gait_logits)
        if self.nonnegative_stride_smoothing:
            coordinates = torch.stack(
                (
                    torch.clamp(coordinates[..., 0], min=0.0),
                    coordinates[..., 1],
                    coordinates[..., 2],
                    self.gait_parameter_limits[3]
                    * torch.clamp(
                        self.smoothing_logit_gain
                        * torch.tanh(gait_logits[..., 3]),
                        min=0.0,
                        max=1.0,
                    ),
                ),
                dim=-1,
            )
        if self.degraded_stride_min_scale < 1.0:
            confidence = observation[..., self.confidence_offset]
            valid = observation[..., self.confidence_offset + 1]
            normalized_age = observation[..., self.confidence_offset + 2]
            recovery_separating_ramp = torch.clamp(
                (confidence - self.degraded_stride_confidence_low)
                / (
                    self.degraded_stride_confidence_high
                    - self.degraded_stride_confidence_low
                ),
                min=0.0,
                max=1.0,
            ).pow(self.degraded_stride_envelope_power)
            valid_envelope = self.degraded_stride_min_scale + (
                1.0 - self.degraded_stride_min_scale
            ) * recovery_separating_ramp
            # A live quality loss accumulates age more slowly than confidence
            # falls, while post-outage recovery starts older and gets fresher.
            # This separates the two directions using only the deployable
            # confidence contract, without simulator phase or ground truth.
            degrading = (valid > 0.5) & (
                normalized_age
                <= self.degraded_stride_age_ratio_max
                * torch.clamp(1.0 - confidence, min=0.0)
                + 1.0e-6
            )
            stride_envelope = torch.where(
                degrading,
                valid_envelope,
                torch.ones_like(valid_envelope),
            )
            coordinates = torch.cat(
                (
                    coordinates[..., 0:1] * stride_envelope.unsqueeze(-1),
                    coordinates[..., 1:],
                ),
                dim=-1,
            )
        return coordinates

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy_observation = observation[..., : self.legacy_observation_dim]
        legacy_action = self.backbone(legacy_observation)
        confidence = observation[..., self.confidence_offset]
        valid = observation[..., self.confidence_offset + 1]
        safe_scale = torch.clamp(valid, 0.0, 1.0) * torch.clamp(
            (confidence - 0.2) / 0.8,
            0.0,
            1.0,
        )
        safe_observation = legacy_observation.clone()
        safe_observation[
            ...,
            self.command_offset : self.command_offset + self.command_dimension,
        ] *= safe_scale.unsqueeze(-1)
        safe_action = self.backbone(safe_observation)
        intent_action = legacy_action + self.intent_blend(
            observation, legacy_action
        ) * (safe_action - legacy_action)

        gait_coordinates = self.gait_coordinates(observation, intent_action)
        previous_action = observation[
            ...,
            self.previous_action_offset : self.previous_action_offset + 12,
        ]
        structured_delta = (
            gait_coordinates[..., 0:1] * (-intent_action)
            + gait_coordinates[..., 1:2] * self.crouch_basis
            + gait_coordinates[..., 2:3] * self.stance_width_basis
            + gait_coordinates[..., 3:4] * (previous_action - intent_action)
        )
        gait_gate = (
            torch.clamp(valid, 0.0, 1.0)
            if self.suppress_gait_when_tracking_invalid
            else torch.ones_like(valid)
        )
        if self.gait_delta_safe_scale_power > 0.0:
            gait_gate = gait_gate * torch.clamp(
                safe_scale, 0.0, 1.0
            ).pow(self.gait_delta_safe_scale_power)
        return (
            intent_action
            + ((1.0 - safe_scale) * gait_gate).unsqueeze(-1) * structured_delta
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


class SlamConfidenceGaitModeActorCritic(ActorCritic):
    """Fresh critic with a frozen model1450 and external gait-mode governor."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        legacy_observation_dim: int = 48,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs.get("state_dependent_std", False):
            raise ValueError("gait-mode actor requires state-independent action noise")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        num_actor_obs = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        self.actor = FrozenBackboneGaitModeActor(
            observation_dim=num_actor_obs,
            num_actions=num_actions,
            backbone_hidden_dims=kwargs.get("actor_hidden_dims", (128, 128, 128)),
            activation=kwargs.get("activation", "elu"),
            legacy_observation_dim=legacy_observation_dim,
        )
        if hasattr(self, "std"):
            self.std.requires_grad_(False)
        elif hasattr(self, "log_std"):
            self.log_std.requires_grad_(False)
        print(f"Confidence gait-mode actor: {self.actor}")


class SlamConfidenceStructuredGaitActorCritic(ActorCritic):
    """Fresh critic plus a frozen model1450 and four-coordinate gait head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        gait_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        gait_parameter_limits: tuple[float, ...] | list[float] = (0.6, 0.3, 0.2, 0.5),
        legacy_observation_dim: int = 48,
        confidence_offset: int = 48,
        previous_action_offset: int = 36,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs.get("state_dependent_std", False):
            raise ValueError("structured gait actor requires state-independent action noise")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        num_actor_obs = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        self.actor = FrozenBackboneStructuredGaitActor(
            observation_dim=num_actor_obs,
            num_actions=num_actions,
            backbone_hidden_dims=kwargs.get("actor_hidden_dims", (128, 128, 128)),
            gait_hidden_dims=gait_hidden_dims,
            activation=kwargs.get("activation", "elu"),
            legacy_observation_dim=legacy_observation_dim,
            confidence_offset=confidence_offset,
            previous_action_offset=previous_action_offset,
            gait_parameter_limits=gait_parameter_limits,
        )
        if hasattr(self, "std"):
            self.std.requires_grad_(False)
        elif hasattr(self, "log_std"):
            self.log_std.requires_grad_(False)
        print(f"Confidence structured-gait actor: {self.actor}")


class SlamConfidenceIntentGaitActorCritic(ActorCritic):
    """Fresh critic plus frozen model1450 and five-coordinate intent/gait head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        gait_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        gait_parameter_limits: tuple[float, ...] | list[float] = (1.0, 0.6, 0.3, 0.2, 0.5),
        legacy_observation_dim: int = 48,
        confidence_offset: int = 48,
        previous_action_offset: int = 36,
        command_offset: int = 9,
        command_dimension: int = 3,
        intent_logit_gain: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs.get("state_dependent_std", False):
            raise ValueError("intent gait actor requires state-independent action noise")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        num_actor_obs = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        self.actor = FrozenBackboneIntentGaitActor(
            observation_dim=num_actor_obs,
            num_actions=num_actions,
            backbone_hidden_dims=kwargs.get("actor_hidden_dims", (128, 128, 128)),
            gait_hidden_dims=gait_hidden_dims,
            activation=kwargs.get("activation", "elu"),
            legacy_observation_dim=legacy_observation_dim,
            confidence_offset=confidence_offset,
            previous_action_offset=previous_action_offset,
            command_offset=command_offset,
            command_dimension=command_dimension,
            intent_logit_gain=intent_logit_gain,
            gait_parameter_limits=gait_parameter_limits,
        )
        if hasattr(self, "std"):
            self.std.requires_grad_(False)
        elif hasattr(self, "log_std"):
            self.log_std.requires_grad_(False)
        print(f"Confidence intent-gait actor: {self.actor}")


class SlamConfidenceAuxIntentGaitActorCritic(ActorCritic):
    """Fresh critic with separate auxiliary intent and PPO gait heads."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        gait_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        gait_parameter_limits: tuple[float, ...] | list[float] = (0.6, 0.3, 0.2, 0.5),
        legacy_observation_dim: int = 48,
        confidence_offset: int = 48,
        previous_action_offset: int = 36,
        command_offset: int = 9,
        command_dimension: int = 3,
        nonnegative_stride_smoothing: bool = False,
        smoothing_logit_gain: float = 1.0,
        degraded_stride_min_scale: float = 1.0,
        degraded_stride_confidence_low: float = 0.2,
        degraded_stride_confidence_high: float = 1.0,
        degraded_stride_age_ratio_max: float = 0.45,
        degraded_stride_envelope_power: float = 1.0,
        suppress_gait_when_tracking_invalid: bool = False,
        gait_delta_safe_scale_power: float = 0.0,
        intent_blend_max: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs.get("state_dependent_std", False):
            raise ValueError("aux intent gait actor requires state-independent noise")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        num_actor_obs = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        self.actor = FrozenBackboneAuxIntentGaitActor(
            observation_dim=num_actor_obs,
            num_actions=num_actions,
            backbone_hidden_dims=kwargs.get("actor_hidden_dims", (128, 128, 128)),
            gait_hidden_dims=gait_hidden_dims,
            activation=kwargs.get("activation", "elu"),
            legacy_observation_dim=legacy_observation_dim,
            confidence_offset=confidence_offset,
            previous_action_offset=previous_action_offset,
            command_offset=command_offset,
            command_dimension=command_dimension,
            gait_parameter_limits=gait_parameter_limits,
            nonnegative_stride_smoothing=nonnegative_stride_smoothing,
            smoothing_logit_gain=smoothing_logit_gain,
            degraded_stride_min_scale=degraded_stride_min_scale,
            degraded_stride_confidence_low=degraded_stride_confidence_low,
            degraded_stride_confidence_high=degraded_stride_confidence_high,
            degraded_stride_age_ratio_max=degraded_stride_age_ratio_max,
            degraded_stride_envelope_power=degraded_stride_envelope_power,
            suppress_gait_when_tracking_invalid=suppress_gait_when_tracking_invalid,
            gait_delta_safe_scale_power=gait_delta_safe_scale_power,
            intent_blend_max=intent_blend_max,
        )
        if hasattr(self, "std"):
            self.std.requires_grad_(False)
        elif hasattr(self, "log_std"):
            self.log_std.requires_grad_(False)
        print(f"Confidence auxiliary intent-gait actor: {self.actor}")
