"""PPO critic plus explicit confidence-conditioned residual teacher."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


class SlamConfidenceTeacherPPO(PPO):
    """Train the residual actor toward a reachable neutral-action target.

    PPO continues to train the fresh critic.  The formal actor backbone and
    action noise remain frozen, while the residual adapter is updated from an
    explicit teacher target after every rollout.  This removes the long credit
    path between invalid-state velocity penalties and joint-position actions.
    """

    def __init__(
        self,
        *args,
        teacher_num_epochs: int = 5,
        teacher_num_mini_batches: int = 4,
        teacher_loss_coef: float = 1.0,
        teacher_target_limit_fraction: float = 0.95,
        teacher_target_mode: str = "nominal_action_zero",
        teacher_learning_rate: float | None = None,
        teacher_gain_loss_coef: float = 0.0,
        teacher_gain_target: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if teacher_num_epochs <= 0 or teacher_num_mini_batches <= 0:
            raise ValueError("teacher epochs and mini-batches must be positive")
        if teacher_loss_coef <= 0.0:
            raise ValueError("teacher_loss_coef must be positive")
        if not 0.0 < teacher_target_limit_fraction < 1.0:
            raise ValueError("teacher_target_limit_fraction must be in (0, 1)")
        if teacher_target_mode not in {
            "nominal_action_zero",
            "confidence_scaled_formal_command",
        }:
            raise ValueError("unsupported teacher_target_mode")
        actor = self.policy.actor
        if not hasattr(actor, "backbone") or not (
            hasattr(actor, "residual") or hasattr(actor, "safe_command_gain")
        ):
            raise TypeError(
                "SlamConfidenceTeacherPPO requires a residual or safe-command actor"
            )
        if teacher_learning_rate is not None and teacher_learning_rate <= 0.0:
            raise ValueError("teacher_learning_rate must be positive when set")
        if teacher_gain_loss_coef < 0.0:
            raise ValueError("teacher_gain_loss_coef must be non-negative")
        if not 0.0 < teacher_gain_target <= 1.0:
            raise ValueError("teacher_gain_target must be in (0, 1]")
        self.teacher_num_epochs = teacher_num_epochs
        self.teacher_num_mini_batches = teacher_num_mini_batches
        self.teacher_loss_coef = teacher_loss_coef
        self.teacher_target_limit_fraction = teacher_target_limit_fraction
        self.teacher_target_mode = teacher_target_mode
        self.teacher_learning_rate = teacher_learning_rate
        self.teacher_gain_loss_coef = teacher_gain_loss_coef
        self.teacher_gain_target = teacher_gain_target
        if hasattr(actor, "safe_command_gain_limit") and teacher_gain_target > float(
            actor.safe_command_gain_limit
        ):
            raise ValueError("teacher_gain_target exceeds actor safe-command limit")
        if teacher_learning_rate is not None:
            adapter_parameters = list(self._adapter_parameters())
            adapter_ids = {id(parameter) for parameter in adapter_parameters}
            for group in self.optimizer.param_groups:
                group["params"] = [
                    parameter
                    for parameter in group["params"]
                    if id(parameter) not in adapter_ids
                ]
            self.optimizer.add_param_group(
                {
                    "params": adapter_parameters,
                    "lr": teacher_learning_rate,
                }
            )

    def _adapter_parameters(self):
        for name, parameter in self.policy.actor.named_parameters():
            if (
                name.startswith("residual.")
                or name.startswith("action_skip.")
                or name == "safe_command_gain"
            ):
                yield parameter

    def _set_adapter_trainable(self, trainable: bool) -> None:
        for name, parameter in self.policy.actor.named_parameters():
            if (
                name.startswith("residual.")
                or name.startswith("action_skip.")
                or name == "safe_command_gain"
            ):
                parameter.requires_grad_(trainable)

    def _teacher_update(self) -> float:
        actor = self.policy.actor
        observations = self.storage.observations["policy"].flatten(0, 1)
        batch_size = observations.shape[0]
        mini_batch_size = batch_size // self.teacher_num_mini_batches
        if mini_batch_size == 0:
            raise RuntimeError("teacher mini-batch is empty")
        mean_loss = 0.0
        updates = 0

        for _ in range(self.teacher_num_epochs):
            indices = torch.randperm(batch_size, device=observations.device)
            for mini_batch in range(self.teacher_num_mini_batches):
                start = mini_batch * mini_batch_size
                stop = (mini_batch + 1) * mini_batch_size
                observation = observations[indices[start:stop]]
                with torch.no_grad():
                    legacy_action = actor.backbone(
                        observation[..., : actor.legacy_observation_dim]
                    )
                    target_limit = (
                        getattr(actor, "residual_action_limit", 1.0)
                        * self.teacher_target_limit_fraction
                    )
                    confidence = observation[..., actor.confidence_offset]
                    valid = observation[..., actor.confidence_offset + 1]
                    safe_scale = torch.clamp(valid, 0.0, 1.0) * torch.clamp(
                        (confidence - 0.2) / 0.8,
                        0.0,
                        1.0,
                    )
                    severity = 1.0 - safe_scale
                    if self.teacher_target_mode == "nominal_action_zero":
                        target_delta = -legacy_action
                    else:
                        safe_observation = observation[
                            ..., : actor.legacy_observation_dim
                        ].clone()
                        command_slice = slice(
                            actor.command_offset,
                            actor.command_offset + actor.command_dimension,
                        )
                        safe_observation[..., command_slice] *= safe_scale.unsqueeze(-1)
                        safe_action = actor.backbone(safe_observation)
                        # The actor multiplies its residual by severity.  Teach
                        # the pre-gate delta so the final action matches the
                        # formal actor evaluated at the safe command.
                        target_delta = (safe_action - legacy_action) / torch.clamp(
                            severity.unsqueeze(-1),
                            min=0.05,
                        )
                    target_delta = torch.clamp(
                        target_delta,
                        -target_limit,
                        target_limit,
                    )
                if hasattr(actor, "safe_command_gain"):
                    if self.teacher_target_mode != "confidence_scaled_formal_command":
                        raise ValueError(
                            "safe-command actor requires confidence-scaled formal target"
                        )
                    predicted_action = actor(observation)
                    target_action = legacy_action + self.teacher_gain_target * (
                        safe_action - legacy_action
                    )
                    squared_error = torch.mean(
                        torch.square(predicted_action - target_action),
                        dim=-1,
                    )
                else:
                    adapter_input = torch.cat((observation, legacy_action), dim=-1)
                    predicted_delta = actor.residual_action_limit * torch.tanh(
                        actor.residual(adapter_input) + actor.action_skip(legacy_action)
                    )
                    squared_error = torch.mean(
                        torch.square(predicted_delta - target_delta),
                        dim=-1,
                    )
                teacher_loss = torch.sum(severity * squared_error) / torch.clamp(
                    torch.sum(severity),
                    min=1.0,
                )
                if hasattr(actor, "safe_command_gain"):
                    effective_gain = torch.clamp(
                        actor.safe_command_gain,
                        0.0,
                        actor.safe_command_gain_limit,
                    )
                    teacher_loss = teacher_loss + self.teacher_gain_loss_coef * torch.mean(
                        torch.square(self.teacher_gain_target - effective_gain)
                    )

                self.optimizer.zero_grad()
                (self.teacher_loss_coef * teacher_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for name, parameter in actor.named_parameters()
                        if (
                            name.startswith("residual.")
                            or name.startswith("action_skip.")
                            or name == "safe_command_gain"
                        )
                        and parameter.requires_grad
                    ],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                mean_loss += float(teacher_loss.detach())
                updates += 1
        return mean_loss / updates

    def update(self) -> dict[str, float]:
        # Keep the actor outside the high-variance PPO surrogate.  The fresh
        # critic still uses PPO returns/value loss from the same rollouts.
        self._set_adapter_trainable(False)
        loss_dict = super().update()
        self._set_adapter_trainable(True)
        loss_dict["teacher_stop"] = self._teacher_update()
        return loss_dict
