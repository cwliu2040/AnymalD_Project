"""PPO gait adaptation plus an explicit safe-command gain target."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


class SlamConfidenceGaitPPO(PPO):
    """Train only the gait residual with PPO and the safe gain by supervision.

    The model1450 backbone and action noise stay frozen.  PPO gradients reach
    the fresh critic and bounded gait residual, so confidence-conditioned gait
    rewards can affect joint strategy.  A separate deterministic update moves
    the per-joint safe-command gain toward a reachable bounded target.
    """

    def __init__(
        self,
        *args,
        safe_gain_target: float = 0.8,
        safe_gain_learning_rate: float = 5.0e-3,
        safe_gain_num_epochs: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        actor = self.policy.actor
        if not hasattr(actor, "residual") or not hasattr(actor, "safe_command_gain"):
            raise TypeError("SlamConfidenceGaitPPO requires the safe-gait actor")
        if not 0.0 < safe_gain_target <= float(actor.safe_command_gain_limit):
            raise ValueError("safe_gain_target must be within the actor gain limit")
        if safe_gain_learning_rate <= 0.0 or safe_gain_num_epochs <= 0:
            raise ValueError("safe gain learning settings must be positive")
        self.safe_gain_target = float(safe_gain_target)
        self.safe_gain_num_epochs = int(safe_gain_num_epochs)

        gain = actor.safe_command_gain
        for group in self.optimizer.param_groups:
            group["params"] = [parameter for parameter in group["params"] if parameter is not gain]
        self.optimizer.add_param_group(
            {"params": [gain], "lr": float(safe_gain_learning_rate)}
        )

    def _set_residual_trainable(self, trainable: bool) -> None:
        for parameter in self.policy.actor.residual.parameters():
            parameter.requires_grad_(trainable)

    def _safe_gain_update(self) -> float:
        gain = self.policy.actor.safe_command_gain
        target = torch.full_like(gain, self.safe_gain_target)
        mean_loss = 0.0
        for _ in range(self.safe_gain_num_epochs):
            loss = torch.mean(torch.square(gain - target))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([gain], self.max_grad_norm)
            self.optimizer.step()
            mean_loss += float(loss.detach())
        return mean_loss / self.safe_gain_num_epochs

    def update(self) -> dict[str, float]:
        gain = self.policy.actor.safe_command_gain
        gain.requires_grad_(False)
        self._set_residual_trainable(True)
        loss_dict = super().update()

        self._set_residual_trainable(False)
        gain.requires_grad_(True)
        loss_dict["safe_gain_teacher"] = self._safe_gain_update()
        self._set_residual_trainable(True)
        return loss_dict
