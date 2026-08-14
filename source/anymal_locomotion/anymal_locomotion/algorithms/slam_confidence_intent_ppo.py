"""PPO with a short-credit auxiliary update for locomotion intent."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


class SlamConfidenceIntentPPO(PPO):
    """Keep PPO gait learning while explicitly teaching the 1-D intent head."""

    def __init__(
        self,
        *args,
        intent_learning_rate: float = 5.0e-3,
        intent_num_epochs: int = 5,
        intent_num_mini_batches: int = 4,
        intent_loss_coef: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        actor = self.policy.actor
        if not getattr(actor, "intent_auxiliary", False) or not hasattr(
            actor, "intent_head"
        ):
            raise TypeError("SlamConfidenceIntentPPO requires an auxiliary intent actor")
        if intent_learning_rate <= 0.0 or intent_num_epochs <= 0:
            raise ValueError("intent learning rate and epochs must be positive")
        if intent_num_mini_batches <= 0 or intent_loss_coef <= 0.0:
            raise ValueError("intent mini-batches and loss coefficient must be positive")
        self.intent_num_epochs = int(intent_num_epochs)
        self.intent_num_mini_batches = int(intent_num_mini_batches)
        self.intent_loss_coef = float(intent_loss_coef)

        intent_parameters = list(actor.intent_head.parameters())
        intent_ids = {id(parameter) for parameter in intent_parameters}
        for group in self.optimizer.param_groups:
            group["params"] = [
                parameter
                for parameter in group["params"]
                if id(parameter) not in intent_ids
            ]
        self.optimizer.add_param_group(
            {"params": intent_parameters, "lr": float(intent_learning_rate)}
        )

    def _intent_update(self) -> float:
        actor = self.policy.actor
        observations = self.storage.observations["policy"].flatten(0, 1)
        batch_size = observations.shape[0]
        mini_batch_size = batch_size // self.intent_num_mini_batches
        if mini_batch_size == 0:
            raise RuntimeError("intent auxiliary mini-batch is empty")
        mean_loss = 0.0
        updates = 0
        for _ in range(self.intent_num_epochs):
            indices = torch.randperm(batch_size, device=observations.device)
            for mini_batch in range(self.intent_num_mini_batches):
                start = mini_batch * mini_batch_size
                stop = (mini_batch + 1) * mini_batch_size
                observation = observations[indices[start:stop]]
                confidence = observation[..., actor.confidence_offset]
                valid = observation[..., actor.confidence_offset + 1]
                safe_scale = torch.clamp(valid, 0.0, 1.0) * torch.clamp(
                    (confidence - 0.2) / 0.8,
                    0.0,
                    1.0,
                )
                severity = 1.0 - safe_scale
                prediction = actor.intent_blend(observation).squeeze(-1)
                squared_error = torch.square(1.0 - prediction)
                loss = torch.sum(severity * squared_error) / torch.clamp(
                    torch.sum(severity), min=1.0
                )
                self.optimizer.zero_grad()
                (self.intent_loss_coef * loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    actor.intent_head.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                mean_loss += float(loss.detach())
                updates += 1
        return mean_loss / updates

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict["intent_auxiliary"] = self._intent_update()
        return loss_dict
