"""PPO fine-tuning with a non-zero frozen-model behavior anchor."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


class BehaviorAnchoredPPO(PPO):
    """Apply the same model1450 action anchor to J1 and J2 after each PPO update."""

    def __init__(
        self,
        *args,
        behavior_anchor_coef: float = 0.25,
        behavior_anchor_num_epochs: int = 1,
        behavior_anchor_num_mini_batches: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self.policy, "reference_action"):
            raise TypeError("BehaviorAnchoredPPO requires a frozen behavior reference")
        if behavior_anchor_coef <= 0.0:
            raise ValueError("behavior_anchor_coef must remain positive")
        if behavior_anchor_num_epochs <= 0 or behavior_anchor_num_mini_batches <= 0:
            raise ValueError("behavior anchor epochs and mini-batches must be positive")
        self.behavior_anchor_coef = float(behavior_anchor_coef)
        self.behavior_anchor_num_epochs = int(behavior_anchor_num_epochs)
        self.behavior_anchor_num_mini_batches = int(behavior_anchor_num_mini_batches)

    def _behavior_anchor_update(self) -> float:
        observations = self.storage.observations["policy"].flatten(0, 1)
        batch_size = observations.shape[0]
        mini_batch_size = batch_size // self.behavior_anchor_num_mini_batches
        if mini_batch_size == 0:
            raise RuntimeError("behavior anchor mini-batch is empty")
        total_loss = 0.0
        updates = 0
        for _ in range(self.behavior_anchor_num_epochs):
            indices = torch.randperm(batch_size, device=observations.device)
            for mini_batch in range(self.behavior_anchor_num_mini_batches):
                start = mini_batch * mini_batch_size
                stop = (mini_batch + 1) * mini_batch_size
                observation = observations[indices[start:stop]]
                with torch.no_grad():
                    target = self.policy.reference_action(observation)
                prediction = self.policy.actor(observation)
                loss = torch.mean(torch.square(prediction - target))
                self.optimizer.zero_grad()
                (self.behavior_anchor_coef * loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                total_loss += float(loss.detach())
                updates += 1
        return total_loss / updates

    def update(self) -> dict[str, float]:
        losses = super().update()
        losses["behavior_anchor"] = self._behavior_anchor_update()
        return losses
