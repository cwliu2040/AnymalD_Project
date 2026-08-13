"""Small temporal MLP used for supervised body-velocity estimation."""

from __future__ import annotations

import torch


class ProprioceptiveVelocityEstimator(torch.nn.Module):
    """Temporal GRU over 0.4-second history -> body-frame linear velocity."""

    def __init__(self, input_dimension: int = 740, step_dimension: int = 37) -> None:
        super().__init__()
        if input_dimension % step_dimension != 0:
            raise ValueError("input_dimension must be divisible by step_dimension")
        self.history_length = input_dimension // step_dimension
        self.step_dimension = step_dimension
        self.hidden_dimension = 128
        self.recurrent_layers = 2
        self.recurrent = torch.nn.GRU(
            input_size=step_dimension,
            hidden_size=self.hidden_dimension,
            num_layers=self.recurrent_layers,
            batch_first=True,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 3),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        sequence = history.reshape(-1, self.history_length, self.step_dimension)
        initial_state = sequence.new_zeros(
            (self.recurrent_layers, sequence.shape[0], self.hidden_dimension)
        )
        recurrent_output, _ = self.recurrent(sequence, initial_state)
        return self.head(recurrent_output[:, -1, :])
