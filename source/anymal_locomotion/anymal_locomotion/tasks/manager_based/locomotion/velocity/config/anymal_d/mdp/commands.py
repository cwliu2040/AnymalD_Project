"""Project-owned velocity command distributions for robust locomotion."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from isaaclab.utils import configclass


class EdgeBiasedVelocityCommand(UniformVelocityCommand):
    """Mix the full command envelope with deployment-critical edge cases."""

    cfg: EdgeBiasedVelocityCommandCfg

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if not 0.0 <= self.cfg.edge_probability <= 1.0:
            raise ValueError("edge_probability must be in [0, 1]")

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0 or self.cfg.edge_probability == 0.0:
            return
        edge_mask = (
            torch.rand(ids.numel(), device=self.device)
            < self.cfg.edge_probability
        )
        edge_ids = ids[edge_mask]
        if edge_ids.numel() == 0:
            return
        modes = torch.randint(
            low=0,
            high=7,
            size=(edge_ids.numel(),),
            device=self.device,
        )
        self.is_standing_env[edge_ids] = False
        for mode in range(7):
            selected = edge_ids[modes == mode]
            if selected.numel() == 0:
                continue
            random_values = torch.empty(
                (selected.numel(), 3),
                device=self.device,
            )
            if mode == 0:
                ranges = ((2.5, 3.0), (-0.15, 0.15), (-0.15, 0.15))
            elif mode == 1:
                ranges = ((-2.0, -1.5), (-0.15, 0.15), (-0.15, 0.15))
            elif mode == 2:
                ranges = ((-0.2, 0.2), (1.2, 1.5), (-0.15, 0.15))
            elif mode == 3:
                ranges = ((-0.2, 0.2), (-1.5, -1.2), (-0.15, 0.15))
            elif mode == 4:
                ranges = ((-0.15, 0.15), (-0.15, 0.15), (1.5, 2.0))
            elif mode == 5:
                ranges = ((-0.15, 0.15), (-0.15, 0.15), (-2.0, -1.5))
            else:
                ranges = ((0.5, 3.0), (-0.2, 0.2), (-1.0, 1.0))
            for component, limits in enumerate(ranges):
                random_values[:, component].uniform_(*limits)
            self.vel_command_b[selected] = random_values


@configclass
class EdgeBiasedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the full-envelope plus edge-case mixture."""

    class_type: type = EdgeBiasedVelocityCommand
    edge_probability: float = 0.6

