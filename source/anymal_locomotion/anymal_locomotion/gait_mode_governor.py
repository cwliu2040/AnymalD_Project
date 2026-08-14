"""Batched ROS-independent gait-mode command governor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class GaitMode(IntEnum):
    TRACK = 0
    DECELERATE = 1
    HOLD = 2
    RECOVER = 3


@dataclass(frozen=True)
class GaitModeGovernorConfig:
    degrade_below: float = 0.45
    degrade_dwell_s: float = 0.10
    recover_at_or_above: float = 0.55
    recover_dwell_s: float = 0.50
    hold_minimum_dwell_s: float = 0.50
    command_scale_rate_down_per_s: float = 0.8
    command_scale_rate_up_per_s: float = 2.0

    def validate(self) -> None:
        if not 0.0 <= self.degrade_below < self.recover_at_or_above <= 1.0:
            raise ValueError("gait-mode confidence thresholds are invalid")
        values = (
            self.degrade_dwell_s,
            self.recover_dwell_s,
            self.hold_minimum_dwell_s,
            self.command_scale_rate_down_per_s,
            self.command_scale_rate_up_per_s,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("gait-mode dwell times and rates must be positive")


class BatchedGaitModeGovernor:
    """Maintain one deterministic mode and command scale per environment."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        config: GaitModeGovernorConfig | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.config = config or GaitModeGovernorConfig()
        self.config.validate()
        self.mode = torch.full(
            (num_envs,), int(GaitMode.HOLD), dtype=torch.int64, device=device
        )
        self.command_scale = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.degrade_elapsed_s = torch.zeros_like(self.command_scale)
        self.hold_elapsed_s = torch.zeros_like(self.command_scale)
        self.recover_elapsed_s = torch.zeros_like(self.command_scale)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(
        self,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> None:
        confidence, valid = self._inputs(confidence, valid)
        mask = (
            torch.ones_like(valid, dtype=torch.bool)
            if reset_mask is None
            else reset_mask.to(device=self.mode.device, dtype=torch.bool)
        )
        healthy = valid & (confidence >= self.config.recover_at_or_above)
        self.mode[mask] = torch.where(
            healthy[mask],
            torch.full_like(self.mode[mask], int(GaitMode.TRACK)),
            torch.full_like(self.mode[mask], int(GaitMode.HOLD)),
        )
        self.command_scale[mask] = healthy[mask].to(self.command_scale.dtype)
        self.degrade_elapsed_s[mask] = 0.0
        self.hold_elapsed_s[mask] = 0.0
        self.recover_elapsed_s[mask] = 0.0
        self.initialized[mask] = True

    def _inputs(
        self, confidence: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        confidence = confidence.to(device=self.mode.device, dtype=torch.float32)
        valid = valid.to(device=self.mode.device) >= 0.5
        if confidence.shape != self.command_scale.shape or valid.shape != self.command_scale.shape:
            raise ValueError("confidence and validity must match governor batch shape")
        if not bool(torch.all(torch.isfinite(confidence))):
            raise ValueError("confidence contains NaN or Inf")
        return torch.clamp(confidence, 0.0, 1.0), valid

    def update(
        self,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        dt_s: float,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        confidence, valid = self._inputs(confidence, valid)
        requested_reset = (
            torch.zeros_like(valid)
            if reset_mask is None
            else reset_mask.to(device=self.mode.device, dtype=torch.bool)
        )
        initialize = requested_reset | ~self.initialized
        if bool(torch.any(initialize)):
            self.reset(confidence, valid, initialize)
        active = ~initialize
        cfg = self.config
        desired_scale = valid.to(torch.float32) * torch.clamp(
            (confidence - 0.2) / 0.8, min=0.0, max=1.0
        )

        track = active & (self.mode == int(GaitMode.TRACK))
        unhealthy = desired_scale < (1.0 - 1.0e-6)
        self.degrade_elapsed_s = torch.where(
            track & unhealthy,
            self.degrade_elapsed_s + dt_s,
            torch.where(track, torch.zeros_like(self.degrade_elapsed_s), self.degrade_elapsed_s),
        )
        leave_track = track & (
            (~valid) | (self.degrade_elapsed_s + 1.0e-6 >= cfg.degrade_dwell_s)
        )
        self.mode[leave_track] = int(GaitMode.DECELERATE)

        decelerate = active & (self.mode == int(GaitMode.DECELERATE))
        self.command_scale[decelerate] = torch.maximum(
            desired_scale[decelerate],
            torch.clamp(
                self.command_scale[decelerate]
                - cfg.command_scale_rate_down_per_s * dt_s,
                min=0.0,
            ),
        )
        reached_hold = decelerate & (self.command_scale <= 0.0)
        self.mode[reached_hold] = int(GaitMode.HOLD)
        self.hold_elapsed_s[reached_hold] = 0.0
        self.recover_elapsed_s[reached_hold] = 0.0

        hold = active & (self.mode == int(GaitMode.HOLD))
        self.command_scale[hold] = 0.0
        self.hold_elapsed_s[hold] += dt_s
        recover_ready = valid & (confidence >= cfg.recover_at_or_above)
        self.recover_elapsed_s = torch.where(
            hold & recover_ready,
            self.recover_elapsed_s + dt_s,
            torch.where(hold, torch.zeros_like(self.recover_elapsed_s), self.recover_elapsed_s),
        )
        leave_hold = (
            hold
            & (self.hold_elapsed_s + 1.0e-6 >= cfg.hold_minimum_dwell_s)
            & (self.recover_elapsed_s + 1.0e-6 >= cfg.recover_dwell_s)
        )
        self.mode[leave_hold] = int(GaitMode.RECOVER)

        recover = active & (self.mode == int(GaitMode.RECOVER))
        abort_recovery = recover & (
            (~valid) | (desired_scale + 1.0e-6 < self.command_scale)
        )
        self.mode[abort_recovery] = int(GaitMode.DECELERATE)
        recover = recover & ~abort_recovery
        self.command_scale[recover] = torch.minimum(
            desired_scale[recover],
            torch.clamp(
                self.command_scale[recover]
                + cfg.command_scale_rate_up_per_s * dt_s,
                max=1.0,
            ),
        )
        reached_track = recover & (self.command_scale >= 1.0)
        self.mode[reached_track] = int(GaitMode.TRACK)
        self.command_scale[reached_track] = 1.0
        self.degrade_elapsed_s[reached_track] = 0.0
        return self.command_scale
