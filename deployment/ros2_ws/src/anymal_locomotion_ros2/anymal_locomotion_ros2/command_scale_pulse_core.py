"""Deterministic untreated-prefix command-scale pulse semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class CommandScalePulse:
    enabled: bool = False
    start_s: float = 7.25
    duration_s: float = 0.75
    scale: float = 1.0
    scales_xyz: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        component_scales = self.component_scales
        values = (self.start_s, self.duration_s, self.scale, *component_scales)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("command pulse values must be finite")
        if self.start_s < 0.0:
            raise ValueError("command pulse start must be nonnegative")
        if self.duration_s <= 0.0:
            raise ValueError("command pulse duration must be positive")
        if not all(0.0 < value <= 1.0 for value in component_scales):
            raise ValueError("command pulse component scales must be in (0, 1]")

    @property
    def component_scales(self) -> tuple[float, float, float]:
        if self.scales_xyz is None:
            return (self.scale, self.scale, self.scale)
        if len(self.scales_xyz) != 3:
            raise ValueError("command pulse scales_xyz must contain exactly three values")
        return tuple(float(value) for value in self.scales_xyz)

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def scale_at(self, elapsed_s: float) -> float:
        scales = self.scales_at(elapsed_s)
        if not (scales[0] == scales[1] == scales[2]):
            raise ValueError("anisotropic pulse has no scalar scale_at value")
        return scales[0]

    def scales_at(self, elapsed_s: float) -> tuple[float, float, float]:
        if not math.isfinite(elapsed_s):
            raise ValueError("elapsed time must be finite")
        return self.component_scales if self.active_at(elapsed_s) else (1.0, 1.0, 1.0)

    def active_at(self, elapsed_s: float) -> bool:
        if not math.isfinite(elapsed_s):
            raise ValueError("elapsed time must be finite")
        return bool(self.enabled and self.start_s <= elapsed_s < self.end_s)

    def apply(
        self, command: Sequence[float], elapsed_s: float,
    ) -> tuple[float, float, float]:
        if len(command) != 3:
            raise ValueError("command pulse requires xyz command")
        values = tuple(float(value) for value in command)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("command must be finite")
        scales = self.scales_at(elapsed_s)
        return tuple(scales[index] * value for index, value in enumerate(values))  # type: ignore[return-value]

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "start_s": self.start_s,
            "duration_s": self.duration_s,
            "end_s": self.end_s,
            "scale": self.scale,
            "scales_xyz": list(self.component_scales),
        }
