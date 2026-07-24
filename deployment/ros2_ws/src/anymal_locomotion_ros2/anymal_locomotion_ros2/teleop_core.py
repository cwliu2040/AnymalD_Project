"""ROS-independent command state for press-and-hold keyboard teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def normalize_tk_key(
    keysym: object,
    *,
    char: object = "",
    keysym_num: object = None,
) -> str:
    """Normalize main-keyboard and numeric-keypad Tk key events."""
    normalized_keysym = str(keysym).lower()
    normalized_char = str(char)
    try:
        normalized_keysym_num = int(keysym_num)
    except (TypeError, ValueError):
        normalized_keysym_num = -1

    # X11 keysyms are stable even when Tk/desktop combinations report an
    # unexpected localized keysym string. 0xFFAB and 0xFFAD are KP_Add and
    # KP_Subtract; 0x002B and 0x002D are the regular + and - keys.
    if normalized_keysym_num in (0x002B, 0x003D, 0xFFAB):
        return "+"
    if normalized_keysym_num in (0x002D, 0x005F, 0xFFAD):
        return "-"
    if normalized_char in ("+", "="):
        return "+"
    if normalized_char in ("-", "_"):
        return "-"
    if normalized_keysym in ("plus", "equal", "kp_add", "kp_plus", "add"):
        return "+"
    if normalized_keysym in (
        "minus",
        "underscore",
        "kp_subtract",
        "kp_minus",
        "subtract",
    ):
        return "-"
    if normalized_keysym == "space" or normalized_char == " ":
        return "space"
    return normalized_keysym


@dataclass(frozen=True)
class TeleopLimits:
    """Base speeds and policy command limits."""

    forward_speed: float = 0.5
    lateral_speed: float = 0.3
    yaw_speed: float = 0.5
    max_forward: float = 3.0
    max_backward: float = 2.0
    max_lateral: float = 1.5
    max_yaw: float = 2.0
    scale_step: float = 0.25
    min_scale: float = 0.25
    max_scale: float = 6.0

    def validate(self) -> None:
        values = tuple(float(value) for value in self.__dict__.values())
        if not all(np.isfinite(values)) or any(value <= 0.0 for value in values):
            raise ValueError("Teleop speeds, limits, and scale settings must be finite and positive")
        if self.min_scale > self.max_scale:
            raise ValueError("Teleop min_scale cannot exceed max_scale")


@dataclass
class TeleopState:
    """Held keys and the resulting body-frame [vx, vy, wz] command."""

    limits: TeleopLimits = field(default_factory=TeleopLimits)
    speed_scale: float = 1.0
    held_keys: set[str] = field(default_factory=set)

    MOTION_KEYS = frozenset(("w", "s", "a", "d", "q", "e"))

    def __post_init__(self) -> None:
        self.limits.validate()
        self.speed_scale = float(
            np.clip(self.speed_scale, self.limits.min_scale, self.limits.max_scale)
        )

    def press(self, key: str) -> None:
        normalized = key.lower()
        if normalized in self.MOTION_KEYS:
            self.held_keys.add(normalized)
        elif normalized in ("space", " "):
            self.stop()

    def release(self, key: str) -> None:
        self.held_keys.discard(key.lower())

    def stop(self) -> None:
        self.held_keys.clear()

    def increase_speed(self) -> None:
        self.speed_scale = min(
            self.limits.max_scale,
            self.speed_scale + self.limits.scale_step,
        )

    def decrease_speed(self) -> None:
        self.speed_scale = max(
            self.limits.min_scale,
            self.speed_scale - self.limits.scale_step,
        )

    def command(self) -> np.ndarray:
        vx = self.limits.forward_speed * self.speed_scale * (
            float("w" in self.held_keys) - float("s" in self.held_keys)
        )
        vy = self.limits.lateral_speed * self.speed_scale * (
            float("q" in self.held_keys) - float("e" in self.held_keys)
        )
        wz = self.limits.yaw_speed * self.speed_scale * (
            float("a" in self.held_keys) - float("d" in self.held_keys)
        )
        return np.asarray(
            [
                np.clip(vx, -self.limits.max_backward, self.limits.max_forward),
                np.clip(vy, -self.limits.max_lateral, self.limits.max_lateral),
                np.clip(wz, -self.limits.max_yaw, self.limits.max_yaw),
            ],
            dtype=np.float32,
        )
