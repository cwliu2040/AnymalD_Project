"""Pure NumPy contract shared by data generation, training, and deployment tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

STEP_DIMENSION = 37
HISTORY_LENGTH = 20
MAXIMUM_INTER_SAMPLE_GAP_S = 0.04
FOOT_ORDER = ("LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT")


def _finite_vector(name: str, values: Sequence[float], length: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), received {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or Inf")
    return vector


def assemble_proprioceptive_step(
    *,
    angular_velocity: Sequence[float],
    linear_acceleration: Sequence[float],
    projected_gravity: Sequence[float],
    relative_joint_position: Sequence[float],
    joint_velocity: Sequence[float],
    foot_contact_probability: Sequence[float],
) -> np.ndarray:
    """Assemble one 37-D sample without SLAM or ground-truth velocity."""
    contacts = _finite_vector(
        "foot_contact_probability", foot_contact_probability, 4
    )
    if np.any(contacts < 0.0) or np.any(contacts > 1.0):
        raise ValueError("foot_contact_probability must be in [0, 1]")
    step = np.concatenate(
        (
            _finite_vector("angular_velocity", angular_velocity, 3),
            _finite_vector("linear_acceleration", linear_acceleration, 3),
            _finite_vector("projected_gravity", projected_gravity, 3),
            _finite_vector("relative_joint_position", relative_joint_position, 12),
            _finite_vector("joint_velocity", joint_velocity, 12),
            contacts,
        )
    ).astype(np.float32, copy=False)
    if step.shape != (STEP_DIMENSION,):
        raise AssertionError(f"internal step dimension mismatch: {step.shape}")
    return step


class HistoryBuffer:
    """Timestamp-aware fixed history that resets on gaps or time regression."""

    def __init__(
        self,
        history_length: int = HISTORY_LENGTH,
        maximum_gap_s: float = MAXIMUM_INTER_SAMPLE_GAP_S,
    ) -> None:
        if history_length <= 0 or maximum_gap_s <= 0.0:
            raise ValueError("history_length and maximum_gap_s must be positive")
        self.history_length = int(history_length)
        self.maximum_gap_s = float(maximum_gap_s)
        self._samples: deque[np.ndarray] = deque(maxlen=self.history_length)
        self._last_stamp_s: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._last_stamp_s = None

    def append(self, stamp_s: float, step: Sequence[float]) -> np.ndarray | None:
        stamp = float(stamp_s)
        sample = _finite_vector("step", step, STEP_DIMENSION)
        if not np.isfinite(stamp) or stamp < 0.0:
            self.reset()
            raise ValueError("sample stamp must be finite and non-negative")
        if self._last_stamp_s is not None:
            gap = stamp - self._last_stamp_s
            if gap <= 0.0 or gap > self.maximum_gap_s:
                self.reset()
        self._samples.append(sample.copy())
        self._last_stamp_s = stamp
        if len(self._samples) != self.history_length:
            return None
        return np.concatenate(tuple(self._samples)).astype(np.float32, copy=False)

    @property
    def ready(self) -> bool:
        return len(self._samples) == self.history_length

    @property
    def sample_count(self) -> int:
        return len(self._samples)
