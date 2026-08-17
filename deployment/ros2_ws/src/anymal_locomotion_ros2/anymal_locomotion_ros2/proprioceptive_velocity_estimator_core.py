"""ROS-independent runtime core for the learned body-velocity estimator."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STEP_DIMENSION = 37
HISTORY_LENGTH = 20
INPUT_DIMENSION = STEP_DIMENSION * HISTORY_LENGTH
FOOT_ORDER = ("LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT")


@dataclass(frozen=True)
class EstimatorInputBundle:
    """One timestamp-selected estimator input bundle."""

    joint_stamp_ns: int
    joint: Any
    imu_stamp_ns: int
    imu: Any
    contact_stamp_ns: int
    contacts: Any
    synchronized: bool


class EstimatorInputSynchronizer:
    """Select nearest inputs by source stamp, independent of callback order."""

    def __init__(self, tolerance_s: float = 0.025) -> None:
        if tolerance_s < 0.0:
            raise ValueError("estimator synchronization tolerance must be non-negative")
        self.tolerance_ns = int(round(float(tolerance_s) * 1.0e9))
        self.reset()

    def reset(self) -> None:
        self._imu: dict[int, Any] = {}
        self._contacts: dict[int, Any] = {}
        self._joints: dict[int, Any] = {}
        self._latest = {"imu": -1, "contacts": -1, "joint": -1}

    def _push(self, stream: str, stamp_ns: int, value: Any) -> list[EstimatorInputBundle]:
        stamp = int(stamp_ns)
        if stamp < 0:
            self.reset()
            raise ValueError("estimator input timestamp must be non-negative")
        latest = self._latest[stream]
        if stamp < latest:
            self.reset()
            raise ValueError(f"estimator {stream} timestamp regressed")
        self._latest[stream] = stamp
        buffers = {
            "imu": self._imu,
            "contacts": self._contacts,
            "joint": self._joints,
        }
        buffers[stream][stamp] = value
        return self._pop_ready()

    def push_imu(self, stamp_ns: int, value: Any) -> list[EstimatorInputBundle]:
        return self._push("imu", stamp_ns, value)

    def push_contacts(self, stamp_ns: int, value: Any) -> list[EstimatorInputBundle]:
        return self._push("contacts", stamp_ns, value)

    def push_joint(self, stamp_ns: int, value: Any) -> list[EstimatorInputBundle]:
        return self._push("joint", stamp_ns, value)

    @staticmethod
    def _nearest(samples: dict[int, Any], stamp_ns: int) -> tuple[int, Any]:
        selected = min(samples, key=lambda value: (abs(value - stamp_ns), value))
        return selected, samples[selected]

    def _pop_ready(self) -> list[EstimatorInputBundle]:
        bundles: list[EstimatorInputBundle] = []
        while self._joints:
            joint_stamp = min(self._joints)
            if (
                self._latest["imu"] < joint_stamp
                or self._latest["contacts"] < joint_stamp
            ):
                break
            joint = self._joints.pop(joint_stamp)
            imu_stamp, imu = self._nearest(self._imu, joint_stamp)
            contact_stamp, contacts = self._nearest(self._contacts, joint_stamp)
            synchronized = max(
                abs(imu_stamp - joint_stamp),
                abs(contact_stamp - joint_stamp),
            ) <= self.tolerance_ns
            bundles.append(
                EstimatorInputBundle(
                    joint_stamp_ns=joint_stamp,
                    joint=joint,
                    imu_stamp_ns=imu_stamp,
                    imu=imu,
                    contact_stamp_ns=contact_stamp,
                    contacts=contacts,
                    synchronized=synchronized,
                )
            )
            oldest_needed = joint_stamp - self.tolerance_ns
            self._imu = {
                stamp: value for stamp, value in self._imu.items()
                if stamp >= oldest_needed
            }
            self._contacts = {
                stamp: value for stamp, value in self._contacts.items()
                if stamp >= oldest_needed
            }
        return bundles


def _vector(name: str, values: Sequence[float], size: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def assemble_step(
    angular_velocity: Sequence[float],
    linear_acceleration: Sequence[float],
    projected_gravity: Sequence[float],
    relative_joint_position: Sequence[float],
    joint_velocity: Sequence[float],
    foot_contact_probability: Sequence[float],
) -> np.ndarray:
    contacts = _vector("foot contacts", foot_contact_probability, 4)
    if np.any(contacts < 0.0) or np.any(contacts > 1.0):
        raise ValueError("foot contacts must be probabilities in [0, 1]")
    return np.concatenate(
        (
            _vector("angular velocity", angular_velocity, 3),
            _vector("linear acceleration", linear_acceleration, 3),
            _vector("projected gravity", projected_gravity, 3),
            _vector("relative joint position", relative_joint_position, 12),
            _vector("joint velocity", joint_velocity, 12),
            contacts,
        )
    ).astype(np.float32, copy=False)


def reorder_foot_contacts(
    names: Sequence[str], probabilities: Sequence[float]
) -> np.ndarray:
    actual_names = tuple(str(name) for name in names)
    if len(actual_names) != 4 or len(set(actual_names)) != 4:
        raise ValueError("foot contact names must contain four unique values")
    missing = set(FOOT_ORDER) - set(actual_names)
    unexpected = set(actual_names) - set(FOOT_ORDER)
    if missing or unexpected:
        raise ValueError(f"foot contact mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    values = _vector("foot contact probabilities", probabilities, 4)
    return values[[actual_names.index(name) for name in FOOT_ORDER]]


class EstimatorHistory:
    def __init__(self, maximum_gap_s: float = 0.04) -> None:
        if maximum_gap_s <= 0.0:
            raise ValueError("maximum_gap_s must be positive")
        self.maximum_gap_s = float(maximum_gap_s)
        self.samples: deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
        self.last_stamp_s: float | None = None

    def reset(self) -> None:
        self.samples.clear()
        self.last_stamp_s = None

    def append(self, stamp_s: float, sample: Sequence[float]) -> np.ndarray | None:
        stamp = float(stamp_s)
        step = _vector("estimator step", sample, STEP_DIMENSION)
        if not np.isfinite(stamp) or stamp < 0.0:
            self.reset()
            raise ValueError("estimator timestamp must be finite and non-negative")
        if self.last_stamp_s is not None:
            gap = stamp - self.last_stamp_s
            if gap <= 0.0 or gap > self.maximum_gap_s:
                self.reset()
        self.samples.append(step.copy())
        self.last_stamp_s = stamp
        if len(self.samples) < HISTORY_LENGTH:
            return None
        return np.concatenate(tuple(self.samples)).astype(np.float32, copy=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_estimator_metadata(path: str | Path) -> tuple[dict, Path]:
    metadata_path = Path(path).expanduser().resolve()
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "contract_id": "anymal-d-proprioceptive-velocity-v1",
        "history_length": HISTORY_LENGTH,
        "step_dimension": STEP_DIMENSION,
        "input_dimension": INPUT_DIMENSION,
        "output_frame": "base_link",
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f"estimator metadata {key} must be {value!r}")
    onnx = document.get("artifacts", {}).get("onnx", {})
    model_path = metadata_path.parent / str(onnx.get("path", ""))
    if not model_path.is_file() or _sha256(model_path) != onnx.get("sha256"):
        raise ValueError("estimator ONNX artifact is missing or its SHA-256 does not match")
    return document, model_path


class EstimatorRuntime:
    def __init__(
        self,
        backend: Callable[[np.ndarray], np.ndarray],
        *,
        maximum_absolute_velocity_mps: float = 8.0,
    ) -> None:
        self.backend = backend
        self.maximum_absolute_velocity_mps = float(maximum_absolute_velocity_mps)
        self.history = EstimatorHistory()

    def reset(self) -> None:
        self.history.reset()

    def step(self, stamp_s: float, sample: Sequence[float]) -> np.ndarray | None:
        history = self.history.append(stamp_s, sample)
        if history is None:
            return None
        output = np.asarray(self.backend(history.reshape(1, -1)), dtype=np.float32)
        if output.shape == (1, 3):
            output = output[0]
        if output.shape != (3,) or not np.all(np.isfinite(output)):
            self.reset()
            raise ValueError("velocity estimator returned an invalid output")
        if np.any(np.abs(output) > self.maximum_absolute_velocity_mps):
            self.reset()
            raise ValueError("velocity estimator exceeded the absolute velocity guard")
        return output
