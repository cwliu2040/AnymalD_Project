"""ROS-independent locomotion stability diagnostics.

The simulation host records project-owned ground-truth traces.  This module
keeps the summarization and event-order classifier independent of Isaac Sim so
that the diagnostic contract can be unit tested without starting Kit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class DiagnosticTraceWriter:
    """Append policy-step samples without rewriting the complete report.

    The canonical ``*.json`` report is intentionally produced once after the
    simulation stops.  During a long interactive run, rewriting that growing
    report at every flush boundary turns the diagnostics path into an
    increasingly expensive O(n²) workload.  JSON Lines keeps the runtime cost
    proportional to the newly recorded samples and leaves the most recent
    flushed data recoverable if the host is interrupted.
    """

    def __init__(self, report_path: Path) -> None:
        self.report_path = report_path
        self.path = report_path.with_suffix(".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open(
            "w",
            encoding="utf-8",
            buffering=1024 * 1024,
        )
        self.sample_count = 0

    def append(self, sample: dict[str, Any]) -> None:
        """Queue one compact JSONL sample for the next flush."""
        self._stream.write(
            json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        self.sample_count += 1

    def flush(self) -> None:
        """Make all queued samples visible to readers."""
        self._stream.flush()

    def close(self) -> None:
        """Flush and close the trace stream exactly once."""
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()


def load_diagnostic_trace(path: Path) -> list[dict[str, Any]]:
    """Load complete JSONL samples written by :class:`DiagnosticTraceWriter`."""
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid diagnostic JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(sample, dict):
                raise ValueError(
                    "diagnostic JSONL samples must be JSON objects: "
                    f"line {line_number}"
                )
            samples.append(sample)
    return samples


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Frozen thresholds used after a stable baseline has been collected."""

    stance_min_normal_force_n: float = 5.0
    foot_slip_speed_mps: float | None = None
    body_roll_pitch_rad: float | None = None
    min_base_height_m: float | None = None
    consecutive_samples: int = 2
    coincidence_window_s: float = 0.1

    def __post_init__(self) -> None:
        positive_values = (
            self.stance_min_normal_force_n,
            self.coincidence_window_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive_values):
            raise ValueError("force and coincidence thresholds must be positive")
        if self.consecutive_samples <= 0:
            raise ValueError("consecutive_samples must be positive")
        for value in (
            self.foot_slip_speed_mps,
            self.body_roll_pitch_rad,
            self.min_base_height_m,
        ):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError("optional diagnostic thresholds must be positive")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> DiagnosticThresholds:
        """Construct thresholds from a versioned YAML mapping."""
        expected = {
            "stance_min_normal_force_n",
            "foot_slip_speed_mps",
            "body_roll_pitch_rad",
            "min_base_height_m",
            "consecutive_samples",
            "coincidence_window_s",
        }
        unexpected = set(values) - expected
        missing = expected - set(values)
        if unexpected or missing:
            raise ValueError(
                "diagnostic threshold keys do not match the contract: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        return cls(**values)


def quaternion_to_rpy_wxyz(
    quaternion: tuple[float, float, float, float] | list[float],
) -> tuple[float, float, float]:
    """Return roll, pitch and yaw for one normalized ``wxyz`` quaternion."""
    if len(quaternion) != 4:
        raise ValueError("quaternion must contain four values")
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("quaternion has near-zero or non-finite norm")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return roll, pitch, yaw


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between zero and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values),
        "min": min(values) if values else None,
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "max": max(values) if values else None,
    }


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _first_sustained_time(
    samples: list[dict[str, Any]],
    predicate,
    consecutive_samples: int,
) -> float | None:
    run = 0
    first_time: float | None = None
    for sample in samples:
        if predicate(sample):
            if run == 0:
                first_time = float(sample["time_s"])
            run += 1
            if run >= consecutive_samples:
                return first_time
        else:
            run = 0
            first_time = None
    return None


def classify_instability(
    samples: list[dict[str, Any]],
    thresholds: DiagnosticThresholds,
) -> dict[str, Any]:
    """Classify whether stance-foot slip or body instability occurred first."""

    def is_slipping(sample: dict[str, Any]) -> bool:
        if thresholds.foot_slip_speed_mps is None:
            return False
        return any(
            float(foot["normal_force_n"]) >= thresholds.stance_min_normal_force_n
            and float(foot["tangential_speed_mps"]) >= thresholds.foot_slip_speed_mps
            for foot in sample["feet"].values()
        )

    def is_body_unstable(sample: dict[str, Any]) -> bool:
        angle_failure = (
            thresholds.body_roll_pitch_rad is not None
            and max(abs(float(sample["roll_rad"])), abs(float(sample["pitch_rad"])))
            >= thresholds.body_roll_pitch_rad
        )
        height_failure = (
            thresholds.min_base_height_m is not None
            and float(sample["base_height_m"]) <= thresholds.min_base_height_m
        )
        return angle_failure or height_failure

    slip_time = _first_sustained_time(
        samples,
        is_slipping,
        thresholds.consecutive_samples,
    )
    body_time = _first_sustained_time(
        samples,
        is_body_unstable,
        thresholds.consecutive_samples,
    )
    hard_failure_time = next(
        (
            float(sample["time_s"])
            for sample in samples
            if bool(sample.get("terminated")) or bool(sample.get("truncated"))
        ),
        None,
    )

    if body_time is None and hard_failure_time is None:
        classification = "no_instability"
    elif slip_time is None and body_time is None:
        classification = (
            "indeterminate" if hard_failure_time is not None else "no_instability"
        )
    elif slip_time is None:
        classification = "body_instability_first"
    elif body_time is None:
        classification = "foot_slip_first"
    elif abs(slip_time - body_time) <= thresholds.coincidence_window_s:
        classification = "indeterminate"
    elif slip_time < body_time:
        classification = "foot_slip_first"
    else:
        classification = "body_instability_first"

    return {
        "classification": classification,
        "first_foot_slip_time_s": slip_time,
        "first_body_instability_time_s": body_time,
        "first_hard_failure_time_s": hard_failure_time,
        "contained_foot_slip": (
            slip_time is not None
            and body_time is None
            and hard_failure_time is None
        ),
        "thresholds": asdict(thresholds),
    }


def summarize_samples(
    samples: list[dict[str, Any]],
    *,
    thresholds: DiagnosticThresholds | None = None,
) -> dict[str, Any]:
    """Build compact stability, tracking and per-foot summaries."""
    if not samples:
        raise ValueError("at least one diagnostic sample is required")
    times = [float(sample["time_s"]) for sample in samples]
    if any(after <= before for before, after in zip(times, times[1:], strict=False)):
        raise ValueError("diagnostic sample timestamps must increase strictly")

    foot_names = tuple(samples[0]["feet"])
    if len(foot_names) != 4:
        raise ValueError(f"expected four feet, received {foot_names}")
    for sample in samples:
        if tuple(sample["feet"]) != foot_names:
            raise ValueError("foot ordering changed within a diagnostic trace")

    base_heights = [float(sample["base_height_m"]) for sample in samples]
    absolute_roll = [abs(float(sample["roll_rad"])) for sample in samples]
    absolute_pitch = [abs(float(sample["pitch_rad"])) for sample in samples]
    base_contact_forces = [
        float(sample["base_contact_force_n"]) for sample in samples
    ]
    stationary_window_s = 5.0
    steady_start_time = times[-1] - stationary_window_s
    steady_samples = [
        sample for sample in samples if float(sample["time_s"]) >= steady_start_time
    ]
    steady_planar_displacement_m: float | None = None
    steady_yaw_change_rad: float | None = None
    if all("base_position_w_m" in sample for sample in steady_samples):
        start_position = steady_samples[0]["base_position_w_m"]
        end_position = steady_samples[-1]["base_position_w_m"]
        steady_planar_displacement_m = math.hypot(
            float(end_position[0]) - float(start_position[0]),
            float(end_position[1]) - float(start_position[1]),
        )
        steady_yaw_change_rad = abs(
            _wrap_angle(
                float(steady_samples[-1]["yaw_rad"])
                - float(steady_samples[0]["yaw_rad"])
            )
        )
    terminated_count = sum(bool(sample.get("terminated")) for sample in samples)
    truncated_count = sum(bool(sample.get("truncated")) for sample in samples)
    non_finite_count = 0
    for sample in samples:
        numeric_values = (
            float(sample["base_height_m"]),
            float(sample["roll_rad"]),
            float(sample["pitch_rad"]),
            *[float(value) for value in sample["actual_velocity"]],
        )
        non_finite_count += int(not all(math.isfinite(value) for value in numeric_values))

    active_samples = [
        sample
        for sample in samples
        if max(abs(float(value)) for value in sample["command"]) >= 0.05
    ]
    peak_commands = []
    for index in range(3):
        peak_commands.append(
            max(
                (float(sample["command"][index]) for sample in samples),
                key=abs,
            )
        )
    commanded_indices = [
        index for index, value in enumerate(peak_commands) if abs(value) >= 0.05
    ]
    target_samples = [
        sample
        for sample in samples
        if commanded_indices
        and all(
            abs(float(sample["command"][index]) - peak_commands[index])
            <= 0.05 * abs(peak_commands[index])
            for index in commanded_indices
        )
    ]
    tracking: dict[str, Any] = {
        "active_sample_count": len(active_samples),
        "target_sample_count": len(target_samples),
        "peak_command": peak_commands,
    }
    axis_names = ("vx_mps", "vy_mps", "wz_radps")
    for index, axis_name in enumerate(axis_names):
        all_actual = [
            float(sample["actual_velocity"][index]) for sample in samples
        ]
        steady_actual = [
            float(sample["actual_velocity"][index])
            for sample in steady_samples
        ]
        actual = [float(sample["actual_velocity"][index]) for sample in active_samples]
        commands = [float(sample["command"][index]) for sample in active_samples]
        errors = [
            float(sample["actual_velocity"][index])
            - float(sample["command"][index])
            for sample in active_samples
        ]
        tracking[axis_name] = {
            "all_absolute_actual": _distribution(
                [abs(value) for value in all_actual]
            ),
            "steady_state_mean_absolute_actual": (
                sum(abs(value) for value in steady_actual)
                / len(steady_actual)
            ),
            "mean_command": (
                sum(commands) / len(commands) if commands else None
            ),
            "mean_actual": sum(actual) / len(actual) if actual else None,
            "mean_absolute_error": (
                sum(abs(value) for value in errors) / len(errors)
                if errors
                else None
            ),
            "root_mean_squared_error": (
                math.sqrt(sum(value * value for value in errors) / len(errors))
                if errors
                else None
            ),
            "target_mean_actual": (
                sum(
                    float(sample["actual_velocity"][index])
                    for sample in target_samples
                )
                / len(target_samples)
                if target_samples
                else None
            ),
            "target_mean_absolute_error": (
                sum(
                    abs(
                        float(sample["actual_velocity"][index])
                        - float(sample["command"][index])
                    )
                    for sample in target_samples
                )
                / len(target_samples)
                if target_samples
                else None
            ),
        }

    stance_threshold = (
        thresholds.stance_min_normal_force_n if thresholds is not None else 5.0
    )
    feet: dict[str, Any] = {}
    for name in foot_names:
        normal_forces = [
            float(sample["feet"][name]["normal_force_n"]) for sample in samples
        ]
        tangential_forces = [
            float(sample["feet"][name]["tangential_force_n"]) for sample in samples
        ]
        stance_speeds = [
            float(sample["feet"][name]["tangential_speed_mps"])
            for sample in samples
            if float(sample["feet"][name]["normal_force_n"]) >= stance_threshold
        ]
        feet[name] = {
            "normal_force_n": _distribution(normal_forces),
            "tangential_force_n": _distribution(tangential_forces),
            "stance_tangential_speed_mps": _distribution(stance_speeds),
        }

    classification = (
        classify_instability(samples, thresholds)
        if thresholds is not None
        else {
            "classification": "not_evaluated",
            "reason": "stable baseline thresholds have not been frozen",
        }
    )
    return {
        "sample_count": len(samples),
        "duration_s": times[-1] - times[0],
        "hard_failures": {
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "non_finite_sample_count": non_finite_count,
        },
        "base": {
            "height_m": _distribution(base_heights),
            "absolute_roll_rad": _distribution(absolute_roll),
            "absolute_pitch_rad": _distribution(absolute_pitch),
            "contact_force_n": _distribution(base_contact_forces),
            "steady_state_window_s": stationary_window_s,
            "steady_state_planar_displacement_m": (
                steady_planar_displacement_m
            ),
            "steady_state_yaw_change_rad": steady_yaw_change_rad,
        },
        "tracking": tracking,
        "feet": feet,
        "event_order": classification,
    }


def build_diagnostic_report(
    *,
    samples: list[dict[str, Any]],
    metadata: dict[str, Any],
    thresholds: DiagnosticThresholds | None = None,
) -> dict[str, Any]:
    """Return the versioned on-disk diagnostic document."""
    return {
        "schema_version": 1,
        "metadata": metadata,
        "summary": summarize_samples(samples, thresholds=thresholds),
        "samples": samples,
    }


def evaluate_stability_gate(
    summary: dict[str, Any],
    *,
    target: tuple[float, float, float],
    config: dict[str, Any],
    driver_passed: bool = True,
) -> dict[str, Any]:
    """Evaluate a frozen hard-failure and command-tracking gate."""
    if config.get("schema_version") != 1:
        raise ValueError("stability config schema_version must be 1")
    hard_gate = config.get("hard_gate")
    tracking_gate = config.get("tracking_gate")
    if not isinstance(hard_gate, dict) or not isinstance(tracking_gate, dict):
        raise ValueError("stability config is missing gate mappings")

    failures: list[str] = []
    if not driver_passed:
        failures.append("benchmark_driver_failed")
    hard_failures = summary["hard_failures"]
    if (
        hard_gate.get("require_zero_terminations")
        and hard_failures["terminated_count"] != 0
    ):
        failures.append(
            f"terminated_count={hard_failures['terminated_count']}"
        )
    if (
        hard_gate.get("require_zero_truncations")
        and hard_failures["truncated_count"] != 0
    ):
        failures.append(
            f"truncated_count={hard_failures['truncated_count']}"
        )
    if (
        hard_gate.get("require_finite_samples")
        and hard_failures["non_finite_sample_count"] != 0
    ):
        failures.append(
            "non_finite_sample_count="
            f"{hard_failures['non_finite_sample_count']}"
        )
    event_classification = summary["event_order"]["classification"]
    if (
        hard_gate.get("require_no_instability_event")
        and event_classification != "no_instability"
    ):
        failures.append(f"event_order={event_classification}")

    tracking = summary["tracking"]
    stationary = max(abs(value) for value in target) < 0.05
    axis_names = ("vx_mps", "vy_mps", "wz_radps")
    if stationary:
        limits = tracking_gate.get(
            "stationary_maximum_mean_absolute_velocity",
            {},
        )
        for axis_name in axis_names:
            actual = tracking[axis_name]["steady_state_mean_absolute_actual"]
            limit = limits.get(axis_name)
            if limit is None or actual is None:
                raise ValueError(
                    f"stationary tracking gate is incomplete for {axis_name}"
                )
            if float(actual) > float(limit):
                failures.append(
                    f"{axis_name}_stationary_mean_abs="
                    f"{actual:.6g}>{limit:.6g}"
                )
        displacement = summary["base"]["steady_state_planar_displacement_m"]
        displacement_limit = tracking_gate.get(
            "stationary_maximum_planar_displacement_m"
        )
        yaw_change = summary["base"]["steady_state_yaw_change_rad"]
        yaw_limit = tracking_gate.get(
            "stationary_maximum_yaw_change_rad"
        )
        if (
            displacement is None
            or displacement_limit is None
            or yaw_change is None
            or yaw_limit is None
        ):
            raise ValueError("stationary pose-drift gate is incomplete")
        if float(displacement) > float(displacement_limit):
            failures.append(
                "stationary_planar_displacement_m="
                f"{displacement:.6g}>{displacement_limit:.6g}"
            )
        if float(yaw_change) > float(yaw_limit):
            failures.append(
                f"stationary_yaw_change_rad={yaw_change:.6g}>{yaw_limit:.6g}"
            )
    else:
        minimum_target_samples = int(
            tracking_gate.get("minimum_target_samples", 0)
        )
        if tracking["target_sample_count"] < minimum_target_samples:
            failures.append(
                "target_sample_count="
                f"{tracking['target_sample_count']}<{minimum_target_samples}"
            )
        limits = tracking_gate.get("maximum_mean_absolute_error", {})
        for axis_name in axis_names:
            error = tracking[axis_name]["target_mean_absolute_error"]
            limit = limits.get(axis_name)
            if limit is None or error is None:
                raise ValueError(
                    f"active tracking gate is incomplete for {axis_name}"
                )
            if float(error) > float(limit):
                failures.append(
                    f"{axis_name}_mae={error:.6g}>{limit:.6g}"
                )

    return {
        "passed": not failures,
        "target": list(target),
        "driver_passed": driver_passed,
        "failures": failures,
    }
