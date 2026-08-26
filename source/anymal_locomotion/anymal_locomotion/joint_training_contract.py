"""Pure contracts for confidence-conditioned full-policy joint training.

This module intentionally imports neither Isaac Lab nor ROS 2.  It defines the
deployable J1/J2 observation layout and future evaluation reductions before any
PPO execution is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LEGACY_OBSERVATION_DIM = 48
HISTORY_FRAME_DIM = 51
HISTORY_LENGTH = 20
FULL_POLICY_OBSERVATION_DIM = (
    LEGACY_OBSERVATION_DIM + HISTORY_LENGTH * HISTORY_FRAME_DIM
)
COMMAND_SLICE = slice(9, 12)
LOCALIZATION_SLICE = slice(48, 51)
NEUTRAL_LOCALIZATION = np.asarray((1.0, 1.0, 0.0), dtype=np.float32)


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, received {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def validate_localization_state(value: Any) -> np.ndarray:
    """Validate backend-neutral confidence, validity and normalized age."""
    state = _finite_array(value, (3,), "localization state")
    confidence, valid, age = (float(item) for item in state)
    if not 0.0 <= confidence <= 1.0 or not 0.0 <= age <= 1.0:
        raise ValueError("confidence and normalized age must be in [0, 1]")
    if valid not in (0.0, 1.0):
        raise ValueError("tracking validity must be exactly 0 or 1")
    return state


def build_history_frame(
    legacy_observation: Any,
    localization_state: Any,
    *,
    requested_command: Any,
) -> np.ndarray:
    """Build one causal 51-D frame without altering the requested command."""
    legacy = _finite_array(
        legacy_observation, (LEGACY_OBSERVATION_DIM,), "legacy observation"
    )
    command = _finite_array(requested_command, (3,), "requested command")
    if not np.array_equal(legacy[COMMAND_SLICE], command):
        raise ValueError("legacy observation command is not the original request")
    localization = validate_localization_state(localization_state)
    return np.concatenate((legacy, localization)).astype(np.float32, copy=False)


def compose_full_policy_observation(
    current_legacy_observation: Any,
    causal_history_frames: Sequence[Any],
    *,
    current_requested_command: Any,
    arm: str,
    history_length: int = HISTORY_LENGTH,
) -> np.ndarray:
    """Compose the equal-width J1/J2 input in oldest-to-newest order.

    Startup history is left-padded with the oldest available causal frame. J1
    retains the same proprioceptive and command history as J2 but receives a
    neutral localization triplet in every frame. J2 receives the validated
    backend-neutral triplets. No future sample is accepted or synthesized.
    """
    if arm not in {"J1", "J2"}:
        raise ValueError("full-policy history arm must be J1 or J2")
    if history_length != HISTORY_LENGTH:
        raise ValueError(f"history length is frozen at {HISTORY_LENGTH}")
    current = _finite_array(
        current_legacy_observation,
        (LEGACY_OBSERVATION_DIM,),
        "current legacy observation",
    )
    requested = _finite_array(current_requested_command, (3,), "current requested command")
    if not np.array_equal(current[COMMAND_SLICE], requested):
        raise ValueError("current policy command is not the original request")
    if not causal_history_frames:
        raise ValueError("at least one causal history frame is required")
    if len(causal_history_frames) > history_length:
        raise ValueError("history contains more than the frozen causal window")

    frames = [
        _finite_array(frame, (HISTORY_FRAME_DIM,), f"history frame {index}").copy()
        for index, frame in enumerate(causal_history_frames)
    ]
    for frame in frames:
        validate_localization_state(frame[LOCALIZATION_SLICE])
        if arm == "J1":
            frame[LOCALIZATION_SLICE] = NEUTRAL_LOCALIZATION
    padded = [frames[0].copy() for _ in range(history_length - len(frames))] + frames
    result = np.concatenate((current, np.concatenate(padded))).astype(np.float32, copy=False)
    if result.shape != (FULL_POLICY_OBSERVATION_DIM,):
        raise RuntimeError("full-policy observation layout changed unexpectedly")
    if not np.array_equal(result[COMMAND_SLICE], requested):
        raise RuntimeError("observation construction modified the original command")
    return result


def history_localization_channels(observation: Any) -> np.ndarray:
    """Return the 20x3 localization view from one 1068-D policy input."""
    value = _finite_array(
        observation, (FULL_POLICY_OBSERVATION_DIM,), "full-policy observation"
    )
    history = value[LEGACY_OBSERVATION_DIM:].reshape(HISTORY_LENGTH, HISTORY_FRAME_DIM)
    return history[:, LOCALIZATION_SLICE]


def accumulated_command_progress(
    times_s: Any,
    requested_commands: Any,
    actual_linear_velocity_body_mps: Any,
    actual_yaw_rate_radps: Any,
    *,
    active_linear_threshold_mps: float = 0.25,
    active_yaw_threshold_radps: float = 0.25,
    stopped_linear_threshold_mps: float = 0.10,
    stopped_yaw_threshold_radps: float = 0.10,
) -> dict[str, float | None]:
    """Reduce accumulated command-aligned motion without endpoint projection.

    Translational progress integrates signed velocity along the requested body
    direction. Yaw progress integrates signed yaw rate along requested yaw.
    Neither value aliases when a curved route approaches an earlier segment.
    """
    times = np.asarray(times_s, dtype=np.float64)
    commands = np.asarray(requested_commands, dtype=np.float64)
    linear = np.asarray(actual_linear_velocity_body_mps, dtype=np.float64)
    yaw_rate = np.asarray(actual_yaw_rate_radps, dtype=np.float64)
    count = len(times)
    if times.shape != (count,) or commands.shape != (count, 3):
        raise ValueError("times and commands must have shapes [N] and [N,3]")
    if linear.shape not in {(count, 2), (count, 3)} or yaw_rate.shape != (count,):
        raise ValueError("actual velocity inputs have invalid shapes")
    if count < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(commands)):
        raise ValueError("progress inputs require at least two finite samples")
    if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(yaw_rate)):
        raise ValueError("actual velocity inputs must be finite")
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("timestamps must increase strictly")

    command_xy = commands[:-1, :2]
    command_speed = np.linalg.norm(command_xy, axis=1)
    linear_active = command_speed >= active_linear_threshold_mps
    direction = np.zeros_like(command_xy)
    direction[linear_active] = command_xy[linear_active] / command_speed[linear_active, None]
    requested_linear_distance = float(np.sum(command_speed[linear_active] * dt[linear_active]))
    realized_linear_progress = float(
        np.sum(np.sum(linear[:-1, :2] * direction, axis=1)[linear_active] * dt[linear_active])
    )

    command_yaw = commands[:-1, 2]
    yaw_active = np.abs(command_yaw) >= active_yaw_threshold_radps
    requested_yaw_distance = float(np.sum(np.abs(command_yaw[yaw_active]) * dt[yaw_active]))
    realized_yaw_progress = float(
        np.sum(
            yaw_rate[:-1][yaw_active]
            * np.sign(command_yaw[yaw_active])
            * dt[yaw_active]
        )
    )
    motion_active = linear_active | yaw_active
    stopped = (
        np.linalg.norm(linear[:-1, :2], axis=1) < stopped_linear_threshold_mps
    ) & (np.abs(yaw_rate[:-1]) < stopped_yaw_threshold_radps)
    return {
        "requested_linear_distance_m": requested_linear_distance,
        "realized_command_aligned_linear_progress_m": realized_linear_progress,
        "linear_progress_ratio": (
            realized_linear_progress / requested_linear_distance
            if requested_linear_distance > 1.0e-9 else None
        ),
        "requested_yaw_distance_rad": requested_yaw_distance,
        "realized_command_aligned_yaw_progress_rad": realized_yaw_progress,
        "yaw_progress_ratio": (
            realized_yaw_progress / requested_yaw_distance
            if requested_yaw_distance > 1.0e-9 else None
        ),
        "stopped_fraction": (
            float(np.mean(stopped[motion_active])) if np.any(motion_active) else None
        ),
    }


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def evaluate_paired_noninferiority(
    candidate: Mapping[str, Any],
    comparator: Mapping[str, Any],
    metric_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply paired relative/difference gates plus independent physical limits."""
    results: dict[str, Any] = {}
    failures: list[str] = []
    for name, spec in metric_specs.items():
        treatment = _finite_scalar(candidate[name], f"candidate {name}")
        control = _finite_scalar(comparator[name], f"comparator {name}")
        mode = str(spec["mode"])
        tolerance = float(spec.get("tolerance", 0.0))
        if mode == "upper_relative":
            paired_pass = treatment <= control * (1.0 + tolerance)
        elif mode == "upper_difference":
            paired_pass = treatment - control <= tolerance
        elif mode == "lower_difference":
            paired_pass = treatment - control >= -tolerance
        elif mode == "two_sided_difference":
            paired_pass = abs(treatment - control) <= tolerance
        else:
            raise ValueError(f"unsupported paired gate mode for {name}: {mode}")
        absolute_pass = True
        if "absolute_minimum" in spec:
            absolute_pass &= treatment >= float(spec["absolute_minimum"])
        if "absolute_maximum" in spec:
            absolute_pass &= treatment <= float(spec["absolute_maximum"])
        passed = bool(paired_pass and absolute_pass)
        results[name] = {
            "candidate": treatment,
            "comparator": control,
            "difference": treatment - control,
            "paired_pass": bool(paired_pass),
            "absolute_pass": bool(absolute_pass),
            "passed": passed,
        }
        if not passed:
            failures.append(name)
    return {"passed": not failures, "failures": failures, "metrics": results}


def event_aligned_body_lidar_summary(
    times_s: Any,
    evaluation_localization_state: Any,
    body_linear_velocity_mps: Any,
    body_angular_velocity_radps: Any,
    requested_commands: Any,
    *,
    post_event_window_s: float = 0.50,
    lidar_scan_time_s: float = 0.10,
) -> dict[str, Any]:
    """Measure motion after observed localization transitions, not route timers."""
    times = np.asarray(times_s, dtype=np.float64)
    localization = np.asarray(evaluation_localization_state, dtype=np.float64)
    linear = np.asarray(body_linear_velocity_mps, dtype=np.float64)
    angular = np.asarray(body_angular_velocity_radps, dtype=np.float64)
    commands = np.asarray(requested_commands, dtype=np.float64)
    count = len(times)
    expected = (count, 3)
    if any(value.shape != expected for value in (localization, linear, angular, commands)):
        raise ValueError("event-aligned arrays must all have shape [N,3]")
    if count < 3 or not all(np.all(np.isfinite(value)) for value in (times, localization, linear, angular, commands)):
        raise ValueError("event-aligned inputs require at least three finite samples")
    if post_event_window_s <= 0.0 or lidar_scan_time_s <= 0.0:
        raise ValueError("event and scan windows must be positive")
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("timestamps must increase strictly")
    confidence_drop = localization[1:, 0] < localization[:-1, 0] - 0.05
    validity_change = localization[1:, 1] != localization[:-1, 1]
    event_indices = np.flatnonzero(confidence_drop | validity_change) + 1
    selected = np.zeros(count, dtype=bool)
    windows = []
    for index in event_indices:
        stop = int(np.searchsorted(times, times[index] + post_event_window_s, side="right"))
        selected[index:stop] = True
        windows.append({"start_s": float(times[index]), "stop_s": float(times[stop - 1])})
    if not np.any(selected):
        return {"available": False, "event_count": 0, "windows": []}

    yaw_error = angular[:, 2] - commands[:, 2]
    linear_error = linear[:, :2] - commands[:, :2]
    roll_pitch_rate = np.linalg.norm(angular[:, :2], axis=1)
    lidar_rotation = lidar_scan_time_s * np.sqrt(
        np.square(roll_pitch_rate) + np.square(yaw_error)
    )
    lidar_translation = lidar_scan_time_s * np.sqrt(
        np.sum(np.square(linear_error), axis=1) + np.square(linear[:, 2])
    )
    angular_acceleration = np.linalg.norm(np.diff(angular, axis=0) / dt[:, None], axis=1)
    acceleration_selected = selected[1:]

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values))))

    return {
        "available": True,
        "event_count": int(len(event_indices)),
        "selected_sample_count": int(np.count_nonzero(selected)),
        "windows": windows,
        "roll_pitch_rate_rms_radps": rms(roll_pitch_rate[selected]),
        "yaw_tracking_error_rms_radps": rms(yaw_error[selected]),
        "linear_tracking_error_rms_mps": rms(np.linalg.norm(linear_error[selected], axis=1)),
        "angular_acceleration_rms_radps2": rms(angular_acceleration[acceleration_selected]),
        "lidar_scan_translation_error_rms_m": rms(lidar_translation[selected]),
        "lidar_scan_rotation_error_rms_rad": rms(lidar_rotation[selected]),
    }


def evaluate_matched_motion(
    times_s: Any,
    candidate_requested_commands: Any,
    comparator_requested_commands: Any,
    candidate_linear_velocity_body_mps: Any,
    comparator_linear_velocity_body_mps: Any,
    candidate_yaw_rate_radps: Any,
    comparator_yaw_rate_radps: Any,
    *,
    maximum_abs_moving_linear_speed_difference_mps: float = 0.05,
    maximum_abs_realized_yaw_rate_difference_radps: float = 0.05,
    minimum_linear_progress_ratio: float = 0.98,
    minimum_yaw_progress_ratio: float = 0.98,
    maximum_stopped_fraction_excess: float = 0.0,
    minimum_comparator_requested_progress_fraction: float = 0.50,
    maximum_candidate_stopped_fraction: float = 0.10,
) -> dict[str, Any]:
    """Require exact commands and matched realized motion before attribution."""
    times = np.asarray(times_s, dtype=np.float64)
    candidate_commands = np.asarray(candidate_requested_commands, dtype=np.float64)
    comparator_commands = np.asarray(comparator_requested_commands, dtype=np.float64)
    candidate_linear = np.asarray(candidate_linear_velocity_body_mps, dtype=np.float64)
    comparator_linear = np.asarray(comparator_linear_velocity_body_mps, dtype=np.float64)
    candidate_yaw = np.asarray(candidate_yaw_rate_radps, dtype=np.float64)
    comparator_yaw = np.asarray(comparator_yaw_rate_radps, dtype=np.float64)
    if not np.array_equal(candidate_commands, comparator_commands):
        return {"passed": False, "failures": ["requested_command_mismatch"]}
    count = len(times)
    if candidate_commands.shape != (count, 3):
        raise ValueError("matched-motion command arrays must have shape [N,3]")
    if candidate_linear.shape not in {(count, 2), (count, 3)} or comparator_linear.shape != candidate_linear.shape:
        raise ValueError("matched-motion linear velocity arrays have invalid shapes")
    if candidate_yaw.shape != (count,) or comparator_yaw.shape != (count,):
        raise ValueError("matched-motion yaw arrays must have shape [N]")
    candidate_progress = accumulated_command_progress(
        times, candidate_commands, candidate_linear, candidate_yaw
    )
    comparator_progress = accumulated_command_progress(
        times, comparator_commands, comparator_linear, comparator_yaw
    )
    command_linear_active = np.linalg.norm(candidate_commands[:, :2], axis=1) >= 0.25
    command_yaw_active = np.abs(candidate_commands[:, 2]) >= 0.25
    candidate_speed = np.linalg.norm(candidate_linear[:, :2], axis=1)
    comparator_speed = np.linalg.norm(comparator_linear[:, :2], axis=1)
    moving_linear_speed_difference = (
        float(abs(np.mean(candidate_speed[command_linear_active]) - np.mean(comparator_speed[command_linear_active])))
        if np.any(command_linear_active) else 0.0
    )
    realized_yaw_rate_difference = (
        float(abs(np.mean(np.abs(candidate_yaw[command_yaw_active])) - np.mean(np.abs(comparator_yaw[command_yaw_active]))))
        if np.any(command_yaw_active) else 0.0
    )

    def relative(candidate: float | None, comparator: float | None) -> float | None:
        if candidate is None or comparator is None:
            return None
        return candidate / comparator if abs(comparator) > 1.0e-9 else None

    linear_progress_relative = relative(
        candidate_progress["realized_command_aligned_linear_progress_m"],
        comparator_progress["realized_command_aligned_linear_progress_m"],
    )
    yaw_progress_relative = relative(
        candidate_progress["realized_command_aligned_yaw_progress_rad"],
        comparator_progress["realized_command_aligned_yaw_progress_rad"],
    )
    candidate_stopped = candidate_progress["stopped_fraction"]
    comparator_stopped = comparator_progress["stopped_fraction"]
    stopped_excess = (
        float(candidate_stopped - comparator_stopped)
        if candidate_stopped is not None and comparator_stopped is not None
        else 0.0
    )
    comparator_linear_requested_fraction = comparator_progress["linear_progress_ratio"]
    comparator_yaw_requested_fraction = comparator_progress["yaw_progress_ratio"]
    checks = {
        "moving_linear_speed": moving_linear_speed_difference <= maximum_abs_moving_linear_speed_difference_mps,
        "realized_yaw_rate": realized_yaw_rate_difference <= maximum_abs_realized_yaw_rate_difference_radps,
        "linear_progress": linear_progress_relative is None or linear_progress_relative >= minimum_linear_progress_ratio,
        "yaw_progress": yaw_progress_relative is None or yaw_progress_relative >= minimum_yaw_progress_ratio,
        "stopped_fraction": stopped_excess <= maximum_stopped_fraction_excess,
        "comparator_linear_motion": (
            comparator_linear_requested_fraction is None
            or comparator_linear_requested_fraction >= minimum_comparator_requested_progress_fraction
        ),
        "comparator_yaw_motion": (
            comparator_yaw_requested_fraction is None
            or comparator_yaw_requested_fraction >= minimum_comparator_requested_progress_fraction
        ),
        "candidate_absolute_stopped_fraction": (
            candidate_stopped is None or candidate_stopped <= maximum_candidate_stopped_fraction
        ),
    }
    return {
        "passed": all(checks.values()),
        "failures": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "moving_linear_speed_difference_mps": moving_linear_speed_difference,
        "realized_yaw_rate_difference_radps": realized_yaw_rate_difference,
        "linear_progress_relative_to_comparator": linear_progress_relative,
        "yaw_progress_relative_to_comparator": yaw_progress_relative,
        "stopped_fraction_excess": stopped_excess,
        "candidate_progress": candidate_progress,
        "comparator_progress": comparator_progress,
    }


def evaluate_directional_metrics(
    candidate: Mapping[str, Any],
    comparator: Mapping[str, Any],
    directions: Mapping[str, str],
    *,
    minimum_improvement_fraction: float = 0.05,
    maximum_regression_fraction: float = 0.10,
    required_improvements: int = 1,
) -> dict[str, Any]:
    """Prespecified relative direction gate for body/LiDAR or SLAM metrics."""
    if not 0.0 <= minimum_improvement_fraction < 1.0:
        raise ValueError("minimum improvement fraction must be in [0,1)")
    if maximum_regression_fraction < 0.0 or required_improvements <= 0:
        raise ValueError("directional gate thresholds are invalid")
    metrics: dict[str, Any] = {}
    improvements = 0
    regressions: list[str] = []
    for name, direction in directions.items():
        treatment = _finite_scalar(candidate[name], f"candidate {name}")
        control = _finite_scalar(comparator[name], f"comparator {name}")
        denominator = max(abs(control), 1.0e-9)
        if direction == "lower":
            signed_improvement = (control - treatment) / denominator
        elif direction == "higher":
            signed_improvement = (treatment - control) / denominator
        else:
            raise ValueError(f"direction for {name} must be lower or higher")
        improved = signed_improvement >= minimum_improvement_fraction
        regressed = signed_improvement < -maximum_regression_fraction
        improvements += int(improved)
        if regressed:
            regressions.append(name)
        metrics[name] = {
            "candidate": treatment,
            "comparator": control,
            "signed_improvement_fraction": signed_improvement,
            "improved": improved,
            "regressed": regressed,
        }
    return {
        "passed": improvements >= required_improvements and not regressions,
        "improvement_count": improvements,
        "required_improvements": required_improvements,
        "regressions": regressions,
        "metrics": metrics,
    }


def evaluate_body_lidar_mechanism(
    candidate: Mapping[str, Any],
    comparator: Mapping[str, Any],
    *,
    minimum_improvement_fraction: float = 0.05,
    maximum_regression_fraction: float = 0.10,
) -> dict[str, Any]:
    """Require repeatable improvement in both body and LiDAR mechanisms."""
    if not bool(candidate.get("available")) or not bool(comparator.get("available")):
        return {"passed": False, "failures": ["event_aligned_window_unavailable"]}
    body_names = {
        "roll_pitch_rate_rms_radps": "lower",
        "angular_acceleration_rms_radps2": "lower",
    }
    lidar_names = {
        "lidar_scan_translation_error_rms_m": "lower",
        "lidar_scan_rotation_error_rms_rad": "lower",
    }
    tracking_names = {
        "yaw_tracking_error_rms_radps": "lower",
        "linear_tracking_error_rms_mps": "lower",
    }
    body = evaluate_directional_metrics(
        candidate, comparator, body_names,
        minimum_improvement_fraction=minimum_improvement_fraction,
        maximum_regression_fraction=maximum_regression_fraction,
        required_improvements=1,
    )
    lidar = evaluate_directional_metrics(
        candidate, comparator, lidar_names,
        minimum_improvement_fraction=minimum_improvement_fraction,
        maximum_regression_fraction=maximum_regression_fraction,
        required_improvements=1,
    )
    tracking = evaluate_directional_metrics(
        candidate, comparator, tracking_names,
        minimum_improvement_fraction=minimum_improvement_fraction,
        maximum_regression_fraction=maximum_regression_fraction,
        required_improvements=1,
    )
    # Tracking does not need to improve, but >10% regression is forbidden.
    tracking_noninferior = not tracking["regressions"]
    failures = []
    if not body["passed"]:
        failures.append("body_mechanism")
    if not lidar["passed"]:
        failures.append("lidar_mechanism")
    if not tracking_noninferior:
        failures.append("tracking_regression")
    return {
        "passed": not failures,
        "failures": failures,
        "body": body,
        "lidar": lidar,
        "tracking": tracking,
    }


def decide_joint_training_route(
    stratum_reports: Iterable[Mapping[str, Any]],
    *,
    required_supported_blocks: int = 3,
) -> dict[str, Any]:
    """Apply the frozen J0/J1/J2 go/no-go interpretation."""
    rows = list(stratum_reports)
    if not rows or required_supported_blocks <= 0:
        raise ValueError("go/no-go requires reports and a positive block requirement")
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        comparison = str(row["comparison"])
        if comparison not in {"J1-J0", "J2-J1"}:
            raise ValueError(f"unsupported comparison: {comparison}")
        key = (comparison, str(row["backend"]), str(row["profile"]))
        grouped.setdefault(key, []).append(row)

    supported = {}
    for key, group in grouped.items():
        count = sum(
            bool(row["integrity_passed"])
            and bool(row["matched_motion_passed"])
            and bool(row["body_lidar_passed"])
            and bool(row["slam_direction_passed"])
            and bool(row["anti_collapse_passed"])
            and not bool(row["safety_event_excess"])
            for row in group
        )
        supported["/".join(key)] = {
            "supported_blocks": count,
            "required_supported_blocks": required_supported_blocks,
            "supported": count >= required_supported_blocks,
        }
    generic = any(
        value["supported"] for key, value in supported.items() if key.startswith("J1-J0/")
    )
    localization = any(
        value["supported"] for key, value in supported.items() if key.startswith("J2-J1/")
    )
    if localization:
        status = "LOCALIZATION_AWARE_HEADROOM_CONTINUE_TO_TEACHER_ADAPTATION"
    elif generic:
        status = "GENERIC_SMOOTHER_LOCOMOTION_ONLY"
    else:
        status = "STOP_FULL_POLICY_JOINT_TRAINING_ROUTE"
    return {
        "status": status,
        "generic_smoother_locomotion_supported": generic,
        "localization_aware_increment_supported": localization,
        "strata": supported,
    }
