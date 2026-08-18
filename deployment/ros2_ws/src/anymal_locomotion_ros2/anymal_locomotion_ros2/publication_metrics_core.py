"""Pure run-level metrics for the SLAM-confidence publication protocol."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _rms(values: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else None


def _reference_route(samples: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    points = [np.asarray(samples[0]["base_position_w_m"][:2], dtype=np.float64)]
    yaw = float(samples[0]["yaw_rad"])
    lengths = [0.0]
    for before, after in zip(samples[:-1], samples[1:], strict=True):
        dt = float(after["time_s"]) - float(before["time_s"])
        command = np.asarray(before["command"], dtype=np.float64)
        mid_yaw = yaw + 0.5 * command[2] * dt
        cosine, sine = math.cos(mid_yaw), math.sin(mid_yaw)
        world_velocity = np.asarray(
            (cosine * command[0] - sine * command[1], sine * command[0] + cosine * command[1])
        )
        following = points[-1] + world_velocity * dt
        points.append(following)
        lengths.append(lengths[-1] + float(np.linalg.norm(following - points[-2])))
        yaw += command[2] * dt
    return np.asarray(points), np.asarray(lengths)


def _route_progress(point: np.ndarray, route: np.ndarray, lengths: np.ndarray) -> float:
    best_distance = math.inf
    best_progress = 0.0
    for index, (start, end) in enumerate(zip(route[:-1], route[1:], strict=True)):
        segment = end - start
        squared_length = float(segment @ segment)
        if squared_length <= 1e-12:
            continue
        fraction = float(np.clip((point - start) @ segment / squared_length, 0.0, 1.0))
        projection = start + fraction * segment
        distance = float(np.linalg.norm(point - projection))
        if distance < best_distance:
            best_distance = distance
            best_progress = float(lengths[index] + fraction * math.sqrt(squared_length))
    return best_progress


def summarize_locomotion(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Reduce correlated frame samples to one experimental-unit record."""
    ordered = sorted(samples, key=lambda value: float(value["time_s"]))
    if len(ordered) < 2:
        raise ValueError("locomotion metrics require at least two samples")
    times = np.asarray([float(value["time_s"]) for value in ordered])
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("locomotion timestamps must be strictly increasing")
    commands = np.asarray([value["command"] for value in ordered], dtype=np.float64)
    velocities = np.asarray(
        [value["actual_linear_velocity_body_mps"] for value in ordered], dtype=np.float64,
    )
    active = np.linalg.norm(commands[:, :2], axis=1) >= 0.25
    speed = np.linalg.norm(velocities[:, :2], axis=1)
    roll = np.unwrap(np.asarray([float(value["roll_rad"]) for value in ordered]))
    pitch = np.unwrap(np.asarray([float(value["pitch_rad"]) for value in ordered]))
    tilt = np.hypot(roll, pitch)
    roll_pitch_rate = np.hypot(np.diff(roll) / dt, np.diff(pitch) / dt)
    actions = np.asarray([value["applied_raw_action"] for value in ordered], dtype=np.float64)
    action_rate = np.linalg.norm(np.diff(actions, axis=0), axis=1) / dt
    base_contact = np.asarray([float(value["base_contact_force_n"]) for value in ordered])
    heights = np.asarray([float(value["base_height_m"]) for value in ordered])
    first_instability_index = next(
        (
            index
            for index, value in enumerate(ordered)
            if bool(value.get("terminated"))
            or bool(value.get("truncated"))
            or heights[index] < 0.45
            or max(abs(roll[index]), abs(pitch[index])) >= 0.2617993877991494
            or base_contact[index] > 0.0
        ),
        None,
    )
    stable_sample_count = (
        first_instability_index
        if first_instability_index is not None
        else len(ordered)
    )
    while_stable_rate = roll_pitch_rate[: max(0, stable_sample_count - 1)]
    stance_speeds: list[float] = []
    slip_event = False
    consecutive_by_foot: dict[str, int] = {}
    for sample in ordered:
        for foot, state in sample["feet"].items():
            in_stance = float(state["normal_force_n"]) >= 50.0
            tangential_speed = float(state["tangential_speed_mps"])
            if in_stance:
                stance_speeds.append(tangential_speed)
            slipping = in_stance and tangential_speed >= 0.6
            consecutive_by_foot[foot] = consecutive_by_foot.get(foot, 0) + 1 if slipping else 0
            slip_event = slip_event or consecutive_by_foot[foot] >= 2
    route, route_lengths = _reference_route(ordered)
    route_length = float(route_lengths[-1])
    actual_positions = np.asarray([value["base_position_w_m"][:2] for value in ordered])
    progresses = np.asarray(
        [_route_progress(point, route, route_lengths) for point in actual_positions]
    )
    normalized = progresses / route_length if route_length > 1e-9 else np.zeros_like(progresses)
    completion_time: float | None = None
    for index, value in enumerate(normalized):
        if value < 0.95:
            continue
        end = int(np.searchsorted(times, times[index] + 0.5, side="left"))
        if end < len(times) and np.all(normalized[index : end + 1] >= 0.95):
            completion_time = float(times[index] - times[0])
            break
    duration = float(times[-1] - times[0])
    return {
        "duration_s": duration,
        "fall": bool(np.any(heights < 0.45) or any(value["terminated"] for value in ordered)),
        "base_contact": bool(np.any(base_contact > 0.0)),
        "base_contact_force_time_integral_n_s": float(np.sum(base_contact[:-1] * dt)),
        "foot_slip_event": slip_event,
        "stance_weighted_foot_slip_rms_mps": _rms(np.asarray(stance_speeds)),
        "body_tilt_rms_rad": _rms(tilt),
        "body_tilt_p95_rad": float(np.quantile(tilt, 0.95)),
        "roll_pitch_rate_rms_radps": _rms(roll_pitch_rate),
        "roll_pitch_rate_p95_radps": float(np.quantile(roll_pitch_rate, 0.95)),
        "while_stable_roll_pitch_rate_rms_radps": _rms(while_stable_rate),
        "while_stable_roll_pitch_rate_p95_radps": (
            float(np.quantile(while_stable_rate, 0.95))
            if while_stable_rate.size else None
        ),
        "while_stable_sample_count": stable_sample_count,
        "first_instability_time_s": (
            float(times[first_instability_index] - times[0])
            if first_instability_index is not None else None
        ),
        "action_rate_rms_per_s": _rms(action_rate),
        "action_rate_p95_per_s": float(np.quantile(action_rate, 0.95)),
        "reference_route_length_m": route_length,
        "normalized_progress": float(normalized[-1]) if route_length > 1e-9 else None,
        "normalized_progress_per_elapsed_second": (
            float(normalized[-1] / duration) if route_length > 1e-9 and duration > 0.0 else None
        ),
        "completion": completion_time is not None,
        "completion_time_s": completion_time,
        "moving_speed_mps": float(np.mean(speed[active])) if np.any(active) else None,
        "stopped_fraction": float(np.mean(speed[active] < 0.10)) if np.any(active) else None,
    }


def summarize_mechanism(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"available": False}
    fields = (
        "safe_scale", "intent_blend", "raw_stride_attenuation",
        "applied_stride_attenuation", "crouch", "stance_width",
        "action_smoothing", "structured_delta_l2",
    )
    degraded = [record for record in records if float(record["safe_scale"]) < 0.999]
    result: dict[str, Any] = {"available": True, "sample_count": len(records), "degraded_sample_count": len(degraded)}
    for field in fields:
        values = np.asarray([float(record[field]) for record in records])
        degraded_values = np.asarray([float(record[field]) for record in degraded])
        result[f"{field}_mean"] = float(np.mean(values))
        result[f"{field}_degraded_mean"] = float(np.mean(degraded_values)) if degraded else None
        result[f"{field}_degraded_nonzero_fraction"] = (
            float(np.mean(np.abs(degraded_values) > 1e-8)) if degraded else None
        )
    return result
