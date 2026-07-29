"""ROS-independent motion profiles and trajectory metrics for LIO-SAM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class PoseSample:
    stamp_s: float
    x: float
    y: float
    z: float
    yaw: float
    linear_speed_mps: float = 0.0
    angular_speed_rps: float = 0.0
    yaw_rate_rps: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0


@dataclass(frozen=True)
class MotionProfile:
    name: str
    target: tuple[float, float, float]
    warmup_s: float
    ramp_s: float
    hold_s: float
    settle_s: float
    sequence: tuple[
        tuple[
            float,
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        ...,
    ] = ()
    command_publish_until_s: float | None = None

    @property
    def duration_s(self) -> float:
        if self.sequence:
            return sum(segment[0] for segment in self.sequence)
        return self.warmup_s + 2.0 * self.ramp_s + self.hold_s + self.settle_s

    def command_at(self, elapsed_s: float) -> tuple[float, float, float]:
        if elapsed_s < 0.0 or elapsed_s >= self.duration_s:
            return (0.0, 0.0, 0.0)
        if self.sequence:
            for duration_s, start, end in self.sequence:
                if elapsed_s < duration_s:
                    phase = elapsed_s / duration_s
                    scale = phase * phase * (3.0 - 2.0 * phase)
                    return tuple(
                        start[index] + scale * (end[index] - start[index])
                        for index in range(3)
                    )
                elapsed_s -= duration_s
            return (0.0, 0.0, 0.0)
        ramp_up_start = self.warmup_s
        hold_start = ramp_up_start + self.ramp_s
        ramp_down_start = hold_start + self.hold_s
        settle_start = ramp_down_start + self.ramp_s
        if elapsed_s < ramp_up_start or elapsed_s >= settle_start:
            scale = 0.0
        elif elapsed_s < hold_start:
            phase = (elapsed_s - ramp_up_start) / self.ramp_s
            scale = phase * phase * (3.0 - 2.0 * phase)
        elif elapsed_s < ramp_down_start:
            scale = 1.0
        else:
            phase = (elapsed_s - ramp_down_start) / self.ramp_s
            smooth = phase * phase * (3.0 - 2.0 * phase)
            scale = 1.0 - smooth
        return tuple(scale * value for value in self.target)

    def should_publish_command(self, elapsed_s: float) -> bool:
        """Return whether the driver should refresh /cmd_vel at this time."""
        return (
            self.command_publish_until_s is None
            or elapsed_s < self.command_publish_until_s
        )


def _out_and_back_sequence(
    first_velocity_mps: float,
    *,
    closed: bool,
) -> tuple[
    tuple[float, tuple[float, float, float], tuple[float, float, float]],
    ...,
]:
    zero = (0.0, 0.0, 0.0)
    first = (first_velocity_mps, 0.0, 0.0)
    second_velocity_mps = -1.0 if first_velocity_mps > 0.0 else 1.5
    second = (second_velocity_mps, 0.0, 0.0)
    first_hold_s = 2.5 if first_velocity_mps > 0.0 else 4.0
    second_hold_s = 4.0 if second_velocity_mps < 0.0 else 2.5
    sequence = [(5.0, zero, zero)]
    sequence.extend(
        (
            (0.5, zero, first),
            (first_hold_s, first, first),
            (0.5, first, zero),
        )
    )
    if closed:
        sequence.extend(
            (
                (1.0, zero, zero),
                (0.5, zero, second),
                (second_hold_s, second, second),
                (0.5, second, zero),
            )
        )
    sequence.append((3.0, zero, zero))
    return tuple(sequence)


_PROFILES = {
    "stationary": MotionProfile(
        "stationary",
        (0.0, 0.0, 0.0),
        warmup_s=5.0,
        ramp_s=1.0,
        hold_s=2.0,
        settle_s=2.0,
    ),
    "forward_0_5": MotionProfile(
        "forward_0_5",
        (0.5, 0.0, 0.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=4.0,
        settle_s=3.0,
    ),
    "forward_1_5": MotionProfile(
        "forward_1_5",
        (1.5, 0.0, 0.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=3.0,
        settle_s=3.0,
    ),
    "forward_3_0": MotionProfile(
        "forward_3_0",
        (3.0, 0.0, 0.0),
        warmup_s=5.0,
        ramp_s=1.5,
        hold_s=1.0,
        settle_s=3.0,
    ),
    "backward_0_5": MotionProfile(
        "backward_0_5",
        (-0.5, 0.0, 0.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=4.0,
        settle_s=3.0,
    ),
    "backward_1_0": MotionProfile(
        "backward_1_0",
        (-1.0, 0.0, 0.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=3.0,
        settle_s=3.0,
    ),
    "backward_2_0": MotionProfile(
        "backward_2_0",
        (-2.0, 0.0, 0.0),
        warmup_s=5.0,
        ramp_s=1.5,
        hold_s=1.0,
        settle_s=3.0,
    ),
    "lateral_0_75": MotionProfile(
        "lateral_0_75",
        (0.0, 0.75, 0.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=3.0,
        settle_s=3.0,
    ),
    "lateral_1_5": MotionProfile(
        "lateral_1_5",
        (0.0, 1.5, 0.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=2.0,
        settle_s=3.0,
    ),
    "lateral_right_0_75": MotionProfile(
        "lateral_right_0_75",
        (0.0, -0.75, 0.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=3.0,
        settle_s=3.0,
    ),
    "lateral_right_1_5": MotionProfile(
        "lateral_right_1_5",
        (0.0, -1.5, 0.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=2.0,
        settle_s=3.0,
    ),
    "yaw_0_5": MotionProfile(
        "yaw_0_5",
        (0.0, 0.0, 0.5),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=4.0,
        settle_s=3.0,
    ),
    "yaw_2_0": MotionProfile(
        "yaw_2_0",
        (0.0, 0.0, 2.0),
        warmup_s=5.0,
        ramp_s=1.0,
        hold_s=1.0,
        settle_s=3.0,
    ),
    "yaw_left_1_0": MotionProfile(
        "yaw_left_1_0",
        (0.0, 0.0, 1.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=4.0,
        settle_s=3.0,
    ),
    "yaw_right_0_5": MotionProfile(
        "yaw_right_0_5",
        (0.0, 0.0, -0.5),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=4.0,
        settle_s=3.0,
    ),
    "yaw_right_1_0": MotionProfile(
        "yaw_right_1_0",
        (0.0, 0.0, -1.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=4.0,
        settle_s=3.0,
    ),
    "yaw_right_2_0": MotionProfile(
        "yaw_right_2_0",
        (0.0, 0.0, -2.0),
        warmup_s=5.0,
        ramp_s=1.0,
        hold_s=1.0,
        settle_s=3.0,
    ),
    "curve_0_5_left_0_5": MotionProfile(
        "curve_0_5_left_0_5",
        (0.5, 0.0, 0.5),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=5.0,
        settle_s=3.0,
    ),
    "curve_0_5_left_0_25_long": MotionProfile(
        "curve_0_5_left_0_25_long",
        (0.5, 0.0, 0.25),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=23.0,
        settle_s=10.0,
    ),
    "curve_0_5_left_0_25_long_shifted": MotionProfile(
        "curve_0_5_left_0_25_long_shifted",
        (0.5, 0.0, 0.25),
        warmup_s=5.9,
        ramp_s=2.0,
        hold_s=23.0,
        settle_s=10.0,
    ),
    "curve_0_5_right_0_5": MotionProfile(
        "curve_0_5_right_0_5",
        (0.5, 0.0, -0.5),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=5.0,
        settle_s=3.0,
    ),
    "curve_1_5_left_1_0": MotionProfile(
        "curve_1_5_left_1_0",
        (1.5, 0.0, 1.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=5.0,
        settle_s=3.0,
    ),
    "curve_1_5_right_1_0": MotionProfile(
        "curve_1_5_right_1_0",
        (1.5, 0.0, -1.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=5.0,
        settle_s=3.0,
    ),
    "curve_3_0_left_0_5": MotionProfile(
        "curve_3_0_left_0_5",
        (3.0, 0.0, 0.5),
        warmup_s=5.0,
        ramp_s=1.5,
        hold_s=1.0,
        settle_s=3.0,
    ),
    "curve_3_0_right_0_5": MotionProfile(
        "curve_3_0_right_0_5",
        (3.0, 0.0, -0.5),
        warmup_s=5.0,
        ramp_s=1.5,
        hold_s=1.0,
        settle_s=3.0,
    ),
    # Out-and-back motion revisits the same spatial path and start pose while
    # using only the pure-translation commands qualified by the stability matrix.
    "loop_out_and_back": MotionProfile(
        "loop_out_and_back",
        (1.5, 0.0, 0.0),
        warmup_s=0.0,
        ramp_s=0.0,
        hold_s=0.0,
        settle_s=0.0,
        sequence=_out_and_back_sequence(1.5, closed=True),
    ),
    "loop_back_and_forth": MotionProfile(
        "loop_back_and_forth",
        (-1.0, 0.0, 0.0),
        warmup_s=0.0,
        ramp_s=0.0,
        hold_s=0.0,
        settle_s=0.0,
        sequence=_out_and_back_sequence(-1.0, closed=True),
    ),
    "loop_open_forward": MotionProfile(
        "loop_open_forward",
        (1.5, 0.0, 0.0),
        warmup_s=0.0,
        ramp_s=0.0,
        hold_s=0.0,
        settle_s=0.0,
        sequence=_out_and_back_sequence(1.5, closed=False),
    ),
    "loop_open_backward": MotionProfile(
        "loop_open_backward",
        (-1.0, 0.0, 0.0),
        warmup_s=0.0,
        ramp_s=0.0,
        hold_s=0.0,
        settle_s=0.0,
        sequence=_out_and_back_sequence(-1.0, closed=False),
    ),
    "combined": MotionProfile(
        "combined",
        (1.5, 0.0, 1.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=5.0,
        settle_s=3.0,
    ),
    "combined_long": MotionProfile(
        "combined_long",
        (1.5, 0.0, 1.0),
        warmup_s=5.0,
        ramp_s=2.0,
        hold_s=20.0,
        settle_s=10.0,
    ),
    "warehouse_mapping_stress": MotionProfile(
        "warehouse_mapping_stress",
        (2.3, 0.0, 2.0),
        warmup_s=0.0,
        ramp_s=0.0,
        hold_s=0.0,
        settle_s=0.0,
        sequence=(
            (5.00, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (0.50, (0.0, 0.0, 0.0), (2.3, 0.0, 2.0)),
            (0.94, (2.3, 0.0, 2.0), (2.3, 0.0, 2.0)),
            (0.02, (2.3, 0.0, 2.0), (2.3, 0.0, 0.0)),
            (9.54, (2.3, 0.0, 0.0), (2.3, 0.0, 0.0)),
            (0.02, (2.3, 0.0, 0.0), (2.3, 0.0, -2.0)),
            (0.16, (2.3, 0.0, -2.0), (2.3, 0.0, -2.0)),
            (0.02, (2.3, 0.0, -2.0), (2.3, 0.0, 0.0)),
            (5.92, (2.3, 0.0, 0.0), (2.3, 0.0, 0.0)),
            (0.02, (2.3, 0.0, 0.0), (2.3, 0.0, -2.0)),
            (0.16, (2.3, 0.0, -2.0), (2.3, 0.0, -2.0)),
            (0.02, (2.3, 0.0, -2.0), (2.3, 0.0, 0.0)),
            (3.74, (2.3, 0.0, 0.0), (2.3, 0.0, 0.0)),
            (0.02, (2.3, 0.0, 0.0), (2.3, 0.0, 2.0)),
            (2.94, (2.3, 0.0, 2.0), (2.3, 0.0, 2.0)),
            (10.00, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
        # Stop refreshing /cmd_vel at the last high-speed left turn.  The
        # policy watchdog therefore owns the transition to the final 10 s
        # zero-command recovery interval, matching interactive teleoperation.
        command_publish_until_s=29.02,
    ),
}


def get_motion_profile(name: str) -> MotionProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown motion profile {name!r}; expected one of {sorted(_PROFILES)}"
        ) from exc


def registered_overlap_separation(
    previous_xyz: np.ndarray,
    current_xyz: np.ndarray,
    *,
    max_points: int = 1200,
    max_correspondence_m: float = 0.3,
    percentile: float = 95.0,
    min_correspondences: int = 100,
) -> float | None:
    """Measure separation of overlapping registered surfaces.

    Points without a nearby surface in the preceding scan are new visibility,
    not map cracks, so they are excluded by ``max_correspondence_m``.
    """
    previous = np.asarray(previous_xyz, dtype=np.float64)
    current = np.asarray(current_xyz, dtype=np.float64)
    if (
        previous.ndim != 2
        or current.ndim != 2
        or previous.shape[1:] != (3,)
        or current.shape[1:] != (3,)
    ):
        raise ValueError("registered clouds must have shape (N, 3)")
    if max_points <= 0 or min_correspondences <= 0:
        raise ValueError("point and correspondence limits must be positive")
    if (
        not math.isfinite(max_correspondence_m)
        or max_correspondence_m <= 0.0
        or not 0.0 < percentile <= 100.0
    ):
        raise ValueError("invalid overlap separation configuration")

    previous = previous[np.isfinite(previous).all(axis=1)]
    current = current[np.isfinite(current).all(axis=1)]
    if not len(previous) or not len(current):
        return None
    if len(current) > max_points:
        current = current[
            np.linspace(0, len(current) - 1, max_points, dtype=np.int64)
        ]

    cell_size = max_correspondence_m
    previous_cells = np.floor(previous / cell_size).astype(np.int64)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(previous_cells):
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        grid.setdefault(key, []).append(index)

    plane_distances: list[float] = []
    current_cells = np.floor(current / cell_size).astype(np.int64)
    for query, cell in zip(current, current_cells, strict=True):
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates.extend(
                        grid.get(
                            (
                                int(cell[0]) + dx,
                                int(cell[1]) + dy,
                                int(cell[2]) + dz,
                            ),
                            (),
                        )
                    )
        if len(candidates) < 8:
            continue
        candidate_points = previous[np.asarray(candidates)]
        delta = candidate_points - query
        squared_distance = np.sum(delta * delta, axis=1)
        nearby = squared_distance <= max_correspondence_m**2
        if int(np.count_nonzero(nearby)) < 8:
            continue
        candidate_points = candidate_points[nearby]
        squared_distance = squared_distance[nearby]
        neighbor_count = min(16, len(candidate_points))
        nearest = np.argpartition(
            squared_distance,
            neighbor_count - 1,
        )[:neighbor_count]
        neighborhood = candidate_points[nearest]
        centroid = neighborhood.mean(axis=0)
        centered = neighborhood - centroid
        covariance = centered.T @ centered / neighbor_count
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        # Reject edges and line-like feature groups. Only a well-supported
        # local plane provides a meaningful wall-normal separation.
        if eigenvalues[1] <= 1.0e-8 or eigenvalues[0] / eigenvalues[1] > 0.15:
            continue
        normal = eigenvectors[:, 0]
        # The acceptance contract is wall separation. Exclude horizontal
        # floor and ceiling planes so vertical gait oscillation cannot be
        # misreported as a split wall.
        if abs(float(normal[2])) > 0.35:
            continue
        plane_distances.append(float(abs(np.dot(query - centroid, normal))))
    if len(plane_distances) < min_correspondences:
        return None
    return float(np.percentile(plane_distances, percentile))


def interpolate_pose_sample(
    samples: list[PoseSample],
    stamp_s: float,
) -> PoseSample:
    stamps = np.asarray([sample.stamp_s for sample in samples])
    upper = int(np.searchsorted(stamps, stamp_s, side="left"))
    if upper == 0:
        return samples[0]
    if upper >= len(samples):
        return samples[-1]
    before = samples[upper - 1]
    after = samples[upper]
    duration = after.stamp_s - before.stamp_s
    if duration <= 0.0:
        return before
    ratio = (stamp_s - before.stamp_s) / duration
    yaw_delta = wrap_angle(after.yaw - before.yaw)
    return PoseSample(
        stamp_s=stamp_s,
        x=before.x + ratio * (after.x - before.x),
        y=before.y + ratio * (after.y - before.y),
        z=before.z + ratio * (after.z - before.z),
        yaw=wrap_angle(before.yaw + ratio * yaw_delta),
        linear_speed_mps=before.linear_speed_mps
        + ratio * (after.linear_speed_mps - before.linear_speed_mps),
        angular_speed_rps=before.angular_speed_rps
        + ratio * (after.angular_speed_rps - before.angular_speed_rps),
        yaw_rate_rps=before.yaw_rate_rps
        + ratio * (after.yaw_rate_rps - before.yaw_rate_rps),
        roll=before.roll + ratio * (after.roll - before.roll),
        pitch=before.pitch + ratio * (after.pitch - before.pitch),
    )


def evaluate_trajectory(
    ground_truth: list[PoseSample],
    estimate: list[PoseSample],
    *,
    ground_truth_sensor_offset_xyz: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
) -> dict[str, float | int]:
    """Align the initial SE(2) pose and evaluate LIO trajectory errors."""
    ground_truth = sorted(ground_truth, key=lambda sample: sample.stamp_s)
    estimate = sorted(estimate, key=lambda sample: sample.stamp_s)
    if len(ground_truth) < 2 or len(estimate) < 2:
        raise ValueError("Trajectory evaluation requires at least two samples")

    matched_estimate = [
        sample
        for sample in estimate
        if ground_truth[0].stamp_s <= sample.stamp_s <= ground_truth[-1].stamp_s
    ]
    if len(matched_estimate) < 2:
        raise ValueError("Estimate timestamps do not overlap ground truth")
    matched_truth = [
        interpolate_pose_sample(ground_truth, sample.stamp_s)
        for sample in matched_estimate
    ]

    first_truth = matched_truth[0]
    first_estimate = matched_estimate[0]
    yaw_alignment = wrap_angle(first_truth.yaw - first_estimate.yaw)
    cosine = math.cos(yaw_alignment)
    sine = math.sin(yaw_alignment)
    sensor_offset = np.asarray(
        ground_truth_sensor_offset_xyz,
        dtype=np.float64,
    )
    if sensor_offset.shape != (3,) or not np.isfinite(sensor_offset).all():
        raise ValueError("ground-truth sensor offset must contain finite XYZ")
    truth_positions_values = []
    for sample in matched_truth:
        sr, cr = math.sin(sample.roll), math.cos(sample.roll)
        sp, cp = math.sin(sample.pitch), math.cos(sample.pitch)
        sy, cy = math.sin(sample.yaw), math.cos(sample.yaw)
        rotation_world_from_body = np.asarray(
            (
                (
                    cy * cp,
                    cy * sp * sr - sy * cr,
                    cy * sp * cr + sy * sr,
                ),
                (
                    sy * cp,
                    sy * sp * sr + cy * cr,
                    sy * sp * cr - cy * sr,
                ),
                (-sp, cp * sr, cp * cr),
            )
        )
        truth_positions_values.append(
            np.asarray((sample.x, sample.y, sample.z))
            + rotation_world_from_body @ sensor_offset
        )
    truth_positions = np.asarray(truth_positions_values, dtype=np.float64)
    first_truth_position = truth_positions[0]
    aligned_positions = []
    aligned_yaws = []
    for sample in matched_estimate:
        dx = sample.x - first_estimate.x
        dy = sample.y - first_estimate.y
        aligned_positions.append(
            (
                first_truth_position[0] + cosine * dx - sine * dy,
                first_truth_position[1] + sine * dx + cosine * dy,
                first_truth_position[2] + sample.z - first_estimate.z,
            )
        )
        aligned_yaws.append(wrap_angle(sample.yaw + yaw_alignment))

    aligned_positions_array = np.asarray(aligned_positions, dtype=np.float64)
    position_errors = np.linalg.norm(
        aligned_positions_array - truth_positions,
        axis=1,
    )
    yaw_errors = np.asarray(
        [
            wrap_angle(aligned - truth.yaw)
            for aligned, truth in zip(
                aligned_yaws,
                matched_truth,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    estimate_steps = np.diff(aligned_positions_array, axis=0)
    truth_steps = np.diff(truth_positions, axis=0)
    translation_jump_residuals = np.linalg.norm(
        estimate_steps - truth_steps,
        axis=1,
    )
    estimate_yaw_steps = np.asarray(
        [
            wrap_angle(after - before)
            for before, after in zip(
                aligned_yaws[:-1],
                aligned_yaws[1:],
                strict=True,
            )
        ]
    )
    truth_yaw_steps = np.asarray(
        [
            wrap_angle(after.yaw - before.yaw)
            for before, after in zip(
                matched_truth[:-1],
                matched_truth[1:],
                strict=True,
            )
        ]
    )
    yaw_jump_residuals = np.abs(
        np.asarray(
            [
                wrap_angle(estimate_step - truth_step)
                for estimate_step, truth_step in zip(
                    estimate_yaw_steps,
                    truth_yaw_steps,
                    strict=True,
                )
            ]
        )
    )
    path_length = float(np.linalg.norm(truth_steps, axis=1).sum())
    initial_truth = matched_truth[0]
    truth_roll = np.asarray(
        [wrap_angle(sample.roll - initial_truth.roll) for sample in matched_truth]
    )
    truth_pitch = np.asarray(
        [
            wrap_angle(sample.pitch - initial_truth.pitch)
            for sample in matched_truth
        ]
    )
    truth_z = np.asarray(
        [sample.z for sample in matched_truth],
        dtype=np.float64,
    )

    return {
        "matched_samples": len(matched_estimate),
        "path_length_m": path_length,
        "translation_ate_rmse_m": float(
            np.sqrt(np.mean(np.square(position_errors)))
        ),
        "translation_error_max_m": float(position_errors.max()),
        "yaw_rmse_deg": math.degrees(
            float(np.sqrt(np.mean(np.square(yaw_errors))))
        ),
        "yaw_error_max_deg": math.degrees(float(np.abs(yaw_errors).max())),
        "translation_jump_residual_max_m": float(
            translation_jump_residuals.max()
        ),
        "yaw_jump_residual_max_deg": math.degrees(
            float(yaw_jump_residuals.max())
        ),
        "ground_truth_linear_speed_max_mps": max(
            abs(sample.linear_speed_mps) for sample in matched_truth
        ),
        "ground_truth_high_speed_scan_count": sum(
            abs(sample.linear_speed_mps) >= 2.5 for sample in matched_truth
        ),
        "ground_truth_angular_speed_max_rps": max(
            abs(sample.angular_speed_rps) for sample in matched_truth
        ),
        "ground_truth_yaw_rate_max_rps": max(
            abs(sample.yaw_rate_rps) for sample in matched_truth
        ),
        "ground_truth_high_yaw_rate_scan_count": sum(
            abs(sample.yaw_rate_rps) >= 1.5 for sample in matched_truth
        ),
        "ground_truth_roll_delta_max_deg": math.degrees(
            float(np.abs(truth_roll).max())
        ),
        "ground_truth_pitch_delta_max_deg": math.degrees(
            float(np.abs(truth_pitch).max())
        ),
        "ground_truth_z_min_m": float(truth_z.min()),
        "ground_truth_z_max_m": float(truth_z.max()),
        "ground_truth_height_drop_max_m": float(
            initial_truth.z - truth_z.min()
        ),
        "ground_truth_final_displacement_xyz_m": [
            float(value)
            for value in truth_positions[-1] - truth_positions[0]
        ],
    }
