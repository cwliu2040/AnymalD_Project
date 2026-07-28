"""Pure helpers for full SE(3) LiDAR motion compensation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimedPose:
    stamp_s: float
    position_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    state_id: int = 0


@dataclass(frozen=True)
class TimedAngularVelocity:
    stamp_s: float
    angular_velocity_xyz: tuple[float, float, float]


def _normalized_quaternion_xyzw(
    quaternion: tuple[float, float, float, float],
) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("pose quaternion has near-zero or non-finite norm")
    return values / norm


def _slerp_xyzw(
    before: np.ndarray,
    after: np.ndarray,
    ratio: float,
) -> np.ndarray:
    dot = float(np.dot(before, after))
    if dot < 0.0:
        after = -after
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = before + ratio * (after - before)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    sine = math.sin(angle)
    return (
        math.sin((1.0 - ratio) * angle) / sine * before
        + math.sin(ratio * angle) / sine * after
    )


def interpolate_pose(samples: list[TimedPose], stamp_s: float) -> TimedPose:
    """Interpolate one timestamp that is bracketed by ordered pose samples."""
    if len(samples) < 2:
        raise ValueError("at least two odometry samples are required")
    stamps = np.asarray([sample.stamp_s for sample in samples])
    upper = int(np.searchsorted(stamps, stamp_s, side="left"))
    if upper < len(samples) and math.isclose(
        samples[upper].stamp_s,
        stamp_s,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        return samples[upper]
    if upper == 0 or upper >= len(samples):
        raise ValueError(f"odometry does not bracket timestamp {stamp_s:.9f}")
    before = samples[upper - 1]
    after = samples[upper]
    duration = after.stamp_s - before.stamp_s
    if duration <= 0.0:
        raise ValueError("odometry timestamps must increase monotonically")
    if before.state_id != after.state_id:
        raise ValueError("odometry state changed across interpolation interval")
    ratio = (stamp_s - before.stamp_s) / duration
    before_position = np.asarray(before.position_xyz, dtype=np.float64)
    after_position = np.asarray(after.position_xyz, dtype=np.float64)
    orientation = _slerp_xyzw(
        _normalized_quaternion_xyzw(before.orientation_xyzw),
        _normalized_quaternion_xyzw(after.orientation_xyzw),
        ratio,
    )
    return TimedPose(
        stamp_s=stamp_s,
        position_xyz=tuple(
            before_position + ratio * (after_position - before_position)
        ),
        orientation_xyzw=tuple(orientation),
        state_id=before.state_id,
    )


def relative_translation_in_start_frame(
    start: TimedPose,
    end: TimedPose,
) -> np.ndarray:
    """Return end-position displacement expressed in the start sensor frame."""
    if start.state_id != end.state_id:
        raise ValueError("odometry state changed during the LiDAR scan")
    x, y, z, w = _normalized_quaternion_xyzw(start.orientation_xyzw)
    rotation_world_from_start = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    delta_world = np.asarray(end.position_xyz) - np.asarray(
        start.position_xyz
    )
    return rotation_world_from_start.T @ delta_world


def translational_offsets_from_columns(
    column_ids: np.ndarray,
    *,
    horizon_scan: int,
    scan_translation_xyz: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate start-frame translation by Ouster scan column."""
    columns = np.asarray(column_ids)
    translation = np.asarray(scan_translation_xyz, dtype=np.float64)
    if columns.ndim != 1:
        raise ValueError("column_ids must be one-dimensional")
    if not np.issubdtype(columns.dtype, np.integer):
        raise ValueError("column_ids must contain integers")
    if horizon_scan <= 0:
        raise ValueError("horizon_scan must be positive")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("scan_translation_xyz must contain three finite values")
    if np.any(columns < 0) or np.any(columns >= horizon_scan):
        raise ValueError("column_ids fall outside the configured horizon")

    # LIO-SAM maps +X to Horizon/2. The Isaac OS1 profile rotates clockwise,
    # so decreasing columns correspond to increasing acquisition time.
    phase = np.mod(horizon_scan // 2 - columns, horizon_scan) / horizon_scan
    if phase.size:
        phase = phase - phase.min()
    return phase[:, np.newaxis] * translation[np.newaxis, :]


def translational_offsets_from_odometry(
    column_ids: np.ndarray,
    *,
    horizon_scan: int,
    scan_start_s: float,
    scan_period_s: float,
    samples: list[TimedPose],
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each LiDAR column from high-rate odometry positions.

    The first retained range-image column is the reference used by upstream
    LIO-SAM's rotational deskew. Expressing every position relative to that
    same instant avoids assuming constant velocity across a 100 ms scan.
    """
    columns = np.asarray(column_ids)
    if columns.ndim != 1:
        raise ValueError("column_ids must be one-dimensional")
    if not np.issubdtype(columns.dtype, np.integer):
        raise ValueError("column_ids must contain integers")
    if horizon_scan <= 0:
        raise ValueError("horizon_scan must be positive")
    if np.any(columns < 0) or np.any(columns >= horizon_scan):
        raise ValueError("column_ids fall outside the configured horizon")
    if (
        not math.isfinite(scan_start_s)
        or not math.isfinite(scan_period_s)
        or scan_period_s <= 0.0
    ):
        raise ValueError("scan timing must be finite and positive")
    if not columns.size:
        return np.empty((0, 3), dtype=np.float64), np.zeros(3)
    if len(samples) < 2:
        raise ValueError("at least two odometry samples are required")

    stamps = np.asarray([sample.stamp_s for sample in samples], dtype=np.float64)
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0.0):
        raise ValueError("odometry timestamps must increase monotonically")
    positions = np.asarray(
        [sample.position_xyz for sample in samples],
        dtype=np.float64,
    )
    if positions.shape != (len(samples), 3) or not np.isfinite(
        positions
    ).all():
        raise ValueError("odometry positions must contain finite XYZ values")

    phase = (
        np.mod(horizon_scan // 2 - columns, horizon_scan) / horizon_scan
    )
    reference_phase = float(phase.min())
    query_phases = np.concatenate((phase, np.asarray([reference_phase])))
    query_stamps = scan_start_s + scan_period_s * query_phases
    if query_stamps.min() < stamps[0] or query_stamps.max() > stamps[-1]:
        raise ValueError("odometry does not bracket all LiDAR columns")

    upper = np.searchsorted(stamps, query_stamps, side="right")
    upper = np.clip(upper, 1, len(samples) - 1)
    lower = upper - 1
    state_ids = np.asarray([sample.state_id for sample in samples])
    if np.any(state_ids[lower] != state_ids[upper]):
        raise ValueError("odometry state changed during the LiDAR scan")
    durations = stamps[upper] - stamps[lower]
    ratios = (query_stamps - stamps[lower]) / durations
    interpolated_positions = positions[lower] + ratios[:, np.newaxis] * (
        positions[upper] - positions[lower]
    )

    reference_pose = interpolate_pose(
        samples,
        scan_start_s + scan_period_s * reference_phase,
    )
    x, y, z, w = _normalized_quaternion_xyzw(
        reference_pose.orientation_xyzw
    )
    rotation_world_from_reference = np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )
    reference_position = interpolated_positions[-1]
    deltas_world = interpolated_positions[:-1] - reference_position
    offsets = deltas_world @ rotation_world_from_reference
    end_index = int(np.argmax(phase))
    return offsets, offsets[end_index].copy()


def select_range_image_points(
    points_xyz: np.ndarray,
    rings: np.ndarray,
    relative_times_s: np.ndarray,
    *,
    n_scan: int,
    horizon_scan: int,
    downsample_rate: int = 1,
    min_range_m: float = 0.5,
    max_range_m: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce LIO-SAM's Ouster range-image point selection.

    Input order matters: upstream keeps the first point in each ring/column
    cell. The project adapter sorts a scan by acquisition time before either
    consumer receives it, so stable first-cell selection also recovers the
    point whose timestamp upstream used for rotational deskew.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    ring_values = np.asarray(rings)
    times = np.asarray(relative_times_s, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points_xyz must have shape (N, 3)")
    if ring_values.shape != (len(points),) or times.shape != (len(points),):
        raise ValueError("rings and relative_times_s must match points_xyz")
    if not np.issubdtype(ring_values.dtype, np.integer):
        raise ValueError("rings must contain integers")
    if n_scan <= 0 or horizon_scan <= 0 or downsample_rate <= 0:
        raise ValueError("range-image dimensions must be positive")
    if (
        not math.isfinite(min_range_m)
        or not math.isfinite(max_range_m)
        or min_range_m < 0.0
        or max_range_m <= min_range_m
    ):
        raise ValueError("invalid LiDAR range limits")

    ranges = np.linalg.norm(points, axis=1)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(times)
        & (times >= 0.0)
        & (ranges >= min_range_m)
        & (ranges <= max_range_m)
        & (ring_values >= 0)
        & (ring_values < n_scan)
        & (ring_values % downsample_rate == 0)
    )
    points = points[valid]
    ring_values = ring_values[valid].astype(np.int64, copy=False)
    times = times[valid]
    if not len(points):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
        )

    # Match imageProjection.cpp:
    #   horizonAngle = atan2(x, y)
    #   column = -round((horizonAngle - 90 deg) / resolution) + H/2
    horizon_angle_deg = np.rad2deg(np.arctan2(points[:, 0], points[:, 1]))
    scaled = (horizon_angle_deg - 90.0) / (360.0 / horizon_scan)
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    columns = -rounded.astype(np.int64) + horizon_scan // 2
    columns[columns >= horizon_scan] -= horizon_scan
    column_valid = (columns >= 0) & (columns < horizon_scan)
    points = points[column_valid]
    ring_values = ring_values[column_valid]
    times = times[column_valid]
    columns = columns[column_valid]

    cells = ring_values * horizon_scan + columns
    _, first_indices = np.unique(cells, return_index=True)
    selected_cells = cells[first_indices]
    row_major = np.argsort(selected_cells, kind="stable")
    selected = first_indices[row_major]
    return (
        points[selected],
        times[selected],
        columns[selected].astype(np.int32),
    )


def _interpolate_pose_arrays(
    samples: list[TimedPose],
    query_stamps_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    queries = np.asarray(query_stamps_s, dtype=np.float64)
    if queries.ndim != 1 or not len(queries) or not np.isfinite(queries).all():
        raise ValueError("query timestamps must be a non-empty finite vector")
    if len(samples) < 2:
        raise ValueError("at least two odometry samples are required")

    stamps = np.asarray([sample.stamp_s for sample in samples], dtype=np.float64)
    positions = np.asarray(
        [sample.position_xyz for sample in samples],
        dtype=np.float64,
    )
    quaternions = np.asarray(
        [
            _normalized_quaternion_xyzw(sample.orientation_xyzw)
            for sample in samples
        ],
        dtype=np.float64,
    )
    state_ids = np.asarray([sample.state_id for sample in samples])
    if (
        not np.isfinite(stamps).all()
        or np.any(np.diff(stamps) <= 0.0)
        or positions.shape != (len(samples), 3)
        or not np.isfinite(positions).all()
    ):
        raise ValueError("odometry samples must be finite and time ordered")
    if queries.min() < stamps[0] or queries.max() > stamps[-1]:
        raise ValueError("odometry does not bracket all LiDAR points")

    upper = np.searchsorted(stamps, queries, side="right")
    upper = np.clip(upper, 1, len(samples) - 1)
    lower = upper - 1
    if np.any(state_ids[lower] != state_ids[upper]):
        raise ValueError("odometry state changed during the LiDAR scan")
    durations = stamps[upper] - stamps[lower]
    ratios = (queries - stamps[lower]) / durations
    interpolated_positions = positions[lower] + ratios[:, np.newaxis] * (
        positions[upper] - positions[lower]
    )

    before = quaternions[lower]
    after = quaternions[upper].copy()
    dots = np.sum(before * after, axis=1)
    flip = dots < 0.0
    after[flip] *= -1.0
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    interpolated_quaternions = np.empty_like(before)
    nearly_linear = dots > 0.9995
    if np.any(nearly_linear):
        linear_ratios = ratios[nearly_linear, np.newaxis]
        values = before[nearly_linear] + linear_ratios * (
            after[nearly_linear] - before[nearly_linear]
        )
        interpolated_quaternions[nearly_linear] = values / np.linalg.norm(
            values,
            axis=1,
            keepdims=True,
        )
    spherical = ~nearly_linear
    if np.any(spherical):
        angles = np.arccos(dots[spherical])
        sine = np.sin(angles)
        spherical_ratios = ratios[spherical]
        before_scale = np.sin((1.0 - spherical_ratios) * angles) / sine
        after_scale = np.sin(spherical_ratios * angles) / sine
        interpolated_quaternions[spherical] = (
            before_scale[:, np.newaxis] * before[spherical]
            + after_scale[:, np.newaxis] * after[spherical]
        )
    return interpolated_positions, interpolated_quaternions


def _rotate_vectors_xyzw(
    quaternions_xyzw: np.ndarray,
    vectors_xyz: np.ndarray,
) -> np.ndarray:
    vector_part = quaternions_xyzw[:, :3]
    scalar_part = quaternions_xyzw[:, 3:4]
    first_cross = np.cross(vector_part, vectors_xyz)
    return (
        vectors_xyz
        + 2.0 * scalar_part * first_cross
        + 2.0 * np.cross(vector_part, first_cross)
    )


def _multiply_quaternions_xyzw(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float64,
    )


def relative_orientations_from_imu(
    query_stamps_s: np.ndarray,
    samples: list[TimedAngularVelocity],
) -> np.ndarray:
    """Integrate body-frame gyro samples from the earliest query timestamp."""
    queries = np.asarray(query_stamps_s, dtype=np.float64)
    if queries.ndim != 1 or not len(queries) or not np.isfinite(queries).all():
        raise ValueError("query timestamps must be a non-empty finite vector")
    if len(samples) < 2:
        raise ValueError("at least two IMU samples are required")
    imu_stamps = np.asarray(
        [sample.stamp_s for sample in samples],
        dtype=np.float64,
    )
    angular_velocities = np.asarray(
        [sample.angular_velocity_xyz for sample in samples],
        dtype=np.float64,
    )
    if (
        not np.isfinite(imu_stamps).all()
        or np.any(np.diff(imu_stamps) <= 0.0)
        or angular_velocities.shape != (len(samples), 3)
        or not np.isfinite(angular_velocities).all()
    ):
        raise ValueError("IMU samples must be finite and time ordered")
    start = float(queries.min())
    end = float(queries.max())
    if start < imu_stamps[0] or end > imu_stamps[-1]:
        raise ValueError("IMU does not bracket all LiDAR points")

    interior = imu_stamps[(imu_stamps > start) & (imu_stamps < end)]
    breakpoints = np.concatenate(
        (np.asarray([start]), interior, np.asarray([end]))
    )
    breakpoint_orientations = np.empty((len(breakpoints), 4))
    breakpoint_orientations[0] = (0.0, 0.0, 0.0, 1.0)
    for index in range(len(breakpoints) - 1):
        before = breakpoints[index]
        after = breakpoints[index + 1]
        duration = after - before
        midpoint = 0.5 * (before + after)
        omega = np.asarray(
            [
                np.interp(midpoint, imu_stamps, angular_velocities[:, axis])
                for axis in range(3)
            ]
        )
        angle = float(np.linalg.norm(omega) * duration)
        if angle < 1.0e-12:
            delta = np.asarray((0.0, 0.0, 0.0, 1.0))
        else:
            axis = omega / np.linalg.norm(omega)
            delta = np.concatenate(
                (axis * math.sin(0.5 * angle), [math.cos(0.5 * angle)])
            )
        orientation = _multiply_quaternions_xyzw(
            breakpoint_orientations[index],
            delta,
        )
        breakpoint_orientations[index + 1] = (
            orientation / np.linalg.norm(orientation)
        )

    upper = np.searchsorted(breakpoints, queries, side="right")
    upper = np.clip(upper, 1, len(breakpoints) - 1)
    lower = upper - 1
    durations = breakpoints[upper] - breakpoints[lower]
    ratios = np.divide(
        queries - breakpoints[lower],
        durations,
        out=np.zeros_like(queries),
        where=durations > 0.0,
    )
    before = breakpoint_orientations[lower]
    after = breakpoint_orientations[upper].copy()
    dots = np.sum(before * after, axis=1)
    flip = dots < 0.0
    after[flip] *= -1.0
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    result = np.empty_like(before)
    nearly_linear = dots > 0.9995
    if np.any(nearly_linear):
        values = before[nearly_linear] + ratios[nearly_linear, np.newaxis] * (
            after[nearly_linear] - before[nearly_linear]
        )
        result[nearly_linear] = values / np.linalg.norm(
            values,
            axis=1,
            keepdims=True,
        )
    spherical = ~nearly_linear
    if np.any(spherical):
        angles = np.arccos(dots[spherical])
        sine = np.sin(angles)
        before_scale = (
            np.sin((1.0 - ratios[spherical]) * angles) / sine
        )
        after_scale = np.sin(ratios[spherical] * angles) / sine
        result[spherical] = (
            before_scale[:, np.newaxis] * before[spherical]
            + after_scale[:, np.newaxis] * after[spherical]
        )
    return result


def deskew_points_from_imu_and_odometry(
    points_xyz: np.ndarray,
    point_stamps_s: np.ndarray,
    imu_samples: list[TimedAngularVelocity],
    odometry_samples: list[TimedPose],
    *,
    apply_translation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Deskew raw points with gyro rotation and odometry-only translation."""
    points = np.asarray(points_xyz, dtype=np.float64)
    point_stamps = np.asarray(point_stamps_s, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points_xyz must have shape (N, 3)")
    if point_stamps.shape != (len(points),) or not len(points):
        raise ValueError("point_stamps_s must provide one timestamp per point")
    if not np.isfinite(points).all() or not np.isfinite(point_stamps).all():
        raise ValueError("LiDAR points and timestamps must be finite")

    relative_orientations = relative_orientations_from_imu(
        point_stamps,
        imu_samples,
    )
    corrected = _rotate_vectors_xyzw(relative_orientations, points)
    if not apply_translation:
        return corrected, np.zeros(3, dtype=np.float64)

    reference_stamp = float(point_stamps.min())
    queries = np.concatenate((point_stamps, np.asarray([reference_stamp])))
    positions, orientations = _interpolate_pose_arrays(
        odometry_samples,
        queries,
    )
    reference_position = positions[-1]
    reference_inverse = np.repeat(
        orientations[-1][np.newaxis, :],
        len(points),
        axis=0,
    )
    reference_inverse[:, :3] *= -1.0
    translation_offsets = _rotate_vectors_xyzw(
        reference_inverse,
        positions[:-1] - reference_position,
    )
    corrected += translation_offsets
    latest_index = int(np.argmax(point_stamps))
    return corrected, translation_offsets[latest_index].copy()


def deskew_points_from_odometry(
    points_xyz: np.ndarray,
    point_stamps_s: np.ndarray,
    samples: list[TimedPose],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform raw LiDAR points into the first point's full SE(3) frame."""
    points = np.asarray(points_xyz, dtype=np.float64)
    point_stamps = np.asarray(point_stamps_s, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points_xyz must have shape (N, 3)")
    if point_stamps.shape != (len(points),) or not len(points):
        raise ValueError("point_stamps_s must provide one timestamp per point")
    if not np.isfinite(points).all() or not np.isfinite(point_stamps).all():
        raise ValueError("LiDAR points and timestamps must be finite")

    reference_stamp = float(point_stamps.min())
    queries = np.concatenate(
        (point_stamps, np.asarray([reference_stamp], dtype=np.float64))
    )
    positions, orientations = _interpolate_pose_arrays(samples, queries)
    point_positions = positions[:-1]
    point_orientations = orientations[:-1]
    reference_position = positions[-1]
    reference_orientation = orientations[-1]

    world_points = (
        _rotate_vectors_xyzw(point_orientations, points) + point_positions
    )
    reference_inverse = np.repeat(
        reference_orientation[np.newaxis, :],
        len(points),
        axis=0,
    )
    reference_inverse[:, :3] *= -1.0
    corrected = _rotate_vectors_xyzw(
        reference_inverse,
        world_points - reference_position,
    )

    latest_index = int(np.argmax(point_stamps))
    displacement_world = point_positions[latest_index] - reference_position
    displacement_reference = _rotate_vectors_xyzw(
        reference_inverse[:1],
        displacement_world[np.newaxis, :],
    )[0]
    return corrected, displacement_reference
