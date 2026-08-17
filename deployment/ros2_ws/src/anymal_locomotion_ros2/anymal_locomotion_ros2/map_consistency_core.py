"""Offline map-to-reference consistency metrics with one frozen SE(2) fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MapConsistencyResult:
    alignment_yaw_rad: float
    alignment_translation_xyz_m: tuple[float, float, float]
    reference_distance_p50_m: float
    reference_distance_p95_m: float
    off_reference_fraction: float
    duplicate_surface_fraction: float
    evaluated_point_count: int


@dataclass(frozen=True)
class SurfacePose:
    stamp_ns: int
    x: float
    y: float
    z: float
    yaw: float
    roll: float = 0.0
    pitch: float = 0.0


def _points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20:
        raise ValueError(f"{name} must have shape [N, 3] with at least 20 points")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite points")
    return points


def _deterministic_sample(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def _nearest(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty(len(query), dtype=np.int64)
    distances = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 256):
        block = query[start : start + 256]
        squared = np.sum((block[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(squared, axis=1)
        indices[start : start + len(block)] = nearest
        distances[start : start + len(block)] = np.sqrt(
            squared[np.arange(len(block)), nearest]
        )
    return indices, distances


def _interpolate_pose(poses: Sequence[SurfacePose], stamp_ns: int) -> SurfacePose | None:
    if len(poses) < 2 or stamp_ns < poses[0].stamp_ns or stamp_ns > poses[-1].stamp_ns:
        return None
    stamps = np.fromiter((pose.stamp_ns for pose in poses), dtype=np.int64)
    right = int(np.searchsorted(stamps, stamp_ns, side="right"))
    if right == 0:
        return poses[0]
    if right == len(poses):
        return poses[-1]
    left = poses[right - 1]
    following = poses[right]
    span = following.stamp_ns - left.stamp_ns
    fraction = (stamp_ns - left.stamp_ns) / span
    def angle_delta(end: float, start: float) -> float:
        return float(np.arctan2(np.sin(end - start), np.cos(end - start)))
    return SurfacePose(
        stamp_ns=stamp_ns,
        x=left.x + fraction * (following.x - left.x),
        y=left.y + fraction * (following.y - left.y),
        z=left.z + fraction * (following.z - left.z),
        yaw=left.yaw + fraction * angle_delta(following.yaw, left.yaw),
        roll=left.roll + fraction * angle_delta(following.roll, left.roll),
        pitch=left.pitch + fraction * angle_delta(following.pitch, left.pitch),
    )


def build_observed_surface(
    scans: Sequence[tuple[int, np.ndarray]],
    poses: Sequence[SurfacePose],
    *,
    lidar_translation_xyz_m: tuple[float, float, float] = (0.20, 0.0, 0.35),
    points_per_scan: int = 64,
) -> np.ndarray:
    """Project identical raw scans with a pose stream to form an offline surface."""
    ordered_poses = sorted(poses, key=lambda value: value.stamp_ns)
    if len(ordered_poses) < 2:
        raise ValueError("at least two poses are required")
    surfaces: list[np.ndarray] = []
    lidar_translation = np.asarray(lidar_translation_xyz_m, dtype=np.float64)
    for stamp_ns, raw_points in scans:
        pose = _interpolate_pose(ordered_poses, int(stamp_ns))
        if pose is None:
            continue
        points = _points(raw_points, "scan")
        points = _deterministic_sample(points, points_per_scan) + lidar_translation
        cr, sr = np.cos(pose.roll), np.sin(pose.roll)
        cp, sp = np.cos(pose.pitch), np.sin(pose.pitch)
        cy, sy = np.cos(pose.yaw), np.sin(pose.yaw)
        rotation = np.array(
            (
                (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                (-sp, cp * sr, cp * cr),
            )
        )
        points = points @ rotation.T + (pose.x, pose.y, pose.z)
        surfaces.append(points)
    if not surfaces:
        raise ValueError("no scan timestamps overlap the pose stream")
    return np.concatenate(surfaces, axis=0)


def _se2_step(source_xy: np.ndarray, target_xy: np.ndarray) -> tuple[float, np.ndarray]:
    source_center = np.mean(source_xy, axis=0)
    target_center = np.mean(target_xy, axis=0)
    covariance = (source_xy - source_center).T @ (target_xy - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    translation = target_center - source_center @ rotation.T
    return yaw, translation


def evaluate_map_consistency(
    estimated_points_xyz: np.ndarray,
    reference_points_xyz: np.ndarray,
    *,
    off_reference_threshold_m: float = 0.10,
    duplicate_surface_threshold_m: float = 0.15,
    maximum_points: int = 3000,
    icp_iterations: int = 20,
) -> MapConsistencyResult:
    """Fit one robust SE(2)+Z translation and score against a full reference map."""
    estimated = _deterministic_sample(_points(estimated_points_xyz, "estimated"), maximum_points)
    reference = _deterministic_sample(_points(reference_points_xyz, "reference"), maximum_points)
    if not 0.0 < off_reference_threshold_m <= duplicate_surface_threshold_m:
        raise ValueError("map thresholds must satisfy 0 < off_reference <= duplicate")
    aligned = estimated.copy()
    translation_xy = np.median(reference[:, :2], axis=0) - np.median(aligned[:, :2], axis=0)
    aligned[:, :2] += translation_xy
    total_rotation = np.eye(2)
    total_translation = translation_xy.copy()
    for _ in range(icp_iterations):
        nearest, distances = _nearest(reference, aligned)
        cutoff = min(0.75, float(np.quantile(distances, 0.70)))
        inlier = distances <= max(cutoff, 0.03)
        if np.count_nonzero(inlier) < 10:
            raise ValueError("insufficient ICP inliers")
        yaw, step_translation = _se2_step(
            aligned[inlier, :2], reference[nearest[inlier], :2]
        )
        cosine, sine = np.cos(yaw), np.sin(yaw)
        step_rotation = np.array(((cosine, -sine), (sine, cosine)))
        aligned[:, :2] = aligned[:, :2] @ step_rotation.T + step_translation
        total_translation = total_translation @ step_rotation.T + step_translation
        total_rotation = step_rotation @ total_rotation
        if abs(yaw) < 1e-8 and np.linalg.norm(step_translation) < 1e-8:
            break
    z_translation = float(np.median(reference[:, 2]) - np.median(aligned[:, 2]))
    aligned[:, 2] += z_translation
    _, distances = _nearest(reference, aligned)
    return MapConsistencyResult(
        alignment_yaw_rad=float(np.arctan2(total_rotation[1, 0], total_rotation[0, 0])),
        alignment_translation_xyz_m=(
            float(total_translation[0]), float(total_translation[1]), z_translation,
        ),
        reference_distance_p50_m=float(np.quantile(distances, 0.50)),
        reference_distance_p95_m=float(np.quantile(distances, 0.95)),
        off_reference_fraction=float(np.mean(distances > off_reference_threshold_m)),
        duplicate_surface_fraction=float(
            np.mean(distances > duplicate_surface_threshold_m)
        ),
        evaluated_point_count=len(aligned),
    )
