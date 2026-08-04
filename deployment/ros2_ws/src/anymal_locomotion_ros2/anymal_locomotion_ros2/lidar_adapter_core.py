"""ROS-independent conversion from RTX XYZ(I) clouds to LIO-SAM Ouster points."""

from __future__ import annotations

import math

import numpy as np

# Ouster OS1 REV6 32-channel elevation angles, in hardware ring order.
OS1_32_ELEVATION_DEG = np.asarray(
    (
        22.10,
        20.67,
        19.25,
        17.82,
        16.40,
        14.97,
        13.55,
        12.12,
        10.69,
        9.27,
        7.84,
        6.42,
        4.99,
        3.56,
        2.14,
        0.71,
        -0.71,
        -2.14,
        -3.56,
        -4.99,
        -6.42,
        -7.84,
        -9.27,
        -10.69,
        -12.12,
        -13.55,
        -14.97,
        -16.40,
        -17.82,
        -19.25,
        -20.67,
        -22.10,
    ),
    dtype=np.float32,
)

OUSTER_POINT_DTYPE = np.dtype(
    {
        "names": (
            "x",
            "y",
            "z",
            "intensity",
            "t",
            "reflectivity",
            "ring",
            "noise",
            "range",
        ),
        "formats": (
            "<f4",
            "<f4",
            "<f4",
            "<f4",
            "<u4",
            "<u2",
            "u1",
            "<u2",
            "<u4",
        ),
        "offsets": (0, 4, 8, 12, 16, 20, 22, 24, 28),
        "itemsize": 32,
    }
)

# The ROS 2 FAST-LIO2 fork registers the Ouster ambient channel instead of
# the project's noise channel.  The binary layout is intentionally identical
# so both adapters receive the same converted XYZ, ring, and per-point time.
FASTLIO_POINT_DTYPE = np.dtype(
    {
        "names": (
            "x",
            "y",
            "z",
            "intensity",
            "t",
            "reflectivity",
            "ring",
            "ambient",
            "range",
        ),
        "formats": (
            "<f4",
            "<f4",
            "<f4",
            "<f4",
            "<u4",
            "<u2",
            "u1",
            "<u2",
            "<u4",
        ),
        "offsets": (0, 4, 8, 12, 16, 20, 22, 24, 28),
        "itemsize": 32,
    }
)


def deterministic_point_indices(
    point_count: int,
    keep_ratio: float,
) -> np.ndarray:
    """Return evenly spaced, deterministic indices for one scan.

    The helper is intentionally independent of ROS and does not shuffle a
    scan.  It is used only for controlled replay degradation; the default
    ratio of ``1.0`` returns every point unchanged.
    """
    if point_count < 0:
        raise ValueError("point_count must be non-negative")
    if not math.isfinite(keep_ratio) or not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be finite and in (0, 1]")
    if point_count == 0:
        return np.empty(0, dtype=np.int64)
    keep_count = max(1, int(round(point_count * keep_ratio)))
    if keep_count >= point_count:
        return np.arange(point_count, dtype=np.int64)
    return np.floor(
        np.arange(keep_count, dtype=np.float64)
        * float(point_count)
        / float(keep_count)
    ).astype(np.int64)


def scan_start_nanoseconds(
    stamp_nanoseconds: int,
    *,
    scan_period_s: float,
    stamp_is_scan_end: bool,
) -> int:
    """Return the scan-start timestamp expected by LIO-SAM."""
    if not math.isfinite(scan_period_s) or scan_period_s <= 0.0:
        raise ValueError("scan_period_s must be finite and positive")
    offset = round(scan_period_s * 1.0e9) if stamp_is_scan_end else 0
    return max(0, int(stamp_nanoseconds) - offset)


def convert_rtx_points_to_ouster(
    xyz: np.ndarray,
    intensity: np.ndarray | None = None,
    *,
    scan_period_s: float = 0.1,
) -> np.ndarray:
    """Add Ouster ring/time fields to one complete clockwise RTX scan."""
    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), received {points.shape}")
    if not math.isfinite(scan_period_s) or scan_period_s <= 0.0:
        raise ValueError("scan_period_s must be finite and positive")

    if intensity is None:
        strengths = np.zeros(points.shape[0], dtype=np.float32)
    else:
        strengths = np.asarray(intensity, dtype=np.float32)
        if strengths.shape != (points.shape[0],):
            raise ValueError(
                "intensity must have one value per point, received "
                f"{strengths.shape} for {points.shape[0]} points"
            )

    finite = np.isfinite(points).all(axis=1) & np.isfinite(strengths)
    points = points[finite]
    strengths = strengths[finite]
    if points.shape[0] == 0:
        return np.empty(0, dtype=OUSTER_POINT_DTYPE)

    horizontal_range = np.hypot(points[:, 0], points[:, 1])
    elevation_deg = np.rad2deg(np.arctan2(points[:, 2], horizontal_range))
    ring = np.abs(
        elevation_deg[:, np.newaxis] - OS1_32_ELEVATION_DEG[np.newaxis, :]
    ).argmin(axis=1)

    # The Isaac OS1 profile rotates clockwise from +X. Convert azimuth to the
    # relative acquisition time in [0, scan_period), then sort so LIO-SAM's
    # use of the final point time produces the true scan end.
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    phase = np.mod(-azimuth, 2.0 * np.pi) / (2.0 * np.pi)
    relative_time_ns = np.rint(phase * scan_period_s * 1.0e9).astype(np.uint32)
    order = np.argsort(relative_time_ns, kind="stable")

    points = points[order]
    strengths = strengths[order]
    ring = ring[order]
    relative_time_ns = relative_time_ns[order]

    output = np.zeros(points.shape[0], dtype=OUSTER_POINT_DTYPE)
    output["x"] = points[:, 0]
    output["y"] = points[:, 1]
    output["z"] = points[:, 2]
    output["intensity"] = strengths
    output["t"] = relative_time_ns
    output["reflectivity"] = np.clip(
        np.rint(strengths * 65535.0),
        0,
        65535,
    ).astype(np.uint16)
    output["ring"] = ring.astype(np.uint8)
    output["range"] = np.clip(
        np.rint(np.linalg.norm(points, axis=1) * 1000.0),
        0,
        np.iinfo(np.uint32).max,
    ).astype(np.uint32)
    return output
