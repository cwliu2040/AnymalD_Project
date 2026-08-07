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

# Isaac Sim's OS1_REV6_32ch10hz1024res profile uses the same emitter timing
# table as the physical OS1 REV6 32-channel sensor.  The RTX scan-buffer
# writer preserves one complete firing column in ring order (0..31), but it
# does not expose the native Ouster ``t`` field in the ROS PointCloud2.  The
# table is retained for an explicit diagnostic mode; the official Ouster ROS
# native PointCloud2 contract stores one timestamp per horizontal column.
OS1_32_FIRE_TIME_NS = np.asarray(
    (
        1525,
        4576,
        7627,
        10678,
        13729,
        16780,
        19831,
        22882,
        25933,
        28984,
        32035,
        35086,
        38137,
        41188,
        44239,
        47290,
        50341,
        53392,
        56443,
        59494,
        62545,
        65596,
        68647,
        71698,
        74749,
        77800,
        80851,
        83902,
        86953,
        90004,
        93055,
        96106,
    ),
    dtype=np.float64,
)
OS1_32_HORIZONTAL_RESOLUTION = 1024
OS1_32_SCAN_PERIOD_S = 0.1

# Ouster's native organized cloud destaggers each ring by the integer pixel
# shift implied by the OS1 REV6 beam azimuth offsets.  The four-beam pattern is
# repeated for the 32-channel profile at 1024 columns.  RTX scan-buffer input
# remains staggered/acquisition ordered; this table is only used when packing
# the native PointCloud2 output.
OS1_32_PIXEL_SHIFT_BY_ROW = np.asarray(
    (12, 4, -4, -12) * 8,
    dtype=np.int64,
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


def _sensor_order_column_indices(
    ring: np.ndarray,
    *,
    scan_period_s: float,
    horizontal_resolution: int,
) -> np.ndarray:
    """Infer horizontal firing-column indices from RTX acquisition order."""
    rings = np.asarray(ring, dtype=np.int64)
    if rings.ndim != 1:
        raise ValueError(f"ring must have shape (N,), received {rings.shape}")
    if not np.all((0 <= rings) & (rings < OS1_32_FIRE_TIME_NS.size)):
        raise ValueError("ring values must be in [0, 31]")
    if not math.isfinite(scan_period_s) or scan_period_s <= 0.0:
        raise ValueError("scan_period_s must be finite and positive")
    if not isinstance(horizontal_resolution, (int, np.integer)):
        raise ValueError("horizontal_resolution must be an integer")
    if horizontal_resolution <= 0:
        raise ValueError("horizontal_resolution must be positive")
    if rings.size == 0:
        return np.empty(0, dtype=np.int64)

    # The scan buffer can omit invalid returns, but it retains the ring order.
    # A strict decrease therefore marks the next horizontal firing column.
    wraps = np.zeros(rings.size, dtype=np.int64)
    if rings.size > 1:
        wraps[1:] = rings[1:] < rings[:-1]
    return np.cumsum(wraps, dtype=np.int64)


def _sensor_order_destaggered_column_indices(
    ring: np.ndarray,
    *,
    scan_period_s: float,
    horizontal_resolution: int,
) -> np.ndarray:
    """Map raw acquisition columns to official Ouster destaggered columns."""
    rings = np.asarray(ring, dtype=np.int64)
    column_index = _sensor_order_column_indices(
        rings,
        scan_period_s=scan_period_s,
        horizontal_resolution=horizontal_resolution,
    )
    if rings.size == 0:
        return np.empty(0, dtype=np.int64)
    shift = np.rint(
        OS1_32_PIXEL_SHIFT_BY_ROW[rings].astype(np.float64)
        * float(horizontal_resolution)
        / float(OS1_32_HORIZONTAL_RESOLUTION)
    ).astype(np.int64)
    return np.mod(column_index + shift, horizontal_resolution)


def sensor_order_time_nanoseconds(
    ring: np.ndarray,
    *,
    scan_period_s: float = OS1_32_SCAN_PERIOD_S,
    horizontal_resolution: int = OS1_32_HORIZONTAL_RESOLUTION,
) -> np.ndarray:
    """Reconstruct the official Ouster column time from RTX scan-buffer order.

    The RTX buffer is emitted in acquisition order: each horizontal firing
    column contains the 32 beams in ring order.  Invalid returns are removed,
    so a new column is detected when the inferred ring index wraps back to a
    lower value.  The official Ouster ROS native point type assigns the same
    ``t`` timestamp to every ring in one column; it does not copy the profile's
    per-emitter ``fireTimeNs`` values into the ROS field.
    """
    rings = np.asarray(ring, dtype=np.int64)
    column_index = _sensor_order_column_indices(
        rings,
        scan_period_s=scan_period_s,
        horizontal_resolution=horizontal_resolution,
    )

    column_period_ns = scan_period_s * 1.0e9 / float(horizontal_resolution)
    relative_time_ns = column_index.astype(np.float64) * column_period_ns
    if np.any(relative_time_ns < 0.0) or np.any(
        relative_time_ns > np.iinfo(np.uint32).max
    ):
        raise ValueError("reconstructed point time does not fit uint32")
    return np.rint(relative_time_ns).astype(np.uint32)


def sensor_order_fire_time_nanoseconds(
    ring: np.ndarray,
    *,
    scan_period_s: float = OS1_32_SCAN_PERIOD_S,
    horizontal_resolution: int = OS1_32_HORIZONTAL_RESOLUTION,
) -> np.ndarray:
    """Diagnostic variant that adds the profile's per-emitter fire offset."""
    rings = np.asarray(ring, dtype=np.int64)
    column_index = _sensor_order_column_indices(
        rings,
        scan_period_s=scan_period_s,
        horizontal_resolution=horizontal_resolution,
    )
    if rings.size == 0:
        return np.empty(0, dtype=np.uint32)
    column_period_ns = scan_period_s * 1.0e9 / float(horizontal_resolution)
    column_time_ns = column_index.astype(np.float64) * column_period_ns
    relative_time_ns = column_time_ns + OS1_32_FIRE_TIME_NS[rings] * (
        scan_period_s / OS1_32_SCAN_PERIOD_S
    )
    if np.any(relative_time_ns > np.iinfo(np.uint32).max):
        raise ValueError("reconstructed point time does not fit uint32")
    return np.rint(relative_time_ns).astype(np.uint32)


def sensor_order_destaggered_time_nanoseconds(
    ring: np.ndarray,
    *,
    scan_period_s: float = OS1_32_SCAN_PERIOD_S,
    horizontal_resolution: int = OS1_32_HORIZONTAL_RESOLUTION,
) -> np.ndarray:
    """Diagnostic time that follows the destaggered output column index.

    This is not the physical capture time.  It isolates consumers that assume
    the last point in a ring-major cloud is near scan end; formal
    ``sensor_order`` retains raw capture-column time.
    """
    rings = np.asarray(ring, dtype=np.int64)
    column_index = _sensor_order_destaggered_column_indices(
        rings,
        scan_period_s=scan_period_s,
        horizontal_resolution=horizontal_resolution,
    )
    column_period_ns = scan_period_s * 1.0e9 / float(horizontal_resolution)
    relative_time_ns = column_index.astype(np.float64) * column_period_ns
    if np.any(relative_time_ns > np.iinfo(np.uint32).max):
        raise ValueError("reconstructed point time does not fit uint32")
    return np.rint(relative_time_ns).astype(np.uint32)


def convert_rtx_points_to_ouster(
    xyz: np.ndarray,
    intensity: np.ndarray | None = None,
    *,
    scan_period_s: float = 0.1,
    clockwise: bool = True,
    time_source: str = "sensor_order",
    point_order: str = "destaggered",
    horizontal_resolution: int = OS1_32_HORIZONTAL_RESOLUTION,
) -> np.ndarray:
    """Add Ouster ring/time fields to one complete RTX scan.

    ``time_source="sensor_order"`` is the official Ouster ROS-compatible RTX
    contract.  ``sensor_order_fire_time`` and the legacy ``azimuth`` source
    remain available for offline diagnostics only.  ``clockwise`` only applies
    to the azimuth diagnostic source.  Sensor-order output is packed in the
    native driver's ring-major order (ring outer, horizontal column inner),
    while the RTX input buffer itself is column-major. ``point_order`` selects
    the driver's default destaggered packing, an explicit staggered A/B mode,
    or the raw column-major diagnostic mode.
    """
    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), received {points.shape}")
    if not math.isfinite(scan_period_s) or scan_period_s <= 0.0:
        raise ValueError("scan_period_s must be finite and positive")
    time_source = str(time_source).strip().lower()
    if time_source not in {
        "sensor_order",
        "sensor_order_fire_time",
        "sensor_order_destaggered",
        "azimuth",
    }:
        raise ValueError(
            "time_source must be sensor_order, sensor_order_fire_time, "
            "sensor_order_destaggered, or azimuth"
        )
    point_order = str(point_order).strip().lower()
    if point_order not in {
        "destaggered",
        "staggered",
        "column",
        "column_shift1",
    }:
        raise ValueError(
            "point_order must be destaggered, staggered, column, or column_shift1"
        )

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

    if time_source == "sensor_order":
        relative_time_ns = sensor_order_time_nanoseconds(
            ring,
            scan_period_s=scan_period_s,
            horizontal_resolution=horizontal_resolution,
        )
    elif time_source == "sensor_order_fire_time":
        relative_time_ns = sensor_order_fire_time_nanoseconds(
            ring,
            scan_period_s=scan_period_s,
            horizontal_resolution=horizontal_resolution,
        )
    elif time_source == "sensor_order_destaggered":
        relative_time_ns = sensor_order_destaggered_time_nanoseconds(
            ring,
            scan_period_s=scan_period_s,
            horizontal_resolution=horizontal_resolution,
        )
    else:
        # Diagnostic only: the Isaac OS1 profile rotates clockwise from +X.
        # This mode intentionally retains the old approximation for A/B tests.
        azimuth = np.arctan2(points[:, 1], points[:, 0])
        phase_angle = -azimuth if clockwise else azimuth
        phase = np.mod(phase_angle, 2.0 * np.pi) / (2.0 * np.pi)
        relative_time_ns = np.rint(
            phase * scan_period_s * 1.0e9
        ).astype(np.uint32)
    if time_source in {
        "sensor_order",
        "sensor_order_fire_time",
        "sensor_order_destaggered",
    }:
        # Ouster's native PointCloud2 is packed ring-major (row/ring outer,
        # horizontal column inner), even though the RTX scan buffer arrives
        # column-major.  Match that order so FAST-LIO2's deterministic
        # point_filter_num keeps columns within every ring instead of
        # accidentally dropping half of the vertical channels.
        if point_order == "destaggered":
            column_index = _sensor_order_destaggered_column_indices(
                ring,
                scan_period_s=scan_period_s,
                horizontal_resolution=horizontal_resolution,
            )
        else:
            column_index = _sensor_order_column_indices(
                ring,
                scan_period_s=scan_period_s,
                horizontal_resolution=horizontal_resolution,
            )
        if point_order in {"column", "column_shift1"}:
            ring_key = ring if point_order == "column" else np.mod(ring - 1, 32)
            order = np.lexsort((ring_key, column_index))
        else:
            order = np.lexsort((column_index, ring))
    else:
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
