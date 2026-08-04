"""Tests for deterministic RTX-to-LIO-SAM point conversion."""

from __future__ import annotations

import numpy as np

from anymal_locomotion_ros2.lidar_adapter_core import (
    FASTLIO_POINT_DTYPE,
    OS1_32_ELEVATION_DEG,
    OUSTER_POINT_DTYPE,
    convert_rtx_points_to_ouster,
    deterministic_point_indices,
    scan_start_nanoseconds,
)


def _point(elevation_deg: float, azimuth_deg: float, distance: float = 10.0) -> list[float]:
    elevation = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)
    horizontal = distance * np.cos(elevation)
    return [
        horizontal * np.cos(azimuth),
        horizontal * np.sin(azimuth),
        distance * np.sin(elevation),
    ]


def test_conversion_assigns_hardware_ring_and_monotonic_relative_time() -> None:
    xyz = np.asarray(
        [
            _point(float(OS1_32_ELEVATION_DEG[31]), -270.0),
            _point(float(OS1_32_ELEVATION_DEG[0]), 0.0),
            _point(float(OS1_32_ELEVATION_DEG[15]), -90.0),
            _point(float(OS1_32_ELEVATION_DEG[16]), -180.0),
        ],
        dtype=np.float32,
    )
    converted = convert_rtx_points_to_ouster(
        xyz,
        np.asarray([0.4, 0.1, 0.2, 0.3], dtype=np.float32),
    )

    assert converted.dtype == OUSTER_POINT_DTYPE
    np.testing.assert_array_equal(converted["ring"], [0, 15, 16, 31])
    np.testing.assert_allclose(converted["t"], [0, 25_000_000, 50_000_000, 75_000_000])
    assert np.all(np.diff(converted["t"].astype(np.int64)) >= 0)
    np.testing.assert_allclose(converted["range"], 10_000, atol=1)


def test_conversion_drops_nonfinite_points_and_supplies_zero_intensity() -> None:
    xyz = np.asarray(
        [
            _point(float(OS1_32_ELEVATION_DEG[4]), 0.0),
            [np.nan, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    converted = convert_rtx_points_to_ouster(xyz)
    assert converted.shape == (1,)
    assert converted["ring"][0] == 4
    assert converted["intensity"][0] == 0.0


def test_scan_start_timestamp_accounts_for_full_scan_accumulation() -> None:
    assert (
        scan_start_nanoseconds(
            12_345_000_000,
            scan_period_s=0.1,
            stamp_is_scan_end=True,
        )
        == 12_245_000_000
    )
    assert (
        scan_start_nanoseconds(
            12_345_000_000,
            scan_period_s=0.1,
            stamp_is_scan_end=False,
        )
        == 12_345_000_000
    )


def test_fastlio_point_layout_uses_ambient_at_the_existing_noise_offset() -> None:
    assert FASTLIO_POINT_DTYPE.itemsize == OUSTER_POINT_DTYPE.itemsize
    assert FASTLIO_POINT_DTYPE.fields["ambient"][1] == 24
    assert "noise" not in FASTLIO_POINT_DTYPE.names


def test_backend_input_layouts_preserve_identical_point_bytes() -> None:
    converted = convert_rtx_points_to_ouster(
        np.asarray(((1.0, 0.0, 0.1), (0.0, -2.0, -0.2)), dtype=np.float32),
        np.asarray((0.25, 0.75), dtype=np.float32),
    )
    fastlio_view = np.frombuffer(converted.tobytes(), dtype=FASTLIO_POINT_DTYPE)
    for field in ("x", "y", "z", "intensity", "t", "reflectivity", "ring", "range"):
        np.testing.assert_array_equal(converted[field], fastlio_view[field])
    np.testing.assert_array_equal(converted["noise"], fastlio_view["ambient"])


def test_deterministic_point_indices_keep_reproducible_even_coverage() -> None:
    first = deterministic_point_indices(10, 0.5)
    second = deterministic_point_indices(10, 0.5)
    np.testing.assert_array_equal(first, [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(first, second)


def test_deterministic_point_indices_keep_all_points_at_full_density() -> None:
    np.testing.assert_array_equal(
        deterministic_point_indices(4, 1.0),
        [0, 1, 2, 3],
    )
