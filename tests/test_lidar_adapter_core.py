"""Tests for deterministic RTX-to-LIO-SAM point conversion."""

from __future__ import annotations

import numpy as np

from anymal_locomotion_ros2.lidar_adapter_core import (
    FASTLIO_POINT_DTYPE,
    OS1_32_ELEVATION_DEG,
    OS1_32_FIRE_TIME_NS,
    OUSTER_POINT_DTYPE,
    convert_rtx_points_to_ouster,
    deterministic_point_indices,
    scan_start_nanoseconds,
    sensor_order_fire_time_nanoseconds,
    sensor_order_time_nanoseconds,
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
        time_source="azimuth",
    )

    assert converted.dtype == OUSTER_POINT_DTYPE
    np.testing.assert_array_equal(converted["ring"], [0, 15, 16, 31])
    np.testing.assert_allclose(converted["t"], [0, 25_000_000, 50_000_000, 75_000_000])
    assert np.all(np.diff(converted["t"].astype(np.int64)) >= 0)
    np.testing.assert_allclose(converted["range"], 10_000, atol=1)


def test_conversion_exposes_reverse_time_direction_for_offline_diagnostics() -> None:
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
        clockwise=False,
        time_source="azimuth",
    )

    # The diagnostic switch reverses azimuth-to-time ordering while retaining
    # the same point fields and monotonic timestamps.
    np.testing.assert_array_equal(converted["ring"], [0, 31, 16, 15])
    assert np.all(np.diff(converted["t"].astype(np.int64)) >= 0)


def test_sensor_order_time_keeps_beams_in_one_firing_column() -> None:
    ring = np.asarray([0, 1, 2, 31, 0, 1, 4, 5], dtype=np.uint8)
    relative_time_ns = sensor_order_time_nanoseconds(ring)

    # Official Ouster ROS native output uses one timestamp per horizontal
    # column, so every beam in the first column shares t=0.
    np.testing.assert_array_equal(relative_time_ns[:4], [0, 0, 0, 0])
    # A missing return does not create a fake column; only ring wrap does.
    assert relative_time_ns[4] == 97_656
    assert np.all(np.diff(relative_time_ns.astype(np.int64)) >= 0)


def test_sensor_order_fire_time_is_explicitly_diagnostic() -> None:
    ring = np.asarray([0, 1, 31, 0], dtype=np.uint8)
    relative_time_ns = sensor_order_fire_time_nanoseconds(ring)

    np.testing.assert_array_equal(
        relative_time_ns[:3],
        OS1_32_FIRE_TIME_NS[[0, 1, 31]].astype(np.uint32),
    )
    assert relative_time_ns[3] == 99_181


def test_sensor_order_conversion_matches_official_ring_major_order() -> None:
    xyz = np.asarray(
        [
            _point(float(OS1_32_ELEVATION_DEG[ring]), -4.0)
            for ring in (0, 1, 2, 31, 0, 1)
        ],
        dtype=np.float32,
    )
    converted = convert_rtx_points_to_ouster(xyz, time_source="sensor_order")

    # RTX acquisition is column-major, but the official Ouster ROS native
    # cloud is packed ring-major (ring/row outer, column inner).  The adapter
    # must emit that layout so FAST-LIO2's point_filter_num acts per ring.
    np.testing.assert_array_equal(converted["ring"], [0, 0, 1, 1, 2, 31])
    np.testing.assert_array_equal(converted["t"], [0, 97_656, 0, 97_656, 0, 0])


def test_sensor_order_conversion_applies_os1_destagger_shifts() -> None:
    xyz = np.asarray(
        [
            _point(float(OS1_32_ELEVATION_DEG[ring]), -4.0 + column)
            for column, ring in ((0, 0), (0, 1), (0, 2), (0, 3))
        ],
        dtype=np.float32,
    )
    intensities = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    converted = convert_rtx_points_to_ouster(
        xyz,
        intensities,
        time_source="sensor_order",
    )

    # With one point per ring, ring-major order is unchanged, but the
    # underlying source columns become 12, 4, 1020, and 1012 after OS1's
    # [12, 4, -4, -12] destagger shifts.  Preserve the source column time.
    np.testing.assert_array_equal(converted["intensity"], intensities)
    np.testing.assert_array_equal(converted["t"], [0, 0, 0, 0])


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
