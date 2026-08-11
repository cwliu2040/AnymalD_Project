"""Tests for exact-stamp FAST-LIO2 confidence signal assembly."""

from __future__ import annotations

import itertools
import math

import pytest

from anymal_locomotion_ros2.fastlio_confidence_core import (
    FastlioAssemblerConfig,
    FastlioSignalAssembler,
)
from anymal_locomotion_ros2.slam_confidence_core import (
    DegradationReason,
    SlamConfidenceStateMachine,
    SourceObservation,
    StateMachineConfig,
    TrackingState,
    seconds_to_nanoseconds,
)


POSE = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)


def _config(*, max_pending: int = 64) -> FastlioAssemblerConfig:
    return FastlioAssemblerConfig(
        odometry_timeout_ns=seconds_to_nanoseconds(0.30),
        lidar_timeout_ns=seconds_to_nanoseconds(0.30),
        imu_timeout_ns=seconds_to_nanoseconds(0.05),
        diagnostic_match_timeout_ns=seconds_to_nanoseconds(0.05),
        future_tolerance_ns=seconds_to_nanoseconds(0.005),
        max_pending_bundles=max_pending,
    )


def _assembler(*, max_pending: int = 64) -> FastlioSignalAssembler:
    return FastlioSignalAssembler(_config(max_pending=max_pending))


def _observe_component(
    assembler: FastlioSignalAssembler,
    component: str,
    *,
    stamp_ns: int,
    observed_at_ns: int | None = None,
    effective_count: int = 100,
) -> None:
    arrival_ns = stamp_ns if observed_at_ns is None else observed_at_ns
    if component == "native":
        assembler.observe_native_odometry(
            stamp_ns=stamp_ns,
            observed_at_ns=arrival_ns,
            pose=POSE,
            frame_id="camera_init",
            child_frame_id="body",
        )
    elif component == "canonical":
        assembler.observe_canonical_odometry(
            stamp_ns=stamp_ns,
            observed_at_ns=arrival_ns,
            pose=POSE,
            frame_id="map",
            child_frame_id="base_link",
        )
    elif component == "effective":
        assembler.observe_effective_points(
            stamp_ns=stamp_ns,
            observed_at_ns=arrival_ns,
            point_count=effective_count,
            layout_valid=True,
        )
    else:
        raise AssertionError(component)


def _complete_bundle(
    assembler: FastlioSignalAssembler,
    stamp_ns: int,
    *,
    effective_count: int = 100,
) -> None:
    for component in ("native", "canonical", "effective"):
        _observe_component(
            assembler,
            component,
            stamp_ns=stamp_ns,
            effective_count=effective_count,
        )


def _prime_inputs(assembler: FastlioSignalAssembler, stamp_ns: int) -> None:
    assembler.observe_lidar_input(
        stamp_ns=stamp_ns,
        has_points=True,
        layout_valid=True,
    )
    assembler.observe_imu_input(stamp_ns=stamp_ns, finite=True)


def test_all_cross_topic_arrival_orders_produce_the_same_bundle() -> None:
    fingerprints: set[bytes] = set()
    for order in itertools.permutations(("native", "canonical", "effective")):
        assembler = _assembler()
        for component in order:
            _observe_component(assembler, component, stamp_ns=123)
        completed = assembler.take_completed()
        assert len(completed) == 1
        assert completed[0].source_stamp_ns == 123
        assert completed[0].effective_point_count == 100
        fingerprints.add(completed[0].content_fingerprint)

    assert len(fingerprints) == 1


def test_odom_first_is_not_missing_during_join_grace() -> None:
    assembler = _assembler()
    stamp_ns = 1_000_000_000
    _prime_inputs(assembler, stamp_ns)
    _observe_component(assembler, "native", stamp_ns=stamp_ns)
    _observe_component(assembler, "canonical", stamp_ns=stamp_ns)

    within_grace = assembler.assess(evaluation_ns=stamp_ns + 50_000_000)
    assert not within_grace.hard_reasons & DegradationReason.SIGNAL_MISSING
    expired = assembler.assess(evaluation_ns=stamp_ns + 50_000_001)
    assert expired.hard_reasons & DegradationReason.SIGNAL_MISSING


def test_join_grace_starts_at_first_arrival_not_old_source_stamp() -> None:
    assembler = _assembler()
    stamp_ns = 1_000_000_000
    arrival_ns = stamp_ns + 60_000_000
    _prime_inputs(assembler, arrival_ns)
    _observe_component(
        assembler,
        "native",
        stamp_ns=stamp_ns,
        observed_at_ns=arrival_ns,
    )
    _observe_component(
        assembler,
        "canonical",
        stamp_ns=stamp_ns,
        observed_at_ns=arrival_ns,
    )

    within_grace = assembler.assess(
        evaluation_ns=arrival_ns + 50_000_000
    )
    assert not within_grace.hard_reasons & DegradationReason.SIGNAL_MISSING
    expired = assembler.assess(evaluation_ns=arrival_ns + 50_000_001)
    assert expired.hard_reasons & DegradationReason.SIGNAL_MISSING


def test_one_nanosecond_stamp_mismatch_never_joins() -> None:
    assembler = _assembler()
    stamp_ns = 1_000_000_000
    _prime_inputs(assembler, stamp_ns)
    _observe_component(assembler, "native", stamp_ns=stamp_ns)
    _observe_component(assembler, "canonical", stamp_ns=stamp_ns)
    _observe_component(assembler, "effective", stamp_ns=stamp_ns + 1)

    assert assembler.take_completed() == []
    assessment = assembler.assess(evaluation_ns=stamp_ns + 60_000_000)
    assert assessment.hard_reasons & DegradationReason.SIGNAL_MISSING


def test_zero_effect_is_a_complete_hard_fault_bundle() -> None:
    assembler = _assembler()
    _complete_bundle(assembler, 100, effective_count=0)

    completed = assembler.take_completed()
    assert len(completed) == 1
    assert completed[0].source_stamp_ns == 100
    assert completed[0].effective_point_count == 0
    assert completed[0].hard_reasons & (
        DegradationReason.INSUFFICIENT_SUPPORT
    )
    assert not completed[0].hard_reasons & DegradationReason.SIGNAL_MISSING


def test_low_positive_support_is_not_a_hard_gate_before_calibration() -> None:
    assembler = _assembler()
    _complete_bundle(assembler, 100, effective_count=1)

    completed = assembler.take_completed()
    assert len(completed) == 1
    assert completed[0].hard_reasons == DegradationReason.NONE


def test_lidar_and_imu_never_advance_complete_source() -> None:
    assembler = _assembler()
    _prime_inputs(assembler, 5_000_000_000)

    assert assembler.last_complete_stamp_ns is None
    assert assembler.take_completed() == []


def test_fresh_odom_with_missing_effect_is_not_odom_stale() -> None:
    assembler = _assembler()
    stamp_ns = 1_000_000_000
    _prime_inputs(assembler, stamp_ns + 100_000_000)
    _observe_component(assembler, "native", stamp_ns=stamp_ns)
    _observe_component(assembler, "canonical", stamp_ns=stamp_ns)

    assessment = assembler.assess(evaluation_ns=stamp_ns + 60_000_000)
    assert assessment.hard_reasons & DegradationReason.SIGNAL_MISSING
    assert not assessment.hard_reasons & DegradationReason.ODOMETRY_STALE


def test_newer_complete_bundle_turns_old_gap_into_one_shot_event() -> None:
    assembler = _assembler()
    _observe_component(assembler, "native", stamp_ns=100)
    _observe_component(assembler, "canonical", stamp_ns=100)
    _complete_bundle(assembler, 200)

    completed = assembler.take_completed()
    assert [bundle.source_stamp_ns for bundle in completed] == [200]
    assert assembler.take_events() & DegradationReason.SIGNAL_MISSING
    assert assembler._first_arrival_ns == {}

    _prime_inputs(assembler, 200)
    assessment = assembler.assess(evaluation_ns=200)
    assert not assessment.hard_reasons & DegradationReason.SIGNAL_MISSING


def test_old_gap_event_is_invalid_once_then_allows_recovery() -> None:
    machine = SlamConfidenceStateMachine(
        StateMachineConfig(
            source_timeout_ns=seconds_to_nanoseconds(10.0),
            future_tolerance_ns=seconds_to_nanoseconds(0.005),
            degrade_below=0.45,
            degrade_dwell_ns=seconds_to_nanoseconds(0.10),
            invalidate_below=0.25,
            invalidate_dwell_ns=seconds_to_nanoseconds(0.20),
            recover_at_or_above=0.55,
            recover_dwell_ns=seconds_to_nanoseconds(0.50),
        ),
        backend_id="fastlio2",
        calibration_id="test-calibration",
    )
    machine.observe_source(
        SourceObservation(source_stamp_ns=0, content_fingerprint=b"initial"),
        ros_now_ns=0,
    )
    machine.evaluate(evaluation_ns=0, confidence=0.8)
    assert machine.evaluate(
        evaluation_ns=500_000_000,
        confidence=0.8,
    ).state == TrackingState.TRACKING

    assembler = _assembler()
    _observe_component(assembler, "native", stamp_ns=600_000_000)
    _observe_component(assembler, "canonical", stamp_ns=600_000_000)
    _complete_bundle(assembler, 700_000_000)
    event = assembler.take_events()
    machine.observe_hard_event(event, ros_now_ns=700_000_000)
    bundle = assembler.take_completed()[0]
    machine.observe_source(
        SourceObservation(
            source_stamp_ns=bundle.source_stamp_ns,
            content_fingerprint=bundle.content_fingerprint,
            hard_reasons=bundle.hard_reasons,
        ),
        ros_now_ns=700_000_000,
    )

    failed = machine.evaluate(evaluation_ns=700_000_000, confidence=0.8)
    assert failed.state == TrackingState.LOST
    assert failed.reasons & DegradationReason.SIGNAL_MISSING
    recovering = machine.evaluate(
        evaluation_ns=800_000_000,
        confidence=0.8,
    )
    assert recovering.state == TrackingState.LOST
    recovered = machine.evaluate(
        evaluation_ns=1_300_000_000,
        confidence=0.8,
    )
    assert recovered.state == TrackingState.TRACKING
    assert not recovered.reasons & DegradationReason.SIGNAL_MISSING


def test_each_input_has_an_independent_freshness_reason() -> None:
    assembler = _assembler()
    stamp_ns = 1_000_000_000
    _complete_bundle(assembler, stamp_ns)
    _prime_inputs(assembler, stamp_ns)
    assembler.take_completed()

    imu_stale = assembler.assess(evaluation_ns=stamp_ns + 50_000_001)
    assert imu_stale.hard_reasons & DegradationReason.IMU_STALE
    assert not imu_stale.hard_reasons & DegradationReason.LIDAR_STALE
    assert not imu_stale.hard_reasons & DegradationReason.ODOMETRY_STALE

    all_stale = assembler.assess(evaluation_ns=stamp_ns + 300_000_001)
    assert all_stale.hard_reasons & DegradationReason.IMU_STALE
    assert all_stale.hard_reasons & DegradationReason.LIDAR_STALE
    assert all_stale.hard_reasons & DegradationReason.ODOMETRY_STALE


def test_freshness_exact_boundary_remains_valid() -> None:
    assembler = _assembler()
    stamp_ns = 1_000_000_000
    _complete_bundle(assembler, stamp_ns)
    _prime_inputs(assembler, stamp_ns)

    assessment = assembler.assess(evaluation_ns=stamp_ns + 50_000_000)
    assert not assessment.hard_reasons & DegradationReason.IMU_STALE


def test_native_numeric_and_frame_errors_are_visible_before_adapter() -> None:
    assembler = _assembler()
    invalid_pose = (math.nan, *POSE[1:])
    assembler.observe_native_odometry(
        stamp_ns=100,
        observed_at_ns=100,
        pose=invalid_pose,
        frame_id="camera_init",
        child_frame_id="body",
    )
    assembler.observe_native_odometry(
        stamp_ns=101,
        observed_at_ns=101,
        pose=POSE,
        frame_id="wrong",
        child_frame_id="body",
    )

    events = assembler.take_events()
    assert events & DegradationReason.NUMERIC_INVALID
    assert events & DegradationReason.BACKEND_ERROR


def test_zero_quaternion_is_numeric_invalid() -> None:
    assembler = _assembler()
    assembler.observe_native_odometry(
        stamp_ns=100,
        observed_at_ns=100,
        pose=(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0),
        frame_id="camera_init",
        child_frame_id="body",
    )

    assert assembler.take_events() & DegradationReason.NUMERIC_INVALID


def test_duplicate_is_idempotent_but_conflict_and_regression_are_invalid() -> None:
    assembler = _assembler()
    _observe_component(assembler, "effective", stamp_ns=100, effective_count=5)
    _observe_component(assembler, "effective", stamp_ns=100, effective_count=5)
    assert assembler.take_events() == DegradationReason.NONE

    _observe_component(assembler, "effective", stamp_ns=100, effective_count=6)
    assert assembler.take_events() & DegradationReason.TIMESTAMP_INVALID
    _observe_component(assembler, "effective", stamp_ns=99, effective_count=5)
    assert assembler.take_events() & DegradationReason.TIMESTAMP_INVALID


def test_malformed_effect_is_missing_not_zero_support() -> None:
    assembler = _assembler()
    assembler.observe_effective_points(
        stamp_ns=100,
        observed_at_ns=100,
        point_count=0,
        layout_valid=False,
    )

    events = assembler.take_events()
    assert events & DegradationReason.NUMERIC_INVALID
    assert events & DegradationReason.SIGNAL_MISSING
    assert not events & DegradationReason.INSUFFICIENT_SUPPORT


def test_empty_lidar_does_not_refresh_backend_input() -> None:
    assembler = _assembler()
    assembler.observe_lidar_input(
        stamp_ns=100,
        has_points=False,
        layout_valid=True,
    )

    assert assembler.latest_lidar_stamp_ns is None
    assert assembler.take_events() & DegradationReason.SIGNAL_MISSING


def test_future_input_stamp_is_timestamp_invalid() -> None:
    assembler = _assembler()
    stamp_ns = 10_000_000
    _complete_bundle(assembler, stamp_ns)
    _prime_inputs(assembler, stamp_ns)

    assessment = assembler.assess(evaluation_ns=4_999_999)
    assert assessment.hard_reasons & DegradationReason.TIMESTAMP_INVALID


def test_pending_caches_are_bounded() -> None:
    assembler = _assembler(max_pending=2)
    for stamp_ns in (1, 2, 3):
        _observe_component(assembler, "canonical", stamp_ns=stamp_ns)

    assert len(assembler._canonical) == 2
    assert list(assembler._canonical) == [2, 3]
    assert assembler.take_events() & DegradationReason.SIGNAL_MISSING


def test_reset_clears_old_epoch_pending_and_completed_state() -> None:
    assembler = _assembler()
    _complete_bundle(assembler, 100)
    assembler.reset()

    assert assembler.last_complete_stamp_ns is None
    assert assembler.take_completed() == []
    assert assembler.take_events() == DegradationReason.NONE


def test_assembler_nanosecond_fields_require_exact_integers() -> None:
    values = vars(_config()).copy()
    values["diagnostic_match_timeout_ns"] = 0.05
    with pytest.raises(TypeError, match="integer"):
        FastlioAssemblerConfig(**values)
