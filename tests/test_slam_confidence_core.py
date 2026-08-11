"""Deterministic tests for the backend-neutral confidence state machine."""

from __future__ import annotations

import math
import struct

import pytest

from anymal_locomotion_ros2.slam_confidence_core import (
    ConfidenceSnapshot,
    DegradationReason,
    HealthAssessment,
    SlamConfidenceStateMachine,
    SourceDisposition,
    SourceObservation,
    StateMachineConfig,
    TrackingState,
    seconds_to_nanoseconds,
)


NS = 1_000_000_000


def _config(*, source_timeout_s: float = 10.0) -> StateMachineConfig:
    return StateMachineConfig(
        source_timeout_ns=seconds_to_nanoseconds(source_timeout_s),
        future_tolerance_ns=seconds_to_nanoseconds(0.005),
        degrade_below=0.45,
        degrade_dwell_ns=seconds_to_nanoseconds(0.10),
        invalidate_below=0.25,
        invalidate_dwell_ns=seconds_to_nanoseconds(0.20),
        recover_at_or_above=0.55,
        recover_dwell_ns=seconds_to_nanoseconds(0.50),
    )


def _machine(*, source_timeout_s: float = 10.0) -> SlamConfidenceStateMachine:
    return SlamConfidenceStateMachine(
        _config(source_timeout_s=source_timeout_s),
        backend_id="fastlio2",
        calibration_id="test-calibration",
    )


def _source(
    stamp_ns: int,
    *,
    fingerprint: bytes | None = None,
    hard: DegradationReason = DegradationReason.NONE,
    quality: DegradationReason = DegradationReason.NONE,
) -> SourceObservation:
    return SourceObservation(
        source_stamp_ns=stamp_ns,
        content_fingerprint=fingerprint or f"sample-{stamp_ns}".encode(),
        hard_reasons=hard,
        quality_reasons=quality,
    )


def _observe_and_evaluate(
    machine: SlamConfidenceStateMachine,
    stamp_ns: int,
    confidence: float,
    *,
    hard: DegradationReason = DegradationReason.NONE,
    quality: DegradationReason = DegradationReason.NONE,
) -> ConfidenceSnapshot:
    disposition = machine.observe_source(
        _source(
            stamp_ns,
            hard=hard,
            quality=quality,
        ),
        ros_now_ns=stamp_ns,
    )
    assert disposition == SourceDisposition.ACCEPTED
    return machine.evaluate(
        evaluation_ns=stamp_ns,
        confidence=confidence,
    )


def _reach_tracking(
    machine: SlamConfidenceStateMachine,
    *,
    confidence: float = 0.8,
) -> ConfidenceSnapshot:
    snapshot: ConfidenceSnapshot | None = None
    for step in range(6):
        stamp_ns = step * 100_000_000
        snapshot = _observe_and_evaluate(machine, stamp_ns, confidence)
    assert snapshot is not None
    assert snapshot.state == TrackingState.TRACKING
    return snapshot


def _previous_float32(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits - 1))[0]


def test_no_source_is_initializing_and_fail_closed() -> None:
    snapshot = _machine().evaluate(evaluation_ns=0)

    assert snapshot.source_stamp_ns is None
    assert snapshot.confidence == 0.0
    assert snapshot.confidence_age_ns == 0
    assert snapshot.state == TrackingState.INITIALIZING
    assert snapshot.tracking_valid is False
    assert snapshot.reasons & DegradationReason.INITIALIZING
    assert snapshot.reasons & DegradationReason.SIGNAL_MISSING


def test_ros_time_zero_is_a_valid_source_stamp() -> None:
    machine = _machine()
    snapshot = _observe_and_evaluate(machine, 0, 0.8)

    assert snapshot.source_stamp_ns == 0
    assert snapshot.confidence_age_ns == 0
    assert snapshot.state == TrackingState.INITIALIZING


def test_recovery_dwell_uses_time_not_source_count() -> None:
    machine = _machine()
    machine.observe_source(_source(0), ros_now_ns=0)

    assert machine.evaluate(
        evaluation_ns=0,
        confidence=0.8,
    ).state == TrackingState.INITIALIZING
    assert machine.evaluate(
        evaluation_ns=499_999_999,
        confidence=0.8,
    ).state == (
        TrackingState.INITIALIZING
    )
    assert machine.evaluate(
        evaluation_ns=500_000_000,
        confidence=0.8,
    ).state == (
        TrackingState.TRACKING
    )


def test_repeated_equal_evaluation_time_does_not_advance_dwell() -> None:
    machine = _machine()
    machine.observe_source(_source(0), ros_now_ns=0)
    for _ in range(20):
        assert machine.evaluate(evaluation_ns=0, confidence=0.8).state == (
            TrackingState.INITIALIZING
        )


def test_degrade_and_invalidate_timers_accumulate_concurrently() -> None:
    machine = _machine()
    _reach_tracking(machine)
    start_ns = 600_000_000
    _observe_and_evaluate(machine, start_ns, 0.24)

    assert machine.evaluate(
        evaluation_ns=start_ns + 99_999_999,
        confidence=0.24,
    ).state == (
        TrackingState.TRACKING
    )
    degraded = machine.evaluate(
        evaluation_ns=start_ns + 100_000_000,
        confidence=0.24,
    )
    assert degraded.state == TrackingState.DEGRADED
    assert degraded.tracking_valid is True
    assert machine.evaluate(
        evaluation_ns=start_ns + 199_999_999,
        confidence=0.24,
    ).state == (
        TrackingState.DEGRADED
    )
    lost = machine.evaluate(
        evaluation_ns=start_ns + 200_000_000,
        confidence=0.24,
    )
    assert lost.state == TrackingState.LOST
    assert lost.tracking_valid is False
    assert lost.reasons & DegradationReason.LOW_CONFIDENCE


def test_exact_wire_thresholds_do_not_trigger_lower_bands() -> None:
    machine = _machine()
    _reach_tracking(machine)
    config = _config()

    _observe_and_evaluate(machine, 600_000_000, config.degrade_below)
    assert machine.evaluate(
        evaluation_ns=800_000_000,
        confidence=config.degrade_below,
    ).state == (
        TrackingState.TRACKING
    )

    _observe_and_evaluate(machine, 900_000_000, config.invalidate_below)
    assert machine.evaluate(
        evaluation_ns=1_200_000_000,
        confidence=config.invalidate_below,
    ).state != (
        TrackingState.LOST
    )


def test_one_float32_ulp_below_degrade_threshold_triggers_dwell() -> None:
    machine = _machine()
    _reach_tracking(machine)
    score = _previous_float32(_config().degrade_below)

    _observe_and_evaluate(machine, 600_000_000, score)
    assert machine.evaluate(
        evaluation_ns=700_000_000,
        confidence=score,
    ).state == (
        TrackingState.DEGRADED
    )


def test_degrade_condition_interruption_resets_dwell() -> None:
    machine = _machine()
    _reach_tracking(machine)
    _observe_and_evaluate(machine, 600_000_000, 0.4)
    machine.evaluate(evaluation_ns=650_000_000, confidence=0.4)
    _observe_and_evaluate(machine, 700_000_000, 0.8)
    _observe_and_evaluate(machine, 750_000_000, 0.4)

    assert machine.evaluate(
        evaluation_ns=849_999_999,
        confidence=0.4,
    ).state == (
        TrackingState.TRACKING
    )
    assert machine.evaluate(
        evaluation_ns=850_000_000,
        confidence=0.4,
    ).state == (
        TrackingState.DEGRADED
    )


def test_degraded_requires_recovery_dwell() -> None:
    machine = _machine()
    _reach_tracking(machine)
    _observe_and_evaluate(machine, 600_000_000, 0.4)
    machine.evaluate(evaluation_ns=700_000_000, confidence=0.4)
    _observe_and_evaluate(machine, 800_000_000, 0.8)

    pending = machine.evaluate(
        evaluation_ns=1_299_999_999,
        confidence=0.8,
    )
    assert pending.state == TrackingState.DEGRADED
    assert pending.reasons & DegradationReason.RECOVERY_PENDING
    assert machine.evaluate(
        evaluation_ns=1_300_000_000,
        confidence=0.8,
    ).state == (
        TrackingState.TRACKING
    )


def test_hard_fault_is_immediate_and_root_reason_latches() -> None:
    machine = _machine()
    _reach_tracking(machine, confidence=0.95)

    failed = machine.evaluate(
        evaluation_ns=600_000_000,
        health=HealthAssessment(
            hard_reasons=DegradationReason.IMU_STALE,
        ),
    )
    assert failed.state == TrackingState.LOST
    assert failed.confidence == pytest.approx(0.95)
    assert failed.reasons & DegradationReason.IMU_STALE

    recovering = machine.evaluate(
        evaluation_ns=700_000_000,
        confidence=0.95,
    )
    assert recovering.state == TrackingState.LOST
    assert recovering.reasons & DegradationReason.IMU_STALE
    assert recovering.reasons & DegradationReason.RECOVERY_PENDING
    recovered = machine.evaluate(
        evaluation_ns=1_200_000_000,
        confidence=0.95,
    )
    assert recovered.state == TrackingState.TRACKING
    assert not recovered.reasons & DegradationReason.IMU_STALE


def test_stale_high_confidence_keeps_score_but_is_invalid() -> None:
    machine = _machine(source_timeout_s=0.30)
    machine.observe_source(_source(0), ros_now_ns=0)

    boundary = machine.evaluate(
        evaluation_ns=300_000_000,
        confidence=0.95,
    )
    assert not boundary.reasons & DegradationReason.SOURCE_STALE
    stale = machine.evaluate(evaluation_ns=300_000_001)
    assert stale.confidence == pytest.approx(0.95)
    assert stale.state == TrackingState.LOST
    assert stale.reasons & DegradationReason.SOURCE_STALE


def test_score_recomputes_each_tick_and_hard_fault_holds_last_score() -> None:
    machine = _machine()
    machine.observe_source(_source(0), ros_now_ns=0)

    first = machine.evaluate(evaluation_ns=0, confidence=0.8)
    second = machine.evaluate(evaluation_ns=50_000_000, confidence=0.6)
    failed = machine.evaluate(
        evaluation_ns=100_000_000,
        health=HealthAssessment(
            hard_reasons=DegradationReason.ODOMETRY_STALE,
        ),
        confidence=0.1,
    )

    assert first.confidence == struct.unpack("<f", struct.pack("<f", 0.8))[0]
    assert second.confidence == struct.unpack("<f", struct.pack("<f", 0.6))[0]
    assert failed.confidence == second.confidence
    assert failed.reasons & DegradationReason.ODOMETRY_STALE


def test_zero_support_can_advance_source_and_fail_hard() -> None:
    machine = _machine()
    snapshot = _observe_and_evaluate(
        machine,
        123,
        0.7,
        hard=DegradationReason.INSUFFICIENT_SUPPORT,
    )

    assert snapshot.source_stamp_ns == 123
    assert snapshot.state == TrackingState.LOST
    assert snapshot.reasons & DegradationReason.INSUFFICIENT_SUPPORT


def test_same_reason_can_be_soft_or_hard_but_not_both() -> None:
    soft = _source(
        0,
        quality=DegradationReason.INSUFFICIENT_SUPPORT,
    )
    assert soft.quality_reasons & DegradationReason.INSUFFICIENT_SUPPORT

    with pytest.raises(ValueError, match="both hard and quality"):
        _source(
            0,
            hard=DegradationReason.INSUFFICIENT_SUPPORT,
            quality=DegradationReason.INSUFFICIENT_SUPPORT,
        )


def test_duplicate_is_idempotent_and_conflict_is_invalid() -> None:
    machine = _machine()
    sample = _source(100, fingerprint=b"same")
    assert machine.observe_source(sample, ros_now_ns=100) == (
        SourceDisposition.ACCEPTED
    )
    assert machine.observe_source(sample, ros_now_ns=100) == (
        SourceDisposition.DUPLICATE
    )
    conflict = _source(100, fingerprint=b"different")
    assert machine.observe_source(conflict, ros_now_ns=100) == (
        SourceDisposition.REJECTED_CONFLICT
    )
    snapshot = machine.evaluate(evaluation_ns=100)
    assert snapshot.source_stamp_ns == 100
    assert snapshot.state == TrackingState.LOST
    assert snapshot.reasons & DegradationReason.TIMESTAMP_INVALID


def test_source_regression_preserves_last_good_source() -> None:
    machine = _machine()
    machine.observe_source(_source(200), ros_now_ns=200)
    machine.evaluate(evaluation_ns=200, confidence=0.8)
    disposition = machine.observe_source(_source(199), ros_now_ns=201)

    assert disposition == SourceDisposition.REJECTED_REGRESSION
    snapshot = machine.evaluate(evaluation_ns=201)
    assert snapshot.source_stamp_ns == 200
    assert snapshot.confidence == pytest.approx(0.8)
    assert snapshot.reasons & DegradationReason.TIMESTAMP_INVALID


def test_transient_hard_event_between_ticks_is_reported_lost() -> None:
    machine = _machine()
    _reach_tracking(machine)
    machine.observe_hard_event(
        DegradationReason.NUMERIC_INVALID,
        ros_now_ns=600_000_000,
    )
    machine.observe_source(
        _source(700_000_000),
        ros_now_ns=700_000_000,
    )

    snapshot = machine.evaluate(evaluation_ns=700_000_000)
    assert snapshot.state == TrackingState.LOST
    assert snapshot.reasons & DegradationReason.NUMERIC_INVALID


def test_hard_source_between_ticks_is_latched_until_evaluation() -> None:
    machine = _machine()
    _reach_tracking(machine)
    machine.observe_source(
        _source(
            600_000_000,
            hard=DegradationReason.INSUFFICIENT_SUPPORT,
        ),
        ros_now_ns=600_000_000,
    )
    machine.observe_source(
        _source(700_000_000),
        ros_now_ns=700_000_000,
    )

    snapshot = machine.evaluate(
        evaluation_ns=700_000_000,
        confidence=0.8,
    )
    assert snapshot.state == TrackingState.LOST
    assert snapshot.reasons & DegradationReason.INSUFFICIENT_SUPPORT


def test_future_source_between_ticks_is_latched_until_evaluation() -> None:
    machine = _machine()
    _reach_tracking(machine)
    machine.observe_source(
        _source(606_000_000),
        ros_now_ns=600_000_000,
    )
    machine.observe_source(
        _source(700_000_000),
        ros_now_ns=700_000_000,
    )

    snapshot = machine.evaluate(
        evaluation_ns=700_000_000,
        confidence=0.8,
    )
    assert snapshot.state == TrackingState.LOST
    assert snapshot.reasons & DegradationReason.TIMESTAMP_INVALID


def test_new_source_without_new_score_fails_closed() -> None:
    machine = _machine()
    machine.observe_source(_source(0), ros_now_ns=0)
    first = machine.evaluate(evaluation_ns=0, confidence=0.8)
    assert first.confidence > 0.0
    machine.observe_source(_source(100_000_000), ros_now_ns=100_000_000)

    missing = machine.evaluate(evaluation_ns=100_000_000)
    assert missing.source_stamp_ns == 100_000_000
    assert missing.confidence == first.confidence
    assert missing.state == TrackingState.LOST
    assert missing.reasons & DegradationReason.SIGNAL_MISSING


def test_same_source_healthy_tick_without_new_score_fails_closed() -> None:
    machine = _machine()
    machine.observe_source(_source(0), ros_now_ns=0)
    first = machine.evaluate(evaluation_ns=0, confidence=0.8)

    missing = machine.evaluate(evaluation_ns=50_000_000)
    assert missing.source_stamp_ns == 0
    assert missing.confidence == first.confidence
    assert missing.state == TrackingState.LOST
    assert missing.reasons & DegradationReason.SIGNAL_MISSING


def test_reason_severity_classes_cannot_be_swapped() -> None:
    with pytest.raises(ValueError, match="hard-only"):
        HealthAssessment(
            quality_reasons=DegradationReason.UNCALIBRATED,
        )
    with pytest.raises(ValueError, match="soft-only"):
        HealthAssessment(
            hard_reasons=DegradationReason.DEGENERATE_GEOMETRY,
        )


def test_nanosecond_config_fields_reject_float_and_bool() -> None:
    values = vars(_config()).copy()
    values["degrade_dwell_ns"] = 0.1
    with pytest.raises(TypeError, match="integer"):
        StateMachineConfig(**values)

    values = vars(_config()).copy()
    values["source_timeout_ns"] = True
    with pytest.raises(TypeError, match="integer"):
        StateMachineConfig(**values)


def test_pre_source_hard_event_latches_into_recovery() -> None:
    machine = _machine()
    machine.observe_hard_event(
        DegradationReason.NUMERIC_INVALID,
        ros_now_ns=0,
    )
    initializing = machine.evaluate(evaluation_ns=0)
    assert initializing.state == TrackingState.INITIALIZING
    assert initializing.reasons & DegradationReason.NUMERIC_INVALID

    machine.observe_source(_source(100_000_000), ros_now_ns=100_000_000)
    recovering = machine.evaluate(
        evaluation_ns=100_000_000,
        confidence=0.8,
    )
    assert recovering.state == TrackingState.INITIALIZING
    assert recovering.reasons & DegradationReason.NUMERIC_INVALID


def test_pre_source_event_is_order_independent_of_first_evaluation() -> None:
    snapshots: list[ConfidenceSnapshot] = []
    for evaluate_before_source in (False, True):
        machine = _machine()
        machine.observe_hard_event(
            DegradationReason.NUMERIC_INVALID,
            ros_now_ns=0,
        )
        if evaluate_before_source:
            machine.evaluate(evaluation_ns=0)
        machine.observe_source(_source(100_000_000), ros_now_ns=100_000_000)
        snapshots.append(
            machine.evaluate(
                evaluation_ns=100_000_000,
                confidence=0.8,
            )
        )

    assert snapshots[0].state == snapshots[1].state == (
        TrackingState.INITIALIZING
    )
    assert snapshots[0].reasons == snapshots[1].reasons
    assert snapshots[0].reasons & DegradationReason.NUMERIC_INVALID


def test_future_stamp_tolerance_and_numeric_score() -> None:
    machine = _machine()
    machine.observe_source(_source(5_000_000), ros_now_ns=0)

    at_tolerance = machine.evaluate(evaluation_ns=0, confidence=math.nan)
    assert at_tolerance.reasons & DegradationReason.NUMERIC_INVALID
    assert not at_tolerance.reasons & DegradationReason.TIMESTAMP_INVALID
    assert at_tolerance.confidence == 0.0

    machine = _machine()
    machine.observe_source(_source(5_000_001), ros_now_ns=0)
    beyond = machine.evaluate(evaluation_ns=0, confidence=0.8)
    assert beyond.reasons & DegradationReason.TIMESTAMP_INVALID


def test_clock_reset_clears_source_and_dwell() -> None:
    machine = _machine()
    machine.observe_source(_source(NS), ros_now_ns=NS)
    machine.evaluate(evaluation_ns=NS, confidence=0.8)

    reset = machine.evaluate(evaluation_ns=0)
    assert reset.source_stamp_ns is None
    assert reset.state == TrackingState.INITIALIZING
    assert reset.reasons & DegradationReason.CLOCK_RESET
    assert machine.observe_source(_source(0), ros_now_ns=0) == (
        SourceDisposition.ACCEPTED
    )


def test_instance_switch_clears_source_and_requires_reinitialization() -> None:
    machine = _machine()
    _reach_tracking(machine)

    assert machine.switch_instance(
        backend_id="fastlio2",
        calibration_id="new-calibration",
        ros_now_ns=600_000_000,
    )
    snapshot = machine.evaluate(evaluation_ns=600_000_000)
    assert snapshot.source_stamp_ns is None
    assert snapshot.state == TrackingState.INITIALIZING
    assert snapshot.calibration_id == "new-calibration"


def test_wire_identity_bounds_and_reason_invariants() -> None:
    with pytest.raises(ValueError, match="32-byte"):
        SlamConfidenceStateMachine(
            _config(),
            backend_id="x" * 33,
            calibration_id="ok",
        )

    machine = _machine()
    snapshots = [
        machine.evaluate(evaluation_ns=0),
        _observe_and_evaluate(
            machine,
            1,
            0.8,
            hard=DegradationReason.UNCALIBRATED,
        ),
    ]
    for snapshot in snapshots:
        assert snapshot.tracking_valid == (
            snapshot.state
            in {
                TrackingState.TRACKING,
                TrackingState.DEGRADED,
            }
        )
        if snapshot.state != TrackingState.TRACKING:
            assert snapshot.reasons != DegradationReason.NONE
