"""ROS-independent state machine for backend-neutral SLAM confidence."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import Enum, IntEnum, IntFlag


class TrackingState(IntEnum):
    """Wire-compatible tracking states from ``SlamConfidence.msg``."""

    INITIALIZING = 0
    TRACKING = 1
    DEGRADED = 2
    LOST = 3


class DegradationReason(IntFlag):
    """Wire-compatible reason bits from ``SlamConfidence.msg``."""

    NONE = 0
    INITIALIZING = 1
    SOURCE_STALE = 2
    ODOMETRY_STALE = 4
    LIDAR_STALE = 8
    IMU_STALE = 16
    INSUFFICIENT_SUPPORT = 32
    DEGENERATE_GEOMETRY = 64
    HIGH_RESIDUAL = 128
    NOT_CONVERGED = 256
    ESTIMATOR_RESET = 512
    TIMESTAMP_INVALID = 1024
    NUMERIC_INVALID = 2048
    BACKEND_ERROR = 4096
    SIGNAL_MISSING = 8192
    LOW_CONFIDENCE = 16384
    RECOVERY_PENDING = 32768
    CLOCK_RESET = 65536
    UNCALIBRATED = 131072


_LIFECYCLE_REASONS = (
    DegradationReason.INITIALIZING
    | DegradationReason.LOW_CONFIDENCE
    | DegradationReason.RECOVERY_PENDING
)
_IMMEDIATE_HARD_REASONS = (
    DegradationReason.SOURCE_STALE
    | DegradationReason.ODOMETRY_STALE
    | DegradationReason.LIDAR_STALE
    | DegradationReason.IMU_STALE
    | DegradationReason.ESTIMATOR_RESET
    | DegradationReason.TIMESTAMP_INVALID
    | DegradationReason.NUMERIC_INVALID
    | DegradationReason.BACKEND_ERROR
    | DegradationReason.CLOCK_RESET
    | DegradationReason.UNCALIBRATED
)
_CALIBRATED_SOFT_REASONS = (
    DegradationReason.DEGENERATE_GEOMETRY
    | DegradationReason.HIGH_RESIDUAL
    | DegradationReason.NOT_CONVERGED
)
_CONDITIONAL_REASONS = (
    DegradationReason.INSUFFICIENT_SUPPORT
    | DegradationReason.SIGNAL_MISSING
)
_ALLOWED_HARD_REASONS = _IMMEDIATE_HARD_REASONS | _CONDITIONAL_REASONS
_ALLOWED_QUALITY_REASONS = _CALIBRATED_SOFT_REASONS | _CONDITIONAL_REASONS
_KNOWN_REASON_MASK = DegradationReason.NONE
for _reason in DegradationReason:
    _KNOWN_REASON_MASK |= _reason


class SourceDisposition(Enum):
    """Result of ingesting a source observation."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    REJECTED_CONFLICT = "rejected_conflict"
    REJECTED_REGRESSION = "rejected_regression"
    REJECTED_CLOCK_RESET = "rejected_clock_reset"


@dataclass(frozen=True)
class StateMachineConfig:
    """Logical-time thresholds for the shared state machine."""

    source_timeout_ns: int
    future_tolerance_ns: int
    degrade_below: float
    degrade_dwell_ns: int
    invalidate_below: float
    invalidate_dwell_ns: int
    recover_at_or_above: float
    recover_dwell_ns: int

    def __post_init__(self) -> None:
        for field_name in (
            "source_timeout_ns",
            "future_tolerance_ns",
            "degrade_dwell_ns",
            "invalidate_dwell_ns",
            "recover_dwell_ns",
        ):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an integer")
        if self.source_timeout_ns <= 0:
            raise ValueError("source_timeout_ns must be positive")
        if self.future_tolerance_ns < 0:
            raise ValueError("future_tolerance_ns must be non-negative")
        thresholds = tuple(
            _wire_float32(value)
            for value in (
                self.invalidate_below,
                self.degrade_below,
                self.recover_at_or_above,
            )
        )
        if not all(math.isfinite(value) for value in thresholds):
            raise ValueError("confidence thresholds must be finite")
        if not (
            0.0
            < thresholds[0]
            < thresholds[1]
            < thresholds[2]
            <= 1.0
        ):
            raise ValueError("confidence thresholds are not ordered")
        object.__setattr__(self, "invalidate_below", thresholds[0])
        object.__setattr__(self, "degrade_below", thresholds[1])
        object.__setattr__(self, "recover_at_or_above", thresholds[2])
        if self.degrade_dwell_ns <= 0:
            raise ValueError("degrade_dwell_ns must be positive")
        if self.invalidate_dwell_ns < self.degrade_dwell_ns:
            raise ValueError(
                "invalidate_dwell_ns must be at least degrade_dwell_ns"
            )
        if self.recover_dwell_ns <= self.invalidate_dwell_ns:
            raise ValueError(
                "recover_dwell_ns must exceed invalidate_dwell_ns"
            )


@dataclass(frozen=True)
class SourceObservation:
    """One complete source-aligned diagnostic bundle."""

    source_stamp_ns: int
    content_fingerprint: bytes
    hard_reasons: DegradationReason = DegradationReason.NONE
    quality_reasons: DegradationReason = DegradationReason.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.source_stamp_ns, int):
            raise TypeError("source_stamp_ns must be an integer")
        if not self.content_fingerprint:
            raise ValueError("content_fingerprint must not be empty")
        _validate_reason_masks(self.hard_reasons, self.quality_reasons)


@dataclass(frozen=True)
class HealthAssessment:
    """Dynamic health conditions evaluated independently of source ingest."""

    hard_reasons: DegradationReason = DegradationReason.NONE
    quality_reasons: DegradationReason = DegradationReason.NONE

    def __post_init__(self) -> None:
        _validate_reason_masks(self.hard_reasons, self.quality_reasons)


@dataclass(frozen=True)
class ConfidenceSnapshot:
    """ROS-independent contents of one authoritative confidence message."""

    backend_id: str
    calibration_id: str
    source_stamp_ns: int | None
    evaluation_stamp_ns: int
    confidence_age_ns: int
    confidence: float
    tracking_valid: bool
    state: TrackingState
    reasons: DegradationReason


def seconds_to_nanoseconds(value: float) -> int:
    """Convert a finite, non-negative duration to integer nanoseconds."""

    if not math.isfinite(value) or value < 0.0:
        raise ValueError("duration must be finite and non-negative")
    return int(round(value * 1_000_000_000.0))


def _wire_float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _validate_identity(value: str, *, field: str, maximum: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if len(normalized.encode("utf-8")) > maximum:
        raise ValueError(f"{field} exceeds its {maximum}-byte wire bound")
    return normalized


def _validate_reason_masks(
    hard_reasons: DegradationReason,
    quality_reasons: DegradationReason,
) -> None:
    hard_value = int(hard_reasons)
    quality_value = int(quality_reasons)
    known_value = int(_KNOWN_REASON_MASK)
    if hard_value & ~known_value or quality_value & ~known_value:
        raise ValueError("reason mask contains an unknown bit")
    if hard_value & quality_value:
        raise ValueError("a reason cannot be both hard and quality-only")
    if (hard_reasons | quality_reasons) & _LIFECYCLE_REASONS:
        raise ValueError("lifecycle reasons are owned by the state machine")
    if hard_reasons & ~_ALLOWED_HARD_REASONS:
        raise ValueError("hard mask contains a calibrated-soft-only reason")
    if quality_reasons & ~_ALLOWED_QUALITY_REASONS:
        raise ValueError("quality mask contains a hard-only reason")


class SlamConfidenceStateMachine:
    """Deterministic logical-time SLAM confidence state machine."""

    def __init__(
        self,
        config: StateMachineConfig,
        *,
        backend_id: str,
        calibration_id: str,
    ) -> None:
        self._config = config
        self._backend_id = _validate_identity(
            backend_id,
            field="backend_id",
            maximum=32,
        )
        self._calibration_id = _validate_identity(
            calibration_id,
            field="calibration_id",
            maximum=128,
        )
        self._latest_clock_ns: int | None = None
        self._last_evaluation_ns: int | None = None
        self._source: SourceObservation | None = None
        self._ever_had_source = False
        self._confidence = 0.0
        self._state = TrackingState.INITIALIZING
        self._latched_reasons = DegradationReason.NONE
        self._source_integrity_fault = DegradationReason.NONE
        self._active_event_faults = DegradationReason.NONE
        self._pending_event_faults = DegradationReason.NONE
        self._degrade_since_ns: int | None = None
        self._invalidate_since_ns: int | None = None
        self._recovery_since_ns: int | None = None

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def calibration_id(self) -> str:
        return self._calibration_id

    @property
    def state(self) -> TrackingState:
        return self._state

    def reset_for_clock(self, *, ros_now_ns: int) -> None:
        """Start a new ROS-clock epoch and discard all prior source state."""

        self._source = None
        self._ever_had_source = False
        self._confidence = 0.0
        self._state = TrackingState.INITIALIZING
        self._latched_reasons = DegradationReason.CLOCK_RESET
        self._source_integrity_fault = DegradationReason.NONE
        self._active_event_faults = DegradationReason.NONE
        self._pending_event_faults = DegradationReason.NONE
        self._clear_dwell()
        self._latest_clock_ns = ros_now_ns
        self._last_evaluation_ns = ros_now_ns

    def switch_instance(
        self,
        *,
        backend_id: str,
        calibration_id: str,
        ros_now_ns: int,
    ) -> bool:
        """Reset state when backend or calibration provenance changes."""

        next_backend = _validate_identity(
            backend_id,
            field="backend_id",
            maximum=32,
        )
        next_calibration = _validate_identity(
            calibration_id,
            field="calibration_id",
            maximum=128,
        )
        if (
            next_backend == self._backend_id
            and next_calibration == self._calibration_id
        ):
            return False
        self._backend_id = next_backend
        self._calibration_id = next_calibration
        self._source = None
        self._ever_had_source = False
        self._confidence = 0.0
        self._state = TrackingState.INITIALIZING
        self._latched_reasons = DegradationReason.NONE
        self._source_integrity_fault = DegradationReason.NONE
        self._active_event_faults = DegradationReason.NONE
        self._pending_event_faults = DegradationReason.NONE
        self._clear_dwell()
        self._latest_clock_ns = ros_now_ns
        self._last_evaluation_ns = None
        return True

    def observe_source(
        self,
        sample: SourceObservation,
        *,
        ros_now_ns: int,
    ) -> SourceDisposition:
        """Ingest a complete source bundle without advancing dwell timers."""

        if self._clock_went_backwards(ros_now_ns):
            self.reset_for_clock(ros_now_ns=ros_now_ns)
            return SourceDisposition.REJECTED_CLOCK_RESET
        self._observe_clock(ros_now_ns)

        normalized = self._normalized_observation(sample)
        if self._source is not None:
            if normalized.source_stamp_ns < self._source.source_stamp_ns:
                self._record_source_fault(DegradationReason.TIMESTAMP_INVALID)
                return SourceDisposition.REJECTED_REGRESSION
            if normalized.source_stamp_ns == self._source.source_stamp_ns:
                if self._same_observation(normalized, self._source):
                    return SourceDisposition.DUPLICATE
                self._record_source_fault(DegradationReason.TIMESTAMP_INVALID)
                return SourceDisposition.REJECTED_CONFLICT

        if not self._ever_had_source:
            self._latched_reasons |= (
                self._active_event_faults | self._pending_event_faults
            )
            self._active_event_faults = DegradationReason.NONE
            self._pending_event_faults = DegradationReason.NONE

        self._source = normalized
        self._ever_had_source = True
        self._source_integrity_fault = DegradationReason.NONE
        self._active_event_faults = DegradationReason.NONE
        if normalized.hard_reasons:
            self._pending_event_faults |= normalized.hard_reasons
        if (
            normalized.source_stamp_ns - ros_now_ns
            > self._config.future_tolerance_ns
        ):
            self._record_source_fault(DegradationReason.TIMESTAMP_INVALID)
        return SourceDisposition.ACCEPTED

    def observe_hard_event(
        self,
        reasons: DegradationReason,
        *,
        ros_now_ns: int,
    ) -> None:
        """Latch a callback-time hard fault until at least one evaluation."""

        _validate_reason_masks(reasons, DegradationReason.NONE)
        if reasons == DegradationReason.NONE:
            raise ValueError("hard event reasons must not be empty")
        if self._clock_went_backwards(ros_now_ns):
            self.reset_for_clock(ros_now_ns=ros_now_ns)
            return
        self._observe_clock(ros_now_ns)
        self._active_event_faults |= reasons
        self._pending_event_faults |= reasons

    def evaluate(
        self,
        *,
        evaluation_ns: int,
        health: HealthAssessment = HealthAssessment(),
        confidence: float | None = None,
    ) -> ConfidenceSnapshot:
        """Advance freshness and hysteresis on the ROS logical-time grid."""

        if self._clock_went_backwards(evaluation_ns):
            self.reset_for_clock(ros_now_ns=evaluation_ns)
            return self._snapshot_without_source(
                evaluation_ns,
                health,
                extra_reasons=DegradationReason.CLOCK_RESET,
            )
        self._observe_clock(evaluation_ns)
        self._last_evaluation_ns = evaluation_ns

        if self._source is None:
            snapshot = self._snapshot_without_source(evaluation_ns, health)
            self._pending_event_faults = DegradationReason.NONE
            return snapshot

        age_ns = max(0, evaluation_ns - self._source.source_stamp_ns)
        hard_reasons = (
            self._source.hard_reasons
            | health.hard_reasons
            | self._source_integrity_fault
            | self._active_event_faults
            | self._pending_event_faults
        )
        quality_reasons = (
            self._source.quality_reasons | health.quality_reasons
        )
        if (
            self._source.source_stamp_ns - evaluation_ns
            > self._config.future_tolerance_ns
        ):
            hard_reasons |= DegradationReason.TIMESTAMP_INVALID
        if age_ns > self._config.source_timeout_ns:
            hard_reasons |= DegradationReason.SOURCE_STALE

        next_confidence = self._confidence
        if confidence is not None:
            if math.isfinite(confidence):
                next_confidence = _wire_float32(
                    min(1.0, max(0.0, float(confidence)))
                )
            else:
                hard_reasons |= DegradationReason.NUMERIC_INVALID
        elif not hard_reasons:
            hard_reasons |= DegradationReason.SIGNAL_MISSING
        if not hard_reasons:
            self._confidence = next_confidence

        if hard_reasons:
            self._state = TrackingState.LOST
            self._latched_reasons |= hard_reasons
            self._clear_dwell()
            reasons = self._latched_reasons | quality_reasons
        else:
            reasons = self._advance_without_hard_fault(
                evaluation_ns=evaluation_ns,
                confidence=self._confidence,
                quality_reasons=quality_reasons,
            )

        self._pending_event_faults = DegradationReason.NONE
        return self._make_snapshot(
            evaluation_ns=evaluation_ns,
            age_ns=age_ns,
            confidence=self._confidence,
            reasons=reasons,
        )

    def _snapshot_without_source(
        self,
        evaluation_ns: int,
        health: HealthAssessment,
        *,
        extra_reasons: DegradationReason = DegradationReason.NONE,
    ) -> ConfidenceSnapshot:
        self._state = TrackingState.INITIALIZING
        self._clear_dwell()
        self._latched_reasons |= (
            self._source_integrity_fault
            | self._active_event_faults
            | self._pending_event_faults
        )
        reasons = (
            DegradationReason.INITIALIZING
            | DegradationReason.SIGNAL_MISSING
            | self._latched_reasons
            | health.hard_reasons
            | health.quality_reasons
            | extra_reasons
        )
        return ConfidenceSnapshot(
            backend_id=self._backend_id,
            calibration_id=self._calibration_id,
            source_stamp_ns=None,
            evaluation_stamp_ns=evaluation_ns,
            confidence_age_ns=0,
            confidence=0.0,
            tracking_valid=False,
            state=self._state,
            reasons=reasons,
        )

    def _advance_without_hard_fault(
        self,
        *,
        evaluation_ns: int,
        confidence: float,
        quality_reasons: DegradationReason,
    ) -> DegradationReason:
        if self._state in {TrackingState.INITIALIZING, TrackingState.LOST}:
            self._degrade_since_ns = None
            self._invalidate_since_ns = None
            if confidence >= self._config.recover_at_or_above:
                self._recovery_since_ns = self._start_or_keep(
                    self._recovery_since_ns,
                    evaluation_ns,
                )
                if self._elapsed(
                    self._recovery_since_ns,
                    evaluation_ns,
                    self._config.recover_dwell_ns,
                ):
                    self._state = TrackingState.TRACKING
                    self._latched_reasons = DegradationReason.NONE
                    self._recovery_since_ns = None
                    return quality_reasons
            else:
                self._recovery_since_ns = None

            if self._state == TrackingState.INITIALIZING:
                reasons = (
                    DegradationReason.INITIALIZING
                    | DegradationReason.RECOVERY_PENDING
                    | self._latched_reasons
                    | quality_reasons
                )
            else:
                reasons = self._latched_reasons | quality_reasons
                if confidence >= self._config.degrade_below:
                    reasons |= DegradationReason.RECOVERY_PENDING
            if confidence < self._config.degrade_below:
                reasons |= DegradationReason.LOW_CONFIDENCE
            if reasons == DegradationReason.NONE:
                reasons = DegradationReason.RECOVERY_PENDING
            return reasons

        self._degrade_since_ns = self._condition_start(
            self._degrade_since_ns,
            confidence < self._config.degrade_below,
            evaluation_ns,
        )
        self._invalidate_since_ns = self._condition_start(
            self._invalidate_since_ns,
            confidence < self._config.invalidate_below,
            evaluation_ns,
        )

        if self._elapsed(
            self._invalidate_since_ns,
            evaluation_ns,
            self._config.invalidate_dwell_ns,
        ):
            self._state = TrackingState.LOST
            self._latched_reasons |= DegradationReason.LOW_CONFIDENCE
            self._recovery_since_ns = None
            return self._latched_reasons | quality_reasons

        if self._state == TrackingState.TRACKING and self._elapsed(
            self._degrade_since_ns,
            evaluation_ns,
            self._config.degrade_dwell_ns,
        ):
            self._state = TrackingState.DEGRADED

        if self._state == TrackingState.TRACKING:
            self._recovery_since_ns = None
            reasons = quality_reasons
            if confidence < self._config.degrade_below:
                reasons |= DegradationReason.LOW_CONFIDENCE
            return reasons

        if confidence >= self._config.recover_at_or_above:
            self._recovery_since_ns = self._start_or_keep(
                self._recovery_since_ns,
                evaluation_ns,
            )
            if self._elapsed(
                self._recovery_since_ns,
                evaluation_ns,
                self._config.recover_dwell_ns,
            ):
                self._state = TrackingState.TRACKING
                self._latched_reasons = DegradationReason.NONE
                self._degrade_since_ns = None
                self._invalidate_since_ns = None
                self._recovery_since_ns = None
                return quality_reasons
        else:
            self._recovery_since_ns = None

        reasons = quality_reasons
        if confidence < self._config.degrade_below:
            reasons |= DegradationReason.LOW_CONFIDENCE
        else:
            reasons |= DegradationReason.RECOVERY_PENDING
        return reasons

    def _normalized_observation(
        self,
        sample: SourceObservation,
    ) -> SourceObservation:
        return SourceObservation(
            source_stamp_ns=sample.source_stamp_ns,
            content_fingerprint=sample.content_fingerprint,
            hard_reasons=sample.hard_reasons,
            quality_reasons=sample.quality_reasons,
        )

    @staticmethod
    def _same_observation(
        first: SourceObservation,
        second: SourceObservation,
    ) -> bool:
        return (
            first.content_fingerprint == second.content_fingerprint
            and first.hard_reasons == second.hard_reasons
            and first.quality_reasons == second.quality_reasons
        )

    def _record_source_fault(self, reason: DegradationReason) -> None:
        self._source_integrity_fault |= reason
        self._pending_event_faults |= reason

    def _make_snapshot(
        self,
        *,
        evaluation_ns: int,
        age_ns: int,
        confidence: float,
        reasons: DegradationReason,
    ) -> ConfidenceSnapshot:
        tracking_valid = self._state in {
            TrackingState.TRACKING,
            TrackingState.DEGRADED,
        }
        if not tracking_valid and reasons == DegradationReason.NONE:
            raise RuntimeError("an invalid tracking state must have a reason")
        return ConfidenceSnapshot(
            backend_id=self._backend_id,
            calibration_id=self._calibration_id,
            source_stamp_ns=self._source.source_stamp_ns if self._source else None,
            evaluation_stamp_ns=evaluation_ns,
            confidence_age_ns=age_ns,
            confidence=confidence,
            tracking_valid=tracking_valid,
            state=self._state,
            reasons=reasons,
        )

    def _clear_dwell(self) -> None:
        self._degrade_since_ns = None
        self._invalidate_since_ns = None
        self._recovery_since_ns = None

    def _clock_went_backwards(self, value_ns: int) -> bool:
        return self._latest_clock_ns is not None and value_ns < self._latest_clock_ns

    def _observe_clock(self, value_ns: int) -> None:
        if not isinstance(value_ns, int):
            raise TypeError("ROS time must be expressed as integer nanoseconds")
        self._latest_clock_ns = value_ns

    @staticmethod
    def _start_or_keep(current: int | None, now_ns: int) -> int:
        return now_ns if current is None else current

    @staticmethod
    def _condition_start(
        current: int | None,
        condition: bool,
        now_ns: int,
    ) -> int | None:
        if not condition:
            return None
        return now_ns if current is None else current

    @staticmethod
    def _elapsed(start_ns: int | None, now_ns: int, dwell_ns: int) -> bool:
        return start_ns is not None and now_ns - start_ns >= dwell_ns
