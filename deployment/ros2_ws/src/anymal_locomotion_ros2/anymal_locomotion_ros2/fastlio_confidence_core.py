"""ROS-independent FAST-LIO2 signal assembly for SLAM confidence."""

from __future__ import annotations

import hashlib
import math
import struct
from collections import OrderedDict
from dataclasses import dataclass

from anymal_locomotion_ros2.slam_confidence_core import (
    DegradationReason,
    HealthAssessment,
)


PoseTuple = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class FastlioAssemblerConfig:
    """Logical-time thresholds and exact join constraints."""

    odometry_timeout_ns: int
    lidar_timeout_ns: int
    imu_timeout_ns: int
    diagnostic_match_timeout_ns: int
    future_tolerance_ns: int
    max_pending_bundles: int = 64
    native_frame_id: str = "camera_init"
    native_child_frame_id: str = "body"
    canonical_frame_id: str = "map"
    canonical_child_frame_id: str = "base_link"

    def __post_init__(self) -> None:
        for field_name in (
            "odometry_timeout_ns",
            "lidar_timeout_ns",
            "imu_timeout_ns",
            "diagnostic_match_timeout_ns",
            "future_tolerance_ns",
            "max_pending_bundles",
        ):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an integer")
        for field_name in (
            "odometry_timeout_ns",
            "lidar_timeout_ns",
            "imu_timeout_ns",
            "diagnostic_match_timeout_ns",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.future_tolerance_ns < 0:
            raise ValueError("future_tolerance_ns must be non-negative")
        if self.max_pending_bundles < 2:
            raise ValueError("max_pending_bundles must be at least two")


@dataclass(frozen=True)
class FastlioBundle:
    """One exact-stamp native/canonical/effective diagnostic bundle."""

    source_stamp_ns: int
    native_pose: PoseTuple
    effective_point_count: int
    content_fingerprint: bytes
    hard_reasons: DegradationReason
    quality_reasons: DegradationReason = DegradationReason.NONE


class FastlioSignalAssembler:
    """Join FAST-LIO2 signals without assuming cross-topic callback order."""

    def __init__(self, config: FastlioAssemblerConfig) -> None:
        self._config = config
        self.reset()

    def reset(self) -> None:
        self._native: OrderedDict[int, PoseTuple] = OrderedDict()
        self._canonical: OrderedDict[int, PoseTuple] = OrderedDict()
        self._effective: OrderedDict[int, int] = OrderedDict()
        self._latest_stamps: dict[str, int] = {}
        self._latest_payloads: dict[str, object] = {}
        self._first_arrival_ns: OrderedDict[int, int] = OrderedDict()
        self._last_complete_stamp_ns: int | None = None
        self._completed: list[FastlioBundle] = []
        self._pending_events = DegradationReason.NONE

    @property
    def last_complete_stamp_ns(self) -> int | None:
        return self._last_complete_stamp_ns

    @property
    def latest_canonical_stamp_ns(self) -> int | None:
        return self._latest_stamps.get("canonical")

    @property
    def latest_lidar_stamp_ns(self) -> int | None:
        return self._latest_stamps.get("lidar")

    @property
    def latest_imu_stamp_ns(self) -> int | None:
        return self._latest_stamps.get("imu")

    def observe_native_odometry(
        self,
        *,
        stamp_ns: int,
        observed_at_ns: int,
        pose: PoseTuple,
        frame_id: str,
        child_frame_id: str,
    ) -> None:
        if (
            frame_id != self._config.native_frame_id
            or child_frame_id != self._config.native_child_frame_id
        ):
            self._record_event(DegradationReason.BACKEND_ERROR)
            return
        normalized = self._validated_pose(pose)
        if normalized is None:
            self._record_event(DegradationReason.NUMERIC_INVALID)
            return
        if self._record_payload(
            "native",
            stamp_ns,
            observed_at_ns,
            normalized,
            self._native,
        ):
            self._maybe_complete(stamp_ns)

    def observe_canonical_odometry(
        self,
        *,
        stamp_ns: int,
        observed_at_ns: int,
        pose: PoseTuple,
        frame_id: str,
        child_frame_id: str,
    ) -> None:
        if (
            frame_id != self._config.canonical_frame_id
            or child_frame_id != self._config.canonical_child_frame_id
        ):
            self._record_event(DegradationReason.BACKEND_ERROR)
            return
        normalized = self._validated_pose(pose)
        if normalized is None:
            self._record_event(DegradationReason.NUMERIC_INVALID)
            return
        if self._record_payload(
            "canonical",
            stamp_ns,
            observed_at_ns,
            normalized,
            self._canonical,
        ):
            self._maybe_complete(stamp_ns)

    def observe_effective_points(
        self,
        *,
        stamp_ns: int,
        observed_at_ns: int,
        point_count: int,
        layout_valid: bool,
    ) -> None:
        if not layout_valid or point_count < 0:
            self._record_event(
                DegradationReason.NUMERIC_INVALID
                | DegradationReason.SIGNAL_MISSING
            )
            return
        if self._record_payload(
            "effective",
            stamp_ns,
            observed_at_ns,
            int(point_count),
            self._effective,
        ):
            self._maybe_complete(stamp_ns)

    def observe_lidar_input(
        self,
        *,
        stamp_ns: int,
        has_points: bool,
        layout_valid: bool,
    ) -> None:
        if not has_points or not layout_valid:
            self._record_event(DegradationReason.SIGNAL_MISSING)
            return
        self._record_freshness_stamp("lidar", stamp_ns)

    def observe_imu_input(self, *, stamp_ns: int, finite: bool) -> None:
        if not finite:
            self._record_event(DegradationReason.NUMERIC_INVALID)
            return
        self._record_freshness_stamp("imu", stamp_ns)

    def take_completed(self) -> list[FastlioBundle]:
        completed = sorted(
            self._completed,
            key=lambda bundle: bundle.source_stamp_ns,
        )
        self._completed = []
        return completed

    def take_events(self) -> DegradationReason:
        events = self._pending_events
        self._pending_events = DegradationReason.NONE
        return events

    def assess(self, *, evaluation_ns: int) -> HealthAssessment:
        hard_reasons = DegradationReason.NONE
        hard_reasons |= self._freshness_reason(
            stamp_ns=self.latest_canonical_stamp_ns,
            evaluation_ns=evaluation_ns,
            timeout_ns=self._config.odometry_timeout_ns,
            stale_reason=DegradationReason.ODOMETRY_STALE,
        )
        hard_reasons |= self._freshness_reason(
            stamp_ns=self.latest_lidar_stamp_ns,
            evaluation_ns=evaluation_ns,
            timeout_ns=self._config.lidar_timeout_ns,
            stale_reason=DegradationReason.LIDAR_STALE,
        )
        hard_reasons |= self._freshness_reason(
            stamp_ns=self.latest_imu_stamp_ns,
            evaluation_ns=evaluation_ns,
            timeout_ns=self._config.imu_timeout_ns,
            stale_reason=DegradationReason.IMU_STALE,
        )

        for stamp_ns in self._pending_bundle_stamps():
            first_arrival_ns = self._first_arrival_ns[stamp_ns]
            if (
                evaluation_ns - first_arrival_ns
                > self._config.diagnostic_match_timeout_ns
            ):
                hard_reasons |= DegradationReason.SIGNAL_MISSING
                break
        return HealthAssessment(hard_reasons=hard_reasons)

    def _record_payload(
        self,
        stream: str,
        stamp_ns: int,
        observed_at_ns: int,
        payload: object,
        cache: OrderedDict,
    ) -> bool:
        if not isinstance(stamp_ns, int):
            raise TypeError("stamp_ns must be an integer")
        if not isinstance(observed_at_ns, int):
            raise TypeError("observed_at_ns must be an integer")
        latest_stamp = self._latest_stamps.get(stream)
        if latest_stamp is not None:
            if stamp_ns < latest_stamp:
                self._record_event(DegradationReason.TIMESTAMP_INVALID)
                return False
            if stamp_ns == latest_stamp:
                if payload == self._latest_payloads[stream]:
                    return False
                self._record_event(DegradationReason.TIMESTAMP_INVALID)
                return False
        self._latest_stamps[stream] = stamp_ns
        self._latest_payloads[stream] = payload
        cache[stamp_ns] = payload
        if stamp_ns not in self._first_arrival_ns:
            self._first_arrival_ns[stamp_ns] = observed_at_ns
        else:
            self._first_arrival_ns[stamp_ns] = min(
                self._first_arrival_ns[stamp_ns],
                observed_at_ns,
            )
        self._prune_pending()
        return True

    def _record_freshness_stamp(self, stream: str, stamp_ns: int) -> None:
        if not isinstance(stamp_ns, int):
            raise TypeError("stamp_ns must be an integer")
        latest_stamp = self._latest_stamps.get(stream)
        if latest_stamp is not None:
            if stamp_ns < latest_stamp:
                self._record_event(DegradationReason.TIMESTAMP_INVALID)
                return
            if stamp_ns == latest_stamp:
                return
        self._latest_stamps[stream] = stamp_ns
        self._latest_payloads[stream] = stamp_ns

    def _maybe_complete(self, stamp_ns: int) -> None:
        if (
            stamp_ns not in self._native
            or stamp_ns not in self._canonical
            or stamp_ns not in self._effective
        ):
            return
        if (
            self._last_complete_stamp_ns is not None
            and stamp_ns <= self._last_complete_stamp_ns
        ):
            return
        self._expire_incomplete_before(stamp_ns)
        count = self._effective[stamp_ns]
        hard_reasons = DegradationReason.NONE
        if count == 0:
            hard_reasons |= DegradationReason.INSUFFICIENT_SUPPORT
        fingerprint = self._bundle_fingerprint(
            stamp_ns=stamp_ns,
            native_pose=self._native[stamp_ns],
            canonical_pose=self._canonical[stamp_ns],
            effective_point_count=count,
        )
        self._completed.append(
            FastlioBundle(
                source_stamp_ns=stamp_ns,
                native_pose=self._native[stamp_ns],
                effective_point_count=count,
                content_fingerprint=fingerprint,
                hard_reasons=hard_reasons,
            )
        )
        self._last_complete_stamp_ns = stamp_ns
        self._first_arrival_ns.pop(stamp_ns, None)
        self._native.pop(stamp_ns, None)
        self._canonical.pop(stamp_ns, None)
        self._effective.pop(stamp_ns, None)

    def _expire_incomplete_before(self, completed_stamp_ns: int) -> None:
        expired_stamps = [
            stamp_ns
            for stamp_ns in self._first_arrival_ns
            if stamp_ns < completed_stamp_ns
        ]
        if expired_stamps:
            self._record_event(DegradationReason.SIGNAL_MISSING)
        for stamp_ns in expired_stamps:
            self._first_arrival_ns.pop(stamp_ns, None)
            self._native.pop(stamp_ns, None)
            self._canonical.pop(stamp_ns, None)
            self._effective.pop(stamp_ns, None)

    def _pending_bundle_stamps(self) -> list[int]:
        return [
            stamp_ns
            for stamp_ns in self._first_arrival_ns
            if not (
                stamp_ns in self._native
                and stamp_ns in self._canonical
                and stamp_ns in self._effective
            )
        ]

    def _freshness_reason(
        self,
        *,
        stamp_ns: int | None,
        evaluation_ns: int,
        timeout_ns: int,
        stale_reason: DegradationReason,
    ) -> DegradationReason:
        if stamp_ns is None:
            return DegradationReason.SIGNAL_MISSING
        if stamp_ns - evaluation_ns > self._config.future_tolerance_ns:
            return DegradationReason.TIMESTAMP_INVALID
        if evaluation_ns - stamp_ns > timeout_ns:
            return stale_reason
        return DegradationReason.NONE

    def _record_event(self, reasons: DegradationReason) -> None:
        self._pending_events |= reasons

    def _prune_pending(self) -> None:
        while len(self._first_arrival_ns) > self._config.max_pending_bundles:
            stamp_ns, _ = self._first_arrival_ns.popitem(last=False)
            self._record_event(DegradationReason.SIGNAL_MISSING)
            self._native.pop(stamp_ns, None)
            self._canonical.pop(stamp_ns, None)
            self._effective.pop(stamp_ns, None)

    @staticmethod
    def _validated_pose(pose: PoseTuple) -> PoseTuple | None:
        if len(pose) != 7 or not all(math.isfinite(value) for value in pose):
            return None
        quaternion_norm = math.sqrt(sum(value * value for value in pose[3:]))
        if quaternion_norm < 1.0e-12:
            return None
        return tuple(float(value) for value in pose)  # type: ignore[return-value]

    @staticmethod
    def _bundle_fingerprint(
        *,
        stamp_ns: int,
        native_pose: PoseTuple,
        canonical_pose: PoseTuple,
        effective_point_count: int,
    ) -> bytes:
        payload = struct.pack(
            "<qQ14d",
            stamp_ns,
            effective_point_count,
            *native_pose,
            *canonical_pose,
        )
        return hashlib.sha256(payload).digest()
