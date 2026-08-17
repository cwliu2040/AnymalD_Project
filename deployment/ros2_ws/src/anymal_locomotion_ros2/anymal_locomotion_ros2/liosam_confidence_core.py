"""ROS-independent LIO-SAM signal assembly for SLAM confidence."""

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
class LiosamAssemblerConfig:
    """Logical-time thresholds and exact-join requirements."""

    odometry_timeout_ns: int
    lidar_timeout_ns: int
    imu_timeout_ns: int
    diagnostic_match_timeout_ns: int
    future_tolerance_ns: int
    edge_feature_min_valid: int = 10
    surface_feature_min_valid: int = 100
    motion_deskew_required: bool = True
    max_pending_bundles: int = 64
    native_frame_id: str = "odom"
    native_child_frame_id: str = "odom_mapping"
    canonical_frame_id: str = "map"
    canonical_child_frame_id: str = "base_link"
    feature_frame_id: str = "lidar_link"

    def __post_init__(self) -> None:
        integer_fields = (
            "odometry_timeout_ns",
            "lidar_timeout_ns",
            "imu_timeout_ns",
            "diagnostic_match_timeout_ns",
            "future_tolerance_ns",
            "edge_feature_min_valid",
            "surface_feature_min_valid",
            "max_pending_bundles",
        )
        for field_name in integer_fields:
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
        if self.edge_feature_min_valid < 0:
            raise ValueError("edge_feature_min_valid must be non-negative")
        if self.surface_feature_min_valid < 0:
            raise ValueError("surface_feature_min_valid must be non-negative")
        if type(self.motion_deskew_required) is not bool:
            raise TypeError("motion_deskew_required must be a boolean")
        if self.max_pending_bundles < 2:
            raise ValueError("max_pending_bundles must be at least two")


@dataclass(frozen=True)
class FeatureObservation:
    corner_count: int
    surface_count: int
    imu_available: bool
    odom_available: bool


@dataclass(frozen=True)
class IncrementalObservation:
    pose: PoseTuple
    degenerate: bool


@dataclass(frozen=True)
class LiosamBundle:
    """One complete exact-stamp LIO-SAM diagnostic bundle."""

    source_stamp_ns: int
    native_pose: PoseTuple
    incremental_pose: PoseTuple
    corner_count: int
    surface_count: int
    imu_available: bool
    odom_available: bool
    degenerate: bool
    motion_deskew_applied: bool | None
    content_fingerprint: bytes
    hard_reasons: DegradationReason
    quality_reasons: DegradationReason


class LiosamSignalAssembler:
    """Join public LIO-SAM topics without changing upstream algorithms."""

    def __init__(self, config: LiosamAssemblerConfig) -> None:
        self._config = config
        self.reset()

    def reset(self) -> None:
        self._native: OrderedDict[int, PoseTuple] = OrderedDict()
        self._canonical: OrderedDict[int, PoseTuple] = OrderedDict()
        self._incremental: OrderedDict[
            int, IncrementalObservation
        ] = OrderedDict()
        self._features: OrderedDict[int, FeatureObservation] = OrderedDict()
        self._motion: OrderedDict[int, bool] = OrderedDict()
        self._latest_stamps: dict[str, int] = {}
        self._latest_payloads: dict[str, object] = {}
        self._first_arrival_ns: OrderedDict[int, int] = OrderedDict()
        self._last_complete_stamp_ns: int | None = None
        self._completed: list[LiosamBundle] = []
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

    def observe_incremental_odometry(
        self,
        *,
        stamp_ns: int,
        observed_at_ns: int,
        pose: PoseTuple,
        degenerate: bool,
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
        observation = IncrementalObservation(normalized, bool(degenerate))
        if self._record_payload(
            "incremental",
            stamp_ns,
            observed_at_ns,
            observation,
            self._incremental,
        ):
            self._maybe_complete(stamp_ns)

    def observe_feature_info(
        self,
        *,
        stamp_ns: int,
        observed_at_ns: int,
        corner_count: int,
        surface_count: int,
        imu_available: bool,
        odom_available: bool,
        frame_id: str,
        layout_valid: bool = True,
    ) -> None:
        if frame_id != self._config.feature_frame_id:
            self._record_event(DegradationReason.BACKEND_ERROR)
            return
        if not layout_valid or corner_count < 0 or surface_count < 0:
            self._record_event(
                DegradationReason.NUMERIC_INVALID
                | DegradationReason.SIGNAL_MISSING
            )
            return
        observation = FeatureObservation(
            int(corner_count),
            int(surface_count),
            bool(imu_available),
            bool(odom_available),
        )
        if self._record_payload(
            "features",
            stamp_ns,
            observed_at_ns,
            observation,
            self._features,
        ):
            self._maybe_complete(stamp_ns)

    def observe_motion_deskew(
        self,
        *,
        stamp_ns: int,
        observed_at_ns: int,
        applied: bool,
    ) -> None:
        if not self._config.motion_deskew_required:
            return
        if self._record_payload(
            "motion",
            stamp_ns,
            observed_at_ns,
            bool(applied),
            self._motion,
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

    def take_completed(self) -> list[LiosamBundle]:
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
        if type(stamp_ns) is not int:
            raise TypeError("stamp_ns must be an integer")
        if type(observed_at_ns) is not int:
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
        first_arrival = self._first_arrival_ns.get(stamp_ns)
        self._first_arrival_ns[stamp_ns] = (
            observed_at_ns
            if first_arrival is None
            else min(first_arrival, observed_at_ns)
        )
        self._prune_pending()
        return True

    def _record_freshness_stamp(self, stream: str, stamp_ns: int) -> None:
        if type(stamp_ns) is not int:
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
        if not self._bundle_complete(stamp_ns):
            return
        if (
            self._last_complete_stamp_ns is not None
            and stamp_ns <= self._last_complete_stamp_ns
        ):
            return
        self._expire_incomplete_before(stamp_ns)
        feature = self._features[stamp_ns]
        incremental = self._incremental[stamp_ns]
        motion = self._motion.get(stamp_ns)
        hard_reasons = DegradationReason.NONE
        quality_reasons = DegradationReason.NONE
        if (
            feature.corner_count <= self._config.edge_feature_min_valid
            or feature.surface_count
            <= self._config.surface_feature_min_valid
        ):
            hard_reasons |= DegradationReason.INSUFFICIENT_SUPPORT
        if not feature.imu_available:
            hard_reasons |= DegradationReason.IMU_STALE
        if self._config.motion_deskew_required and not motion:
            hard_reasons |= DegradationReason.BACKEND_ERROR
        if incremental.degenerate:
            quality_reasons |= DegradationReason.DEGENERATE_GEOMETRY
        if not feature.odom_available:
            quality_reasons |= DegradationReason.SIGNAL_MISSING
        fingerprint = self._bundle_fingerprint(
            stamp_ns=stamp_ns,
            native_pose=self._native[stamp_ns],
            canonical_pose=self._canonical[stamp_ns],
            incremental=incremental,
            feature=feature,
            motion=motion,
        )
        self._completed.append(
            LiosamBundle(
                source_stamp_ns=stamp_ns,
                native_pose=self._native[stamp_ns],
                incremental_pose=incremental.pose,
                corner_count=feature.corner_count,
                surface_count=feature.surface_count,
                imu_available=feature.imu_available,
                odom_available=feature.odom_available,
                degenerate=incremental.degenerate,
                motion_deskew_applied=motion,
                content_fingerprint=fingerprint,
                hard_reasons=hard_reasons,
                quality_reasons=quality_reasons,
            )
        )
        self._last_complete_stamp_ns = stamp_ns
        self._evict_stamp(stamp_ns)

    def _bundle_complete(self, stamp_ns: int) -> bool:
        required = (
            stamp_ns in self._native
            and stamp_ns in self._canonical
            and stamp_ns in self._incremental
            and stamp_ns in self._features
        )
        if self._config.motion_deskew_required:
            required = required and stamp_ns in self._motion
        return required

    def _expire_incomplete_before(self, completed_stamp_ns: int) -> None:
        expired = [
            stamp_ns
            for stamp_ns in self._first_arrival_ns
            if stamp_ns < completed_stamp_ns
        ]
        if any(
            self._is_mapping_source_candidate(stamp_ns)
            and not self._bundle_complete(stamp_ns)
            for stamp_ns in expired
        ):
            self._record_event(DegradationReason.SIGNAL_MISSING)
        for stamp_ns in expired:
            self._evict_stamp(stamp_ns)

    def _pending_bundle_stamps(self) -> list[int]:
        return [
            stamp_ns
            for stamp_ns in self._first_arrival_ns
            if self._is_mapping_source_candidate(stamp_ns)
            and not self._bundle_complete(stamp_ns)
        ]

    def _is_mapping_source_candidate(self, stamp_ns: int) -> bool:
        """Return whether a stamp represents an expected mapping source.

        Feature extraction runs for scans that map optimization may
        intentionally skip because of ``mappingProcessInterval``.  Those
        feature/incremental-only stamps are useful if a mapping source with
        the same stamp arrives, but they must not independently create a
        missing mapping bundle.  Native mapping odometry and its canonical
        adapter output are the two authoritative candidate anchors.
        """
        return stamp_ns in self._native or stamp_ns in self._canonical

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
            stamp_ns = next(iter(self._first_arrival_ns))
            if (
                self._is_mapping_source_candidate(stamp_ns)
                and not self._bundle_complete(stamp_ns)
            ):
                self._record_event(DegradationReason.SIGNAL_MISSING)
            self._evict_stamp(stamp_ns)

    def _evict_stamp(self, stamp_ns: int) -> None:
        self._first_arrival_ns.pop(stamp_ns, None)
        self._native.pop(stamp_ns, None)
        self._canonical.pop(stamp_ns, None)
        self._incremental.pop(stamp_ns, None)
        self._features.pop(stamp_ns, None)
        self._motion.pop(stamp_ns, None)

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
        incremental: IncrementalObservation,
        feature: FeatureObservation,
        motion: bool | None,
    ) -> bytes:
        payload = struct.pack(
            "<q14d7dqq????",
            stamp_ns,
            *native_pose,
            *canonical_pose,
            *incremental.pose,
            feature.corner_count,
            feature.surface_count,
            feature.imu_available,
            feature.odom_available,
            incremental.degenerate,
            bool(motion),
        )
        return hashlib.sha256(payload).digest()
