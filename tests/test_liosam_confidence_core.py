"""Tests for ROS-independent LIO-SAM confidence signal assembly."""

from __future__ import annotations

import itertools

import pytest

from anymal_locomotion_ros2.liosam_confidence_core import (
    LiosamAssemblerConfig,
    LiosamSignalAssembler,
)
from anymal_locomotion_ros2.slam_confidence_core import (
    DegradationReason,
    SlamConfidenceStateMachine,
    SourceObservation,
    StateMachineConfig,
    TrackingState,
    seconds_to_nanoseconds,
)


POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _config(
    *,
    motion_required: bool = True,
    max_pending: int = 64,
) -> LiosamAssemblerConfig:
    return LiosamAssemblerConfig(
        odometry_timeout_ns=seconds_to_nanoseconds(0.50),
        lidar_timeout_ns=seconds_to_nanoseconds(0.50),
        imu_timeout_ns=seconds_to_nanoseconds(0.05),
        diagnostic_match_timeout_ns=seconds_to_nanoseconds(0.15),
        future_tolerance_ns=seconds_to_nanoseconds(0.005),
        motion_deskew_required=motion_required,
        max_pending_bundles=max_pending,
    )


def _assembler(
    *,
    motion_required: bool = True,
    max_pending: int = 64,
) -> LiosamSignalAssembler:
    return LiosamSignalAssembler(
        _config(
            motion_required=motion_required,
            max_pending=max_pending,
        )
    )


def _observe_component(
    assembler: LiosamSignalAssembler,
    component: str,
    stamp_ns: int,
    *,
    observed_at_ns: int | None = None,
    corner_count: int = 20,
    surface_count: int = 200,
    imu_available: bool = True,
    odom_available: bool = True,
    degenerate: bool = False,
    motion_applied: bool = True,
) -> None:
    observed = stamp_ns if observed_at_ns is None else observed_at_ns
    if component == "native":
        assembler.observe_native_odometry(
            stamp_ns=stamp_ns,
            observed_at_ns=observed,
            pose=POSE,
            frame_id="odom",
            child_frame_id="odom_mapping",
        )
    elif component == "canonical":
        assembler.observe_canonical_odometry(
            stamp_ns=stamp_ns,
            observed_at_ns=observed,
            pose=POSE,
            frame_id="map",
            child_frame_id="base_link",
        )
    elif component == "incremental":
        assembler.observe_incremental_odometry(
            stamp_ns=stamp_ns,
            observed_at_ns=observed,
            pose=POSE,
            degenerate=degenerate,
            frame_id="odom",
            child_frame_id="odom_mapping",
        )
    elif component == "features":
        assembler.observe_feature_info(
            stamp_ns=stamp_ns,
            observed_at_ns=observed,
            corner_count=corner_count,
            surface_count=surface_count,
            imu_available=imu_available,
            odom_available=odom_available,
            frame_id="lidar_link",
        )
    elif component == "motion":
        assembler.observe_motion_deskew(
            stamp_ns=stamp_ns,
            observed_at_ns=observed,
            applied=motion_applied,
        )
    else:
        raise AssertionError(component)


def _observe_complete(
    assembler: LiosamSignalAssembler,
    stamp_ns: int,
    *,
    corner_count: int = 20,
    surface_count: int = 200,
    imu_available: bool = True,
    odom_available: bool = True,
    degenerate: bool = False,
    motion_applied: bool = True,
) -> None:
    for component in (
        "features",
        "motion",
        "native",
        "canonical",
        "incremental",
    ):
        _observe_component(
            assembler,
            component,
            stamp_ns,
            corner_count=corner_count,
            surface_count=surface_count,
            imu_available=imu_available,
            odom_available=odom_available,
            degenerate=degenerate,
            motion_applied=motion_applied,
        )


def _observe_fresh_inputs(
    assembler: LiosamSignalAssembler,
    stamp_ns: int,
) -> None:
    assembler.observe_lidar_input(
        stamp_ns=stamp_ns,
        has_points=True,
        layout_valid=True,
    )
    assembler.observe_imu_input(stamp_ns=stamp_ns, finite=True)


def test_required_topic_arrival_order_does_not_change_bundle() -> None:
    expected = None
    components = (
        "native",
        "canonical",
        "incremental",
        "features",
        "motion",
    )
    for order in itertools.permutations(components):
        assembler = _assembler()
        for component in order:
            _observe_component(assembler, component, 100)
        bundles = assembler.take_completed()
        assert len(bundles) == 1
        bundle = bundles[0]
        comparable = (
            bundle.source_stamp_ns,
            bundle.corner_count,
            bundle.surface_count,
            bundle.hard_reasons,
            bundle.quality_reasons,
            bundle.content_fingerprint,
        )
        if expected is None:
            expected = comparable
        assert comparable == expected


def test_degeneracy_and_odom_unavailable_are_soft_features() -> None:
    assembler = _assembler()
    _observe_complete(
        assembler,
        100,
        degenerate=True,
        odom_available=False,
    )

    bundle = assembler.take_completed()[0]
    assert bundle.hard_reasons == DegradationReason.NONE
    assert bundle.quality_reasons == (
        DegradationReason.DEGENERATE_GEOMETRY
        | DegradationReason.SIGNAL_MISSING
    )


@pytest.mark.parametrize(
    ("corner_count", "surface_count"),
    ((10, 200), (20, 100), (0, 0)),
)
def test_conservatively_insufficient_features_are_hard(
    corner_count: int,
    surface_count: int,
) -> None:
    assembler = _assembler()
    _observe_complete(
        assembler,
        100,
        corner_count=corner_count,
        surface_count=surface_count,
    )

    assert assembler.take_completed()[0].hard_reasons & (
        DegradationReason.INSUFFICIENT_SUPPORT
    )


def test_feature_counts_above_upstream_minima_are_not_a_hard_gate() -> None:
    assembler = _assembler()
    _observe_complete(
        assembler,
        100,
        corner_count=11,
        surface_count=101,
    )

    assert assembler.take_completed()[0].hard_reasons == (
        DegradationReason.NONE
    )


def test_cloud_info_imu_unavailable_is_hard() -> None:
    assembler = _assembler()
    _observe_complete(assembler, 100, imu_available=False)

    assert assembler.take_completed()[0].hard_reasons & (
        DegradationReason.IMU_STALE
    )


def test_required_motion_deskew_unavailable_is_hard() -> None:
    assembler = _assembler()
    _observe_complete(assembler, 100, motion_applied=False)

    assert assembler.take_completed()[0].hard_reasons & (
        DegradationReason.BACKEND_ERROR
    )


def test_native_mode_does_not_wait_for_optional_motion_status() -> None:
    assembler = _assembler(motion_required=False)
    for component in ("native", "canonical", "incremental", "features"):
        _observe_component(assembler, component, 100)
    assembler.observe_motion_deskew(
        stamp_ns=200,
        observed_at_ns=200,
        applied=False,
    )
    _observe_fresh_inputs(assembler, 400_000_000)

    assert len(assembler.take_completed()) == 1
    assert assembler.assess(evaluation_ns=400_000_000).hard_reasons & (
        DegradationReason.SIGNAL_MISSING
    ) == DegradationReason.NONE


def test_missing_required_diagnostic_uses_first_arrival_grace() -> None:
    assembler = _assembler()
    for component in ("native", "canonical", "features", "motion"):
        _observe_component(
            assembler,
            component,
            100,
            observed_at_ns=1_000_000_000,
        )
    _observe_fresh_inputs(assembler, 1_150_000_001)

    within = assembler.assess(evaluation_ns=1_150_000_000)
    after = assembler.assess(evaluation_ns=1_150_000_001)
    assert not within.hard_reasons & DegradationReason.SIGNAL_MISSING
    assert after.hard_reasons & DegradationReason.SIGNAL_MISSING


def test_freshness_inputs_do_not_advance_complete_source() -> None:
    assembler = _assembler()
    _observe_complete(assembler, 100)
    assembler.take_completed()
    assembler.observe_lidar_input(
        stamp_ns=200,
        has_points=True,
        layout_valid=True,
    )
    assembler.observe_imu_input(stamp_ns=300, finite=True)

    assert assembler.last_complete_stamp_ns == 100
    assert assembler.latest_lidar_stamp_ns == 200
    assert assembler.latest_imu_stamp_ns == 300


def test_each_input_freshness_reason_is_independent() -> None:
    assembler = _assembler()
    _observe_complete(assembler, 0)
    assembler.take_completed()
    assembler.observe_lidar_input(
        stamp_ns=600_000_000,
        has_points=True,
        layout_valid=True,
    )
    assembler.observe_imu_input(stamp_ns=600_000_000, finite=True)

    reasons = assembler.assess(evaluation_ns=600_000_000).hard_reasons
    assert reasons & DegradationReason.ODOMETRY_STALE
    assert not reasons & DegradationReason.LIDAR_STALE
    assert not reasons & DegradationReason.IMU_STALE


def test_old_incomplete_bundle_becomes_one_shot_event_and_is_evicted() -> None:
    assembler = _assembler()
    for component in ("native", "canonical", "features", "motion"):
        _observe_component(assembler, component, 100)
    _observe_complete(assembler, 200)

    assert [bundle.source_stamp_ns for bundle in assembler.take_completed()] == [
        200
    ]
    assert assembler.take_events() & DegradationReason.SIGNAL_MISSING
    assert assembler.take_events() == DegradationReason.NONE
    _observe_fresh_inputs(assembler, 250)
    assert not assembler.assess(evaluation_ns=250).hard_reasons & (
        DegradationReason.SIGNAL_MISSING
    )


def test_throttled_feature_only_stamps_do_not_create_missing_mapping_bundles() -> None:
    assembler = _assembler(motion_required=False)
    for stamp_ns in (100, 150):
        _observe_component(assembler, "incremental", stamp_ns)
        _observe_component(assembler, "features", stamp_ns)
    _observe_complete(assembler, 200, motion_applied=None)

    assert [bundle.source_stamp_ns for bundle in assembler.take_completed()] == [
        200
    ]
    assert assembler.take_events() == DegradationReason.NONE
    _observe_fresh_inputs(assembler, 250)
    assert not assembler.assess(evaluation_ns=250).hard_reasons & (
        DegradationReason.SIGNAL_MISSING
    )


def test_conflicting_duplicate_and_regression_are_timestamp_invalid() -> None:
    assembler = _assembler()
    _observe_component(assembler, "canonical", 200)
    assembler.observe_canonical_odometry(
        stamp_ns=200,
        observed_at_ns=201,
        pose=(1.0, *POSE[1:]),
        frame_id="map",
        child_frame_id="base_link",
    )
    _observe_component(assembler, "canonical", 100)

    assert assembler.take_events() & DegradationReason.TIMESTAMP_INVALID


def test_invalid_pose_and_frame_fail_closed() -> None:
    assembler = _assembler()
    assembler.observe_native_odometry(
        stamp_ns=100,
        observed_at_ns=100,
        pose=(float("nan"), *POSE[1:]),
        frame_id="odom",
        child_frame_id="odom_mapping",
    )
    assembler.observe_feature_info(
        stamp_ns=100,
        observed_at_ns=100,
        corner_count=20,
        surface_count=200,
        imu_available=True,
        odom_available=True,
        frame_id="wrong",
    )

    reasons = assembler.take_events()
    assert reasons & DegradationReason.NUMERIC_INVALID
    assert reasons & DegradationReason.BACKEND_ERROR


def test_pending_cache_is_bounded() -> None:
    assembler = _assembler(max_pending=2)
    for stamp_ns in (100, 200, 300):
        _observe_component(assembler, "native", stamp_ns)

    assert assembler.take_events() & DegradationReason.SIGNAL_MISSING


def test_config_requires_exact_integer_nanoseconds() -> None:
    values = vars(_config()).copy()
    values["diagnostic_match_timeout_ns"] = 0.15
    with pytest.raises(TypeError, match="must be an integer"):
        LiosamAssemblerConfig(**values)


class _ConfidenceHarness:
    def __init__(self) -> None:
        self.assembler = _assembler()
        self.machine = SlamConfidenceStateMachine(
            StateMachineConfig(
                source_timeout_ns=seconds_to_nanoseconds(0.50),
                future_tolerance_ns=seconds_to_nanoseconds(0.005),
                degrade_below=0.45,
                degrade_dwell_ns=seconds_to_nanoseconds(0.10),
                invalidate_below=0.25,
                invalidate_dwell_ns=seconds_to_nanoseconds(0.20),
                recover_at_or_above=0.55,
                recover_dwell_ns=seconds_to_nanoseconds(0.50),
            ),
            backend_id="liosam",
            calibration_id="synthetic-test-v1",
        )

    def flush(self, now_ns: int) -> None:
        events = self.assembler.take_events()
        if events:
            self.machine.observe_hard_event(events, ros_now_ns=now_ns)
        for bundle in self.assembler.take_completed():
            self.machine.observe_source(
                SourceObservation(
                    source_stamp_ns=bundle.source_stamp_ns,
                    content_fingerprint=bundle.content_fingerprint,
                    hard_reasons=bundle.hard_reasons,
                    quality_reasons=bundle.quality_reasons,
                ),
                ros_now_ns=now_ns,
            )

    def tick(self, now_ns: int):
        self.flush(now_ns)
        return self.machine.evaluate(
            evaluation_ns=now_ns,
            health=self.assembler.assess(evaluation_ns=now_ns),
            confidence=0.90,
        )

    def nominal(
        self,
        stamp_ns: int,
        *,
        bundle: bool = True,
        lidar: bool = True,
        imu: bool = True,
        **bundle_options,
    ) -> None:
        if imu:
            self.assembler.observe_imu_input(stamp_ns=stamp_ns, finite=True)
        if stamp_ns % 100_000_000 == 0:
            if lidar:
                self.assembler.observe_lidar_input(
                    stamp_ns=stamp_ns,
                    has_points=True,
                    layout_valid=True,
                )
            if bundle:
                _observe_complete(
                    self.assembler,
                    stamp_ns,
                    **bundle_options,
                )

    def seed_tracking(self) -> None:
        for stamp_ns in range(0, 600_000_001, 50_000_000):
            self.nominal(stamp_ns)
            snapshot = self.tick(stamp_ns)
        assert snapshot.state == TrackingState.TRACKING


@pytest.mark.parametrize(
    ("fault", "expected_reason", "detected_ns"),
    (
        ("odom", DegradationReason.ODOMETRY_STALE, 1_150_000_000),
        ("lidar", DegradationReason.LIDAR_STALE, 1_150_000_000),
        ("imu", DegradationReason.IMU_STALE, 700_000_000),
    ),
)
def test_stale_high_freeze_matrix(
    fault: str,
    expected_reason: DegradationReason,
    detected_ns: int,
) -> None:
    harness = _ConfidenceHarness()
    harness.seed_tracking()
    detected = None
    for stamp_ns in range(650_000_000, 1_200_000_001, 50_000_000):
        harness.nominal(
            stamp_ns,
            bundle=fault != "odom",
            lidar=fault != "lidar",
            imu=fault != "imu",
        )
        snapshot = harness.tick(stamp_ns)
        if snapshot.reasons & expected_reason:
            detected = snapshot
            break

    assert detected is not None
    assert detected.evaluation_stamp_ns == detected_ns
    assert detected.state == TrackingState.LOST
    assert detected.tracking_valid is False
    assert detected.confidence > 0.89


@pytest.mark.parametrize(
    ("options", "reason"),
    (
        ({"corner_count": 10}, DegradationReason.INSUFFICIENT_SUPPORT),
        ({"surface_count": 100}, DegradationReason.INSUFFICIENT_SUPPORT),
        ({"imu_available": False}, DegradationReason.IMU_STALE),
        ({"motion_applied": False}, DegradationReason.BACKEND_ERROR),
    ),
)
def test_complete_hard_bundle_invalidates_on_same_tick(
    options: dict,
    reason: DegradationReason,
) -> None:
    harness = _ConfidenceHarness()
    harness.seed_tracking()
    harness.nominal(700_000_000, **options)
    snapshot = harness.tick(700_000_000)

    assert snapshot.state == TrackingState.LOST
    assert snapshot.reasons & reason
    assert snapshot.confidence > 0.89


def test_backend_local_soft_features_do_not_independently_invalidate() -> None:
    harness = _ConfidenceHarness()
    harness.seed_tracking()
    harness.nominal(
        700_000_000,
        degenerate=True,
        odom_available=False,
    )
    snapshot = harness.tick(700_000_000)

    assert snapshot.state == TrackingState.TRACKING
    assert snapshot.tracking_valid is True
    assert snapshot.reasons & DegradationReason.DEGENERATE_GEOMETRY
    assert snapshot.reasons & DegradationReason.SIGNAL_MISSING


def test_missing_scan_event_is_one_shot_and_recovers() -> None:
    harness = _ConfidenceHarness()
    harness.seed_tracking()
    harness.nominal(700_000_000, bundle=False)
    for component in ("native", "canonical", "features", "motion"):
        _observe_component(harness.assembler, component, 700_000_000)
    harness.tick(700_000_000)

    recovered = None
    lost = None
    for stamp_ns in range(750_000_000, 1_500_000_001, 50_000_000):
        harness.nominal(stamp_ns)
        snapshot = harness.tick(stamp_ns)
        if (
            lost is None
            and snapshot.reasons & DegradationReason.SIGNAL_MISSING
        ):
            lost = snapshot
        if lost is not None and snapshot.state == TrackingState.TRACKING:
            recovered = snapshot
            break

    assert lost is not None
    assert lost.evaluation_stamp_ns == 800_000_000
    assert recovered is not None
    assert recovered.evaluation_stamp_ns == 1_350_000_000
