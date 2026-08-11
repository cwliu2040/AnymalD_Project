"""Deterministic synthetic hard-fault validation for FAST-LIO2 confidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from anymal_locomotion_ros2.fastlio_confidence_core import (
    FastlioAssemblerConfig,
    FastlioSignalAssembler,
    PoseTuple,
)
from anymal_locomotion_ros2.slam_confidence_core import (
    ConfidenceSnapshot,
    DegradationReason,
    SlamConfidenceStateMachine,
    SourceObservation,
    StateMachineConfig,
    TrackingState,
    seconds_to_nanoseconds,
)


_TICK_NS = seconds_to_nanoseconds(0.05)
_SCAN_NS = seconds_to_nanoseconds(0.10)
_HEALTHY_CONFIDENCE = 0.90
_SUPPORT_COUNT = 1000


@dataclass(frozen=True)
class FaultCaseResult:
    """Outcome of one deterministic fault scenario."""

    name: str
    expected_reason: str
    expected_state: str
    injected_at_ns: int
    detection_deadline_ns: int
    detected_at_ns: int | None
    detection_latency_ns: int | None
    detected_state: str | None
    detected_reasons: int
    confidence_at_detection: float | None
    recovery_required: bool
    recovered_at_ns: int | None
    passed: bool
    failures: tuple[str, ...]


class _Harness:
    """Drive the production pure cores on a fixed 20 Hz logical grid."""

    def __init__(self) -> None:
        self.assembler = FastlioSignalAssembler(
            FastlioAssemblerConfig(
                odometry_timeout_ns=seconds_to_nanoseconds(0.30),
                lidar_timeout_ns=seconds_to_nanoseconds(0.30),
                imu_timeout_ns=seconds_to_nanoseconds(0.05),
                diagnostic_match_timeout_ns=seconds_to_nanoseconds(0.05),
                future_tolerance_ns=seconds_to_nanoseconds(0.005),
                max_pending_bundles=64,
            )
        )
        self.machine = SlamConfidenceStateMachine(
            StateMachineConfig(
                source_timeout_ns=seconds_to_nanoseconds(0.30),
                future_tolerance_ns=seconds_to_nanoseconds(0.005),
                degrade_below=0.45,
                degrade_dwell_ns=seconds_to_nanoseconds(0.10),
                invalidate_below=0.25,
                invalidate_dwell_ns=seconds_to_nanoseconds(0.20),
                recover_at_or_above=0.55,
                recover_dwell_ns=seconds_to_nanoseconds(0.50),
            ),
            backend_id="fastlio2",
            calibration_id="synthetic-fault-matrix-v1",
        )

    @staticmethod
    def _pose(stamp_ns: int, *, offset: float = 0.0) -> PoseTuple:
        return (
            float(stamp_ns) * 1.0e-10 + offset,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    def observe_bundle(
        self,
        *,
        stamp_ns: int,
        observed_at_ns: int,
        effective_count: int = _SUPPORT_COUNT,
        include_native: bool = True,
        include_canonical: bool = True,
        include_effective: bool = True,
    ) -> None:
        pose = self._pose(stamp_ns)
        if include_native:
            self.assembler.observe_native_odometry(
                stamp_ns=stamp_ns,
                observed_at_ns=observed_at_ns,
                pose=pose,
                frame_id="camera_init",
                child_frame_id="body",
            )
        if include_canonical:
            self.assembler.observe_canonical_odometry(
                stamp_ns=stamp_ns,
                observed_at_ns=observed_at_ns,
                pose=pose,
                frame_id="map",
                child_frame_id="base_link",
            )
        if include_effective:
            self.assembler.observe_effective_points(
                stamp_ns=stamp_ns,
                observed_at_ns=observed_at_ns,
                point_count=effective_count,
                layout_valid=True,
            )

    def observe_lidar(self, stamp_ns: int) -> None:
        self.assembler.observe_lidar_input(
            stamp_ns=stamp_ns,
            has_points=True,
            layout_valid=True,
        )

    def observe_imu(self, stamp_ns: int) -> None:
        self.assembler.observe_imu_input(stamp_ns=stamp_ns, finite=True)

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

    def evaluate(self, evaluation_ns: int) -> ConfidenceSnapshot:
        self.flush(evaluation_ns)
        return self.machine.evaluate(
            evaluation_ns=evaluation_ns,
            health=self.assembler.assess(evaluation_ns=evaluation_ns),
            confidence=_HEALTHY_CONFIDENCE,
        )

    def observe_nominal(self, stamp_ns: int) -> None:
        self.observe_imu(stamp_ns)
        if stamp_ns % _SCAN_NS == 0:
            self.observe_lidar(stamp_ns)
            self.observe_bundle(
                stamp_ns=stamp_ns,
                observed_at_ns=stamp_ns,
            )

    def seed_tracking(self) -> int:
        final_ns = seconds_to_nanoseconds(0.60)
        for stamp_ns in range(0, final_ns + _TICK_NS, _TICK_NS):
            self.observe_nominal(stamp_ns)
            snapshot = self.evaluate(stamp_ns)
        if snapshot.state != TrackingState.TRACKING:
            raise RuntimeError("synthetic healthy prefix did not reach TRACKING")
        return final_ns


ScenarioStep = Callable[[_Harness, int], None]


def _validate_case(
    *,
    name: str,
    expected_reason: DegradationReason,
    step: ScenarioStep,
    maximum_detection_latency_ns: int,
    recovery_required: bool = False,
    end_after_ns: int | None = None,
    expected_state: TrackingState = TrackingState.LOST,
    injection_delay_ns: int = _SCAN_NS,
) -> FaultCaseResult:
    harness = _Harness()
    healthy_until_ns = harness.seed_tracking()
    injected_at_ns = healthy_until_ns + injection_delay_ns
    end_ns = end_after_ns or (
        injected_at_ns + maximum_detection_latency_ns + _TICK_NS
    )
    detected: ConfidenceSnapshot | None = None
    recovered_at_ns: int | None = None

    for evaluation_ns in range(
        healthy_until_ns + _TICK_NS,
        end_ns + _TICK_NS,
        _TICK_NS,
    ):
        step(harness, evaluation_ns)
        snapshot = harness.evaluate(evaluation_ns)
        if (
            evaluation_ns >= injected_at_ns
            and detected is None
            and not snapshot.tracking_valid
            and snapshot.reasons & expected_reason
        ):
            detected = snapshot
        elif (
            recovery_required
            and detected is not None
            and snapshot.tracking_valid
            and snapshot.state == TrackingState.TRACKING
        ):
            recovered_at_ns = evaluation_ns
            break

    failures: list[str] = []
    if detected is None:
        failures.append("expected invalid reason was not observed")
    else:
        latency_ns = detected.evaluation_stamp_ns - injected_at_ns
        if latency_ns > maximum_detection_latency_ns:
            failures.append("fault detection exceeded its logical deadline")
        if detected.confidence < 0.89:
            failures.append("hard gate did not preserve stale-high confidence")
        if detected.state != expected_state:
            failures.append(
                f"fault entered {detected.state.name}, expected "
                f"{expected_state.name}"
            )
    if recovery_required and recovered_at_ns is None:
        failures.append("healthy input did not complete recovery")

    return FaultCaseResult(
        name=name,
        expected_reason=expected_reason.name,
        expected_state=expected_state.name,
        injected_at_ns=injected_at_ns,
        detection_deadline_ns=(
            injected_at_ns + maximum_detection_latency_ns
        ),
        detected_at_ns=(
            detected.evaluation_stamp_ns if detected is not None else None
        ),
        detection_latency_ns=(
            detected.evaluation_stamp_ns - injected_at_ns
            if detected is not None
            else None
        ),
        detected_state=detected.state.name if detected is not None else None,
        detected_reasons=int(detected.reasons) if detected is not None else 0,
        confidence_at_detection=(
            detected.confidence if detected is not None else None
        ),
        recovery_required=recovery_required,
        recovered_at_ns=recovered_at_ns,
        passed=not failures,
        failures=tuple(failures),
    )


def _odom_freeze_step(harness: _Harness, stamp_ns: int) -> None:
    harness.observe_imu(stamp_ns)
    if stamp_ns % _SCAN_NS == 0:
        harness.observe_lidar(stamp_ns)


def _lidar_freeze_step(harness: _Harness, stamp_ns: int) -> None:
    harness.observe_imu(stamp_ns)
    if stamp_ns % _SCAN_NS == 0:
        harness.observe_bundle(stamp_ns=stamp_ns, observed_at_ns=stamp_ns)


def _imu_freeze_step(harness: _Harness, stamp_ns: int) -> None:
    if stamp_ns % _SCAN_NS == 0:
        harness.observe_lidar(stamp_ns)
        harness.observe_bundle(stamp_ns=stamp_ns, observed_at_ns=stamp_ns)


def _effect_missing_step(harness: _Harness, stamp_ns: int) -> None:
    harness.observe_imu(stamp_ns)
    if stamp_ns % _SCAN_NS == 0:
        harness.observe_lidar(stamp_ns)
        harness.observe_bundle(
            stamp_ns=stamp_ns,
            observed_at_ns=stamp_ns,
            include_effective=False,
        )


def _zero_support_step(harness: _Harness, stamp_ns: int) -> None:
    harness.observe_imu(stamp_ns)
    if stamp_ns % _SCAN_NS == 0:
        harness.observe_lidar(stamp_ns)
        harness.observe_bundle(
            stamp_ns=stamp_ns,
            observed_at_ns=stamp_ns,
            effective_count=0,
        )


def _numeric_invalid_step(harness: _Harness, stamp_ns: int) -> None:
    harness.observe_imu(stamp_ns)
    if stamp_ns % _SCAN_NS != 0:
        return
    harness.observe_lidar(stamp_ns)
    invalid_pose = (float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    harness.assembler.observe_native_odometry(
        stamp_ns=stamp_ns,
        observed_at_ns=stamp_ns,
        pose=invalid_pose,
        frame_id="camera_init",
        child_frame_id="body",
    )


def _future_stamp_step(harness: _Harness, stamp_ns: int) -> None:
    harness.observe_imu(stamp_ns)
    if stamp_ns % _SCAN_NS != 0:
        return
    harness.observe_lidar(stamp_ns)
    future_ns = stamp_ns + seconds_to_nanoseconds(0.010)
    harness.observe_bundle(
        stamp_ns=future_ns,
        observed_at_ns=stamp_ns,
    )


def _conflicting_duplicate_step(harness: _Harness, stamp_ns: int) -> None:
    harness.observe_imu(stamp_ns)
    if stamp_ns % _SCAN_NS != 0:
        return
    harness.observe_lidar(stamp_ns)
    previous_stamp_ns = stamp_ns - _SCAN_NS
    harness.assembler.observe_canonical_odometry(
        stamp_ns=previous_stamp_ns,
        observed_at_ns=stamp_ns,
        pose=harness._pose(previous_stamp_ns, offset=1.0),
        frame_id="map",
        child_frame_id="base_link",
    )


def _regressing_stamp_step(harness: _Harness, stamp_ns: int) -> None:
    harness.observe_imu(stamp_ns)
    if stamp_ns % _SCAN_NS != 0:
        return
    harness.observe_lidar(stamp_ns)
    harness.assembler.observe_canonical_odometry(
        stamp_ns=stamp_ns - 2 * _SCAN_NS,
        observed_at_ns=stamp_ns,
        pose=harness._pose(stamp_ns - 2 * _SCAN_NS),
        frame_id="map",
        child_frame_id="base_link",
    )


def _malformed_effect_step(harness: _Harness, stamp_ns: int) -> None:
    harness.observe_imu(stamp_ns)
    if stamp_ns % _SCAN_NS != 0:
        return
    harness.observe_lidar(stamp_ns)
    harness.observe_bundle(
        stamp_ns=stamp_ns,
        observed_at_ns=stamp_ns,
        include_effective=False,
    )
    harness.assembler.observe_effective_points(
        stamp_ns=stamp_ns,
        observed_at_ns=stamp_ns,
        point_count=_SUPPORT_COUNT,
        layout_valid=False,
    )


def _bad_native_frame_step(harness: _Harness, stamp_ns: int) -> None:
    harness.observe_imu(stamp_ns)
    if stamp_ns % _SCAN_NS != 0:
        return
    harness.observe_lidar(stamp_ns)
    harness.assembler.observe_native_odometry(
        stamp_ns=stamp_ns,
        observed_at_ns=stamp_ns,
        pose=harness._pose(stamp_ns),
        frame_id="wrong_frame",
        child_frame_id="body",
    )


def _gap_then_recovery_step(harness: _Harness, stamp_ns: int) -> None:
    harness.observe_imu(stamp_ns)
    if stamp_ns % _SCAN_NS != 0:
        return
    harness.observe_lidar(stamp_ns)
    first_fault_stamp_ns = seconds_to_nanoseconds(0.70)
    harness.observe_bundle(
        stamp_ns=stamp_ns,
        observed_at_ns=stamp_ns,
        include_effective=stamp_ns != first_fault_stamp_ns,
    )


def _validate_clock_reset_case() -> FaultCaseResult:
    harness = _Harness()
    harness.seed_tracking()
    reset_stamp_ns = seconds_to_nanoseconds(0.10)
    harness.assembler.reset()
    harness.machine.reset_for_clock(ros_now_ns=reset_stamp_ns)
    snapshot = harness.evaluate(reset_stamp_ns)
    failures: list[str] = []
    if snapshot.tracking_valid:
        failures.append("clock reset did not invalidate tracking")
    if snapshot.state != TrackingState.INITIALIZING:
        failures.append("new clock epoch did not return to INITIALIZING")
    if not snapshot.reasons & DegradationReason.CLOCK_RESET:
        failures.append("clock reset reason was not preserved")
    if snapshot.source_stamp_ns is not None or snapshot.confidence != 0.0:
        failures.append("clock reset did not clear source and score")
    return FaultCaseResult(
        name="clock_backward",
        expected_reason=DegradationReason.CLOCK_RESET.name,
        expected_state=TrackingState.INITIALIZING.name,
        injected_at_ns=reset_stamp_ns,
        detection_deadline_ns=reset_stamp_ns,
        detected_at_ns=reset_stamp_ns,
        detection_latency_ns=0,
        detected_state=snapshot.state.name,
        detected_reasons=int(snapshot.reasons),
        confidence_at_detection=snapshot.confidence,
        recovery_required=False,
        recovered_at_ns=None,
        passed=not failures,
        failures=tuple(failures),
    )


def validate_fastlio_fault_matrix() -> dict:
    """Run the required synthetic FAST hard-fault matrix."""

    cases = (
        _validate_case(
            name="odometry_and_source_freeze",
            expected_reason=DegradationReason.ODOMETRY_STALE,
            step=_odom_freeze_step,
            maximum_detection_latency_ns=seconds_to_nanoseconds(0.30),
        ),
        _validate_case(
            name="lidar_input_freeze",
            expected_reason=DegradationReason.LIDAR_STALE,
            step=_lidar_freeze_step,
            maximum_detection_latency_ns=seconds_to_nanoseconds(0.30),
        ),
        _validate_case(
            name="imu_input_freeze",
            expected_reason=DegradationReason.IMU_STALE,
            step=_imu_freeze_step,
            maximum_detection_latency_ns=seconds_to_nanoseconds(0.05),
            injection_delay_ns=_TICK_NS,
        ),
        _validate_case(
            name="effective_diagnostic_missing",
            expected_reason=DegradationReason.SIGNAL_MISSING,
            step=_effect_missing_step,
            maximum_detection_latency_ns=seconds_to_nanoseconds(0.10),
        ),
        _validate_case(
            name="zero_effective_support",
            expected_reason=DegradationReason.INSUFFICIENT_SUPPORT,
            step=_zero_support_step,
            maximum_detection_latency_ns=0,
        ),
        _validate_case(
            name="native_numeric_invalid",
            expected_reason=DegradationReason.NUMERIC_INVALID,
            step=_numeric_invalid_step,
            maximum_detection_latency_ns=0,
        ),
        _validate_case(
            name="future_source_stamp",
            expected_reason=DegradationReason.TIMESTAMP_INVALID,
            step=_future_stamp_step,
            maximum_detection_latency_ns=0,
        ),
        _validate_case(
            name="conflicting_duplicate",
            expected_reason=DegradationReason.TIMESTAMP_INVALID,
            step=_conflicting_duplicate_step,
            maximum_detection_latency_ns=0,
        ),
        _validate_case(
            name="regressing_source_stamp",
            expected_reason=DegradationReason.TIMESTAMP_INVALID,
            step=_regressing_stamp_step,
            maximum_detection_latency_ns=0,
        ),
        _validate_case(
            name="malformed_effect_layout",
            expected_reason=DegradationReason.NUMERIC_INVALID,
            step=_malformed_effect_step,
            maximum_detection_latency_ns=0,
        ),
        _validate_case(
            name="native_frame_contract_violation",
            expected_reason=DegradationReason.BACKEND_ERROR,
            step=_bad_native_frame_step,
            maximum_detection_latency_ns=0,
        ),
        _validate_clock_reset_case(),
        _validate_case(
            name="missing_scan_then_recovery",
            expected_reason=DegradationReason.SIGNAL_MISSING,
            step=_gap_then_recovery_step,
            maximum_detection_latency_ns=seconds_to_nanoseconds(0.10),
            recovery_required=True,
            end_after_ns=seconds_to_nanoseconds(1.40),
        ),
    )
    return {
        "schema_version": 1,
        "backend_id": "fastlio2",
        "calibration_id": "synthetic-fault-matrix-v1",
        "deployment_calibration": False,
        "logical_tick_ns": _TICK_NS,
        "thresholds_s": {
            "source": 0.30,
            "canonical_odometry": 0.30,
            "lidar_input": 0.30,
            "imu_input": 0.05,
            "diagnostic_match": 0.05,
            "future_stamp_tolerance": 0.005,
            "recovery_dwell": 0.50,
        },
        "case_count": len(cases),
        "passed_case_count": sum(case.passed for case in cases),
        "passed": all(case.passed for case in cases),
        "cases": [asdict(case) for case in cases],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/slam_confidence/fastlio_fault_validation.json"),
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path = output_path.resolve()
    if not (project_root / "configs/slam_confidence_contract.yaml").is_file():
        parser.error(f"not an anymal_locomotion project root: {project_root}")
    if not output_path.is_relative_to(project_root):
        parser.error(f"output must remain inside {project_root}: {output_path}")

    report = validate_fastlio_fault_matrix()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
