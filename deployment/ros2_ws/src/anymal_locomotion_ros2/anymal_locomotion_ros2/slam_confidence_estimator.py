"""Validated runtime loader for the small calibrated confidence estimators."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from anymal_locomotion_ros2.slam_confidence_calibration_core import (
    CausalFeatureTransform,
    FEATURE_SCHEMAS,
    FEATURE_TRANSFORM_CONFIG,
    SUPPORT_FORECAST_GUARDS,
    apply_support_forecast_guard,
    calibration_fingerprint,
    isotonic_predict,
    logistic_predict,
)


ARTIFACT_PROVENANCE_SCHEMA_VERSION = 2
RUNTIME_PROVENANCE_EXPECTATIONS = {
    "fastlio2": {
        "backend_source_revision": "373aa886402b6307db2995ca12b3f4596ef4f633",
        "backend_source_manifest_sha256": "93526dffbcea31d0bb2f7d5211f4f2dcac8bbd816460f2db5bb1bc7de717ab05",
        "estimator_config_sha256": "355e861ec79f3d572735ec065d578e0b35855c1c02c1f9dedc4fcb58fa98b75f",
        "split_manifest_sha256": "b0572014071776d0a6a502bed20a4e60cda488031a471522a6f19a494c9f708f",
        "sensor_model": "isaac_sim_rtx_reconstructed_ouster32",
        "domain": "isaac_sim",
        "sensor_translation_in_body_xyz": [0.2, 0.0, 0.35],
        "timestamp_contract": "scan_end_header_with_sensor_order_column_time",
        "input_adapter_contract": "sensor_order_staggered_native_deskew_v1",
        "score_timing": "first_20hz_evaluation_after_source_then_sample_hold_v1",
    },
    "liosam": {
        "backend_source_revision": "08af3f32f01725372d4269838dc44c19c6d9e76b",
        "backend_source_manifest_sha256": "c39f148b4208771d5f8c1d18190ad439e51af7290831773aced1a8cb5c8c5ee2",
        "estimator_config_sha256": "bac9647a2114c702961c509bb3935fdd42c7177657e720515d1238df10d9e535",
        "split_manifest_sha256": "86976d2ba4be5028ddf375e95d24350ef3c589c09055cde233d624fb1f2e8534",
        "sensor_model": "isaac_sim_rtx_reconstructed_ouster32",
        "domain": "isaac_sim",
        "sensor_translation_in_body_xyz": [0.2, 0.0, 0.35],
        "timestamp_contract": "native_liosam_scan_start_header",
        "input_adapter_contract": "native_deskew_cloudinfo_exact_join_v1",
        "score_timing": "first_20hz_evaluation_after_source_then_sample_hold_v1",
    },
}


class CalibratedConfidenceEstimator:
    def __init__(self, artifact_path: str, *, backend_id: str) -> None:
        path = Path(artifact_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        fingerprint = str(payload.get("artifact_fingerprint_sha256", ""))
        unsigned = dict(payload)
        unsigned.pop("artifact_fingerprint_sha256", None)
        if not fingerprint or calibration_fingerprint(unsigned) != fingerprint:
            raise ValueError("confidence artifact fingerprint mismatch")
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported confidence artifact schema")
        if payload.get("backend_id") != backend_id:
            raise ValueError("confidence artifact backend mismatch")
        if payload.get("deskew_mode") != "native":
            raise ValueError("confidence artifact is not native-deskew calibrated")
        if payload.get("artifact_provenance_schema_version") != ARTIFACT_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("unsupported confidence artifact provenance schema")
        if tuple(payload.get("feature_schema", ())) != FEATURE_SCHEMAS[backend_id]:
            raise ValueError("confidence artifact feature schema mismatch")
        if payload.get("feature_transform") != FEATURE_TRANSFORM_CONFIG:
            raise ValueError("confidence artifact feature transform mismatch")
        if (
            payload.get("support_forecast_guard")
            != SUPPORT_FORECAST_GUARDS[backend_id]
        ):
            raise ValueError("confidence artifact support guard mismatch")
        provenance = payload.get("provenance", {})
        if provenance.get("runtime_contract") != RUNTIME_PROVENANCE_EXPECTATIONS[backend_id]:
            raise ValueError("confidence artifact runtime provenance mismatch")
        calibration_id = str(payload.get("calibration_id", ""))
        report_fingerprint = str(provenance.get("report_fingerprint_sha256", ""))
        if (
            not calibration_id.startswith("native-v1-")
            or calibration_id.removeprefix("native-v1-")
            != report_fingerprint[:12]
        ):
            raise ValueError("confidence calibration ID does not match its report")
        self.backend_id = backend_id
        self.calibration_id = calibration_id
        self.thresholds = dict(payload["thresholds"])
        self._logistic = payload["logistic"]
        self._isotonic = payload["isotonic"]
        self._transform = CausalFeatureTransform(backend_id)
        self._last_confidence = 0.0

    @property
    def confidence(self) -> float:
        return self._last_confidence

    def observe(
        self,
        row: dict[str, Any],
        *,
        update_confidence: bool = True,
    ) -> float:
        vector = self._transform.transform(row)
        if vector is None:
            raise ValueError("runtime confidence feature vector is incomplete")
        raw = logistic_predict(
            self._logistic, np.asarray([vector], dtype=np.float64)
        )
        calibrated = apply_support_forecast_guard(
            self.backend_id,
            np.asarray([vector], dtype=np.float64),
            isotonic_predict(self._isotonic, raw),
        )
        if update_confidence:
            self._last_confidence = float(np.clip(calibrated[0], 0.0, 1.0))
        return self._last_confidence

    def reset(self) -> None:
        self._transform = CausalFeatureTransform(self.backend_id)
        self._last_confidence = 0.0


class CausalPoseMotion:
    """Causal source-stamped speed features shared by both runtime backends."""

    def __init__(self) -> None:
        self._previous: tuple[int, tuple[float, ...]] | None = None

    def reset(self) -> None:
        self._previous = None

    def observe(
        self, stamp_ns: int, pose: tuple[float, float, float, float, float, float, float]
    ) -> dict[str, float]:
        speed = yaw_rate = 0.0
        if self._previous is not None and stamp_ns > self._previous[0]:
            previous_stamp, previous_pose = self._previous
            dt_s = (stamp_ns - previous_stamp) * 1.0e-9
            speed = math.dist(pose[:3], previous_pose[:3]) / dt_s
            yaw = self._yaw(pose)
            previous_yaw = self._yaw(previous_pose)
            delta = math.atan2(math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw))
            yaw_rate = abs(math.degrees(delta) / dt_s)
        if self._previous is None or stamp_ns > self._previous[0]:
            self._previous = (stamp_ns, pose)
        return {"linear_speed_mps": speed, "yaw_rate_deg_s": yaw_rate}

    @staticmethod
    def _yaw(pose: tuple[float, ...]) -> float:
        x, y, z, w = pose[3:]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
