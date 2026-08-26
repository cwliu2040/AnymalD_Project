#!/usr/bin/env python3
"""Validate frozen ANYmal-D kinematics against finite differences and baseline traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS_SOURCE = PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"
sys.path.insert(0, str(ROS_SOURCE))

from anymal_locomotion_ros2.anymal_d_kinematics_core import (  # noqa: E402
    FOOT_ORDER,
    anymal_d_foot_kinematics,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/anymal_d_frozen_kinematics_v1.yaml"
DEFAULT_BASELINE = PROJECT_ROOT / "outputs/slam_low_level_touchdown_headroom_v1_retry2/baseline_envelope"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_world_from_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray((
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    ))


def finite_difference_error(sample_count: int = 64, epsilon: float = 1.0e-6) -> float:
    rng = np.random.default_rng(20260825)
    maximum = 0.0
    for position in rng.uniform(-1.2, 1.2, size=(sample_count, 12)):
        nominal = anymal_d_foot_kinematics(position)
        for joint in range(12):
            perturbation = np.zeros(12)
            perturbation[joint] = epsilon
            plus = anymal_d_foot_kinematics(position + perturbation).foot_position_base_m
            minus = anymal_d_foot_kinematics(position - perturbation).foot_position_base_m
            numeric = (plus - minus) / (2.0 * epsilon)
            error = np.max(np.abs(numeric - nominal.foot_jacobian_per_rad[:, :, joint]))
            maximum = max(maximum, float(error))
    return maximum


def trace_errors(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("*/block_*/zero/locomotion_diagnostics.json"))
    if len(paths) != 36:
        raise ValueError("kinematics trace validation requires the complete 36-run baseline")
    position_errors = []
    vertical_velocity_errors = []
    records = []
    for path in paths:
        trace = json.loads(path.read_text(encoding="utf-8"))
        if tuple(trace["metadata"]["joint_order"]) != (
            "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA", "LF_HFE", "LF_KFE",
            "LH_HFE", "LH_KFE", "RF_HFE", "RF_KFE", "RH_HFE", "RH_KFE",
        ):
            raise ValueError("trace canonical joint order mismatch")
        for sample in trace["samples"]:
            result = anymal_d_foot_kinematics(
                sample["joint_position_rad"], sample["joint_velocity_radps"]
            )
            truth_position = np.asarray([
                sample["feet"][foot]["position_b_m"] for foot in FOOT_ORDER
            ], dtype=np.float64)
            position_errors.append(result.foot_position_base_m - truth_position)
            rotation = _rotation_world_from_body(
                float(sample["roll_rad"]), float(sample["pitch_rad"]), float(sample["yaw_rad"])
            )
            foot_velocity_world = np.asarray([
                sample["feet"][foot]["velocity_w_mps"] for foot in FOOT_ORDER
            ], dtype=np.float64)
            base_linear = np.asarray(sample["actual_linear_velocity_body_mps"], dtype=np.float64)
            base_angular = np.asarray(sample["actual_angular_velocity_body_radps"], dtype=np.float64)
            truth_relative = (
                (rotation.T @ foot_velocity_world.T).T
                - base_linear
                - np.cross(np.broadcast_to(base_angular, (4, 3)), truth_position)
            )
            vertical_velocity_errors.extend(
                (result.foot_velocity_base_mps[:, 2] - truth_relative[:, 2]).tolist()
            )
        records.append({"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha256(path)})
    position = np.asarray(position_errors)
    vertical = np.asarray(vertical_velocity_errors)
    return {
        "run_count": len(paths),
        "sample_count": int(position.shape[0]),
        "foot_position_rmse_m": float(np.sqrt(np.mean(np.square(position)))),
        "foot_position_max_abs_error_m": float(np.max(np.abs(position))),
        "vertical_velocity_rmse_mps": float(np.sqrt(np.mean(np.square(vertical)))),
        "vertical_velocity_abs_error_p99_mps": float(np.quantile(np.abs(vertical), 0.99)),
        "trace_records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "docs/validation/anymal_d_frozen_kinematics_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    baseline_root = args.baseline_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if any(not path.is_relative_to(PROJECT_ROOT) for path in (config_path, baseline_root, output)):
        raise ValueError("kinematics validation paths must remain inside project")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    finite_difference = finite_difference_error()
    traces = trace_errors(baseline_root)
    gates = config["validation_gates"]
    checks = {
        "finite_difference_jacobian": finite_difference <= float(
            gates["finite_difference_jacobian_max_abs_error_m_per_rad"]
        ),
        "trace_foot_position_rmse": traces["foot_position_rmse_m"] <= float(
            gates["baseline_trace_foot_position_rmse_max_m"]
        ),
        "trace_foot_position_max_abs_error": traces["foot_position_max_abs_error_m"] <= float(
            gates["baseline_trace_foot_position_max_abs_error_max_m"]
        ),
        "trace_vertical_velocity_rmse": traces["vertical_velocity_rmse_mps"] <= float(
            gates["baseline_trace_vertical_velocity_rmse_max_mps"]
        ),
    }
    report = {
        "schema_version": 1,
        "kind": "anymal_d_frozen_kinematics_validation",
        "passed": all(checks.values()), "frozen": all(checks.values()),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": _sha256(config_path),
        "checks": checks,
        "finite_difference_jacobian_max_abs_error_m_per_rad": finite_difference,
        "trace_validation": traces,
        "simulator_foot_state_runtime_input_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"], "checks": checks,
        "finite_difference_error": finite_difference,
        "trace_metrics": {key: value for key, value in traces.items() if key != "trace_records"},
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
