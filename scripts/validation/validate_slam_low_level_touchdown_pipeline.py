#!/usr/bin/env python3
"""Replay the complete touchdown residual pipeline on development traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS_SOURCE = PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"
sys.path.insert(0, str(ROS_SOURCE))

from anymal_locomotion_ros2.touchdown_phase_tracker_core import (  # noqa: E402
    initial_touchdown_phase_tracker_state,
)
from anymal_locomotion_ros2.touchdown_residual_pipeline_core import (  # noqa: E402
    apply_touchdown_residual_pipeline,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/slam_low_level_touchdown_pipeline_validation_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "docs/validation/slam_low_level_touchdown_pipeline_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if any(not path.is_relative_to(PROJECT_ROOT) for path in (config_path, output)):
        raise ValueError("pipeline validation paths must remain inside project")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidate_path = (PROJECT_ROOT / config["phase_candidate"]["path"]).resolve()
    if _sha256(candidate_path) != config["phase_candidate"]["sha256"]:
        raise ValueError("phase candidate SHA-256 mismatch")
    for section, path_key, hash_key in (
        ("kinematics", "config_path", "config_sha256"),
        ("kinematics", "report_path", "report_sha256"),
    ):
        path = (PROJECT_ROOT / config[section][path_key]).resolve()
        if not path.is_file() or _sha256(path) != config[section][hash_key]:
            raise ValueError(f"pipeline prerequisite mismatch: {path}")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("frozen") is not False or candidate.get("development_candidate") is not True:
        raise ValueError("static pipeline replay requires the nondeployable development candidate")
    root = (PROJECT_ROOT / config["source"]["baseline_root"]).resolve()
    paths = sorted(root.glob("*/block_*/zero/locomotion_diagnostics.json"))
    if len(paths) != int(config["source"]["expected_run_count"]):
        raise ValueError("pipeline baseline run count mismatch")
    total_ticks = 0
    active_ticks = 0
    maximum_residual = 0.0
    maximum_horizontal = 0.0
    minimum_vertical = float("inf")
    command_exact = True
    noneligible_zero = True
    mode_counts: dict[str, int] = {}
    for path in paths:
        samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
        position = np.asarray([sample["joint_position_rad"] for sample in samples])
        velocity = np.asarray([sample["joint_velocity_radps"] for sample in samples])
        action = np.asarray([sample["applied_raw_action"] for sample in samples])
        state = initial_touchdown_phase_tracker_state()
        for index in range(9, len(samples)):
            result = apply_touchdown_residual_pipeline(
                requested_command=samples[index]["command"],
                backbone_action=action[index],
                joint_position_history=position[:index + 1],
                joint_velocity_history=velocity[:index + 1],
                previous_action_history=action[:index],
                current_joint_position_rad=position[index],
                current_joint_velocity_radps=velocity[index],
                phase_artifact=candidate,
                phase_state=state,
                attenuation_fraction=float(config["mechanism"]["attenuation_fraction"]),
                tracking_valid=True,
                allow_unfrozen_phase_for_validation=True,
            )
            state = result.phase.state
            total_ticks += 1
            mode_counts[result.mode] = mode_counts.get(result.mode, 0) + 1
            command_exact &= bool(np.array_equal(
                result.intervention.effective_command,
                np.asarray(samples[index]["command"], dtype=np.float32),
            ))
            residual = result.intervention.action_residual
            maximum_residual = max(maximum_residual, float(np.max(np.abs(residual))))
            if result.mode != "valid_touchdown_residual":
                noneligible_zero &= bool(np.array_equal(residual, np.zeros(12, dtype=np.float32)))
                continue
            active_ticks += 1
            delta = np.einsum(
                "fij,j->fi", result.kinematics.foot_jacobian_per_policy_action, residual
            )
            active = result.intervention.active_feet
            maximum_horizontal = max(
                maximum_horizontal, float(np.max(np.abs(delta[active, :2])))
            )
            minimum_vertical = min(minimum_vertical, float(np.min(delta[active, 2])))
    gates = config["gates"]
    checks = {
        "active_tick_count": active_ticks >= int(gates["minimum_active_tick_count"]),
        "residual_bound": maximum_residual <= float(
            gates["residual_linf_maximum_with_float32_tolerance"]
        ),
        "horizontal_first_order": maximum_horizontal <= float(
            gates["horizontal_first_order_max_abs_m"]
        ),
        "vertical_correction_direction": minimum_vertical >= float(
            gates["vertical_correction_minimum_m"]
        ),
        "original_command_bit_exact": command_exact,
        "noneligible_exact_zero": noneligible_zero,
    }
    report = {
        "schema_version": 1,
        "kind": "slam_low_level_touchdown_pipeline_static_validation",
        "passed": all(checks.values()), "effect_claim_allowed": False,
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": _sha256(config_path),
        "checks": checks, "total_tick_count": total_ticks,
        "active_tick_count": active_ticks,
        "active_tick_fraction": active_ticks / total_ticks,
        "maximum_residual_linf": maximum_residual,
        "maximum_horizontal_first_order_m": maximum_horizontal,
        "minimum_vertical_correction_m": minimum_vertical,
        "mode_counts": mode_counts,
        "phase_candidate_frozen": False,
        "runtime_simulator_foot_or_contact_input_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
