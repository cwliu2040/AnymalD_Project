#!/usr/bin/env python3
"""Run the candidate policy against both native calibrated SLAM backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS2_WORKSPACE = PROJECT_ROOT / "deployment" / "ros2_ws"
DEFAULT_MATRIX = (
    PROJECT_ROOT / "configs" / "slam_confidence_locomotion_matrix.yaml"
)
DEFAULT_STABILITY_CONFIG = (
    PROJECT_ROOT / "configs" / "stability_diagnostics.yaml"
)
DEFAULT_OUTPUT_PARENT = (
    PROJECT_ROOT
    / "logs"
    / "slam_confidence_locomotion_matrix"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(path: Path, *, must_exist: bool) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path must remain inside {PROJECT_ROOT}: {resolved}")
    if must_exist and not resolved.is_file():
        raise ValueError(f"file does not exist: {resolved}")
    return resolved


def _load_document(path: Path) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"document must contain a mapping: {path}")
    return document


def validate_candidate_artifacts(
    matrix: dict[str, Any],
) -> dict[str, str]:
    candidate = matrix["candidate"]
    policy_path = _project_path(
        PROJECT_ROOT / str(candidate["policy_path"]), must_exist=True
    )
    metadata_path = _project_path(
        PROJECT_ROOT / str(candidate["metadata_path"]), must_exist=True
    )
    parity_path = _project_path(
        PROJECT_ROOT / str(candidate["parity_report_path"]), must_exist=True
    )
    policy_sha256 = _sha256(policy_path)
    if policy_sha256 != str(candidate["policy_sha256"]):
        raise ValueError("candidate ONNX SHA-256 does not match matrix")

    metadata = _load_document(metadata_path)
    if metadata.get("observation", {}).get("dimension") != 51:
        raise ValueError("candidate metadata is not explicitly 51-D")
    metadata_checkpoint_sha256 = str(
        metadata.get("checkpoint", {}).get("sha256", "")
    )
    if metadata_checkpoint_sha256 != str(candidate["checkpoint_sha256"]):
        raise ValueError("candidate checkpoint SHA-256 does not match metadata")
    metadata_policy_sha256 = str(
        metadata.get("artifacts", {}).get("onnx", {}).get("sha256", "")
    )
    if metadata_policy_sha256 != policy_sha256:
        raise ValueError("candidate ONNX SHA-256 does not match metadata")

    parity = _load_document(parity_path)
    parity_artifacts = parity.get("artifacts", {})
    if not parity.get("passed"):
        raise ValueError("candidate export parity did not pass")
    if (
        parity_artifacts.get("checkpoint", {}).get("sha256")
        != metadata_checkpoint_sha256
    ):
        raise ValueError("parity report was generated from another checkpoint")
    if parity_artifacts.get("onnx", {}).get("sha256") != policy_sha256:
        raise ValueError("parity report was generated from another ONNX")
    artifacts = {
        "policy_path": str(policy_path),
        "metadata_path": str(metadata_path),
        "parity_report_path": str(parity_path),
        "checkpoint_sha256": metadata_checkpoint_sha256,
        "policy_sha256": policy_sha256,
        "parity_report_sha256": _sha256(parity_path),
    }
    checkpoint_value = candidate.get("checkpoint_path")
    agent_config_value = candidate.get("agent_config_path")
    if (checkpoint_value is None) != (agent_config_value is None):
        raise ValueError(
            "candidate checkpoint_path and agent_config_path must be configured together"
        )
    if checkpoint_value is not None:
        checkpoint_path = _project_path(
            PROJECT_ROOT / str(checkpoint_value), must_exist=True
        )
        agent_config_path = _project_path(
            PROJECT_ROOT / str(agent_config_value), must_exist=True
        )
        if _sha256(checkpoint_path) != metadata_checkpoint_sha256:
            raise ValueError("candidate checkpoint SHA-256 does not match metadata")
        artifacts.update(
            {
                "checkpoint_path": str(checkpoint_path),
                "agent_config_path": str(agent_config_path),
                "agent_config_sha256": _sha256(agent_config_path),
            }
        )
    estimator_metadata_value = candidate.get("velocity_estimator_metadata_path")
    if estimator_metadata_value is not None:
        estimator_metadata_path = _project_path(
            PROJECT_ROOT / str(estimator_metadata_value), must_exist=True
        )
        estimator_metadata_sha256 = _sha256(estimator_metadata_path)
        if estimator_metadata_sha256 != str(
            candidate.get("velocity_estimator_metadata_sha256", "")
        ):
            raise ValueError("velocity estimator metadata SHA-256 does not match matrix")
        estimator_metadata = _load_document(estimator_metadata_path)
        if estimator_metadata.get("contract_id") != "anymal-d-proprioceptive-velocity-v1":
            raise ValueError("velocity estimator metadata contract is invalid")
        estimator_onnx = estimator_metadata.get("artifacts", {}).get("onnx", {})
        estimator_onnx_path = _project_path(
            estimator_metadata_path.parent / str(estimator_onnx.get("path", "")),
            must_exist=True,
        )
        if _sha256(estimator_onnx_path) != estimator_onnx.get("sha256"):
            raise ValueError("velocity estimator ONNX digest does not match metadata")
        artifacts.update(
            {
                "velocity_estimator_metadata_path": str(estimator_metadata_path),
                "velocity_estimator_metadata_sha256": estimator_metadata_sha256,
                "velocity_estimator_onnx_path": str(estimator_onnx_path),
                "velocity_estimator_onnx_sha256": _sha256(estimator_onnx_path),
            }
        )
    return artifacts


def validate_stability_config(matrix: dict[str, Any]) -> dict[str, str]:
    value = matrix.get("stability_config_path")
    path = _project_path(
        PROJECT_ROOT / str(value) if value is not None else DEFAULT_STABILITY_CONFIG,
        must_exist=True,
    )
    config = _load_document(path)
    if config.get("schema_version") != 1:
        raise ValueError("stability config schema_version must be 1")
    hard_gate = config.get("hard_gate")
    tracking_gate = config.get("tracking_gate")
    if not isinstance(hard_gate, dict) or not isinstance(tracking_gate, dict):
        raise ValueError("stability config is missing gate mappings")
    tracking_enabled = tracking_gate.get("enabled", True)
    if not isinstance(tracking_enabled, bool):
        raise ValueError("tracking_gate.enabled must be boolean")
    return {
        "stability_config_path": str(path),
        "stability_config_sha256": _sha256(path),
    }


def validate_policy_diagnostics(
    diagnostics: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    records = diagnostics.get("records")
    if diagnostics.get("schema_version") != 2 or not isinstance(records, list):
        return {
            "passed": False,
            "failures": ["policy diagnostics schema is not version 2"],
            "record_count": 0,
        }

    minimum_records = int(gate["minimum_records"])
    dimension = int(gate["observation_dimension"])
    tracking_valid_count = 0
    non_fail_closed_count = 0
    fail_closed_count = 0
    watchdog_timeout_count = 0
    finite_record_count = 0
    for record in records:
        observation = record.get("observation")
        if not isinstance(observation, list) or len(observation) != dimension:
            failures.append("policy observation dimension mismatch")
            continue
        try:
            values = tuple(float(value) for value in observation)
        except (TypeError, ValueError):
            failures.append("policy observation contains a non-numeric value")
            continue
        if not all(math.isfinite(value) for value in values):
            failures.append("policy observation contains NaN or Inf")
            continue
        finite_record_count += 1
        confidence, tracking_valid, normalized_age = values[48:51]
        if not 0.0 <= confidence <= 1.0:
            failures.append("policy confidence observation is outside [0, 1]")
        if tracking_valid not in (0.0, 1.0):
            failures.append("policy tracking-valid observation is not binary")
        if not 0.0 <= normalized_age <= 1.0:
            failures.append("policy confidence age is outside [0, 1]")
        tracking_valid_count += int(tracking_valid == 1.0)
        is_fail_closed = (
            confidence == 0.0
            and tracking_valid == 0.0
            and normalized_age == 1.0
        )
        fail_closed_count += int(is_fail_closed)
        non_fail_closed_count += int(not is_fail_closed)
        watchdog_timeout_count += int(bool(record.get("watchdog_timed_out")))

    if len(records) < minimum_records:
        failures.append(
            f"policy record count {len(records)} is below {minimum_records}"
        )
    if finite_record_count != len(records):
        failures.append("not every policy record has a finite observation")
    if gate.get("require_tracking_valid_observation") and not tracking_valid_count:
        failures.append("policy never consumed tracking_valid=1")
    if gate.get("require_non_fail_closed_observation") and not non_fail_closed_count:
        failures.append("policy only consumed fail-closed confidence vectors")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "record_count": len(records),
        "finite_record_count": finite_record_count,
        "tracking_valid_observation_count": tracking_valid_count,
        "non_fail_closed_observation_count": non_fail_closed_count,
        "fail_closed_observation_count": fail_closed_count,
        "watchdog_timeout_count": watchdog_timeout_count,
    }


def _validated_passed_cell(
    path: Path,
    *,
    matrix_id: str,
    matrix_sha256: str,
    backend: str,
    profile: str,
    repetition: int,
    calibration_id: str,
    artifacts: dict[str, str],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = _load_document(path)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return None
    expected_artifacts = {
        key: artifacts[key]
        for key in (
            "checkpoint_sha256",
            "policy_sha256",
            "parity_report_sha256",
            "velocity_estimator_metadata_sha256",
            "velocity_estimator_onnx_sha256",
            "stability_config_sha256",
            "agent_config_sha256",
        )
        if key in artifacts
    }
    identity_matches = bool(
        result.get("passed")
        and result.get("matrix_id") == matrix_id
        and result.get("matrix_sha256") == matrix_sha256
        and result.get("backend") == backend
        and result.get("profile") == profile
        and result.get("repetition") == repetition
        and result.get("expected_calibration_id") == calibration_id
        and all(
            result.get("artifacts", {}).get(key) == value
            for key, value in expected_artifacts.items()
        )
    )
    return result if identity_matches else None


def _matrix_output_root(matrix_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", matrix_id):
        raise ValueError(
            "matrix_id must contain only lowercase letters, digits, '.', '_', or '-'"
        )
    return DEFAULT_OUTPUT_PARENT / matrix_id


def _summary_filename(
    *,
    configured_backends: tuple[str, ...],
    configured_profiles: tuple[str, ...],
    configured_repetitions: int,
    backends: tuple[str, ...],
    profiles: tuple[str, ...],
    repetitions: int,
) -> str:
    if (
        backends == configured_backends
        and profiles == configured_profiles
        and repetitions == configured_repetitions
    ):
        return "matrix_summary.json"
    selection = json.dumps(
        {
            "backends": backends,
            "profiles": profiles,
            "repetitions": repetitions,
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(selection).hexdigest()[:12]
    return f"matrix_summary.selection-{digest}.json"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--backend", action="append", dest="backends")
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    matrix_path = _project_path(args.matrix, must_exist=True)
    matrix = _load_document(matrix_path)
    if matrix.get("schema_version") != 1:
        raise ValueError("locomotion matrix schema_version must be 1")
    matrix_id = str(matrix["matrix_id"])
    matrix_sha256 = _sha256(matrix_path)
    output_root = _project_path(
        args.output_root if args.output_root is not None else _matrix_output_root(matrix_id),
        must_exist=False,
    )
    artifacts = validate_candidate_artifacts(matrix)
    artifacts.update(validate_stability_config(matrix))
    mechanism_gate_config = matrix.get("mechanism_sidecar_gate", {})
    mechanism_enabled = bool(mechanism_gate_config.get("enabled", False))
    if mechanism_enabled and not {
        "checkpoint_path",
        "agent_config_path",
    }.issubset(artifacts):
        raise ValueError(
            "mechanism sidecar requires candidate checkpoint_path and agent_config_path"
        )
    mechanism_action_atol = float(
        mechanism_gate_config.get("maximum_action_reconstruction_error", 1.0e-5)
    )
    if not math.isfinite(mechanism_action_atol) or mechanism_action_atol < 0.0:
        raise ValueError("mechanism action reconstruction tolerance is invalid")
    isaaclab_root = Path(
        os.environ.get("ISAACLAB_ROOT", Path.home() / "IsaacLab")
    ).expanduser()
    mechanism_python = isaaclab_root / "_isaac_sim" / "python.sh"
    if mechanism_enabled and not mechanism_python.is_file():
        raise FileNotFoundError(
            f"mechanism sidecar Python runtime is missing: {mechanism_python}"
        )

    configured_backends = tuple(str(value) for value in matrix["backends"])
    backends = tuple(args.backends) if args.backends else configured_backends
    unknown_backends = sorted(set(backends) - set(configured_backends))
    if unknown_backends:
        raise ValueError(f"unconfigured backends: {unknown_backends}")
    configured_profiles = tuple(str(value) for value in matrix["profiles"])
    profiles = tuple(args.profiles) if args.profiles else configured_profiles
    unknown_profiles = sorted(set(profiles) - set(configured_profiles))
    if unknown_profiles:
        raise ValueError(f"unconfigured profiles: {unknown_profiles}")
    repetitions = (
        int(args.repetitions)
        if args.repetitions is not None
        else int(matrix["repetitions"])
    )
    if repetitions <= 0 or not backends or not profiles:
        raise ValueError("matrix requires positive repetitions and nonempty axes")
    configured_repetitions = int(matrix["repetitions"])

    environment = os.environ.copy()
    environment["ROS_LOG_DIR"] = str(PROJECT_ROOT / "logs" / "ros")
    source_root = str(PROJECT_ROOT / "source" / "anymal_locomotion")
    environment["PYTHONPATH"] = (
        f"{source_root}:{environment.get('PYTHONPATH', '')}"
    )
    results: list[dict[str, Any]] = []
    stop = False
    for backend in backends:
        backend_config = matrix["backends"][backend]
        if backend_config.get("deskew_mode") != "native":
            raise ValueError(f"{backend} matrix arm is not native deskew")
        calibration_id = str(backend_config["expected_calibration_id"])
        policy_odometry_topic = str(
            backend_config.get("policy_odometry_topic", "/slam/odom")
        )
        if not policy_odometry_topic.startswith("/"):
            raise ValueError(
                f"{backend} policy_odometry_topic must be absolute"
            )
        for profile in profiles:
            for repetition in range(1, repetitions + 1):
                run_dir = (
                    output_root / backend / profile / f"run_{repetition:02d}"
                )
                cell_path = run_dir / "cell.json"
                existing = (
                    None
                    if args.rerun
                    else _validated_passed_cell(
                        cell_path,
                        matrix_id=matrix_id,
                        matrix_sha256=matrix_sha256,
                        backend=backend,
                        profile=profile,
                        repetition=repetition,
                        calibration_id=calibration_id,
                        artifacts=artifacts,
                    )
                )
                if existing is not None:
                    print(
                        f"SKIP passed {backend}/{profile}/run_{repetition:02d}",
                        flush=True,
                    )
                    results.append(existing)
                    continue

                print(
                    f"RUN {backend}/{profile}/run_{repetition:02d}",
                    flush=True,
                )
                launch = subprocess.run(
                    (
                        "ros2",
                        "launch",
                        "anymal_locomotion_ros2",
                        str(matrix["launch_file"]),
                        f"slam_backend:={backend}",
                        f"expected_confidence_backend:={backend}",
                        f"expected_calibration_id:={calibration_id}",
                        f"policy_odometry_topic:={policy_odometry_topic}",
                        "enable_confidence:=true",
                        f"profile:={profile}",
                        f"output_dir:={run_dir}",
                        f"simulation_steps:={int(matrix['simulation_steps'])}",
                        f"policy_path:={artifacts['policy_path']}",
                        f"metadata_path:={artifacts['metadata_path']}",
                        *(
                            (
                                "enable_velocity_estimator:=true",
                                "policy_inference_trigger:=estimator_joint_state",
                                "velocity_estimator_metadata_path:="
                                + artifacts["velocity_estimator_metadata_path"],
                            )
                            if "velocity_estimator_metadata_path" in artifacts
                            else ()
                        ),
                    ),
                    cwd=ROS2_WORKSPACE,
                    env=environment,
                    check=False,
                )

                trace_path = run_dir / "locomotion_diagnostics.json"
                driver_path = run_dir / "driver.json"
                policy_diagnostics_path = run_dir / "policy_diagnostics.json"
                gate_path = run_dir / "gate.json"
                mechanism_path = run_dir / "mechanism_sidecar.json"
                evaluation_returncode: int | None = None
                if trace_path.is_file() and driver_path.is_file():
                    evaluation = subprocess.run(
                        (
                            sys.executable,
                            str(
                                PROJECT_ROOT
                                / "scripts"
                                / "validation"
                                / "evaluate_stability_trace.py"
                            ),
                            "--trace",
                            str(trace_path),
                            "--driver",
                            str(driver_path),
                            "--config",
                            artifacts["stability_config_path"],
                            "--output",
                            str(gate_path),
                        ),
                        cwd=PROJECT_ROOT,
                        env=environment,
                        check=False,
                    )
                    evaluation_returncode = evaluation.returncode

                mechanism_returncode: int | None = None
                if mechanism_enabled and policy_diagnostics_path.is_file():
                    mechanism_path.unlink(missing_ok=True)
                    mechanism = subprocess.run(
                        (
                            str(mechanism_python),
                            str(
                                PROJECT_ROOT
                                / "scripts"
                                / "validation"
                                / "build_slam_confidence_mechanism_sidecar.py"
                            ),
                            "--diagnostics",
                            str(policy_diagnostics_path),
                            "--checkpoint",
                            artifacts["checkpoint_path"],
                            "--agent-config",
                            artifacts["agent_config_path"],
                            "--output",
                            str(mechanism_path),
                            "--action-atol",
                            str(mechanism_action_atol),
                        ),
                        cwd=PROJECT_ROOT,
                        env=environment,
                        check=False,
                    )
                    mechanism_returncode = mechanism.returncode

                driver = (
                    _load_document(driver_path) if driver_path.is_file() else {}
                )
                policy_gate = (
                    validate_policy_diagnostics(
                        _load_document(policy_diagnostics_path),
                        matrix["policy_diagnostics_gate"],
                    )
                    if policy_diagnostics_path.is_file()
                    else {
                        "passed": False,
                        "failures": ["policy diagnostics file is missing"],
                        "record_count": 0,
                    }
                )
                stability_gate = (
                    _load_document(gate_path).get("gate", {})
                    if gate_path.is_file()
                    else {}
                )
                mechanism_gate = (
                    _load_document(mechanism_path).get("gate", {})
                    if mechanism_path.is_file()
                    else (
                        {"passed": False, "failures": ["mechanism sidecar is missing"]}
                        if mechanism_enabled
                        else {"passed": True, "enabled": False}
                    )
                )
                passed = bool(
                    launch.returncode == 0
                    and evaluation_returncode == 0
                    and driver.get("passed")
                    and stability_gate.get("passed")
                    and policy_gate.get("passed")
                    and (
                        not mechanism_enabled
                        or mechanism_returncode == 0
                    )
                    and mechanism_gate.get("passed")
                )
                cell = {
                    "schema_version": 1,
                    "matrix_id": matrix_id,
                    "matrix_sha256": matrix_sha256,
                    "backend": backend,
                    "expected_calibration_id": calibration_id,
                    "deskew_mode": "native",
                    "profile": profile,
                    "repetition": repetition,
                    "artifacts": artifacts,
                    "launch_returncode": launch.returncode,
                    "evaluation_returncode": evaluation_returncode,
                    "mechanism_returncode": mechanism_returncode,
                    "driver_passed": bool(driver.get("passed")),
                    "slam_confidence": driver.get("slam_confidence"),
                    "stability_gate": stability_gate,
                    "policy_diagnostics_gate": policy_gate,
                    "mechanism_sidecar_gate": mechanism_gate,
                    "passed": passed,
                }
                _write_json(cell_path, cell)
                results.append(cell)
                if args.fail_fast and not passed:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    failed = [result for result in results if not result.get("passed")]
    expected_count = len(backends) * len(profiles) * repetitions
    summary = {
        "schema_version": 1,
        "matrix_id": matrix_id,
        "matrix_path": str(matrix_path),
        "matrix_sha256": matrix_sha256,
        "output_root": str(output_root),
        "artifacts": artifacts,
        "backends": list(backends),
        "profiles": list(profiles),
        "repetitions": repetitions,
        "selection_is_full_matrix": bool(
            backends == configured_backends
            and profiles == configured_profiles
            and repetitions == configured_repetitions
        ),
        "expected_run_count": expected_count,
        "completed_run_count": len(results),
        "passed_run_count": len(results) - len(failed),
        "failed_run_count": len(failed),
        "passed": not failed and len(results) == expected_count,
        "runs": results,
    }
    summary_path = output_root / _summary_filename(
        configured_backends=configured_backends,
        configured_profiles=configured_profiles,
        configured_repetitions=configured_repetitions,
        backends=backends,
        profiles=profiles,
        repetitions=repetitions,
    )
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "expected_run_count",
                    "completed_run_count",
                    "passed_run_count",
                    "failed_run_count",
                    "passed",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Matrix summary written to: {summary_path}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
