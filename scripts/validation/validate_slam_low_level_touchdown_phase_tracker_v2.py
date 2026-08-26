#!/usr/bin/env python3
"""One-shot fresh-data gate for the frozen touchdown phase-tracker v2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/slam_low_level_touchdown_phase_tracker_v2.yaml"
CANDIDATE_PATH = PROJECT_ROOT / "exported/slam_low_level_touchdown_phase_tracker_v2/candidate.json"
DEVELOPMENT_REPORT_PATH = PROJECT_ROOT / "docs/validation/slam_low_level_touchdown_phase_tracker_v2_development.json"
FITTER_PATH = PROJECT_ROOT / "scripts/validation/fit_slam_low_level_touchdown_phase_tracker_v2.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FITTER = _module("touchdown_phase_tracker_v2_fit", FITTER_PATH)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _integrity(root: Path, config: dict[str, Any]) -> list[str]:
    fresh = config["data_roles"]["fresh_validation"]
    expected = {
        (str(profile), int(block))
        for profile in fresh["profiles"] for block in fresh["blocks"]
    }
    observed = set()
    failures = []
    for trace_path in sorted(root.glob("*/block_*/zero/locomotion_diagnostics.json")):
        relative = trace_path.relative_to(root)
        profile = relative.parts[0]
        block = int(relative.parts[1].removeprefix("block_"))
        observed.add((profile, block))
        run_dir = trace_path.parent
        required = (
            run_dir / "baseline_trace_validation.json",
            run_dir / "stability_gate.json",
            run_dir / "driver.json",
        )
        if not all(path.is_file() for path in required):
            failures.append(f"missing_integrity_artifact:{profile}:{block}")
            continue
        parity, stability, driver = (_load(path) for path in required)
        if parity.get("passed") is not True:
            failures.append(f"model1450_or_command_parity:{profile}:{block}")
        if stability.get("gate", {}).get("passed") is not True:
            failures.append(f"physical_safety:{profile}:{block}")
        if driver.get("passed") is not True:
            failures.append(f"driver:{profile}:{block}")
    if observed != expected:
        if expected - observed:
            failures.append("missing_fresh_identity")
        if observed - expected:
            failures.append("unexpected_fresh_identity")
    if len(observed) != int(fresh["expected_run_count"]):
        failures.append("fresh_run_count")
    return sorted(set(failures))


def _metric_failures(metrics: dict[str, Any], config: dict[str, Any]) -> list[str]:
    gates = config["fresh_validation_gates"]
    aggregate = metrics["aggregate"]
    failures = []
    comparisons = (
        (aggregate["late_swing_precision"] >= float(gates["aggregate_minimum_late_swing_precision"]), "aggregate_precision"),
        (aggregate["false_trigger_fraction"] <= float(gates["aggregate_maximum_false_trigger_fraction"]), "aggregate_false_trigger"),
        (aggregate["late_swing_recall"] >= float(gates["aggregate_minimum_late_swing_recall"]), "aggregate_recall"),
        (aggregate["swing_progress_mae"] <= float(gates["aggregate_maximum_swing_progress_mae"]), "aggregate_progress_mae"),
    )
    failures.extend(name for passed, name in comparisons if not passed)
    for profile, value in metrics["by_profile"].items():
        if value["late_swing_precision"] < float(gates["each_profile_minimum_late_swing_precision"]):
            failures.append(f"profile_precision:{profile}")
        if value["late_swing_recall"] < float(gates["each_profile_minimum_late_swing_recall"]):
            failures.append(f"profile_recall:{profile}")
    if gates["all_four_feet_must_have_positive_predictions"] and any(
        int(value) == 0 for value in aggregate["positive_predictions_by_foot"]
    ):
        failures.append("missing_foot_predictions")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--development-report", type=Path, default=DEVELOPMENT_REPORT_PATH)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frozen-artifact", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        args.records_root.expanduser().resolve(), args.config.expanduser().resolve(),
        args.candidate.expanduser().resolve(), args.development_report.expanduser().resolve(),
        args.report.expanduser().resolve(), args.frozen_artifact.expanduser().resolve(),
    ]
    if any(not path.is_relative_to(PROJECT_ROOT) for path in paths):
        raise ValueError("all phase validation paths must remain inside project")
    root, config_path, candidate_path, development_path, report_path, frozen_path = paths
    if not root.is_dir() or not all(path.is_file() for path in (config_path, candidate_path, development_path)):
        raise ValueError("phase validation inputs are missing")
    config, candidate, development = map(_load, (config_path, candidate_path, development_path))
    if development.get("config_sha256") != _sha256(config_path):
        raise ValueError("development config hash mismatch")
    if development.get("candidate_sha256") != _sha256(candidate_path):
        raise ValueError("development candidate hash mismatch")
    if candidate.get("development_candidate") is not True or candidate.get("frozen") is not False:
        raise ValueError("candidate state is not development-only")
    integrity_failures = _integrity(root, config)
    loader_config = FITTER._v1_compatible_config(config)
    loader_config["source"]["expected_run_count"] = int(
        config["data_roles"]["fresh_validation"]["expected_run_count"]
    )
    rows = FITTER.V1.load_rows(root, loader_config)
    metrics = FITTER.evaluate(rows, candidate)
    metric_failures = _metric_failures(metrics, config)
    failures = sorted(set(integrity_failures + metric_failures))
    passed = not failures
    report = {
        "schema_version": 1,
        "kind": "slam_low_level_touchdown_phase_tracker_v2_fresh_validation",
        "passed": passed, "frozen": passed, "failures": failures,
        "config_sha256": _sha256(config_path),
        "candidate_sha256": _sha256(candidate_path),
        "fresh_blocks": config["data_roles"]["fresh_validation"]["blocks"],
        "metrics": metrics, "gates": config["fresh_validation_gates"],
        "block_597_allowed_next": passed,
        "effect_execution_authorized": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if passed:
        promoted = dict(candidate)
        promoted.update({
            "development_candidate": False, "frozen": True, "passed": True,
            "fresh_validation_report_path": str(report_path.relative_to(PROJECT_ROOT)),
            "fresh_validation_report_sha256": _sha256(report_path),
        })
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_path.write_text(json.dumps(promoted, indent=2) + "\n", encoding="utf-8")
    elif frozen_path.exists():
        raise ValueError("failed validation must not coexist with a frozen artifact")
    print(json.dumps({"passed": passed, "failures": failures, "metrics": metrics}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
