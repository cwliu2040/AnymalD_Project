#!/usr/bin/env python3
"""Build the frozen model1450 anti-collapse envelope from baseline records."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "configs/slam_low_level_touchdown_headroom_v1.yaml"
VALIDATOR_PATH = (
    PROJECT_ROOT / "scripts/validation/validate_slam_low_level_touchdown_headroom_protocol.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validator_module():
    spec = importlib.util.spec_from_file_location("touchdown_protocol", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load touchdown protocol validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(root.glob("**/baseline_envelope_run_record.json"))
    records = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    if any(not isinstance(record, dict) for _, record in records):
        raise ValueError("baseline records must be mappings")
    return records


def _samples(value: Any, metric: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"baseline metric must be finite scalar or 1-D samples: {metric}")
    return array


def build_envelope(
    records_with_paths: list[tuple[Path, dict[str, Any]]],
    protocol: dict[str, Any],
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    expected_rows = _validator_module().build_baseline_schedule(protocol)
    expected = {(str(row["profile"]), int(row["block_id"])) for row in expected_rows}
    observed: list[tuple[str, int]] = []
    failures: list[str] = []
    metrics = tuple(str(value) for value in protocol["baseline_envelope"]["metrics"])
    by_profile: dict[str, dict[str, list[dict[str, float]]]] = {
        str(profile): {metric: [] for metric in metrics}
        for profile in protocol["baseline_envelope"]["profiles"]
    }
    record_hashes = []
    lower_q = float(
        protocol["baseline_envelope"]["bounds_rule"]["within_run_lower_quantile"]
    )
    upper_q = float(
        protocol["baseline_envelope"]["bounds_rule"]["within_run_upper_quantile"]
    )
    for path, record in records_with_paths:
        identity = record.get("identity", {})
        profile = str(identity.get("profile", ""))
        try:
            block = int(identity.get("block_id"))
        except (TypeError, ValueError):
            failures.append("invalid_block_id")
            continue
        observed.append((profile, block))
        if identity.get("stage") != "baseline_envelope":
            failures.append("stage_mismatch")
        if identity.get("arm") != "zero":
            failures.append("baseline_must_use_exact_zero_arm")
        if record.get("dataset_role") != "anti_collapse_calibration_only":
            failures.append("incorrect_dataset_role")
        if record.get("gate", {}).get("passed") is not True:
            failures.append("baseline_integrity_gate_failure")
        checks = record.get("trace", {}).get("checks", {})
        if checks.get("requested_command_exact_original") is not True:
            failures.append("baseline_command_not_exact_original")
        if checks.get("zero_residual_exact_model1450") is not True:
            failures.append("baseline_not_exact_model1450")
        if checks.get("frame_samples_used_as_independent_replicates") is not False:
            failures.append("frame_pseudoreplication_forbidden")
        if profile not in by_profile:
            failures.append("unexpected_profile")
            continue
        sample_units = record.get("sample_units", {})
        values = record.get("metrics", {})
        for metric in metrics:
            if sample_units.get(metric) not in {"complete_run", "complete_stride"}:
                failures.append(f"invalid_sample_unit:{metric}")
                continue
            try:
                array = _samples(values.get(metric), metric)
            except (TypeError, ValueError):
                failures.append(f"invalid_metric:{metric}")
                continue
            by_profile[profile][metric].append({
                "block_id": block,
                "lower": float(np.quantile(array, lower_q)),
                "upper": float(np.quantile(array, upper_q)),
                "sample_count": int(len(array)),
            })
        if path.is_file():
            try:
                relative_path = str(path.resolve().relative_to(PROJECT_ROOT))
            except ValueError:
                relative_path = str(path.resolve())
            record_hashes.append({"path": relative_path, "sha256": _sha256(path)})

    duplicate = sorted({identity for identity in observed if observed.count(identity) > 1})
    observed_set = set(observed)
    if duplicate:
        failures.append("duplicate_baseline_identity")
    if expected - observed_set:
        failures.append("missing_baseline_identity")
    if observed_set - expected:
        failures.append("unexpected_baseline_identity")
    profile_envelopes: dict[str, dict[str, Any]] = {}
    for profile, metric_rows in by_profile.items():
        profile_envelopes[profile] = {}
        for metric, rows in metric_rows.items():
            if len(rows) != int(protocol["baseline_envelope"]["minimum_complete_runs_per_profile"]):
                failures.append(f"incomplete_profile_metric:{profile}:{metric}")
                continue
            profile_envelopes[profile][metric] = {
                "lower": min(row["lower"] for row in rows),
                "upper": max(row["upper"] for row in rows),
                "complete_run_count": len(rows),
                "within_run_summaries": sorted(rows, key=lambda row: int(row["block_id"])),
            }
    failures = sorted(set(failures))
    resolved_protocol = protocol_path.expanduser().resolve()
    return {
        "schema_version": 1,
        "kind": "model1450_original_command_anti_collapse_envelope",
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": _sha256(resolved_protocol) if resolved_protocol.is_file() else None,
        "dataset_role": "anti_collapse_calibration_only",
        "frozen": not failures,
        "passed": not failures,
        "failures": failures,
        "inventory": {
            "expected_run_count": len(expected),
            "observed_run_count": len(records_with_paths),
            "missing_identities": sorted(expected - observed_set),
            "unexpected_identities": sorted(observed_set - expected),
            "duplicate_identities": duplicate,
        },
        "bounds_rule": protocol["baseline_envelope"]["bounds_rule"],
        "record_hashes": record_hashes,
        "profiles": profile_envelopes,
        "candidate_outcomes_used": False,
        "effect_claim_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.expanduser().resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    report = build_envelope(load_records(args.records_root), protocol, protocol_path)
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("output must remain inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "failures": report["failures"],
        "output": str(output),
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
