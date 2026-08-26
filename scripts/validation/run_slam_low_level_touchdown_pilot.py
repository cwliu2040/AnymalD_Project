#!/usr/bin/env python3
"""Execute the hash-locked blocks598--601 touchdown causal pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/slam_low_level_touchdown_headroom_v1.yaml"
RELEASE = ROOT / "configs/slam_low_level_touchdown_pilot_release_v1.yaml"
SHAKEDOWN = ROOT / "scripts/validation/run_slam_low_level_touchdown_shakedown.py"
VALIDATOR = ROOT / "scripts/validation/validate_slam_low_level_touchdown_headroom_protocol.py"
ANALYZER = ROOT / "scripts/validation/analyze_slam_low_level_touchdown_headroom.py"
BUILDER = ROOT / "scripts/validation/build_slam_low_level_touchdown_run_record.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _module("touchdown_shakedown_runner", SHAKEDOWN)
V = _module("touchdown_protocol", VALIDATOR)
A = _module("touchdown_analysis", ANALYZER)
B = _module("touchdown_record_builder", BUILDER)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def validate_release(release: dict, protocol_path: Path, *, for_execution: bool) -> None:
    allowed_status = {"pilot_execution_authorized", "pilot_complete_execution_closed"}
    if release.get("status") not in allowed_status:
        raise ValueError("unsupported pilot release status")
    authorized = release["status"] == "pilot_execution_authorized"
    if for_execution and not authorized:
        raise ValueError("pilot execution is closed")
    if release["protocol"]["sha256"] != _sha(protocol_path):
        raise ValueError("protocol SHA-256 mismatch")
    for section in ("base_policy", "velocity_estimator", "phase_tracker", "kinematics",
                    "baseline_envelope", "stability_gate"):
        value = release[section]
        pairs = [("path", "sha256")]
        if section == "base_policy":
            pairs.append(("metadata_path", "metadata_sha256"))
        elif section == "velocity_estimator":
            pairs = [("metadata_path", "metadata_sha256")]
        for path_key, hash_key in pairs:
            path = (ROOT / value[path_key]).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or _sha(path) != value[hash_key]:
                raise ValueError(f"release artifact mismatch: {value[path_key]}")
    prerequisite = release["wiring_prerequisite"]
    for path_key, hash_key in (("decision_path", "decision_sha256"),
                               ("matrix_summary_path", "matrix_summary_sha256")):
        path = (ROOT / prerequisite[path_key]).resolve()
        if not path.is_file() or _sha(path) != prerequisite[hash_key]:
            raise ValueError("wiring prerequisite mismatch")
    decision = json.loads((ROOT / prerequisite["decision_path"]).read_text())
    if decision.get("decision", {}).get("status") != prerequisite["required_decision"]:
        raise ValueError("wiring prerequisite did not pass")
    boundaries = release["boundaries"]
    if boundaries.get("live_execution_authorized") is not authorized:
        raise ValueError("status and live authorization disagree")
    if boundaries.get("authorized_stages") != (["pilot"] if authorized else []):
        raise ValueError("only pilot may be authorized")
    if boundaries.get("authorized_blocks") != [598, 599, 600, 601]:
        raise ValueError("pilot block authorization mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/slam_low_level_touchdown_headroom_v1")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--domain-id", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reanalyze-existing", action="store_true")
    args = parser.parse_args()
    protocol_path, release_path = args.protocol.resolve(), args.release.resolve()
    protocol, release = _load(protocol_path), _load(release_path)
    validation = V.validate_protocol(protocol)
    if not validation["passed"]:
        raise ValueError(validation["failures"])
    if args.execute and args.reanalyze_existing:
        raise ValueError("execute and reanalyze-existing are mutually exclusive")
    validate_release(release, protocol_path, for_execution=args.execute)
    schedule = V.build_stage_schedule(protocol, "pilot")
    plan = {"stage": "pilot", "selected_count": len(schedule),
            "blocks": [598, 599, 600, 601],
            "profiles": protocol["stages"]["pilot"]["profiles"],
            "backends": protocol["stages"]["pilot"]["backends"],
            "execution_authorized": release["boundaries"]["live_execution_authorized"]}
    if not args.execute and not args.reanalyze_existing:
        print(json.dumps(plan, indent=2))
        return 0
    stage_root = args.output_root.resolve() / "pilot"
    if args.reanalyze_existing:
        if not (stage_root / "matrix_summary.json").is_file():
            raise ValueError("completed pilot evidence is missing")
        for row in schedule:
            run_dir = (stage_root / row["backend"] / row["profile"] /
                       f"block_{row['block_id']}" / row["arm"])
            record = B.build(
                run_dir, row["backend"], row["profile"], int(row["block_id"]),
                row["arm"], "pilot",
            )
            S._write(run_dir / "touchdown_headroom_run_record_v3.json", record)
        decision = A.analyze_records(
            A.load_records(stage_root, "touchdown_headroom_run_record_v3.json"),
            protocol, "pilot",
        )
        S._write(stage_root / "decision_v3.json", decision)
        original = json.loads((stage_root / "matrix_summary.json").read_text())
        S._write(stage_root / "matrix_summary_v3.json", {
            **original,
            "record_revision": "v3_pure_yaw_semantics_and_frozen_anti_collapse_envelope",
            "prior_decisions_preserved": ["decision.json", "decision_v2.json"],
            "decision": decision,
        })
        print(json.dumps({"reanalyzed_count": len(schedule),
                          "integrity": decision["integrity"],
                          "decision": decision["decision"]}, indent=2))
        return 0 if decision["integrity"]["passed"] else 1
    if stage_root.exists():
        raise ValueError("pilot output already exists")
    manifest = {"schema_version": 1, "kind": "touchdown_blocks598_601_causal_pilot",
                **plan, "schedule": schedule, "protocol_sha256": _sha(protocol_path),
                "release_sha256": _sha(release_path), "worktree_snapshot": S._snapshot()}
    S._write(stage_root / "run_manifest.json", manifest)
    results = []
    for row in schedule:
        result = S.execute_cell(row, release, stage_root, args.domain_id)
        results.append(result)
        if result["stop_required"]:
            break
    records = A.load_records(stage_root)
    decision = A.analyze_records(records, protocol, "pilot")
    S._write(stage_root / "decision.json", decision)
    complete = len(results) == len(schedule) and all(row["passed"] for row in results)
    S._write(stage_root / "matrix_summary.json", {
        **manifest, "executed_count": len(results), "collection_passed": complete,
        "stopped_early": len(results) < len(schedule), "decision": decision,
        "results": results,
    })
    print(json.dumps({"selected_count": len(schedule), "executed_count": len(results),
                      "collection_passed": complete, "decision": decision["decision"]}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
