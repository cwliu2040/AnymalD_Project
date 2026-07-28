#!/usr/bin/env python3
"""Apply the frozen Factory stability gate to one diagnostic trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from anymal_locomotion.stability_diagnostics import (
    DiagnosticThresholds,
    build_diagnostic_report,
    evaluate_stability_gate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stability_diagnostics.yaml"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--driver", type=Path, required=True)
parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def _project_file(path: Path, *, must_exist: bool) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        parser.error(f"path must remain inside {PROJECT_ROOT}: {resolved}")
    if must_exist and not resolved.is_file():
        parser.error(f"file does not exist: {resolved}")
    return resolved


def main() -> None:
    trace_path = _project_file(args.trace, must_exist=True)
    driver_path = _project_file(args.driver, must_exist=True)
    config_path = _project_file(args.config, must_exist=True)
    output_path = _project_file(args.output, must_exist=False)

    raw_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    driver = json.loads(driver_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    thresholds = DiagnosticThresholds.from_mapping(
        config["classification"]
    )
    refreshed = build_diagnostic_report(
        samples=raw_trace["samples"],
        metadata={
            **raw_trace["metadata"],
            "stability_diagnostics_config": str(config_path),
        },
        thresholds=thresholds,
    )
    target = tuple(float(value) for value in driver["target"])
    if len(target) != 3:
        raise ValueError(f"driver target must contain three values: {target}")
    gate = evaluate_stability_gate(
        refreshed["summary"],
        target=target,
        config=config,
        driver_passed=bool(driver.get("passed")),
    )
    result = {
        "schema_version": 1,
        "trace_path": str(trace_path),
        "driver_path": str(driver_path),
        "config_path": str(config_path),
        "gate": gate,
        "summary": refreshed["summary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"Stability gate written to: {output_path}")
    if not gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
