#!/usr/bin/env python3
"""Upgrade pilot captures after separating pipeline age from odom outage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value: str, *, exists: bool) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path escapes project: {path}")
    if exists and not path.is_file():
        raise ValueError(f"missing input: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = _project_path(args.input, exists=True)
    output = _project_path(args.output, exists=False)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload["rows"]
    last_source = None
    last_advance_eval = None
    max_gap_ns = 0
    for row in rows:
        source_stamp = row.get("source_stamp_ns")
        if source_stamp is None:
            continue
        if source_stamp != last_source:
            last_source = source_stamp
            last_advance_eval = int(row["evaluation_stamp_ns"])
        assert last_advance_eval is not None
        max_gap_ns = max(
            max_gap_ns,
            int(row["evaluation_stamp_ns"]) - last_advance_eval,
        )
    if max_gap_ns > 300_000_000:
        raise ValueError(
            "capture contains a real source-advance outage; replay it with "
            "the current label core instead of applying this safe migration"
        )
    for row in rows:
        if row.get("usable_next_0_5s") is None:
            continue
        reasons = [
            reason for reason in row.get("failure_reasons", [])
            if reason != "odometry_outage"
        ]
        row["failure_reasons"] = reasons
        row["usable_next_0_5s"] = not reasons
    payload["schema_version"] = 2
    payload["odometry_outage_semantics"] = "source_advance_gap"
    payload["migration_provenance"] = {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "validated_max_source_advance_gap_s": max_gap_ns * 1.0e-9,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
