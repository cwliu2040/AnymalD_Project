#!/usr/bin/env python3
"""Build a compact state-transplant manifest from formal diagnostic traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anymal_locomotion.state_transplant import (
    build_state_transplant_manifest_from_files,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--locomotion-trace", type=Path, required=True)
parser.add_argument("--policy-trace", type=Path, required=True)
parser.add_argument("--source-time-s", type=float, default=25.0)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

output = args.output.expanduser().resolve()
if not output.is_relative_to(PROJECT_ROOT):
    parser.error(f"output must remain inside the project: {output}")
manifest = build_state_transplant_manifest_from_files(
    args.locomotion_trace.expanduser().resolve(),
    args.policy_trace.expanduser().resolve(),
    PROJECT_ROOT / "configs" / "policy_contract.yaml",
    source_time_s=args.source_time_s,
)
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
temporary.replace(output)
print(f"PASS state_transplant_manifest={output}")
