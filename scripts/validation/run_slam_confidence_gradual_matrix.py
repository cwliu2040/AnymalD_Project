#!/usr/bin/env python3
"""Run the registered native gradual-degradation capture matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bag_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.glob("*")):
        if source.is_file():
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("fastlio2", "liosam"), required=True)
    parser.add_argument(
        "--manifest",
        default="configs/slam_confidence_gradual_capture_manifest.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="logs/slam_confidence/gradual_v2",
    )
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--group", action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    output_root = (PROJECT_ROOT / args.output_root / args.backend).resolve()
    if not manifest_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("manifest escapes project")
    if not output_root.is_relative_to(PROJECT_ROOT):
        raise ValueError("output root escapes project")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest["deskew_mode"] != "native":
        raise ValueError("formal confidence matrix must use native deskew")
    if len(manifest["captures"]) < 20:
        raise ValueError("formal matrix requires at least 20 capture groups")
    profile = str(manifest["degradation"]["profile"])
    minimum_density = float(manifest["degradation"]["minimum_density"])
    launch = (
        "fastlio2_replay_benchmark.launch.py"
        if args.backend == "fastlio2"
        else "lio_replay_benchmark.launch.py"
    )
    rows = []
    environment = os.environ.copy()
    environment["ROS_DOMAIN_ID"] = str(args.domain_id)
    environment["ROS_LOCALHOST_ONLY"] = "1"
    environment["ROS_LOG_DIR"] = "/tmp/anymal_slam_confidence_ros_logs"
    selected = [
        capture
        for capture in manifest["captures"]
        if not args.group or capture["group"] in set(args.group)
    ]
    if args.group and len(selected) != len(set(args.group)):
        raise ValueError("one or more requested capture groups are not registered")
    for capture in selected:
        group = str(capture["group"])
        bag = (PROJECT_ROOT / capture["bag"]).resolve()
        if not bag.is_relative_to(PROJECT_ROOT) or not (bag / "metadata.yaml").is_file():
            raise ValueError(f"invalid bag for {group}: {bag}")
        destination = output_root / group
        command = [
            "ros2", "launch", "anymal_locomotion_ros2", launch,
            f"bag_path:={bag}",
            f"output_path:={destination / 'replay.json'}",
            "enable_confidence:=true",
            f"confidence_dataset_path:={destination / 'dataset.json'}",
            f"capture_group:={group}",
            "point_density:=1.0",
            f"point_density_profile:={profile}",
            f"point_density_min:={minimum_density}",
            f"ros_domain_id:={args.domain_id}",
        ]
        rows.append({
            "group": group,
            "bag": str(bag),
            "source_bag_fingerprint_sha256": _bag_fingerprint(bag),
            "command": command,
        })
        if not args.dry_run:
            destination.mkdir(parents=True, exist_ok=True)
            for stale in (destination / "replay.json", destination / "dataset.json"):
                stale.unlink(missing_ok=True)
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"capture failed for {group}: {completed.returncode}")
            if not (destination / "replay.json").is_file() or not (
                destination / "dataset.json"
            ).is_file():
                raise RuntimeError(f"capture outputs incomplete for {group}")
    output_root.mkdir(parents=True, exist_ok=True)
    fingerprints = [row["source_bag_fingerprint_sha256"] for row in rows]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("capture matrix contains duplicate source bag content")
    (output_root / "matrix_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend_id": args.backend,
                "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
                "deskew_mode": "native",
                "capture_count": len(rows),
                "manifest_capture_count": len(manifest["captures"]),
                "captures": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
