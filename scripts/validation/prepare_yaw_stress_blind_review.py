#!/usr/bin/env python3
# flake8: noqa: E402
"""Create separated blind/reveal manifests for yaw-stress visual review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PYTHON = (
    PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"
)
sys.path.insert(0, str(PACKAGE_PYTHON))

from anymal_locomotion_ros2.yaw_stress_core import (  # noqa: E402
    build_blind_review_assignments,
)


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _video_contract(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    duration_s = frame_count / fps if fps > 0.0 else 0.0
    sampled_stddev = []
    if opened and frame_count > 0:
        for frame_index in (0, frame_count // 2, frame_count - 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if success:
                sampled_stddev.append(float(frame.std()))
    capture.release()
    failures = []
    if not opened:
        failures.append("video cannot be opened")
    if width != 720 or height != 720:
        failures.append("video must be 720x720")
    if abs(fps - 10.0) > 0.01:
        failures.append("video must be 10 fps")
    if duration_s < 26.0:
        failures.append("video must cover the 27 s replay")
    if len(sampled_stddev) != 3 or min(sampled_stddev) < 2.0:
        failures.append("video sample frames are missing or visually empty")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": duration_s,
        "sampled_pixel_stddev": sampled_stddev,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--review-seed", type=int, default=20260804)
    parser.add_argument("--require-videos", action="store_true")
    args = parser.parse_args()
    root = args.experiment_root.expanduser().resolve()
    if not root.is_relative_to(PROJECT_ROOT):
        parser.error(f"experiment root must remain inside {PROJECT_ROOT}")
    reports = sorted(root.glob("replays/*/replay.json"))
    assignments = build_blind_review_assignments(
        (path.parent.name for path in reports),
        review_seed=args.review_seed,
    )
    blind_cases = []
    reveal_cases = []
    for assignment in assignments:
        case_name = str(assignment["case_name"])
        blind_id = str(assignment["blind_id"])
        expected_video = root / "visual_review" / "videos" / f"{blind_id}.mp4"
        video_contract = (
            _video_contract(expected_video)
            if expected_video.is_file()
            else {
                "status": "missing",
                "failures": ["expected video is missing"],
            }
        )
        blind_cases.append(
            {
                "review_index": assignment["review_index"],
                "blind_id": blind_id,
                "video": str(expected_video),
                "fixed_view_contract": "camera_init, identical camera pose and timeline",
                "annotation_fields": (
                    "crack_onset_s, severity_0_to_3, recovered, notes"
                ),
                "video_contract": video_contract,
            }
        )
        reveal_cases.append(
            {
                **assignment,
                "report": str(root / "replays" / case_name / "replay.json"),
            }
        )
    common = {
        "schema_version": 1,
        "review_seed": args.review_seed,
        "case_count": len(assignments),
    }
    _write(
        root / "visual_review" / "blind_manifest.json",
        {**common, "cases": blind_cases},
    )
    _write(
        root / "visual_review" / "reveal_manifest.json",
        {**common, "cases": reveal_cases},
    )
    print(json.dumps({"case_count": len(assignments)}))
    if args.require_videos and any(
        case["video_contract"]["status"] != "passed"
        for case in blind_cases
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
