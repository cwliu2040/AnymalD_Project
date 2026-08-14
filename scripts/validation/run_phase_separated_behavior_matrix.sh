#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
checkpoint="${1:?checkpoint path required}"
estimator_metadata="${2:?estimator metadata path required}"
output_dir="${3:?output directory required}"
export PYTHONPATH="${project_root}/source/anymal_locomotion:${PYTHONPATH:-}"
export TERM=xterm
isaaclab_root="${ISAACLAB_ROOT:-/home/ros/IsaacLab}"

profiles=(
  "forward:slam_confidence_behavior_gate.yaml"
  "lateral_left:slam_confidence_behavior_holdout_lateral_left.yaml"
  "lateral_right:slam_confidence_behavior_holdout_lateral_right.yaml"
  "reverse:slam_confidence_behavior_holdout_reverse.yaml"
  "combined:slam_confidence_behavior_holdout_combined.yaml"
)

mkdir -p "${output_dir}"
for item in "${profiles[@]}"; do
  profile="${item%%:*}"
  gate="${item#*:}"
  output="${output_dir}/behavior_gate_${profile}.json"
  if [[ -f "${output}" ]] && python3 -c 'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["behavior_gate"]["passed"])' "${output}"; then
    echo "SKIP passed ${profile}"
    continue
  fi
  "${isaaclab_root}/isaaclab.sh" -p "${project_root}/scripts/rsl_rl/play.py" \
    --task Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-PhaseSeparatedGait-v0 \
    --checkpoint "${checkpoint}" \
    --behavior_gate_config "${project_root}/configs/${gate}" \
    --velocity_estimator_metadata "${estimator_metadata}" \
    --evaluation_output "${output}" \
    --headless
done
