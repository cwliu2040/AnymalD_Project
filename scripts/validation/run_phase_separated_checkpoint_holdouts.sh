#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="${1:-${project_root}/logs/rsl_rl/anymal_d_locomotion_slam_confidence_phase_separated_gait_v1/2026-08-13_19-41-59_phase_separated_invalid_safe_scale_continuation}"
shift || true
models=("${@:-50 60 70 72}")

export PYTHONPATH="${project_root}/source/anymal_locomotion:${PYTHONPATH:-}"
export TERM="xterm"

for model in ${models[*]}; do
  checkpoint="${run_dir}/model_${model}.pt"
  output="${run_dir}/diagnostics/behavior_gate_model${model}_lateral_right.json"
  mkdir -p "$(dirname "${output}")"
  "${ISAACLAB_ROOT:-/home/ros/IsaacLab}/isaaclab.sh" -p \
    "${project_root}/scripts/rsl_rl/play.py" \
    --task Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-PhaseSeparatedGait-v0 \
    --checkpoint "${checkpoint}" \
    --behavior_gate_config "${project_root}/configs/slam_confidence_behavior_holdout_lateral_right.yaml" \
    --velocity_estimator_metadata "${project_root}/exported/proprioceptive_velocity_estimator/v1/clean_candidate_08/velocity_estimator_metadata.json" \
    --evaluation_output "${output}" \
    --headless
done
