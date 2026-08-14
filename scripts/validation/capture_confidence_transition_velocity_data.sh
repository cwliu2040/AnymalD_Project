#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${project_root}/source/anymal_locomotion:${PYTHONPATH:-}"
export TERM=xterm
isaaclab_root="${ISAACLAB_ROOT:-/home/ros/IsaacLab}"
checkpoint="${project_root}/checkpoints/anymal_d_locomotion_v1/recovery_v0.4.0/model_1450.pt"

for profile in left right; do
  gate="${project_root}/configs/slam_confidence_behavior_holdout_lateral_${profile}.yaml"
  "${isaaclab_root}/isaaclab.sh" -p "${project_root}/scripts/rsl_rl/play.py" \
    --task Isaac-Velocity-Flat-Anymal-D-Locomotion-v0 \
    --checkpoint "${checkpoint}" \
    --confidence_supervisor_baseline \
    --behavior_gate_config "${gate}" \
    --velocity_estimator_dataset_output \
      "${project_root}/logs/velocity_estimator/captures/clean_base_confidence_transition_lateral_${profile}.npz" \
    --headless
done
