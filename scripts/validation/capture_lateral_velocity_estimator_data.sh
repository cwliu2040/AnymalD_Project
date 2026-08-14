#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${project_root}/source/anymal_locomotion:${PYTHONPATH:-}"
export TERM=xterm
isaaclab_root="${ISAACLAB_ROOT:-/home/ros/IsaacLab}"
checkpoint="${project_root}/checkpoints/anymal_d_locomotion_v1/recovery_v0.4.0/model_1450.pt"

for profile in left right; do
  if [[ "${profile}" == "left" ]]; then
    vy=1.5
    seed=46
  else
    vy=-1.5
    seed=47
  fi
  "${isaaclab_root}/isaaclab.sh" -p "${project_root}/scripts/rsl_rl/play.py" \
    --task Isaac-Velocity-Flat-Anymal-D-Locomotion-v0 \
    --checkpoint "${checkpoint}" \
    --num_envs 256 --seed "${seed}" --evaluation_steps 600 \
    --velocity_estimator_fixed_command 0.0 "${vy}" 0.0 \
    --velocity_estimator_dataset_output \
      "${project_root}/logs/velocity_estimator/captures/clean_fixed_lateral_${profile}_seed${seed}_256env_600step.npz" \
    --headless
done
