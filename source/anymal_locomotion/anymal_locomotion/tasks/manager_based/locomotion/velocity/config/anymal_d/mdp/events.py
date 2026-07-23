"""Startup validation events for the ANYmal-D policy contract."""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg

from anymal_locomotion.policy_contract import validate_runtime_joint_names


def validate_anymal_d_joint_contract(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Fail environment startup when actuated joints violate the canonical schema."""
    del env_ids  # Required by Isaac Lab's startup-event calling convention.
    print("[INFO][anymal_locomotion] Validating startup joint contract...", flush=True)
    robot = env.scene[asset_cfg.name]
    validate_runtime_joint_names(robot.joint_names)
    print("[INFO][anymal_locomotion] Startup joint contract validated.", flush=True)
