#!/usr/bin/env python3
"""Probe the one-environment Play scene for ROS 2 Bridge integration paths."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
from pxr import UsdPhysics

import anymal_locomotion.tasks  # noqa: F401
from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import PLAY_TASK_ID
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main() -> None:
    env_cfg = load_cfg_from_registry(PLAY_TASK_ID, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = 42
    env = gym.make(PLAY_TASK_ID, cfg=env_cfg)
    try:
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        stage = omni.usd.get_context().get_stage()
        articulation_roots = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        rigid_bodies = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and str(prim.GetPath()).startswith(base_env.scene.env_prim_paths[0])
        ]

        print(f"PLAY_TASK_ID={PLAY_TASK_ID}", flush=True)
        print(f"ENV_PRIM_PATH={base_env.scene.env_prim_paths[0]}", flush=True)
        print(f"ROBOT_CFG_PRIM_EXPR={robot.cfg.prim_path}", flush=True)
        print(f"ARTICULATION_ROOTS={articulation_roots}", flush=True)
        print(f"ROBOT_BODY_NAMES={robot.body_names}", flush=True)
        print(f"ROBOT_JOINT_NAMES={robot.joint_names}", flush=True)
        print(f"SCENE_SENSORS={list(base_env.scene.sensors)}", flush=True)
        print(f"RIGID_BODY_PRIMS={rigid_bodies}", flush=True)
        print(f"ROOT_POSITION_SHAPE={tuple(robot.data.root_pos_w.shape)}", flush=True)
        print(f"ROOT_ORIENTATION_SHAPE={tuple(robot.data.root_quat_w.shape)}", flush=True)
        print(f"ROOT_LINEAR_VELOCITY_SHAPE={tuple(robot.data.root_lin_vel_b.shape)}", flush=True)
        print(f"ROOT_ANGULAR_VELOCITY_SHAPE={tuple(robot.data.root_ang_vel_b.shape)}", flush=True)
        print(f"PROJECTED_GRAVITY_SHAPE={tuple(robot.data.projected_gravity_b.shape)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        simulation_app.close(skip_cleanup=True)
        raise
    else:
        simulation_app.close()
