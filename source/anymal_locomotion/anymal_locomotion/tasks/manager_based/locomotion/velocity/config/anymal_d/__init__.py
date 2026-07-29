"""Register project-owned ANYmal-D Flat environments."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Velocity-Flat-Anymal-D-Locomotion-v0"
PLAY_TASK_ID = "Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0"
ROBUST_TASK_ID = "Isaac-Velocity-Flat-Anymal-D-Locomotion-Robust-v0"
RECOVERY_TASK_ID = "Isaac-Velocity-Flat-Anymal-D-Locomotion-Recovery-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalDLocomotionFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDLocomotionFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id=ROBUST_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionRobustEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionRobustPPORunnerCfg"
        ),
    },
)

gym.register(
    id=RECOVERY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionRecoveryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionRecoveryPPORunnerCfg"
        ),
    },
)

gym.register(
    id=PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalDLocomotionFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDLocomotionFlatPPORunnerCfg"
        ),
    },
)
