"""Register project-owned ANYmal-D Flat environments."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Velocity-Flat-Anymal-D-Locomotion-v0"
PLAY_TASK_ID = "Isaac-Velocity-Flat-Anymal-D-Locomotion-Play-v0"
ROBUST_TASK_ID = "Isaac-Velocity-Flat-Anymal-D-Locomotion-Robust-v0"
RECOVERY_TASK_ID = "Isaac-Velocity-Flat-Anymal-D-Locomotion-Recovery-v0"
RECOVERY_V05_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-Recovery-v0.5"
)
JOINT_TRAINING_J1_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-JointTraining-J1-v0"
)
JOINT_TRAINING_J2_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-JointTraining-J2-v0"
)

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
    id=RECOVERY_V05_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionRecoveryV05EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionRecoveryV05PPORunnerCfg"
        ),
    },
)

gym.register(
    id=JOINT_TRAINING_J1_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionJointTrainingJ1EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionJointTrainingJ1RunnerCfg"
        ),
    },
)

gym.register(
    id=JOINT_TRAINING_J2_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionJointTrainingJ2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionJointTrainingJ2RunnerCfg"
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
