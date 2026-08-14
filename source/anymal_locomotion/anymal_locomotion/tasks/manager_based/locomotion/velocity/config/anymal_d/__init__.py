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
SLAM_CONFIDENCE_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-v0"
)
SLAM_CONFIDENCE_RESIDUAL_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-Residual-v0"
)
SLAM_CONFIDENCE_TEACHER_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-Teacher-v0"
)
SLAM_CONFIDENCE_SAFE_COMMAND_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-SafeCommand-v0"
)
SLAM_CONFIDENCE_BOUNDED_SAFE_COMMAND_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-BoundedSafeCommand-v0"
)
SLAM_CONFIDENCE_GAIT_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-Gait-v0"
)
SLAM_CONFIDENCE_GAIT_MODE_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-GaitMode-v0"
)
SLAM_CONFIDENCE_STRUCTURED_GAIT_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-StructuredGait-v0"
)
SLAM_CONFIDENCE_INTENT_GAIT_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-IntentGait-v0"
)
SLAM_CONFIDENCE_INTENT_GAIT_GAIN20_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-IntentGaitGain20-v0"
)
SLAM_CONFIDENCE_AUX_INTENT_GAIT_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-AuxIntentGait-v0"
)
SLAM_CONFIDENCE_ESTIMATOR_ROBUST_GAIT_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-"
    "SlamConfidence-EstimatorRobustGait-v0"
)
SLAM_CONFIDENCE_RECOVERY_SAFE_GAIT_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-"
    "SlamConfidence-RecoverySafeGait-v0"
)
SLAM_CONFIDENCE_CONSTRAINED_GAIT_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-"
    "SlamConfidence-ConstrainedGait-v0"
)
SLAM_CONFIDENCE_AMPLIFIED_SMOOTHING_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-"
    "SlamConfidence-AmplifiedSmoothing-v0"
)
SLAM_CONFIDENCE_DEGRADED_SLIP_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-"
    "SlamConfidence-DegradedSlip-v0"
)
SLAM_CONFIDENCE_PHASE_SEPARATED_GAIT_TASK_ID = (
    "Isaac-Velocity-Flat-Anymal-D-Locomotion-"
    "SlamConfidence-PhaseSeparatedGait-v0"
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

gym.register(
    id=SLAM_CONFIDENCE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidencePPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_RESIDUAL_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceResidualPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_TEACHER_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_SAFE_COMMAND_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceSafeCommandPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_BOUNDED_SAFE_COMMAND_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceBoundedSafeCommandPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_GAIT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceGaitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceGaitPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_GAIT_MODE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceGaitModeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceGaitModePPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_STRUCTURED_GAIT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceGaitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceStructuredGaitPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_INTENT_GAIT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceGaitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceIntentGaitPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_INTENT_GAIT_GAIN20_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceGaitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceIntentGaitGain20PPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_AUX_INTENT_GAIT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceGaitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceAuxIntentGaitPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_ESTIMATOR_ROBUST_GAIT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceGaitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceEstimatorRobustGaitPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_RECOVERY_SAFE_GAIT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceRecoveryGaitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceRecoverySafeGaitPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_CONSTRAINED_GAIT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceRecoveryGaitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceConstrainedGaitPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_AMPLIFIED_SMOOTHING_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceRecoveryGaitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceAmplifiedSmoothingPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_DEGRADED_SLIP_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceDegradedSlipEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidenceDegradedSlipPPORunnerCfg"
        ),
    },
)

gym.register(
    id=SLAM_CONFIDENCE_PHASE_SEPARATED_GAIT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "AnymalDLocomotionSlamConfidenceRecoveryGaitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "AnymalDLocomotionSlamConfidencePhaseSeparatedGaitPPORunnerCfg"
        ),
    },
)
