"""RSL-RL PPO baseline for project-owned ANYmal-D Flat locomotion."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg import (
    AnymalDFlatPPORunnerCfg,
)
from .. import mdp
class AnchoredFullPolicyActorCriticCfg(RslRlPpoActorCriticCfg):
    """Trainable full policy with a frozen model1450 action reference."""

    class_name: str = (
        "anymal_locomotion.policies.joint_training:"
        "AnchoredFullPolicyActorCritic"
    )
    legacy_observation_dim: int = 48
    full_observation_dim: int = 1068


@configclass
class BehaviorAnchoredPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO plus the same non-zero behavior anchor for J1 and J2."""

    class_name: str = (
        "anymal_locomotion.algorithms.behavior_anchored_ppo:"
        "BehaviorAnchoredPPO"
    )
    behavior_anchor_coef: float = 0.25
    behavior_anchor_num_epochs: int = 1
    behavior_anchor_num_mini_batches: int = 4


@configclass
class AnymalDLocomotionFlatPPORunnerCfg(AnymalDFlatPPORunnerCfg):
    """Official ANYmal-D Flat PPO parameters with a project-local experiment name."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "anymal_d_locomotion_v1"
        self.max_iterations = 1000


@configclass
class AnymalDLocomotionRobustPPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Conservative symmetry-augmented fine-tuning configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.max_iterations = 300
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 0.003
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mdp.compute_symmetric_states,
        )


@configclass
class AnymalDLocomotionRecoveryPPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Low-drift fine-tuning configuration for zero-command recovery."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.max_iterations = 200
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.001


@configclass
class AnymalDLocomotionRecoveryV05PPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Symmetric low-drift fine-tuning for high-speed turn recovery."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.max_iterations = 300
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 0.001
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mdp.compute_symmetric_states,
        )


@configclass
class AnymalDLocomotionJointTrainingRunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Shared, execution-closed J1/J2 full-policy fine-tuning contract."""

    policy = AnchoredFullPolicyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = BehaviorAnchoredPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "anymal_d_locomotion_joint_training_v1"
        self.max_iterations = 300
        self.save_interval = 25
        # Existing symmetry code only understands a single 48/51-D frame.
        # It must not silently permute the 1068-D temporal contract.
        self.algorithm.symmetry_cfg = None


@configclass
class AnymalDLocomotionJointTrainingJ1RunnerCfg(
    AnymalDLocomotionJointTrainingRunnerCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.run_name = "J1_generic_history_anchor"


@configclass
class AnymalDLocomotionJointTrainingJ2RunnerCfg(
    AnymalDLocomotionJointTrainingRunnerCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.run_name = "J2_localization_history_anchor"
