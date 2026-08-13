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


@configclass
class SlamConfidenceResidualActorCriticCfg(RslRlPpoActorCriticCfg):
    """Frozen 48-D backbone plus confidence-gated residual adapter."""

    class_name: str = (
        "anymal_locomotion.policies.slam_confidence_residual:"
        "SlamConfidenceResidualActorCritic"
    )
    residual_hidden_dims: list[int] = [64, 64]
    residual_action_limit: float = 2.0
    legacy_observation_dim: int = 48
    confidence_offset: int = 48
    command_offset: int = 9
    command_dimension: int = 3
    use_action_skip: bool = False


@configclass
class SlamConfidenceSafeCommandActorCriticCfg(RslRlPpoActorCriticCfg):
    """Frozen 48-D backbone plus an exact confidence-scaled command path."""

    class_name: str = (
        "anymal_locomotion.policies.slam_confidence_residual:"
        "SlamConfidenceSafeCommandActorCritic"
    )
    legacy_observation_dim: int = 48
    confidence_offset: int = 48
    command_offset: int = 9
    command_dimension: int = 3
    safe_command_gain_limit: float = 1.0


@configclass
class SlamConfidenceSafeGaitActorCriticCfg(SlamConfidenceSafeCommandActorCriticCfg):
    """Frozen model1450 plus explicit safe command and bounded gait residual."""

    class_name: str = (
        "anymal_locomotion.policies.slam_confidence_residual:"
        "SlamConfidenceSafeGaitActorCritic"
    )
    residual_hidden_dims: list[int] = [64, 64]
    residual_action_limit: float = 0.25
    safe_command_gain_limit: float = 1.0


@configclass
class SlamConfidenceTeacherPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Fresh critic PPO plus explicit residual stop teacher."""

    class_name: str = (
        "anymal_locomotion.algorithms.slam_confidence_teacher_ppo:"
        "SlamConfidenceTeacherPPO"
    )
    teacher_num_epochs: int = 5
    teacher_num_mini_batches: int = 4
    teacher_loss_coef: float = 1.0
    teacher_target_limit_fraction: float = 0.95
    teacher_target_mode: str = "nominal_action_zero"
    teacher_learning_rate: float | None = None
    teacher_gain_loss_coef: float = 0.0
    teacher_gain_target: float = 1.0


@configclass
class SlamConfidenceGaitPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO residual adaptation with a separate bounded safe-gain teacher."""

    class_name: str = (
        "anymal_locomotion.algorithms.slam_confidence_gait_ppo:"
        "SlamConfidenceGaitPPO"
    )
    safe_gain_target: float = 1.0
    safe_gain_learning_rate: float = 5.0e-3
    safe_gain_num_epochs: int = 5


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
class AnymalDLocomotionSlamConfidencePPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Second confidence-conditioned actor-only warm-start configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "anymal_d_locomotion_slam_confidence_v2"
        self.run_name = "second_ppo_actor_only"
        self.max_iterations = 200
        self.save_interval = 25
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.001
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mdp.compute_symmetric_states,
        )


@configclass
class AnymalDLocomotionSlamConfidenceResidualPPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Bounded residual-adapter experiment with a frozen formal backbone."""

    policy = SlamConfidenceResidualActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "anymal_d_locomotion_slam_confidence_residual_v1"
        self.run_name = "bounded_residual_actor_only"
        self.max_iterations = 200
        self.save_interval = 25
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.0
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mdp.compute_symmetric_states,
        )


@configclass
class AnymalDLocomotionSlamConfidenceTeacherPPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Explicit stop-teacher candidate with a frozen model1450 backbone."""

    policy = SlamConfidenceResidualActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        use_action_skip=True,
    )
    algorithm = SlamConfidenceTeacherPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        teacher_target_mode="confidence_scaled_formal_command",
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mdp.compute_symmetric_states,
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "anymal_d_locomotion_slam_confidence_teacher_v1"
        self.run_name = "explicit_stop_teacher"
        self.max_iterations = 100
        self.save_interval = 25


@configclass
class AnymalDLocomotionSlamConfidenceSafeCommandPPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Exact safe-command teacher with a frozen model1450 backbone."""

    policy = SlamConfidenceSafeCommandActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        safe_command_gain_limit=1.0,
    )
    algorithm = SlamConfidenceTeacherPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        teacher_target_mode="confidence_scaled_formal_command",
        teacher_learning_rate=5.0e-3,
        teacher_gain_loss_coef=1.0,
        teacher_gain_target=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mdp.compute_symmetric_states,
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "anymal_d_locomotion_slam_confidence_safe_command_v1"
        self.run_name = "exact_safe_command_teacher"
        self.max_iterations = 15
        self.save_interval = 5


@configclass
class AnymalDLocomotionSlamConfidenceBoundedSafeCommandPPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Stable 0.80-bounded safe-command blend after lateral holdout audit."""

    policy = SlamConfidenceSafeCommandActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        safe_command_gain_limit=0.8,
    )
    algorithm = SlamConfidenceTeacherPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        teacher_target_mode="confidence_scaled_formal_command",
        teacher_learning_rate=5.0e-3,
        teacher_gain_loss_coef=1.0,
        teacher_gain_target=0.8,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mdp.compute_symmetric_states,
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = (
            "anymal_d_locomotion_slam_confidence_bounded_safe_command_v1"
        )
        self.run_name = "bounded_safe_command_teacher"
        self.max_iterations = 15
        self.save_interval = 5


@configclass
class AnymalDLocomotionSlamConfidenceGaitPPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Third candidate: bounded safe stop plus reward-driven gait adaptation."""

    policy = SlamConfidenceSafeGaitActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        residual_hidden_dims=[64, 64],
        residual_action_limit=0.25,
        safe_command_gain_limit=1.0,
    )
    algorithm = SlamConfidenceGaitPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mdp.compute_symmetric_states,
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "anymal_d_locomotion_slam_confidence_gait_v1"
        self.run_name = "safe_command_gait_adaptation"
        self.max_iterations = 100
        self.save_interval = 10
