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
class AnchoredFullPolicyActorCriticCfg(RslRlPpoActorCriticCfg):
    """Trainable full policy with a frozen model1450 action reference."""

    class_name: str = (
        "anymal_locomotion.policies.joint_training:"
        "AnchoredFullPolicyActorCritic"
    )
    legacy_observation_dim: int = 48
    full_observation_dim: int = 1068


@configclass
class ActionConstrainedFullPolicyActorCriticCfg(AnchoredFullPolicyActorCriticCfg):
    """Complete actor with a hard deterministic deviation bound from model1450."""

    class_name: str = (
        "anymal_locomotion.policies.joint_training:"
        "ActionConstrainedFullPolicyActorCritic"
    )
    action_deviation_limit: float = 0.05


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
class SlamConfidenceGaitModeActorCriticCfg(RslRlPpoActorCriticCfg):
    """Frozen model1450 consuming a statefully governed 51-D observation."""

    class_name: str = (
        "anymal_locomotion.policies.slam_confidence_residual:"
        "SlamConfidenceGaitModeActorCritic"
    )
    legacy_observation_dim: int = 48


@configclass
class SlamConfidenceStructuredGaitActorCriticCfg(RslRlPpoActorCriticCfg):
    """Frozen model1450 plus a four-coordinate confidence gait head."""

    class_name: str = (
        "anymal_locomotion.policies.slam_confidence_residual:"
        "SlamConfidenceStructuredGaitActorCritic"
    )
    gait_hidden_dims: list[int] = [64, 64]
    gait_parameter_limits: list[float] = [0.6, 0.3, 0.2, 0.5]
    legacy_observation_dim: int = 48
    confidence_offset: int = 48
    previous_action_offset: int = 36


@configclass
class SlamConfidenceIntentGaitActorCriticCfg(RslRlPpoActorCriticCfg):
    """Frozen model1450 plus locomotion intent and four gait coordinates."""

    class_name: str = (
        "anymal_locomotion.policies.slam_confidence_residual:"
        "SlamConfidenceIntentGaitActorCritic"
    )
    gait_hidden_dims: list[int] = [64, 64]
    gait_parameter_limits: list[float] = [1.0, 0.6, 0.3, 0.2, 0.5]
    legacy_observation_dim: int = 48
    confidence_offset: int = 48
    previous_action_offset: int = 36
    command_offset: int = 9
    command_dimension: int = 3
    intent_logit_gain: float = 1.0


@configclass
class SlamConfidenceAuxIntentGaitActorCriticCfg(RslRlPpoActorCriticCfg):
    """Separate 1-D auxiliary intent and 4-D PPO gait heads."""

    class_name: str = (
        "anymal_locomotion.policies.slam_confidence_residual:"
        "SlamConfidenceAuxIntentGaitActorCritic"
    )
    gait_hidden_dims: list[int] = [64, 64]
    gait_parameter_limits: list[float] = [0.6, 0.3, 0.2, 0.5]
    legacy_observation_dim: int = 48
    confidence_offset: int = 48
    previous_action_offset: int = 36
    command_offset: int = 9
    command_dimension: int = 3
    nonnegative_stride_smoothing: bool = False
    smoothing_logit_gain: float = 1.0
    degraded_stride_min_scale: float = 1.0
    degraded_stride_confidence_low: float = 0.2
    degraded_stride_confidence_high: float = 1.0
    degraded_stride_age_ratio_max: float = 0.45
    degraded_stride_envelope_power: float = 1.0
    suppress_gait_when_tracking_invalid: bool = False
    gait_delta_safe_scale_power: float = 0.0
    intent_blend_max: float = 1.0


@configclass
class SlamConfidenceIntentPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO plus a separate short-credit intent-head auxiliary update."""

    class_name: str = (
        "anymal_locomotion.algorithms.slam_confidence_intent_ppo:"
        "SlamConfidenceIntentPPO"
    )
    intent_learning_rate: float = 5.0e-3
    intent_num_epochs: int = 5
    intent_num_mini_batches: int = 4
    intent_loss_coef: float = 1.0


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


@configclass
class AnymalDLocomotionCausalJointTrainingRunnerCfg(
    AnymalDLocomotionJointTrainingRunnerCfg
):
    """Action-constrained full-policy runner for causal PPO."""

    policy = ActionConstrainedFullPolicyActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        action_deviation_limit=0.05,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "anymal_d_locomotion_constrained_joint_training_v3"


@configclass
class AnymalDLocomotionCausalJointTrainingJ1RunnerCfg(
    AnymalDLocomotionCausalJointTrainingRunnerCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.run_name = "J1_constrained_neutral_localization"


@configclass
class AnymalDLocomotionCausalJointTrainingJ2RunnerCfg(
    AnymalDLocomotionCausalJointTrainingRunnerCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.run_name = "J2_constrained_localization_history"


@configclass
class AnymalDLocomotionConstrainedBarrierRunnerCfg(
    AnymalDLocomotionCausalJointTrainingRunnerCfg
):
    """v4 action-constrained runner with prospectively frozen barriers."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "anymal_d_locomotion_constrained_barrier_v4"


@configclass
class AnymalDLocomotionConstrainedBarrierJ1RunnerCfg(
    AnymalDLocomotionConstrainedBarrierRunnerCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.run_name = "J1_barrier_neutral_localization"


@configclass
class AnymalDLocomotionConstrainedBarrierJ2RunnerCfg(
    AnymalDLocomotionConstrainedBarrierRunnerCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.run_name = "J2_barrier_localization_history"


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


@configclass
class AnymalDLocomotionSlamConfidenceGaitModePPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Architecture-gate runner for the deterministic gait-mode governor."""

    policy = SlamConfidenceGaitModeActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
        self.experiment_name = "anymal_d_locomotion_slam_confidence_gait_mode_v1"
        self.run_name = "gait_mode_architecture_gate"
        self.max_iterations = 1
        self.save_interval = 1


@configclass
class AnymalDLocomotionSlamConfidenceStructuredGaitPPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """C candidate: PPO trains only four structured gait coordinates and critic."""

    policy = SlamConfidenceStructuredGaitActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        gait_hidden_dims=[64, 64],
        gait_parameter_limits=[0.6, 0.3, 0.2, 0.5],
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
        self.experiment_name = (
            "anymal_d_locomotion_slam_confidence_structured_gait_v1"
        )
        self.run_name = "four_coordinate_gait_head"
        self.max_iterations = 25
        self.save_interval = 5


@configclass
class AnymalDLocomotionSlamConfidenceIntentGaitPPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """C-v2 architecture gate for joint PPO locomotion intent and gait control."""

    policy = SlamConfidenceIntentGaitActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        gait_hidden_dims=[64, 64],
        gait_parameter_limits=[1.0, 0.6, 0.3, 0.2, 0.5],
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
        self.experiment_name = (
            "anymal_d_locomotion_slam_confidence_intent_gait_v1"
        )
        self.run_name = "five_coordinate_intent_gait_head"
        self.max_iterations = 25
        self.save_interval = 5


@configclass
class AnymalDLocomotionSlamConfidenceIntentGaitGain20PPORunnerCfg(
    AnymalDLocomotionSlamConfidenceIntentGaitPPORunnerCfg
):
    """C-v3 with higher PPO intent-coordinate control resolution."""

    policy = SlamConfidenceIntentGaitActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        gait_hidden_dims=[64, 64],
        gait_parameter_limits=[1.0, 0.6, 0.3, 0.2, 0.5],
        intent_logit_gain=20.0,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = (
            "anymal_d_locomotion_slam_confidence_intent_gait_gain20_v1"
        )
        self.run_name = "five_coordinate_intent_gait_gain20"


@configclass
class AnymalDLocomotionSlamConfidenceAuxIntentGaitPPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """C-v4 fixed pilot with separate intent auxiliary and PPO gait heads."""

    policy = SlamConfidenceAuxIntentGaitActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        gait_hidden_dims=[64, 64],
        gait_parameter_limits=[0.6, 0.3, 0.2, 0.5],
    )
    algorithm = SlamConfidenceIntentPpoAlgorithmCfg(
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
        intent_learning_rate=5.0e-3,
        intent_num_epochs=5,
        intent_num_mini_batches=4,
        intent_loss_coef=1.0,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = (
            "anymal_d_locomotion_slam_confidence_aux_intent_gait_v1"
        )
        self.run_name = "separate_aux_intent_ppo_gait"
        self.max_iterations = 25
        self.save_interval = 5


@configclass
class AnymalDLocomotionSlamConfidenceEstimatorRobustGaitPPORunnerCfg(
    AnymalDLocomotionSlamConfidenceAuxIntentGaitPPORunnerCfg
):
    """C-v5: train the PPO gait head through the deployable velocity estimator."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = (
            "anymal_d_locomotion_slam_confidence_estimator_robust_gait_v1"
        )
        self.run_name = "closed_loop_estimator_aux_intent_ppo_gait"
        self.max_iterations = 25
        self.save_interval = 5


@configclass
class AnymalDLocomotionSlamConfidenceRecoverySafeGaitPPORunnerCfg(
    AnymalDLocomotionSlamConfidenceEstimatorRobustGaitPPORunnerCfg
):
    """C-v6 fixed pilot with estimator closure and explicit fall/recovery costs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = (
            "anymal_d_locomotion_slam_confidence_recovery_safe_gait_v1"
        )
        self.run_name = "recovery_penalty_estimator_aux_intent_ppo_gait"


@configclass
class AnymalDLocomotionSlamConfidenceConstrainedGaitPPORunnerCfg(
    AnymalDLocomotionSlamConfidenceRecoverySafeGaitPPORunnerCfg
):
    """C-v7: PPO gait coordinates with physically valid smoothing signs."""

    policy = SlamConfidenceAuxIntentGaitActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        gait_hidden_dims=[64, 64],
        gait_parameter_limits=[0.6, 0.3, 0.2, 0.9],
        nonnegative_stride_smoothing=True,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = (
            "anymal_d_locomotion_slam_confidence_constrained_gait_v1"
        )
        self.run_name = "nonnegative_smoothing_recovery_safe_ppo_gait"


@configclass
class AnymalDLocomotionSlamConfidenceAmplifiedSmoothingPPORunnerCfg(
    AnymalDLocomotionSlamConfidenceRecoverySafeGaitPPORunnerCfg
):
    """C-v8: higher PPO control resolution for the safe smoothing coordinate."""

    policy = SlamConfidenceAuxIntentGaitActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        gait_hidden_dims=[64, 64],
        gait_parameter_limits=[0.6, 0.3, 0.2, 0.9],
        nonnegative_stride_smoothing=True,
        smoothing_logit_gain=10.0,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = (
            "anymal_d_locomotion_slam_confidence_amplified_smoothing_v1"
        )
        self.run_name = "gain10_smoothing_recovery_safe_ppo_gait"


@configclass
class AnymalDLocomotionSlamConfidenceDegradedSlipPPORunnerCfg(
    AnymalDLocomotionSlamConfidenceConstrainedGaitPPORunnerCfg
):
    """C-v9 continuation runner for the remaining degradation slip gate."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = (
            "anymal_d_locomotion_slam_confidence_constrained_gait_v1"
        )
        self.run_name = "degraded_slip_model48_continuation"
        self.max_iterations = 15


@configclass
class AnymalDLocomotionSlamConfidencePhaseSeparatedGaitPPORunnerCfg(
    AnymalDLocomotionSlamConfidenceConstrainedGaitPPORunnerCfg
):
    """C-v10: suppress degradation stride without weakening recovery stride."""

    policy = SlamConfidenceAuxIntentGaitActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        gait_hidden_dims=[64, 64],
        gait_parameter_limits=[0.6, 0.3, 0.2, 0.9],
        nonnegative_stride_smoothing=True,
        degraded_stride_min_scale=0.0,
        degraded_stride_confidence_low=0.2,
        degraded_stride_confidence_high=1.0,
        degraded_stride_age_ratio_max=0.45,
        degraded_stride_envelope_power=3.0,
        suppress_gait_when_tracking_invalid=True,
        gait_delta_safe_scale_power=10.0,
        intent_blend_max=0.79,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = (
            "anymal_d_locomotion_slam_confidence_phase_separated_gait_v1"
        )
        self.run_name = "degraded_stride_envelope_recovery_safe_ppo_gait"
