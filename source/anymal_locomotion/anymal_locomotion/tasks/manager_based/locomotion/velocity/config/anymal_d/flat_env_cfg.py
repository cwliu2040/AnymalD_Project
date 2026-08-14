"""Project-owned ANYmal-D Flat v1 environment configuration."""

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CurriculumCfg,
    EventCfg,
    RewardsCfg,
)

from anymal_locomotion.policy_contract import CANONICAL_JOINT_ORDER

from . import mdp


@configclass
class AnymalDLocomotionEventCfg(EventCfg):
    """Official events plus a fail-fast joint-contract check."""

    validate_joint_contract = EventTerm(
        func=mdp.validate_anymal_d_joint_contract,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class AnymalDLocomotionFlatEnvCfg(AnymalDFlatEnvCfg):
    """Official ANYmal-D Flat baseline with documented deployment-facing deviations."""

    events: AnymalDLocomotionEventCfg = AnymalDLocomotionEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Intentional deviation: direct [vx, vy, wz] semantics for future geometry_msgs/Twist.
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_heading_envs = 0.0
        # High-Speed v0.2.0: match the official Spot command envelope.
        command.ranges.lin_vel_x = (-2.0, 3.0)
        command.ranges.lin_vel_y = (-1.5, 1.5)
        command.ranges.ang_vel_z = (-2.0, 2.0)
        command.ranges.heading = None

        # Deterministic actions and proprioception, independent of USD/runtime array ordering.
        joint_names = list(CANONICAL_JOINT_ORDER)
        self.actions.joint_pos.joint_names = joint_names
        self.actions.joint_pos.preserve_order = True
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=joint_names, preserve_order=True
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=joint_names, preserve_order=True
        )


@configclass
class AnymalDLocomotionFlatEnvCfg_PLAY(AnymalDLocomotionFlatEnvCfg):
    """Small deterministic scene for future policy playback."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AnymalDLocomotionRobustRewardsCfg(RewardsCfg):
    """Official rewards plus explicit stance-foot sliding control."""

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*FOOT",
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*FOOT",
            ),
        },
    )


@configclass
class AnymalDLocomotionRobustEnvCfg(AnymalDLocomotionFlatEnvCfg):
    """Fine-tuning task biased toward deployment-critical command edges."""

    rewards: AnymalDLocomotionRobustRewardsCfg = (
        AnymalDLocomotionRobustRewardsCfg()
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity = mdp.EdgeBiasedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            edge_probability=0.6,
            ranges=mdp.EdgeBiasedVelocityCommandCfg.Ranges(
                lin_vel_x=(-2.0, 3.0),
                lin_vel_y=(-1.5, 1.5),
                ang_vel_z=(-2.0, 2.0),
                heading=None,
            ),
        )


@configclass
class AnymalDLocomotionRecoveryV05RewardsCfg(
    AnymalDLocomotionRobustRewardsCfg
):
    """Base rewards plus penalties scoped to the observed failure envelope."""

    high_combined_flat_orientation_l2 = RewTerm(
        func=mdp.high_combined_flat_orientation_l2,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "min_forward_speed": 2.0,
            "min_yaw_speed": 1.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    high_combined_feet_slide = RewTerm(
        func=mdp.high_combined_feet_slide,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "min_forward_speed": 2.0,
            "min_yaw_speed": 1.5,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*FOOT",
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*FOOT",
            ),
        },
    )
    refinery_flat_orientation_l2 = RewTerm(
        func=mdp.refinery_flat_orientation_l2,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "target_forward_speed": 1.898749,
            "target_yaw_speed": 0.817349,
            "command_tolerance": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    refinery_feet_slide = RewTerm(
        func=mdp.refinery_feet_slide,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "target_forward_speed": 1.898749,
            "target_yaw_speed": 0.817349,
            "command_tolerance": 0.1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*FOOT",
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*FOOT",
            ),
        },
    )
    low_yaw_track_ang_vel_z_exp = RewTerm(
        func=mdp.low_yaw_track_ang_vel_z_exp,
        weight=4.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "target_yaw_speed": 0.5,
            "yaw_tolerance": 0.1,
            "max_planar_speed": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    low_curve_track_lin_vel_xy_exp = RewTerm(
        func=mdp.low_curve_track_lin_vel_xy_exp,
        weight=4.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "target_forward_speed": 0.5,
            "target_yaw_speed": 0.5,
            "command_tolerance": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    high_curve_track_lin_vel_xy_exp = RewTerm(
        func=mdp.high_curve_track_lin_vel_xy_exp,
        weight=16.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "target_forward_speed": 3.0,
            "target_yaw_speed": 0.5,
            "command_tolerance": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class AnymalDLocomotionRecoveryEnvCfg(AnymalDLocomotionFlatEnvCfg):
    """Fine-tuning task with frequent moving-to-standing transitions."""

    def __post_init__(self) -> None:
        super().__post_init__()
        command = self.commands.base_velocity
        command.resampling_time_range = (5.0, 5.0)
        command.rel_standing_envs = 0.2


@configclass
class AnymalDLocomotionRecoveryV05EnvCfg(
    AnymalDLocomotionRobustEnvCfg
):
    """Long-horizon high-combined locomotion and stop-recovery fine-tuning."""

    rewards: AnymalDLocomotionRecoveryV05RewardsCfg = (
        AnymalDLocomotionRecoveryV05RewardsCfg()
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 40.0
        # The full RTX/LIO stack exposed sensitivity to small state deviations
        # while executing the warehouse high-combined sequence.  Train through
        # those deviations without changing the deployment friction contract.
        self.events.push_robot.interval_range_s = (5.0, 10.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.75, 0.75),
            "y": (-0.75, 0.75),
            "yaw": (-0.5, 0.5),
        }
        # Preserve baseline turning behavior by applying the additional
        # anti-tilt and anti-slide costs only inside the high-combined
        # warehouse failure envelope.
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.commands.base_velocity = mdp.RecoveryV05VelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(6.0, 12.0),
            rel_standing_envs=0.15,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            high_combined_probability=0.80,
            high_combined_stop_probability=0.25,
            high_combined_straight_probability=0.50,
            warehouse_sequence_probability=0.35,
            refinery_replay_probability=0.15,
            turning_regression_probability=0.40,
            low_yaw_profile_probability=0.25,
            low_curve_profile_probability=0.25,
            high_curve_profile_probability=0.25,
            ranges=mdp.RecoveryV05VelocityCommandCfg.Ranges(
                lin_vel_x=(-2.0, 3.0),
                lin_vel_y=(-1.5, 1.5),
                ang_vel_z=(-2.0, 2.0),
                heading=None,
            ),
        )


@configclass
class AnymalDLocomotionSlamConfidenceRewardsCfg(RewardsCfg):
    """Baseline rewards with confidence-scaled velocity targets."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.confidence_track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "cycle_s": 10.0},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.confidence_track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5, "cycle_s": 10.0},
    )
    invalid_planar_speed_l2 = RewTerm(
        func=mdp.confidence_invalid_planar_speed_l2,
        weight=-2.0,
        params={"cycle_s": 10.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    invalid_yaw_rate_l2 = RewTerm(
        func=mdp.confidence_invalid_yaw_rate_l2,
        weight=-0.5,
        params={"cycle_s": 10.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    invalid_action_l2 = RewTerm(
        func=mdp.confidence_invalid_action_l2,
        weight=-0.1,
        params={"cycle_s": 10.0},
    )


@configclass
class AnymalDLocomotionSlamConfidenceCurriculumCfg(CurriculumCfg):
    """Stage in stronger invalid-state stopping costs after 25 PPO iterations."""

    terrain_levels = None
    invalid_planar_stop = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={
            "term_name": "invalid_planar_speed_l2",
            "weight": -4.0,
            "num_steps": 600,
        },
    )
    invalid_yaw_stop = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={
            "term_name": "invalid_yaw_rate_l2",
            "weight": -1.0,
            "num_steps": 600,
        },
    )
    invalid_action = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={
            "term_name": "invalid_action_l2",
            "weight": -0.5,
            "num_steps": 600,
        },
    )


@configclass
class AnymalDLocomotionSlamConfidenceEnvCfg(AnymalDLocomotionFlatEnvCfg):
    """51-D PPO task with a deployable three-value confidence contract."""

    rewards: AnymalDLocomotionSlamConfidenceRewardsCfg = (
        AnymalDLocomotionSlamConfidenceRewardsCfg()
    )
    curriculum: AnymalDLocomotionSlamConfidenceCurriculumCfg = (
        AnymalDLocomotionSlamConfidenceCurriculumCfg()
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Configclass preserves inherited observation order; assigning the new
        # term here appends it after previous_action at offsets 48..50.
        self.observations.policy.slam_confidence = ObsTerm(
            func=mdp.simulated_slam_confidence,
            params={"cycle_s": 10.0, "phase_offset_mode": "distributed"},
            clip=(0.0, 1.0),
        )
        # The upstream term uses the unscaled command magnitude and conflicts
        # with a deliberate stop during low confidence.
        self.rewards.feet_air_time.weight = 0.0


@configclass
class AnymalDLocomotionSlamConfidenceGaitRewardsCfg(
    AnymalDLocomotionSlamConfidenceRewardsCfg
):
    """Explicit low-confidence gait-quality objectives beyond command scaling."""

    confidence_gait_action_rate_l2 = RewTerm(
        func=mdp.confidence_gait_action_rate_l2,
        weight=-0.02,
        params={"cycle_s": 10.0},
    )
    confidence_gait_lin_vel_z_l2 = RewTerm(
        func=mdp.confidence_gait_lin_vel_z_l2,
        weight=-2.0,
        params={"cycle_s": 10.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    confidence_gait_ang_vel_xy_l2 = RewTerm(
        func=mdp.confidence_gait_ang_vel_xy_l2,
        weight=-0.2,
        params={"cycle_s": 10.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    confidence_gait_flat_orientation_l2 = RewTerm(
        func=mdp.confidence_gait_flat_orientation_l2,
        weight=-2.0,
        params={"cycle_s": 10.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    confidence_gait_feet_slide = RewTerm(
        func=mdp.confidence_gait_feet_slide,
        weight=-0.05,
        params={
            "cycle_s": 10.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT"),
        },
    )


@configclass
class AnymalDLocomotionSlamConfidenceGaitCurriculumCfg(
    AnymalDLocomotionSlamConfidenceCurriculumCfg
):
    """Ramp the five gait objectives after the safe stop path is established."""

    gait_action_smoothness = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={"term_name": "confidence_gait_action_rate_l2", "weight": -0.05, "num_steps": 600},
    )
    gait_vertical_stability = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={"term_name": "confidence_gait_lin_vel_z_l2", "weight": -4.0, "num_steps": 600},
    )
    gait_roll_pitch_stability = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={"term_name": "confidence_gait_ang_vel_xy_l2", "weight": -0.5, "num_steps": 600},
    )
    gait_posture = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={"term_name": "confidence_gait_flat_orientation_l2", "weight": -5.0, "num_steps": 600},
    )
    gait_stance_slip = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={"term_name": "confidence_gait_feet_slide", "weight": -0.2, "num_steps": 600},
    )


@configclass
class AnymalDLocomotionSlamConfidenceGaitEnvCfg(
    AnymalDLocomotionSlamConfidenceEnvCfg
):
    """Safe-command plus learned gait adaptation, without synthetic pushes."""

    rewards: AnymalDLocomotionSlamConfidenceGaitRewardsCfg = (
        AnymalDLocomotionSlamConfidenceGaitRewardsCfg()
    )
    curriculum: AnymalDLocomotionSlamConfidenceGaitCurriculumCfg = (
        AnymalDLocomotionSlamConfidenceGaitCurriculumCfg()
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.push_robot = None
        self.events.base_external_force_torque = None


@configclass
class AnymalDLocomotionSlamConfidenceRecoveryGaitRewardsCfg(
    AnymalDLocomotionSlamConfidenceGaitRewardsCfg
):
    """C-v6 rewards that make estimator-sensitive recovery failures expensive."""

    termination_penalty = RewTerm(func=isaac_mdp.is_terminated, weight=-200.0)
    confidence_recovery_action_rate_l2 = RewTerm(
        func=mdp.confidence_recovery_action_rate_l2,
        weight=-0.1,
        params={"cycle_s": 10.0},
    )
    confidence_recovery_ang_vel_xy_l2 = RewTerm(
        func=mdp.confidence_recovery_ang_vel_xy_l2,
        weight=-2.0,
        params={"cycle_s": 10.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    confidence_recovery_flat_orientation_l2 = RewTerm(
        func=mdp.confidence_recovery_flat_orientation_l2,
        weight=-10.0,
        params={"cycle_s": 10.0, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class AnymalDLocomotionSlamConfidenceRecoveryGaitEnvCfg(
    AnymalDLocomotionSlamConfidenceGaitEnvCfg
):
    """Estimator-closed-loop gait task with an explicit recovery safety signal."""

    rewards: AnymalDLocomotionSlamConfidenceRecoveryGaitRewardsCfg = (
        AnymalDLocomotionSlamConfidenceRecoveryGaitRewardsCfg()
    )


@configclass
class AnymalDLocomotionSlamConfidenceDegradedSlipRewardsCfg(
    AnymalDLocomotionSlamConfidenceRecoveryGaitRewardsCfg
):
    """C-v9 adds the one gait-value term that C-v7 model48 missed."""

    confidence_degradation_feet_slide = RewTerm(
        func=mdp.confidence_degradation_feet_slide,
        weight=-1.0,
        params={
            "cycle_s": 10.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT"),
        },
    )


@configclass
class AnymalDLocomotionSlamConfidenceDegradedSlipEnvCfg(
    AnymalDLocomotionSlamConfidenceRecoveryGaitEnvCfg
):
    """Recovery-safe task with a targeted degradation slip objective."""

    rewards: AnymalDLocomotionSlamConfidenceDegradedSlipRewardsCfg = (
        AnymalDLocomotionSlamConfidenceDegradedSlipRewardsCfg()
    )


@configclass
class AnymalDLocomotionSlamConfidenceGaitModeEnvCfg(
    AnymalDLocomotionSlamConfidenceGaitEnvCfg
):
    """Stateful rate-limited command governor with an exact model1450 actor."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.gait_mode_velocity_command,
            params={
                "command_name": "base_velocity",
                "cycle_s": 10.0,
                "phase_offset_mode": "distributed",
            },
        )
