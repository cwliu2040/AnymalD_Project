"""Project-owned ANYmal-D Flat v1 environment configuration."""

import torch

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
)

from anymal_locomotion.policy_contract import CANONICAL_JOINT_ORDER

from . import mdp
class AnymalDLocomotionEventCfg(EventCfg):
    """Official events plus a fail-fast joint-contract check."""

    validate_joint_contract = EventTerm(
        func=mdp.validate_anymal_d_joint_contract,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
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


def _vector_uniform_noise(data: torch.Tensor, cfg) -> torch.Tensor:
    """Apply Hydra-serializable per-coordinate additive uniform noise."""
    if cfg.operation != "add":
        raise ValueError("joint-training vector noise supports additive operation only")
    n_min = torch.as_tensor(list(cfg.n_min), dtype=data.dtype, device=data.device)
    n_max = torch.as_tensor(list(cfg.n_max), dtype=data.dtype, device=data.device)
    return data + torch.rand_like(data) * (n_max - n_min) + n_min


@configclass
class _VectorUniformNoiseCfg(NoiseCfg):
    func = _vector_uniform_noise
    n_min: list[float] = []
    n_max: list[float] = []


def _joint_training_history_noise() -> _VectorUniformNoiseCfg:
    """Match the legacy per-component observation corruption ranges."""
    limits = (
        [0.1] * 3
        + [0.2] * 3
        + [0.05] * 3
        + [0.0] * 3
        + [0.01] * 12
        + [1.5] * 12
        + [0.0] * 12
        + [0.0] * 3
    )
    if len(limits) != 51:
        raise RuntimeError("joint-training history noise contract must remain 51-D")
    return _VectorUniformNoiseCfg(
        n_min=[-float(value) for value in limits],
        n_max=[float(value) for value in limits],
    )


@configclass
class AnymalDLocomotionJointTrainingObservationsCfg(ObservationsCfg):
    """Current legacy 48-D input followed by 20 causal 51-D frames."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        history_frame = ObsTerm(
            func=mdp.full_policy_history_frame,
            params={
                "command_name": "base_velocity",
                "localization_mode": "actual",
                "cycle_s": 10.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
            noise=_joint_training_history_noise(),
            history_length=20,
            flatten_history_dim=True,
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class AnymalDLocomotionJointTrainingRewardsCfg(
    AnymalDLocomotionRecoveryV05RewardsCfg
):
    """Original-command rewards plus body/LiDAR motion distortion objectives."""

    angular_acceleration_l2 = RewTerm(
        func=mdp.joint_training_angular_acceleration_l2,
        weight=0.0,
        params={"cycle_s": 10.0, "severity_gain": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    linear_jerk_l2 = RewTerm(
        func=mdp.joint_training_linear_jerk_l2,
        weight=0.0,
        params={"cycle_s": 10.0, "severity_gain": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    lidar_scan_translation_distortion_l2 = RewTerm(
        func=mdp.joint_training_lidar_scan_translation_distortion_l2,
        weight=0.0,
        params={
            "scan_time_s": 0.10,
            "cycle_s": 10.0,
            "severity_gain": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    lidar_scan_rotation_distortion_l2 = RewTerm(
        func=mdp.joint_training_lidar_scan_rotation_distortion_l2,
        weight=0.0,
        params={
            "scan_time_s": 0.10,
            "cycle_s": 10.0,
            "severity_gain": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class AnymalDLocomotionJointTrainingCurriculumCfg(CurriculumCfg):
    """Prespecified step gates: first retain walking, then introduce smoothness."""

    terrain_levels = None
    enable_angular_acceleration = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={"term_name": "angular_acceleration_l2", "weight": -1.0e-3, "num_steps": 600},
    )
    enable_lidar_translation = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={
            "term_name": "lidar_scan_translation_distortion_l2",
            "weight": -2.0,
            "num_steps": 600,
        },
    )
    enable_lidar_rotation = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={
            "term_name": "lidar_scan_rotation_distortion_l2",
            "weight": -1.0,
            "num_steps": 600,
        },
    )
    enable_linear_jerk = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={"term_name": "linear_jerk_l2", "weight": -1.0e-5, "num_steps": 1200},
    )
    enable_joint_limit_guard = CurrTerm(
        func=isaac_mdp.modify_reward_weight,
        params={"term_name": "dof_pos_limits", "weight": -1.0, "num_steps": 0},
    )


@configclass
class AnymalDLocomotionJointTrainingEnvCfg(AnymalDLocomotionRecoveryV05EnvCfg):
    """Shared J1/J2 environment; no route replay and no command scaling."""

    observations: AnymalDLocomotionJointTrainingObservationsCfg = (
        AnymalDLocomotionJointTrainingObservationsCfg()
    )
    rewards: AnymalDLocomotionJointTrainingRewardsCfg = (
        AnymalDLocomotionJointTrainingRewardsCfg()
    )
    curriculum: AnymalDLocomotionJointTrainingCurriculumCfg = (
        AnymalDLocomotionJointTrainingCurriculumCfg()
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity = mdp.EdgeBiasedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 8.0),
            rel_standing_envs=0.10,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            edge_probability=0.55,
            ranges=mdp.EdgeBiasedVelocityCommandCfg.Ranges(
                lin_vel_x=(-2.0, 3.0),
                lin_vel_y=(-1.5, 1.5),
                ang_vel_z=(-2.0, 2.0),
                heading=None,
            ),
        )
        joint_names = list(CANONICAL_JOINT_ORDER)
        self.observations.policy.history_frame.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=joint_names, preserve_order=True
        )


@configclass
class AnymalDLocomotionJointTrainingJ1EnvCfg(AnymalDLocomotionJointTrainingEnvCfg):
    """Generic history fine-tuning comparator with neutral localization."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.history_frame.params["localization_mode"] = "neutral"


@configclass
class AnymalDLocomotionJointTrainingJ2EnvCfg(AnymalDLocomotionJointTrainingEnvCfg):
    """Localization-aware history fine-tuning arm with original commands."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.history_frame.params["localization_mode"] = "actual"
