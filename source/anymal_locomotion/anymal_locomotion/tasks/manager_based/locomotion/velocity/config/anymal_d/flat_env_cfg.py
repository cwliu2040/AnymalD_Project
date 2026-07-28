"""Project-owned ANYmal-D Flat v1 environment configuration."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
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
