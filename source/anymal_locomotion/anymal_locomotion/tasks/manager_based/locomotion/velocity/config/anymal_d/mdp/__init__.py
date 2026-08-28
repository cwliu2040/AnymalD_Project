"""Project-specific MDP terms."""

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide

from .commands import (
    EdgeBiasedVelocityCommand,
    EdgeBiasedVelocityCommandCfg,
    RecoveryV05VelocityCommand,
    RecoveryV05VelocityCommandCfg,
)
from .events import validate_anymal_d_joint_contract
from .joint_training import (
    causal_slam_delayed_advantage,
    causal_slam_state,
    full_policy_history_frame,
    joint_training_angular_acceleration_l2,
    joint_training_lidar_scan_rotation_distortion_l2,
    joint_training_lidar_scan_translation_distortion_l2,
    joint_training_linear_jerk_l2,
)
from .rewards import (
    confidence_gait_action_rate_l2,
    confidence_degradation_feet_slide,
    confidence_gait_ang_vel_xy_l2,
    confidence_gait_feet_slide,
    confidence_gait_flat_orientation_l2,
    confidence_gait_lin_vel_z_l2,
    confidence_recovery_action_rate_l2,
    confidence_recovery_ang_vel_xy_l2,
    confidence_recovery_flat_orientation_l2,
    confidence_invalid_action_l2,
    confidence_invalid_planar_speed_l2,
    confidence_invalid_yaw_rate_l2,
    confidence_track_ang_vel_z_exp,
    confidence_track_lin_vel_xy_exp,
    high_curve_track_lin_vel_xy_exp,
    high_combined_feet_slide,
    high_combined_flat_orientation_l2,
    low_curve_track_lin_vel_xy_exp,
    low_yaw_track_ang_vel_z_exp,
    lateral_stance_slip_barrier,
    mixed_yaw_tracking_barrier,
    refinery_feet_slide,
    refinery_flat_orientation_l2,
)
from .slam_confidence import (
    confidence_phase,
    confidence_degradation_mask,
    confidence_recovery_mask,
    confidence_safe_scale,
    gait_mode_command_scale,
    gait_mode_velocity_command,
    simulated_slam_confidence,
)
from .symmetry import compute_symmetric_states
