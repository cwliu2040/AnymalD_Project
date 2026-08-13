from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "source/anymal_locomotion/anymal_locomotion/policies/slam_confidence_residual.py"
ALGORITHM = ROOT / "source/anymal_locomotion/anymal_locomotion/algorithms/slam_confidence_gait_ppo.py"
ENV = (
    ROOT
    / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based"
    / "locomotion/velocity/config/anymal_d/flat_env_cfg.py"
)
AGENT = ENV.parent / "agents/rsl_rl_ppo_cfg.py"
TASKS = ENV.parent / "__init__.py"


def test_safe_gait_actor_has_exact_zero_start_and_bounded_low_confidence_path() -> None:
    source = POLICY.read_text(encoding="utf-8")
    assert "class FrozenBackboneSafeGaitActor" in source
    assert "torch.nn.init.zeros_(final_layer.weight)" in source
    assert "torch.nn.init.zeros_(final_layer.bias)" in source
    assert "transition_gate.unsqueeze(-1) * residual" in source
    assert "4.0 * safe_scale * (1.0 - safe_scale)" in source
    assert "residual_action_limit: float = 0.25" in source
    assert "class SlamConfidenceSafeGaitActorCritic" in source
    assert "rclpy" not in source


def test_gait_ppo_separates_safe_gain_teacher_from_reward_driven_residual() -> None:
    source = ALGORITHM.read_text(encoding="utf-8")
    assert "gain.requires_grad_(False)" in source
    assert "self._set_residual_trainable(True)" in source
    assert "loss_dict = super().update()" in source
    assert 'loss_dict["safe_gain_teacher"]' in source


def test_gait_task_has_explicit_five_metric_rewards_and_pushes_disabled() -> None:
    env = ENV.read_text(encoding="utf-8")
    for term in (
        "confidence_gait_action_rate_l2",
        "confidence_gait_lin_vel_z_l2",
        "confidence_gait_ang_vel_xy_l2",
        "confidence_gait_flat_orientation_l2",
        "confidence_gait_feet_slide",
    ):
        assert term in env
    assert "class AnymalDLocomotionSlamConfidenceGaitCurriculumCfg" in env
    assert "self.events.push_robot = None" in env
    assert "self.events.base_external_force_torque = None" in env


def test_gait_task_and_runner_are_registered_for_actor_only_warm_start() -> None:
    agent = AGENT.read_text(encoding="utf-8")
    tasks = TASKS.read_text(encoding="utf-8")
    assert "SlamConfidenceSafeGaitActorCriticCfg" in agent
    assert "SlamConfidenceGaitPpoAlgorithmCfg" in agent
    assert "Isaac-Velocity-Flat-Anymal-D-Locomotion-SlamConfidence-Gait-v0" in tasks
    assert "AnymalDLocomotionSlamConfidenceGaitPPORunnerCfg" in tasks
