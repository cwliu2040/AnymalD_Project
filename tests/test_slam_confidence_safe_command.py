from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = (
    ROOT
    / "source/anymal_locomotion/anymal_locomotion/policies"
    / "slam_confidence_residual.py"
)
ALGORITHM = (
    ROOT
    / "source/anymal_locomotion/anymal_locomotion/algorithms"
    / "slam_confidence_teacher_ppo.py"
)
AGENT_CONFIG = (
    ROOT
    / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based"
    / "locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py"
)
TASKS = (
    ROOT
    / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based"
    / "locomotion/velocity/config/anymal_d/__init__.py"
)


def test_safe_command_policy_preserves_contract_and_zero_start() -> None:
    source = POLICY.read_text(encoding="utf-8")
    assert "class FrozenBackboneSafeCommandActor" in source
    assert "observation_dim != 51" in source
    assert "legacy_observation_dim != 48" in source
    assert "command_offset != 9" in source
    assert "torch.zeros(num_actions)" in source
    assert "self.safe_command_gain_limit" in source
    assert "safe_command_gain_limit must be in (0, 1]" in source
    assert "safe_observation[..., command_slice] *= self.safe_scale" in source
    assert "rclpy" not in source


def test_safe_command_teacher_has_separate_fresh_adapter_rate() -> None:
    source = ALGORITHM.read_text(encoding="utf-8")
    assert "teacher_learning_rate" in source
    assert "teacher_gain_loss_coef" in source
    assert "teacher_gain_target" in source
    assert "teacher_gain_target exceeds actor safe-command limit" in source
    assert 'name == "safe_command_gain"' in source
    assert "predicted_action - target_action" in source
    assert "self.optimizer.add_param_group" in source


def test_safe_command_task_uses_formal_warm_start_compatible_actor() -> None:
    agent = AGENT_CONFIG.read_text(encoding="utf-8")
    tasks = TASKS.read_text(encoding="utf-8")
    assert "SlamConfidenceSafeCommandActorCriticCfg" in agent
    assert "teacher_target_mode=\"confidence_scaled_formal_command\"" in agent
    assert "teacher_learning_rate=5.0e-3" in agent
    assert "teacher_gain_loss_coef=1.0" in agent
    assert "SlamConfidence-SafeCommand-v0" in tasks
    assert "AnymalDLocomotionSlamConfidenceSafeCommandPPORunnerCfg" in tasks
    assert "SlamConfidence-BoundedSafeCommand-v0" in tasks
    assert "AnymalDLocomotionSlamConfidenceBoundedSafeCommandPPORunnerCfg" in tasks
    assert "safe_command_gain_limit=0.8" in agent
    assert "teacher_gain_target=0.8" in agent


def test_warm_start_and_parity_support_safe_command_actor() -> None:
    warm_start = (ROOT / "scripts/rsl_rl/train.py").read_text(encoding="utf-8")
    parity = (
        ROOT / "scripts/validation/validate_policy_parity.py"
    ).read_text(encoding="utf-8")
    assert '"actor.safe_command_gain"' in warm_start
    assert "safe_command_gain_initialization" in warm_start
    assert 'report["safe_command_gain_limit"]' in warm_start
    assert "class _SafeCommandCheckpointActor" in parity
    assert '"frozen_backbone_exact_safe_command"' in parity
    assert 'policy_config.get("safe_command_gain_limit", 1.0)' in parity
