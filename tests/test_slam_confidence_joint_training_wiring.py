from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FLAT_ENV = ROOT / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py"
JOINT_MDP = ROOT / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion/velocity/config/anymal_d/mdp/joint_training.py"
MOTION_CORE = ROOT / "source/anymal_locomotion/anymal_locomotion/joint_training_motion_core.py"
AGENT_CFG = ROOT / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py"
TASKS = ROOT / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion/velocity/config/anymal_d/__init__.py"
TRAIN = ROOT / "scripts/rsl_rl/train.py"
PREFLIGHT = ROOT / "scripts/validation/preflight_slam_confidence_joint_training.py"
PROTOCOL = ROOT / "configs/slam_confidence_joint_training_v1.yaml"
REWARDS = ROOT / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion/velocity/config/anymal_d/mdp/rewards.py"


def _class_source(path: Path, class_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"missing class {class_name}")


def test_j1_j2_wiring_differs_only_in_localization_mode() -> None:
    base = _class_source(FLAT_ENV, "AnymalDLocomotionJointTrainingEnvCfg")
    j1 = _class_source(FLAT_ENV, "AnymalDLocomotionJointTrainingJ1EnvCfg")
    j2 = _class_source(FLAT_ENV, "AnymalDLocomotionJointTrainingJ2EnvCfg")
    observations = _class_source(FLAT_ENV, "AnymalDLocomotionJointTrainingObservationsCfg")

    assert "history_length=20" in observations
    assert "flatten_history_dim=True" in observations
    assert '"localization_mode": "actual"' in observations
    assert "EdgeBiasedVelocityCommandCfg" in base
    assert "rel_standing_envs=0.10" in base
    assert 'params["localization_mode"] = "neutral"' in j1
    assert 'params["localization_mode"] = "actual"' in j2
    assert "confidence_track" not in base + j1 + j2
    assert "gait_mode_velocity_command" not in base + j1 + j2


def test_history_frame_preserves_original_command_and_canonical_layout() -> None:
    source = JOINT_MDP.read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "full_policy_history_frame"
    )
    function_source = ast.get_source_segment(source, function) or ""
    assert "generated_commands" in function_source
    assert "command *" not in function_source
    assert "safe_scale" not in function_source
    ordered_terms = [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "command,",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    ]
    positions = [function_source.index(term) for term in ordered_terms]
    assert positions == sorted(positions)
    assert "legacy.shape[-1] != 48" in function_source


def test_motion_rewards_do_not_penalize_constant_requested_yaw_or_mean_speed() -> None:
    source = JOINT_MDP.read_text(encoding="utf-8")
    motion_core = MOTION_CORE.read_text(encoding="utf-8")
    rotation = motion_core[motion_core.index("def lidar_scan_rotation_distortion_l2"):]
    assert "body_angular_velocity_radps[:, :2]" in rotation
    assert "body_angular_velocity_radps[:, 2]" not in rotation
    assert "command_manager" not in rotation
    assert "confidence_safe_scale" not in source

    rewards = _class_source(FLAT_ENV, "AnymalDLocomotionJointTrainingRewardsCfg")
    assert "confidence_track_lin_vel_xy_exp" not in rewards
    assert "confidence_track_ang_vel_z_exp" not in rewards
    assert "joint_training_angular_acceleration_l2" in rewards
    assert "joint_training_linear_jerk_l2" in rewards
    assert "joint_training_lidar_scan_translation_distortion_l2" in rewards
    assert "joint_training_lidar_scan_rotation_distortion_l2" in rewards


def test_behavior_anchor_and_runner_are_shared_and_nonzero() -> None:
    policy = _class_source(AGENT_CFG, "AnchoredFullPolicyActorCriticCfg")
    algorithm = _class_source(AGENT_CFG, "BehaviorAnchoredPpoAlgorithmCfg")
    base = _class_source(AGENT_CFG, "AnymalDLocomotionJointTrainingRunnerCfg")
    j1 = _class_source(AGENT_CFG, "AnymalDLocomotionJointTrainingJ1RunnerCfg")
    j2 = _class_source(AGENT_CFG, "AnymalDLocomotionJointTrainingJ2RunnerCfg")
    assert "full_observation_dim: int = 1068" in policy
    assert "behavior_anchor_coef: float = 0.25" in algorithm
    assert "BehaviorAnchoredPpoAlgorithmCfg" in base
    assert "symmetry_cfg = None" in base
    assert "entropy_coef=0.0" in base
    assert "AnymalDLocomotionJointTrainingRunnerCfg" in j1
    assert "AnymalDLocomotionJointTrainingRunnerCfg" in j2


def test_tasks_registered_but_execution_protocol_remains_closed() -> None:
    tasks = TASKS.read_text(encoding="utf-8")
    train = TRAIN.read_text(encoding="utf-8")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert "JointTraining-J1-v0" in tasks
    assert "JointTraining-J2-v0" in tasks
    assert "bootstrap_dense_actor_state" in train
    gates = protocol["execution_gates"]
    assert gates["ppo_training_authorized"] is False
    assert gates["live_ros_wiring_authorized"] is False
    assert gates["default_switch_authorized"] is False
    assert gates["physical_robot_authorized"] is False
    assert gates["nonlearning_preflight_authorized"] is False
    assert protocol["nonlearning_preflight"]["execution_authorized"] is False


def test_preflight_is_fixed_nonlearning_and_fails_closed_before_launcher() -> None:
    source = PREFLIGHT.read_text(encoding="utf-8")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert "from rsl_rl.runners" not in source
    assert "from rsl_rl.algorithms" not in source
    assert "torch.optim" not in source
    assert "runner.learn" not in source
    assert source.index("if not _preview_authorized:") < source.index("AppLauncher(args_cli)")
    assert source.index('ppo_training_authorized') < source.index("AppLauncher(args_cli)")
    assert "bootstrap_dense_actor_state" in source
    assert 'range(int(cfg["total_steps"]))' in source
    preflight = protocol["nonlearning_preflight"]
    assert preflight["total_steps"] == 100
    assert preflight["num_environments"] == 8
    assert preflight["maximum_runtime_bootstrap_action_parity_abs"] == 1.0e-6
    assert protocol["execution_gates"]["ppo_training_authorized"] is False


def test_joint_training_resume_restores_iteration_and_curriculum_state() -> None:
    source = TRAIN.read_text(encoding="utf-8")
    assert "completed_iterations = loaded_iteration + 1" in source
    assert "completed_iterations + int(agent_cfg.max_iterations)" in source
    assert "restored_common_steps = completed_iterations * rollout_steps" in source
    assert "base_env.common_step_counter = restored_common_steps" in source
    assert "base_env.curriculum_manager.compute(env_ids=all_env_ids)" in source
    assert 'run_manifest["joint_training_continuation"]' in source
