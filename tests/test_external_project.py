"""External Project import, registration, and resolved-config tests."""

from __future__ import annotations

import ast
import importlib.util

import pytest

import anymal_locomotion
from anymal_locomotion.artifacts import PROJECT_ROOT
from anymal_locomotion.policy_contract import CANONICAL_JOINT_ORDER, POLICY_CONTRACT


def test_extension_package_import() -> None:
    assert anymal_locomotion.__version__ == "0.1.0"


try:
    ISAAC_RUNTIME_AVAILABLE = importlib.util.find_spec("omni.timeline") is not None
except ModuleNotFoundError:
    ISAAC_RUNTIME_AVAILABLE = False
requires_isaac_runtime = pytest.mark.skipif(
    not ISAAC_RUNTIME_AVAILABLE,
    reason="Isaac Sim runtime is unavailable; run this suite after AppLauncher starts",
)


def _runtime_imports():
    import gymnasium as gym

    import anymal_locomotion.tasks  # noqa: F401
    from anymal_locomotion.tasks.manager_based.locomotion.velocity.config.anymal_d import PLAY_TASK_ID, TASK_ID
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    return gym, TASK_ID, PLAY_TASK_ID, load_cfg_from_registry


@requires_isaac_runtime
def test_task_registration() -> None:
    gym, task_id, play_task_id, _ = _runtime_imports()
    assert gym.spec(task_id).id == task_id
    assert gym.spec(play_task_id).id == play_task_id


@requires_isaac_runtime
def test_resolved_environment_contract() -> None:
    _, task_id, _, load_cfg_from_registry = _runtime_imports()
    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")

    assert cfg.scene.num_envs == 4096
    assert cfg.sim.dt == pytest.approx(0.005)
    assert cfg.decimation == 4
    assert 1.0 / (cfg.sim.dt * cfg.decimation) == pytest.approx(50.0)

    assert cfg.scene.height_scanner is None
    assert cfg.observations.policy.height_scan is None
    assert POLICY_CONTRACT["observation"]["dimension"] == 48

    expected_terms = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    )
    assert all(getattr(cfg.observations.policy, term) is not None for term in expected_terms)

    assert tuple(cfg.actions.joint_pos.joint_names) == CANONICAL_JOINT_ORDER
    assert cfg.actions.joint_pos.preserve_order is True
    assert cfg.actions.joint_pos.scale == pytest.approx(0.5)
    assert len(cfg.actions.joint_pos.joint_names) == 12

    joint_pos_asset = cfg.observations.policy.joint_pos.params["asset_cfg"]
    joint_vel_asset = cfg.observations.policy.joint_vel.params["asset_cfg"]
    assert tuple(joint_pos_asset.joint_names) == CANONICAL_JOINT_ORDER
    assert tuple(joint_vel_asset.joint_names) == CANONICAL_JOINT_ORDER
    assert joint_pos_asset.preserve_order is True
    assert joint_vel_asset.preserve_order is True

    command = cfg.commands.base_velocity
    assert command.heading_command is False
    assert command.rel_heading_envs == 0.0
    assert command.ranges.lin_vel_x == (-1.0, 1.0)
    assert command.ranges.lin_vel_y == (-1.0, 1.0)
    assert command.ranges.ang_vel_z == (-1.0, 1.0)


@requires_isaac_runtime
def test_official_reward_and_ppo_baseline_are_retained() -> None:
    _, task_id, _, load_cfg_from_registry = _runtime_imports()
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")

    expected_weights = {
        "track_lin_vel_xy_exp": 1.0,
        "track_ang_vel_z_exp": 0.5,
        "lin_vel_z_l2": -2.0,
        "ang_vel_xy_l2": -0.05,
        "dof_torques_l2": -2.5e-5,
        "dof_acc_l2": -2.5e-7,
        "action_rate_l2": -0.01,
        "feet_air_time": 0.5,
        "undesired_contacts": -1.0,
        "flat_orientation_l2": -5.0,
        "dof_pos_limits": 0.0,
    }
    for name, weight in expected_weights.items():
        assert getattr(env_cfg.rewards, name).weight == pytest.approx(weight)

    assert agent_cfg.experiment_name == "anymal_d_locomotion_v1"
    assert agent_cfg.num_steps_per_env == 24
    assert agent_cfg.max_iterations == 300
    assert agent_cfg.policy.actor_hidden_dims == [128, 128, 128]
    assert agent_cfg.policy.critic_hidden_dims == [128, 128, 128]
    assert agent_cfg.policy.actor_obs_normalization is False
    assert agent_cfg.algorithm.learning_rate == pytest.approx(1.0e-3)
    assert agent_cfg.algorithm.desired_kl == pytest.approx(0.01)
    assert agent_cfg.algorithm.max_grad_norm == pytest.approx(1.0)


def test_artifact_root_is_not_isaaclab() -> None:
    assert PROJECT_ROOT == PROJECT_ROOT.resolve()
    assert str(PROJECT_ROOT) == "/home/ros/anymal_locomotion"
    assert not str(PROJECT_ROOT / "logs").startswith("/home/ros/IsaacLab/logs")


def test_task_source_declares_project_contract() -> None:
    source_path = (
        PROJECT_ROOT
        / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion/velocity"
        / "config/anymal_d/flat_env_cfg.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "CANONICAL_JOINT_ORDER" in source
    assert "preserve_order = True" in source
    assert "command.heading_command = False" in source
    assert 'joint_names=[".*"]' not in source


def test_startup_event_accepts_required_env_ids() -> None:
    source_path = (
        PROJECT_ROOT
        / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion/velocity"
        / "config/anymal_d/mdp/events.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_anymal_d_joint_contract"
    )
    assert [argument.arg for argument in function.args.args[:2]] == ["env", "env_ids"]
    mandatory_count = len(function.args.args) - len(function.args.defaults)
    assert mandatory_count == 2
