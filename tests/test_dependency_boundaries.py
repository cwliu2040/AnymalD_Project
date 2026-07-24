"""Static dependency and artifact-boundary tests."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PACKAGE = PROJECT_ROOT / "source" / "anymal_locomotion"
EVALUATION_SCRIPT = PROJECT_ROOT / "scripts" / "rsl_rl" / "evaluate.py"
ROS2_POLICY_NODE = (
    PROJECT_ROOT
    / "deployment"
    / "ros2_ws"
    / "src"
    / "anymal_locomotion_ros2"
    / "anymal_locomotion_ros2"
    / "policy_node.py"
)
ROS2_BRIDGE = (
    PROJECT_ROOT
    / "source"
    / "anymal_locomotion"
    / "anymal_locomotion"
    / "simulation"
    / "ros2_bridge.py"
)
ROS2_BRIDGE_HOST = PROJECT_ROOT / "scripts" / "validation" / "validate_ros2_bridge.py"
ONNX_BACKEND = (
    PROJECT_ROOT
    / "deployment"
    / "ros2_ws"
    / "src"
    / "anymal_locomotion_ros2"
    / "anymal_locomotion_ros2"
    / "onnx_backend.py"
)


def test_training_package_does_not_import_rclpy() -> None:
    violations: list[str] = []
    for path in TRAINING_PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name == "rclpy" for alias in node.names):
                violations.append(str(path))
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "rclpy":
                violations.append(str(path))
    assert violations == []


def test_artifact_configuration_is_project_local() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "artifacts.yaml").read_text(encoding="utf-8"))
    root = Path(config["project_root"]).resolve()
    assert root == PROJECT_ROOT
    for key in ("logs_root", "checkpoints_root", "exports_root"):
        path = Path(config[key]).resolve()
        assert root in path.parents
        assert not str(path).startswith("/home/ros/IsaacLab/logs")


def test_evaluation_uses_public_fixed_command_configuration() -> None:
    source = EVALUATION_SCRIPT.read_text(encoding="utf-8")
    assert "command_cfg.ranges.lin_vel_x" in source
    assert "command_cfg.ranges.lin_vel_y" in source
    assert "command_cfg.ranges.ang_vel_z" in source
    assert "command_manager._" not in source
    assert "rclpy" not in source
    assert 'LOG_ROOT / "evaluation"' in source


def test_ros2_policy_node_stays_outside_training_and_uses_bridge_topics() -> None:
    source = ROS2_POLICY_NODE.read_text(encoding="utf-8")
    assert "import rclpy" in source
    assert "Odometry" in source
    assert "Imu" in source
    assert "JointState" in source
    assert '"/joint_command"' in source
    assert "socket" not in source
    assert "udp" not in source.lower()


def test_isaac_ros2_bridge_uses_action_graph_and_name_based_commands() -> None:
    source = ROS2_BRIDGE.read_text(encoding="utf-8")
    for node_type in (
        "ConstantQuatd",
        "ConstantMatrix4d",
        "SetMatrix4Rotation",
        "OgnInvertMatrix",
        "TransformVector",
        "OnImpulseEvent",
        "ROS2PublishClock",
        "ROS2Publisher",
        "ROS2PublishOdometry",
        "ROS2PublishImu",
        "ROS2SubscribeTwist",
        "ROS2SubscribeJointState",
        "IsaacArticulationController",
    ):
        assert node_type in source
    assert "SubscribeJointState.outputs:jointNames" in source
    assert "ComputeOdometry.outputs:linearVelocity" in source
    assert "BodyAngularVelocity.outputs:result" in source
    assert '("PublishOdometry.inputs:publishRawVelocities", True)' in source
    assert "Context.inputs:domain_id" in source
    assert "read_joint_position_command" in source
    assert "read_velocity_command" in source
    assert "write_base_orientation" in source
    assert "write_joint_state" in source
    assert "trigger_policy_step" in source
    assert '"sensor_msgs"' in source
    assert '"JointState"' in source
    assert "outputs:timeStamp" in source
    assert "rclpy" not in source
    assert "socket" not in source
    assert "udp" not in source.lower()


def test_bridge_host_preserves_manager_based_actuator_path() -> None:
    source = ROS2_BRIDGE_HOST.read_text(encoding="utf-8")
    assert "connect_articulation_controller=not args_cli.external_control" in source
    assert "read_joint_position_command" in source
    assert "validate_runtime_joint_names" in source
    assert "env.step(action_for_step)" in source
    assert "joint_command_timeout_steps" in source
    assert "actions.zero_()" in source
    assert "rclpy" not in source


def test_onnx_backend_is_external_and_cpu_only() -> None:
    source = ONNX_BACKEND.read_text(encoding="utf-8")
    assert "ReferenceEvaluator" in source
    assert "np.float32" in source
    assert "rclpy" not in source
