"""Create the Isaac Sim ROS 2 Action Graph for one ANYmal-D articulation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Ros2PolicyBridge:
    """Resolved graph and robot paths for one simulation bridge."""

    graph_path: str
    articulation_root_path: str
    command_topic: str
    joint_state_topic: str
    imu_topic: str
    odometry_topic: str
    joint_command_topic: str
    domain_id: int
    uses_articulation_controller: bool


@dataclass(frozen=True)
class Ros2JointPositionCommand:
    """One named joint-position command received by the Action Graph."""

    names: tuple[str, ...]
    positions: tuple[float, ...]
    timestamp: float


@dataclass(frozen=True)
class Ros2VelocityCommand:
    """Latest body-frame velocity command received by the Action Graph."""

    linear: tuple[float, float, float]
    angular: tuple[float, float, float]


@dataclass(frozen=True)
class Ros2BaseState:
    """Base state computed by the Action Graph before ROS serialization."""

    linear_velocity: tuple[float, float, float]
    world_angular_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


def _resolve_domain_id(domain_id: int | None) -> int:
    if domain_id is None:
        value = os.environ.get("ROS_DOMAIN_ID", "0")
        try:
            domain_id = int(value)
        except ValueError as exc:
            raise ValueError(f"ROS_DOMAIN_ID must be an integer, received {value!r}") from exc
    if not 0 <= domain_id <= 232:
        raise ValueError(f"ROS domain ID must be between 0 and 232, received {domain_id}")
    return domain_id


def _validate_articulation_root(path: str) -> None:
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"Articulation root prim does not exist: {path}")
    if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise ValueError(f"Prim does not have ArticulationRootAPI: {path}")
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"Articulation root is not a rigid body: {path}")


def create_ros2_policy_bridge(
    articulation_root_path: str,
    *,
    graph_path: str = "/ROS2PolicyBridge",
    command_topic: str = "cmd_vel",
    joint_state_topic: str = "joint_states",
    imu_topic: str = "imu",
    odometry_topic: str = "odom",
    joint_command_topic: str = "joint_command",
    domain_id: int | None = None,
    connect_articulation_controller: bool = True,
) -> Ros2PolicyBridge:
    """Create state publishers and a name-based joint-position subscriber."""
    import carb
    import omni.graph.core as og
    import omni.kit.app
    import omni.usd
    import usdrt

    if not omni.kit.app.get_app().get_extension_manager().is_extension_enabled(
        "isaacsim.ros2.bridge"
    ):
        raise RuntimeError(
            "isaacsim.ros2.bridge is not enabled; preload it with AppLauncher kit_args"
        )
    domain_id = _resolve_domain_id(domain_id)
    _validate_articulation_root(articulation_root_path)
    if omni.usd.get_context().get_stage().GetPrimAtPath(graph_path).IsValid():
        raise ValueError(f"Action Graph path already exists: {graph_path}")

    keys = og.Controller.Keys
    target_prim = [usdrt.Sdf.Path(articulation_root_path)]
    nodes = [
        ("PolicyImpulse", "omni.graph.action.OnImpulseEvent"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
        ("BaseOrientation", "omni.graph.nodes.ConstantQuatd"),
        ("IdentityMatrix", "omni.graph.nodes.ConstantMatrix4d"),
        ("OrientationMatrix", "omni.graph.nodes.SetMatrix4Rotation"),
        ("InverseOrientation", "omni.graph.nodes.OgnInvertMatrix"),
        ("BodyAngularVelocity", "omni.graph.nodes.TransformVector"),
        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("PublishJointState", "isaacsim.ros2.bridge.ROS2Publisher"),
        ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ("PublishImu", "isaacsim.ros2.bridge.ROS2PublishImu"),
        ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
        ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
    ]
    values = [
        ("PolicyImpulse.inputs:onlyPlayback", False),
        ("ReadSimTime.inputs:resetOnStop", False),
        ("Context.inputs:domain_id", domain_id),
        ("ComputeOdometry.inputs:chassisPrim", target_prim),
        ("PublishClock.inputs:topicName", "clock"),
        ("PublishJointState.inputs:messagePackage", "sensor_msgs"),
        ("PublishJointState.inputs:messageSubfolder", "msg"),
        ("PublishJointState.inputs:messageName", "JointState"),
        ("PublishJointState.inputs:topicName", joint_state_topic),
        ("PublishOdometry.inputs:topicName", odometry_topic),
        ("PublishOdometry.inputs:odomFrameId", "odom"),
        ("PublishOdometry.inputs:chassisFrameId", "base_link"),
        ("PublishOdometry.inputs:publishRawVelocities", True),
        ("PublishImu.inputs:topicName", imu_topic),
        ("PublishImu.inputs:frameId", "base_link"),
        ("SubscribeTwist.inputs:topicName", command_topic),
        ("SubscribeJointState.inputs:topicName", joint_command_topic),
    ]
    connections = [
        ("PolicyImpulse.outputs:execOut", "PublishClock.inputs:execIn"),
        ("PolicyImpulse.outputs:execOut", "ComputeOdometry.inputs:execIn"),
        ("PolicyImpulse.outputs:execOut", "SubscribeTwist.inputs:execIn"),
        ("PolicyImpulse.outputs:execOut", "SubscribeJointState.inputs:execIn"),
        ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
        ("ComputeOdometry.outputs:execOut", "PublishImu.inputs:execIn"),
        ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
        (
            "ComputeOdometry.outputs:orientation",
            "PublishOdometry.inputs:orientation",
        ),
        (
            "ComputeOdometry.outputs:linearVelocity",
            "PublishOdometry.inputs:linearVelocity",
        ),
        (
            "BodyAngularVelocity.outputs:result",
            "PublishOdometry.inputs:angularVelocity",
        ),
        ("ComputeOdometry.outputs:orientation", "PublishImu.inputs:orientation"),
        (
            "IdentityMatrix.inputs:value",
            "OrientationMatrix.inputs:matrix",
        ),
        (
            "BaseOrientation.inputs:value",
            "OrientationMatrix.inputs:rotationAngle",
        ),
        (
            "OrientationMatrix.outputs:matrix",
            "InverseOrientation.inputs:matrix",
        ),
        (
            "InverseOrientation.outputs:invertedMatrix",
            "BodyAngularVelocity.inputs:matrix",
        ),
        (
            "ComputeOdometry.outputs:angularVelocity",
            "BodyAngularVelocity.inputs:vector",
        ),
        (
            "BodyAngularVelocity.outputs:result",
            "PublishImu.inputs:angularVelocity",
        ),
        (
            "ComputeOdometry.outputs:linearAcceleration",
            "PublishImu.inputs:linearAcceleration",
        ),
        (
            "ReadSimTime.outputs:simulationTime",
            "PublishClock.inputs:timeStamp",
        ),
        (
            "ReadSimTime.outputs:simulationTime",
            "PublishOdometry.inputs:timeStamp",
        ),
        (
            "ReadSimTime.outputs:simulationTime",
            "PublishImu.inputs:timeStamp",
        ),
        ("Context.outputs:context", "PublishClock.inputs:context"),
        ("Context.outputs:context", "PublishJointState.inputs:context"),
        ("Context.outputs:context", "PublishOdometry.inputs:context"),
        ("Context.outputs:context", "PublishImu.inputs:context"),
        ("Context.outputs:context", "SubscribeTwist.inputs:context"),
        ("Context.outputs:context", "SubscribeJointState.inputs:context"),
    ]
    if connect_articulation_controller:
        nodes.append(
            (
                "ArticulationController",
                "isaacsim.core.nodes.IsaacArticulationController",
            )
        )
        values.append(
            ("ArticulationController.inputs:robotPath", articulation_root_path)
        )
        connections.extend(
            [
                (
                    "SubscribeJointState.outputs:execOut",
                    "ArticulationController.inputs:execIn",
                ),
                (
                    "SubscribeJointState.outputs:jointNames",
                    "ArticulationController.inputs:jointNames",
                ),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
            ]
        )

    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: nodes,
            keys.SET_VALUES: values,
            keys.CONNECT: connections,
        },
    )
    # The generic ROS 2 publisher creates message-field inputs dynamically after
    # its message type is resolved. Resolve them without advancing physics.
    settings = carb.settings.get_settings()
    play_simulations = settings.get_as_bool("/app/player/playSimulations")
    settings.set_bool("/app/player/playSimulations", False)
    try:
        omni.kit.app.get_app().update()
    finally:
        settings.set_bool("/app/player/playSimulations", play_simulations)
    joint_state_name = og.Controller.attribute(
        f"{graph_path}/PublishJointState.inputs:name"
    )
    if not joint_state_name.is_valid():
        raise RuntimeError(
            "ROS2 generic JointState publisher did not create its message fields"
        )
    if not og.Controller.set(
        og.Controller.attribute(
            f"{graph_path}/PublishJointState.inputs:header:frame_id"
        ),
        "base_link",
    ):
        raise RuntimeError("Failed to initialize JointState frame_id")
    og.Controller.connect(
        og.Controller.attribute(
            f"{graph_path}/PolicyImpulse.outputs:execOut"
        ),
        og.Controller.attribute(
            f"{graph_path}/PublishJointState.inputs:execIn"
        ),
    )
    return Ros2PolicyBridge(
        graph_path=graph_path,
        articulation_root_path=articulation_root_path,
        command_topic=f"/{command_topic.lstrip('/')}",
        joint_state_topic=f"/{joint_state_topic.lstrip('/')}",
        imu_topic=f"/{imu_topic.lstrip('/')}",
        odometry_topic=f"/{odometry_topic.lstrip('/')}",
        joint_command_topic=f"/{joint_command_topic.lstrip('/')}",
        domain_id=domain_id,
        uses_articulation_controller=connect_articulation_controller,
    )


def trigger_policy_step(bridge: Ros2PolicyBridge) -> None:
    """Evaluate one 50 Hz bridge tick without advancing Kit rendering."""
    import omni.graph.core as og

    impulse = og.Controller.attribute(
        f"{bridge.graph_path}/PolicyImpulse.state:enableImpulse"
    )
    if not og.Controller.set(impulse, True):
        raise RuntimeError("Failed to trigger ROS2 policy bridge impulse")
    og.Controller.evaluate_sync(graph_id=bridge.graph_path)


def read_velocity_command(bridge: Ros2PolicyBridge) -> Ros2VelocityCommand:
    """Read the latest Twist output without importing a ROS Python client."""
    import omni.graph.core as og

    node_path = f"{bridge.graph_path}/SubscribeTwist"
    linear = og.Controller.get(
        og.Controller.attribute(f"{node_path}.outputs:linearVelocity")
    )
    angular = og.Controller.get(
        og.Controller.attribute(f"{node_path}.outputs:angularVelocity")
    )
    if linear is None or angular is None or len(linear) != 3 or len(angular) != 3:
        raise RuntimeError("ROS2 Subscribe Twist did not produce two 3-D vectors")
    return Ros2VelocityCommand(
        linear=tuple(float(value) for value in linear),
        angular=tuple(float(value) for value in angular),
    )


def read_base_state(bridge: Ros2PolicyBridge) -> Ros2BaseState:
    """Read the odometry node outputs used by the ROS state publishers."""
    import omni.graph.core as og

    odometry_node_path = f"{bridge.graph_path}/ComputeOdometry"
    body_angular_velocity_node_path = f"{bridge.graph_path}/BodyAngularVelocity"
    linear_velocity = og.Controller.get(
        og.Controller.attribute(f"{odometry_node_path}.outputs:linearVelocity")
    )
    world_angular_velocity = og.Controller.get(
        og.Controller.attribute(f"{odometry_node_path}.outputs:angularVelocity")
    )
    angular_velocity = og.Controller.get(
        og.Controller.attribute(
            f"{body_angular_velocity_node_path}.outputs:result"
        )
    )
    orientation = og.Controller.get(
        og.Controller.attribute(f"{odometry_node_path}.outputs:orientation")
    )
    if (
        linear_velocity is None
        or world_angular_velocity is None
        or angular_velocity is None
        or orientation is None
        or len(linear_velocity) != 3
        or len(world_angular_velocity) != 3
        or len(angular_velocity) != 3
        or len(orientation) != 4
    ):
        raise RuntimeError("Isaac Compute Odometry did not produce a complete base state")
    return Ros2BaseState(
        linear_velocity=tuple(float(value) for value in linear_velocity),
        world_angular_velocity=tuple(
            float(value) for value in world_angular_velocity
        ),
        angular_velocity=tuple(float(value) for value in angular_velocity),
        orientation_xyzw=tuple(float(value) for value in orientation),
    )


def write_base_orientation(
    bridge: Ros2PolicyBridge,
    orientation_wxyz: Sequence[float],
) -> None:
    """Update the world orientation used for world-to-body vector projection."""
    import omni.graph.core as og

    values = tuple(float(value) for value in orientation_wxyz)
    if len(values) != 4:
        raise ValueError(
            f"Base orientation must contain wxyz quaternion, received {values}"
        )
    norm = sum(value * value for value in values) ** 0.5
    if norm < 1.0e-9:
        raise ValueError("Base orientation quaternion has near-zero norm")
    w, x, y, z = (value / norm for value in values)
    attribute = og.Controller.attribute(
        f"{bridge.graph_path}/BaseOrientation.inputs:value"
    )
    if not og.Controller.set(attribute, (x, y, z, w)):
        raise RuntimeError("Failed to update Action Graph base orientation")


def write_joint_state(
    bridge: Ros2PolicyBridge,
    joint_names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    *,
    timestamp_s: float,
) -> None:
    """Copy one canonical joint-state sample into the ROS 2 Bridge publisher."""
    import omni.graph.core as og

    names = tuple(str(name) for name in joint_names)
    position_values = tuple(float(value) for value in positions)
    velocity_values = tuple(float(value) for value in velocities)
    if not names:
        raise ValueError("Joint state must contain at least one named joint")
    if len(names) != len(position_values) or len(names) != len(velocity_values):
        raise ValueError(
            "Joint-state name/position/velocity length mismatch: "
            f"{len(names)}/{len(position_values)}/{len(velocity_values)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("Joint-state names must be unique")
    if not math.isfinite(timestamp_s) or timestamp_s < 0.0:
        raise ValueError(
            f"Joint-state timestamp must be finite and non-negative, got {timestamp_s}"
        )
    if not all(
        math.isfinite(value) for value in (*position_values, *velocity_values)
    ):
        raise ValueError("Joint state contains NaN or Inf")

    seconds = math.floor(timestamp_s)
    nanoseconds = round((timestamp_s - seconds) * 1.0e9)
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000

    node_path = f"{bridge.graph_path}/PublishJointState"
    values = (
        ("inputs:header:stamp:sec", int(seconds)),
        ("inputs:header:stamp:nanosec", int(nanoseconds)),
        ("inputs:name", list(names)),
        ("inputs:position", list(position_values)),
        ("inputs:velocity", list(velocity_values)),
        ("inputs:effort", []),
    )
    failed = [
        attribute_name
        for attribute_name, value in values
        if not og.Controller.set(
            og.Controller.attribute(f"{node_path}.{attribute_name}"),
            value,
        )
    ]
    if failed:
        raise RuntimeError(
            f"Failed to update ROS2 generic JointState fields: {failed}"
        )


def read_joint_position_command(
    bridge: Ros2PolicyBridge,
) -> Ros2JointPositionCommand | None:
    """Read the latest JointState output without a ROS Python client."""
    import omni.graph.core as og

    node_path = f"{bridge.graph_path}/SubscribeJointState"
    names = og.Controller.get(
        og.Controller.attribute(f"{node_path}.outputs:jointNames")
    )
    positions = og.Controller.get(
        og.Controller.attribute(f"{node_path}.outputs:positionCommand")
    )
    timestamp = og.Controller.get(
        og.Controller.attribute(f"{node_path}.outputs:timeStamp")
    )
    if names is None or positions is None or len(names) == 0:
        return None
    if len(names) != len(positions):
        raise ValueError(
            "ROS joint command name/position length mismatch: "
            f"{len(names)} names, {len(positions)} positions"
        )
    return Ros2JointPositionCommand(
        names=tuple(str(name) for name in names),
        positions=tuple(float(value) for value in positions),
        timestamp=float(timestamp),
    )
