"""Project-owned RTX LiDAR mounting and render-product creation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RtxLidarSensor:
    """Resolved prim and render-product metadata for one simulated LiDAR."""

    mount_path: str
    sensor_path: str
    render_product_path: str
    frame_id: str
    raw_topic: str
    ros2_writer_name: str
    config: str
    variant: str
    mount_translation_xyz: tuple[float, float, float]
    mount_orientation_wxyz: tuple[float, float, float, float]
    scan_rate_hz: float
    channels: int
    horizontal_resolution: int


def create_rtx_lidar_sensor(
    articulation_root_path: str,
    *,
    mount_name: str = "lidar_link",
    sensor_name: str = "rtx_lidar",
    config: str = "OS1",
    variant: str = "OS1_REV6_32ch10hz1024res",
    mount_translation_xyz: tuple[float, float, float] = (0.20, 0.0, 0.35),
    frame_id: str = "lidar_link",
    raw_topic: str = "lidar/points_raw",
) -> RtxLidarSensor:
    """Attach the confirmed OS1 profile and official ROS 2 Bridge writer."""
    import omni.kit.commands
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, UsdGeom

    if not mount_name or "/" in mount_name:
        raise ValueError("mount_name must be one non-empty path component")
    if not sensor_name or "/" in sensor_name:
        raise ValueError("sensor_name must be one non-empty path component")

    stage = omni.usd.get_context().get_stage()
    articulation = stage.GetPrimAtPath(articulation_root_path)
    if not articulation.IsValid():
        raise ValueError(
            f"Articulation root prim does not exist: {articulation_root_path}"
        )
    mount_path = f"{articulation_root_path.rstrip('/')}/{mount_name}"
    if stage.GetPrimAtPath(mount_path).IsValid():
        raise ValueError(f"LiDAR mount prim already exists: {mount_path}")

    mount = UsdGeom.Xform.Define(stage, mount_path)
    UsdGeom.XformCommonAPI(mount).SetTranslate(Gf.Vec3d(*mount_translation_xyz))
    success, sensor = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path=f"/{sensor_name}",
        parent=mount_path,
        config=config,
        variant=variant,
        translation=Gf.Vec3d(0.0, 0.0, 0.0),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        **{"omni:sensor:Core:outputFrameOfReference": "SENSOR"},
    )
    if not success or sensor is None or not sensor.IsValid():
        raise RuntimeError(
            f"Failed to create RTX LiDAR {config}/{variant} under {mount_path}"
        )
    sensor_path = str(sensor.GetPath())
    if not sensor_path.startswith(f"{mount_path}/"):
        raise RuntimeError(
            f"RTX LiDAR resolved outside its mount: {sensor_path} vs {mount_path}"
        )
    frame_attribute = sensor.GetAttribute(
        "omni:sensor:Core:outputFrameOfReference"
    )
    if frame_attribute.IsValid() and frame_attribute.Get() != "SENSOR":
        raise RuntimeError(
            "RTX LiDAR output must remain in sensor coordinates, received "
            f"{frame_attribute.Get()!r}"
        )

    render_product = rep.create.render_product(
        sensor_path,
        resolution=(1, 1),
        name="AnymalLidar",
    )
    render_product_path = str(render_product.path)
    if not render_product_path:
        raise RuntimeError("RTX LiDAR render product path is empty")

    # Isaac Sim 5.1's official standalone RTX LiDAR example attaches the
    # Replicator ROS 2 Bridge writer directly to the render product. The
    # Buffer variant accumulates one complete mechanical scan.
    ros2_writer_name = "RtxLidarROS2PublishPointCloudBuffer"
    ros2_writer = rep.writers.get(ros2_writer_name)
    ros2_writer.initialize(topicName=raw_topic, frameId=frame_id)
    ros2_writer.attach([render_product])

    return RtxLidarSensor(
        mount_path=mount_path,
        sensor_path=sensor_path,
        render_product_path=render_product_path,
        frame_id=frame_id,
        raw_topic=f"/{raw_topic.lstrip('/')}",
        ros2_writer_name=ros2_writer_name,
        config=config,
        variant=variant,
        mount_translation_xyz=mount_translation_xyz,
        mount_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        scan_rate_hz=10.0,
        channels=32,
        horizontal_resolution=1024,
    )
