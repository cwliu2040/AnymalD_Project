"""Project-owned Isaac Sim physics IMU prim spawner."""

from __future__ import annotations

import math
from collections.abc import Callable

import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
import omni.kit.commands
from isaaclab.sim import SpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass
from pxr import Gf, Usd


@clone
def spawn_physics_imu(
    prim_path: str,
    cfg: PhysicsImuSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_: object,
) -> Usd.Prim:
    """Create an Isaac Sim IMU before physics starts."""
    if not math.isfinite(cfg.sensor_period) or cfg.sensor_period <= 0.0:
        raise ValueError(
            f"IMU sensor period must be finite and positive, got {cfg.sensor_period}"
        )
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"IMU sensor prim path already exists: {prim_path}")
    parent_path, sensor_name = prim_path.rsplit("/", 1)
    translation = translation or (0.0, 0.0, 0.0)
    orientation = orientation or (1.0, 0.0, 0.0, 0.0)
    success, sensor = omni.kit.commands.execute(
        "IsaacSensorCreateImuSensor",
        path=f"/{sensor_name}",
        parent=parent_path,
        sensor_period=cfg.sensor_period,
        translation=Gf.Vec3d(*translation),
        orientation=Gf.Quatd(orientation[0], *orientation[1:]),
        linear_acceleration_filter_size=cfg.linear_acceleration_filter_size,
        angular_velocity_filter_size=cfg.angular_velocity_filter_size,
        orientation_filter_size=cfg.orientation_filter_size,
    )
    if not success or sensor is None:
        raise RuntimeError(f"Failed to create physics IMU sensor at {prim_path}")
    if str(sensor.GetPath()) != prim_path:
        raise RuntimeError(
            f"Physics IMU sensor resolved to {sensor.GetPath()}, expected {prim_path}"
        )
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(IsaacSensorSchema.IsaacImuSensor):
        raise RuntimeError(f"Prim is not an Isaac IMU sensor: {prim_path}")
    return prim


@configclass
class PhysicsImuSpawnerCfg(SpawnerCfg):
    """Configuration for an Isaac Sim physics IMU prim."""

    func: Callable = spawn_physics_imu
    sensor_period: float = 0.005
    linear_acceleration_filter_size: int = 1
    angular_velocity_filter_size: int = 1
    orientation_filter_size: int = 1

