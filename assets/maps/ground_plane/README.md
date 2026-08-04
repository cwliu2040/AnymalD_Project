# Feature-poor GroundPlane scene

`GroundPlane.usda` is the reviewable source layer and `GroundPlane.usd` is the
runtime layer used by the SLAM qualification matrix. It contains one flat,
100 m square collision mesh and no walls, props, or visual features. The robot
and RTX LiDAR remain unchanged; this scene is intentionally a point-cloud
degeneracy arm, not a locomotion training environment.

Rebuild the runtime layer after changing the source:

```bash
$ISAACLAB_ROOT/isaaclab.sh -p scripts/validation/build_ground_plane_usd.py
```
