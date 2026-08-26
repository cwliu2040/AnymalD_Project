# Offline component-conditioned SLAM risk model v2

Date: 2026-08-24

## Frozen model

This development-only model uses the 96 richer-identification runs from blocks 577--580. Its
runtime inputs exclude backend, profile, intervention ID, ground truth, and future labels. Compared
with v1, the component family removes duplicate scale/reduction columns and most interactions. It
keeps only translation reduction, yaw reduction, and their interactions with requested yaw. Four
leave-one-block-out folds compare it with state-only and scalar-action ridge models.

The gate was frozen before fitting. It required component Brier no worse than both baselines,
LIO-SAM risk benefit over uniform slowdown without material speed loss, FAST-LIO2 noninferiority,
selection diversity, and no selected safety event. Passing would have unlocked reserved blocks
581--584; failing required stopping before those blocks.

## Result

The frozen decision is `FAIL`:

- component Brier: `0.257138`;
- state-only Brier: `0.250483`;
- scalar-action Brier: `0.254038`;
- LIO-SAM selected-minus-uniform mean hazard: `+0.128289`;
- LIO-SAM selected-minus-uniform mean moving speed: `+0.056488 m/s`;
- FAST-LIO2 selected-minus-control mean hazard: `-0.000356`;
- selected component fraction: `0.50`.

The selector used multiple actions, but it chose a safety-event control run in FAST-LIO2/right/block
579 and another in LIO-SAM/right/block579. More importantly, action selection increased average
LIO-SAM hazard and probability calibration was worse than both baselines. Speed and FAST-LIO2
noninferiority cannot override those failures.

The valid conclusion is narrower than “component actions do not work.” Fixed interventions do show
causal effects, but current pre-treatment state does not reliably identify which component action
will help on a new block. The state-to-action closed loop is therefore not ready. Reserved blocks
581--584 were not executed, and this model must not be connected to ROS, C2, or PPO.

The frozen config SHA-256 is
`cfa9753a8aa5e8ca2f4dea05a0fe05d9098a08217f21f6595e8322c3553f7737`. The report is
`exported/slam_component_risk_model_v2/model_report.json`, SHA-256
`8ac9ed89ccb8a916894ab64adf9df69027793c5ac94fff382c1a67a99f832ace`.
