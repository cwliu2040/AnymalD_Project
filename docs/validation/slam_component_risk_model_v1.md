# Offline component-conditioned SLAM risk model v1

Date: 2026-08-24

## Frozen purpose and boundary

The component-pulse pilot established that preserving translation or yaw can outperform uniform
slowdown. This model asks the separate closed-loop question: can current causal state predict which
candidate XYZ scale will reduce future 0.5 s SLAM failure risk?

The model is development-only. Its 64 source runs were already used to establish component
headroom, so leave-one-block-out evaluation is internal development validation, not a fresh or
formal claim. Backend ID, profile ID, intervention ID, ground truth, and future labels are forbidden
runtime inputs. ROS wiring, live execution, PPO, default switching, and physical use remain false.

## Model and validation

Three fixed L2-ridge probability models use the same four folds, each holding out all 16 runs from
one block 572--575:

- state-only uses confidence, normalized age, requested command, compact base motion, joint-speed
  RMS, and previous-action RMS;
- scalar-action adds only mean XYZ scale and its state interactions, representing a B-like scalar
  action description;
- component-action keeps translation and yaw scales separate and adds frozen confidence/yaw
  interactions.

Preprocessing is refit inside each fold. Candidate selection scores all four arms from the mean
pre-treatment state of each held-out backend/profile/block group. The gate requires component
Brier no worse than both baselines, LIO-SAM risk/progress benefit over uniform slowdown, FAST-LIO2
noninferiority to control, action diversity, and no selected safety event.

## Result

The frozen decision is `FAIL` because probability accuracy did not generalize across blocks:

- state-only Brier: `0.245041`;
- scalar-action Brier: `0.245622`;
- component-action Brier: `0.250866`.

The component selector nevertheless chose a component arm in 14/16 groups, used two distinct
component choices, produced mean LIO-SAM selected-minus-uniform risk `-0.2171`, increased LIO-SAM
moving speed by `0.2528 m/s`, and had FAST selected-minus-control risk `0.0`. Seven of eight
LIO-SAM groups were non-worse than uniform, and no selected arm had a safety event.

These favorable selection outcomes are not enough to override the Brier failure. They are partly
driven by `preserve_translation` being selected in 13/16 groups, while block-to-block risk remains
hard to calibrate from only four blocks. The valid conclusion is:

- component actions have causal headroom;
- the current compact model and 64 development runs do not yet provide a reliable calibrated
  action-conditioned risk probability;
- do not connect this model to the governor or start C2/PPO;
- do not tune features or thresholds on these same folds and relabel the result as validation.

The next design decision is whether to collect fresh identification blocks with richer randomized
component amplitudes, or redesign the offline state representation and reserve entirely fresh
blocks for its validation.

## Artifact

The machine-readable report is
`exported/slam_component_risk_model_v1/model_report.json`, SHA-256
`7ca5e03a7bd90a73958d5305c10c78b9331bdae5b4ae101e56b4b04eaedee0af`.
The frozen config SHA-256 is
`ddfae9b4983dd381f7e9dcae8566ca8e64c0fef9032477a4591b75ccd5f3e767`.
