# 800-cell formal accepted-artifact stratified diagnostics

Date: 2026-08-21

This is an exploratory diagnosis of the accepted-only 800-run formal live matrix. The frozen
confirmatory result remains
`outputs/slam_confidence_publication_v1/formal_analysis_v1/formal_analysis.json`; this diagnosis
does not change its estimands, evidence rule, or `complete_support=false` conclusion.

The reproducible diagnostic command is:

```bash
python3 scripts/validation/diagnose_slam_confidence_publication.py \
  --input-root outputs/slam_confidence_publication_v1/formal_analysis_v1/accepted_records \
  --output outputs/slam_confidence_publication_v1/formal_analysis_v1/stratified_diagnostics.json \
  --markdown-output outputs/slam_confidence_publication_v1/formal_analysis_v1/stratified_diagnostics.md
```

It reads run records, locomotion diagnostics, and C-arm mechanism sidecars. It does not read
rosbags, run SLAM replay, or treat frame samples as independent repetitions. All intervals below
use 10,000 resamples clustered by `(paired_block_id, profile)`.

## Backend and profile heterogeneity

C-D slip differs by backend: FAST-LIO2 is `-0.00597 m/s`, while LIO-SAM is `+0.00499 m/s`.
The preregistered difference in paired effects (FAST minus LIO) is `-0.01096 m/s`, 95% CI
`[-0.02244, -0.00007]`. Body-rate, tracking RMST, and progress backend interactions cross zero.
This is weak evidence of backend-specific slip interaction, not a positive learned-gait result:
neither backend supplies the complete C-D causal chain.

C-B is worse under LIO-SAM (`+0.02293 m/s` slip, `+0.02963 rad/s` body-rate) than under
FAST-LIO2 (`+0.00590 m/s`, `+0.00685 rad/s`). C also moves faster than B under both backends:
`+0.05953 m/s` for FAST and `+0.09382 m/s` for LIO.

Profile diagnosis shows two different failure modes:

- `curve_1_5_left_1_0`: C-B increases slip/body-rate, reduces RMST by `1.024 s`, and reduces
  progress by `0.11985` while moving `0.08466 m/s` faster.
- `lateral_right_1_5`: C-B increases slip by `0.02826 m/s` and body-rate by
  `0.05472 rad/s`, but increases progress by `0.06362`; this is a speed/safety trade-off.
- `curve_1_5_right_1_0`: C-D itself increases slip by `0.00522 m/s` and body-rate by
  `0.00742 rad/s` while moving `0.01804 m/s` faster.
- `warehouse_mapping_stress`: C-D improves progress by `0.01570`, but does not improve
  body-rate or RMST; the extra gait still lacks a safety/survival mechanism.

These are exploratory subgroup intervals and were not multiplicity-confirmatory.

## Matched speed and matched progress

Speed and progress are post-treatment mediators. The results in this section are diagnostics,
not randomized causal effects. Two operationalizations are reported: a linear paired-difference
intercept at zero mediator difference and a threshold subset (`|speed difference| <= 0.05 m/s`
or `|progress difference| <= 0.05`).

For C-D, equal-speed regression gives slip `-0.00114 m/s` (CI
`[-0.00410, 0.00197]`), body-rate `+0.00212 rad/s` (CI
`[-0.00234, 0.00653]`), and RMST `-0.071 s` (CI `[-0.709, 0.475]`). The 146
speed-threshold pairs agree that there is no safety or survival advantage. Extra gait therefore
does not recover value after removing its small speed difference.

For C-B, conclusions are method-sensitive after conditioning on speed. Equal-speed regression
estimates slip `-0.01844 m/s` (CI `[-0.02324, -0.01295]`) and body-rate
`-0.00202 rad/s` (CI crosses zero); the 49 speed-threshold pairs estimate slip
`-0.00688 m/s` (CI crosses zero) and body-rate `-0.01493 rad/s` (CI
`[-0.02513, -0.00575]`). Neither method gives a reliable RMST or progress benefit. In contrast,
equal-progress regression retains C-B harm: slip `+0.01442 m/s` (CI
`[0.00617, 0.02250]`) and body-rate `+0.01826 rad/s` (CI
`[0.00827, 0.02856]`).

The safe conclusion is not that old C is superior at matched speed. It is that much of its raw
C-B safety deficit is entangled with moving faster than conservative B, while the extra speed
does not yield reliable progress or SLAM-survival benefit. A new method must establish matched
speed by design in randomized simulation, rather than recover it with post-hoc adjustment.

## Nominal confidence phase

Nominal windows are healthy `0-3 s`, ramp-down `3-9 s`, low-support hold `9-13 s`, recovery
`13-16 s`, and after-recovery `>=16 s`. Exact LiDAR-adapter phase-origin timestamps are not in
the run record, so a small startup offset is possible. More importantly, route commands have a
5 s warmup. The nominal healthy window is stationary and cannot serve as an active-motion
healthy control; perception phase and command phase are partially aliased.

The useful within-window paired findings are nevertheless clear:

- During ramp-down, C-D increases slip `+0.00704 m/s` (CI
  `[0.00204, 0.01222]`), while-stable body-rate `+0.00742 rad/s` (CI
  `[0.00148, 0.01360]`), and speed `+0.02194 m/s` (CI `[0.00841, 0.03732]`).
- During low-support hold, C-D is essentially null, but C-B is much worse: slip
  `+0.05788 m/s`, body-rate `+0.11554 rad/s`, and speed `+0.10550 m/s`; all three
  intervals exclude zero.
- Recovery and after-recovery provide no reliable C-D slip, body-rate, or speed advantage.

Thus old C reacts too aggressively during degradation onset, remains much more active than B
during low support, and does not produce a measurable recovery dividend.

## Gait-coordinate associations

Run-level C-arm correlations were centered within backend/profile and then repeated with moving
speed partialled out. Most raw gait/slip associations disappear after speed adjustment, showing
that coordinate means are often activity proxies.

Remaining exploratory partial correlations are:

- applied stride attenuation versus tracking RMST: `r=-0.317`, CI `[-0.536, -0.108]`;
- crouch versus body-rate: `r=-0.196`, CI `[-0.351, -0.042]`;
- stance width versus slip: `r=+0.173`, CI `[0.037, 0.312]`;
- stance width versus body-rate: `r=+0.275`, CI `[0.133, 0.409]`;
- action smoothing versus body-rate: `r=+0.201`, CI `[0.041, 0.360]`.

These are ecological associations from a jointly controlled actor, not coordinate ablations.
They are sufficient to reject a blind reward sweep of the same four-coordinate head, but not to
claim that independently changing one coordinate will reproduce the association.

## Estimator boundary and design decision

All formal A/B/C/D arms use estimator15, so the 800-run matrix cannot identify a GT-versus-
estimator effect and estimator15 cannot explain randomized C-B or C-D differences. Prior
simulation still shows a real interaction candidate: C-v4 changed from `0/512` hard terminations
with GT velocity to `78/512` with the early estimator; the finalized model48+estimator15
regression later had `6/2560` across five profiles.

The minimum viable new C should therefore:

1. encode 0.5-1.0 s localization-risk history and distinguish degradation onset, persistent low
   support, and recovery;
2. retain deterministic invalid/stale hard safety, with B as the strong simple baseline;
3. learn richer contact/timing behavior rather than reuse the fixed four-coordinate head alone;
4. randomize estimator error, delay, and stale behavior during training;
5. use randomized matched-command/matched-speed simulation gates plus constant/shuffle/delay
   confidence ablations before any 48-cell live qualification.

No evidence here supports model48 promotion, default switching, or starting the 3,200-cell replay.
