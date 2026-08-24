# Component-specific SLAM command-pulse causal pilot v1

Date: 2026-08-24

## Purpose

The earlier short-pulse experiment changed linear X, linear Y, and angular Z by the same factor.
That established a command-action effect for FAST-LIO2, but it was still a B-like uniform speed
limiter and failed the prespecified LIO-SAM headroom gate. This experiment asks a narrower question:
can preserving one useful command component outperform uniform 0.75 slowdown?

The four arms were frozen before collection:

- `control`: `[1.00, 1.00, 1.00]`;
- `uniform_075`: `[0.75, 0.75, 0.75]`;
- `preserve_yaw`: `[0.75, 0.75, 1.00]`;
- `preserve_translation`: `[1.00, 1.00, 0.75]`.

All arms use frozen model1450 plus Arm B, share the untreated prefix, apply the XYZ pulse at sensor
degradation onset 7.25 s for 0.75 s, and recover exactly. This is an intervention probe, not a new
policy. It does not add confidence to the policy or authorize PPO.

## Frozen decision rule

The primary comparison is each component arm against `uniform_075`, not merely against no
intervention. A LIO-SAM curve stratum supports a component only when at least three of four block
differences are non-worse, mean hazard benefit is at least 0.05, mean moving-speed loss is no more
than 0.02 m/s, and no repeated component-specific safety excess occurs. The same component must
also keep mean FAST-LIO2 hazard within 0.02 of control. Backends and right/left curves are reported
separately; pre-treatment covariates are retained as diagnostics and cannot exclude runs.

## Execution

Restricted-sandbox wiring attempt 1 stopped after one cell because DDS interfaces and the GPU were
unavailable. It is retained as an environment `WIRING_FAIL`, not an experimental outcome.

GPU-visible attempt 2 completed the fresh block-571 wiring smoke 8/8 with `WIRING_PASS`. Four
episode safety events were retained, but none was used to stop or relabel wiring. The separately
authorized randomized pilot then completed 64/64 records over blocks 572--575, two backends, two
curve directions, and all four arms. Every integrity, trace, schedule, prestate-reporting, FAST
safeguard, and repeated-safety condition passed. The 64-run pilot had no fall or base contact.

## Result

The frozen decision is `PASS`, with both component arms qualifying. The useful result is localized,
not universal:

- LIO-SAM right curve: uniform risk was 0.826. `preserve_yaw` reduced it to 0.616 while moving
  0.075 m/s faster; `preserve_translation` reduced it to 0.399 while moving 0.140 m/s faster.
- LIO-SAM left curve: `preserve_translation` matched uniform risk at 0.849 and moved 0.327 m/s
  faster. `preserve_yaw` was worse in one block and did not support that stratum.
- FAST-LIO2 right curve: `preserve_yaw` matched uniform risk and was 0.011 m/s faster;
  `preserve_translation` was 0.066 worse in mean risk than uniform but remained within the frozen
  safeguard when compared with control across both profiles.
- FAST-LIO2 left curve: component and uniform hazards were all 0.500; preserving translation added
  0.308 m/s moving speed.

Therefore the experiment provides development evidence that C should not choose only a scalar
speed limit. It should choose how much translation and yaw to preserve as a function of current
state and causal SLAM risk. The strongest practical seed is `preserve_translation`: it delivered
the largest right-curve LIO-SAM hazard improvement and retained much more motion, while remaining
neutral on left-curve LIO-SAM risk. Directional asymmetry and the FAST right-curve tradeoff must be
represented in the next model rather than hidden by pooling.

The next permitted step is offline component-conditioned risk modeling with blockwise validation.
Live execution, PPO, default switching, formal claims, and physical-robot use are closed. A later
learned C still must use the original requested command, retain B only as missing-risk fallback and
invalid/stale hard safety, and prove it does not collapse back to uniform B-like scaling.

## Locked artifacts

- Protocol: `configs/slam_component_pulse_causal_pilot_v1.yaml`, SHA-256
  `a0646009a2cf28acab99755f0c5e23a3fde2e31fcf0e152472d67e893514c6e5`.
- Accepted wiring decision: `outputs/slam_component_pulse_causal_pilot_v1_attempt2/wiring_smoke/decision.json`,
  SHA-256 `5eeaffeab1a5546660c77ab1172feda160c4c2cae4e4b1c5fceba253e1ef6cce`.
- Pilot manifest: `outputs/slam_component_pulse_causal_pilot_v1_attempt2/causal_pilot/run_manifest.json`,
  SHA-256 `587c64f8d4250fb778dd439f98fcff76ce2338d99544c15ee0401c9bd5403b4b`.
- Pilot decision: `outputs/slam_component_pulse_causal_pilot_v1_attempt2/causal_pilot/decision.json`,
  SHA-256 `ebd2cc10cd9ed79b0622b08ebe4a34e61462bf12779f2864f8ea3cf8456fa710`.
