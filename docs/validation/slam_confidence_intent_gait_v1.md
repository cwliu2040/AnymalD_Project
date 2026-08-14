# SLAM confidence intent-gait experiments

Date: 2026-08-13

Status: C-v10 phase-separated structured-gait candidate passes the fixed
behavior, gait-value and export-parity gates. The exported deployment candidate
has passed the hardest right-lateral live cell once with each backend, but it
has not been promoted; recovery v0.4.0 model1450 remains formal.

## Architecture and invariant

The 51-D actor keeps the official 48-D model1450 actor frozen and adds three
deployable inputs: confidence, tracking-valid, and normalized source age. A
separate supervised 1-D intent head blends raw-command and safe-command
model1450 actions. PPO controls four bounded structured gait coordinates:
stride attenuation, crouch, stance width, and previous-action smoothing. The
head cannot emit an arbitrary 12-D residual.

Healthy confidence makes the raw and safe actions identical and gates off the
structured delta, so the healthy path remains exactly model1450 regardless of
trained head weights. Training uses the exact project TorchScript
proprioceptive velocity estimator in the observation loop; simulator ground
truth velocity is evaluator-only.

C-v10 adds a policy-internal stride envelope. It uses only the three confidence
contract values, not simulator phase or ground truth. A live degradation has
age growing more slowly than confidence falls; post-outage recovery begins
older and becomes fresher. During valid degradation only, the learned positive
stride coordinate is continuously attenuated with a cubic confidence envelope.
Invalid tracking preserves the learned PPO posture while the intent head stops;
recovery preserves the larger learned stride needed to stand and reacquire the
command. This is structured gait adaptation inside the policy, not an external
speed limiter.

## Estimator-closure and fixed pilots

All listed behavior evaluations use seed 43, 512 environments, 1,000 steps,
the frozen behavior profile, and estimator candidate 08 unless noted.

| Version | Change | Final checkpoint | Distinct hard terminations | Result |
|---|---|---|---:|---|
| C-v4 | Separate auxiliary intent + PPO gait | `model_24.pt` (`2bf10c38...`) | 78/512 | Failed recovery safety |
| C-v4 GT diagnostic | Same weights, simulator GT velocity | same | 0/512 | Proved estimator closure was the cause |
| C-v5 | Estimator closed-loop training | `model_24.pt` (`0924dddb...`) | 107/512 | Worse |
| C-v6 | Fall and recovery costs | `model_24.pt` (`f3e87ce0...`) | 61/512 | Improved, still failed |
| C-v7 | Nonnegative stride/smoothing | `model_24.pt` (`327df6aa...`) | 17/512 | Near gate |
| C-v8 | Smoothing gain 10 | `model_24.pt` (`096b4ebc...`) | 63/512 | Failed; gain sweep stopped |
| C-v7 continuation | Fixed 25-more-iteration continuation | `model_48.pt` (`888bc682...`) | 0/512 | Behavior passed |
| C-v9 | Degradation foot-slip reward continuation | `model_62.pt` (`8f69781a...`) | 0/512 | Slip worsened; retired |

C-v7 model48 passed every behavior check, but its degraded stance-foot slip was
`0.14052 m/s`, versus the model1450 supervisor baseline `0.11145 m/s`; the
allowed 1.10 ratio corresponds to approximately `0.12260 m/s`. Its gait-value
gate therefore failed only `degraded_invalid_no_gait_metric_regressed`.

## Causal ablations

The following are diagnostics, not candidates:

- Zero crouch kept 0/512 hard terminations but only improved degraded slip to
  `0.13775 m/s`; crouch was not the main cause.
- Zero stride reduced degraded slip to `0.11692 m/s`, but caused 78/512 hard
  terminations, almost all during recovery. Learned recovery stride is safety
  critical.
- Scaling stride to 0.5 still caused 9/512 hard terminations and left slip at
  `0.13069 m/s`; one global stride scale cannot satisfy both gates.
- The first late, abrupt degradation-only envelope was safe at 1/512 but raised
  slip to `0.16792 m/s`. A continuous linear envelope reduced it to
  `0.13103 m/s`; the final cubic envelope reduced it below the frozen limit.

## Passing candidate

Checkpoint:

`logs/rsl_rl/anymal_d_locomotion_slam_confidence_constrained_gait_v1/2026-08-13_18-18-26_nonnegative_smoothing_recovery_safe_ppo_gait/model_48.pt`

SHA-256:

`888bc682eda65e2c427eccd445d8b9cb1e787635b5da86e6fa8a85b83fda4fa0`

The checkpoint is evaluated with
`AnymalDLocomotionSlamConfidencePhaseSeparatedGaitPPORunnerCfg`; the candidate
is the checkpoint plus this policy architecture/config, not a modified binary
checkpoint.

Behavior gate:

- Passed all checks.
- 2/512 distinct environments hard-terminated (`0.390625%`, limit `1%`).
- Degraded-late stance-foot slip: `0.117835 m/s`.
- Degraded/recovery mean gait coordinates were approximately
  `[0.0006, -0.0542, 0.0155, 0.1805]` and
  `[0.1451, -0.0602, 0.0185, 0.1999]`, demonstrating the intended phase split.

Gait-value A/B against model1450 plus the identical confidence supervisor:

- Passed all nine checks.
- Degraded+invalid composite ratio: `0.770354` (limit `0.95`; lower is better).
- Worst degraded/invalid individual ratio: stance-foot slip `1.05730`
  (limit `1.10`).
- Healthy and recovery tracking/gait non-regression checks passed.

Machine-readable reports:

- `docs/validation/slam_confidence_phase_separated_model48_behavior.json`
- `docs/validation/slam_confidence_phase_separated_model48_gait_value.json`

## Verification and boundary

The full Python suite after the interactive deployment entry is
`336 passed, 3 skipped`;
the three skips are the existing Isaac-Sim-runtime-only external project tests.
`git diff --check` passes.

Simulation behavior, export parity, DDS fault handling and one hardest
right-lateral live cell per backend have passed. The full FAST-LIO2/LIO-SAM
replay/live matrix, manual LIO-SAM operation, efficiency/false-stop analysis,
physical ANYmal-D and formal promotion remain open. No replacement, commit, or
push was performed.
