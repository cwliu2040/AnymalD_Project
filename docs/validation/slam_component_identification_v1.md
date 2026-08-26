# Richer component identification v1

Date: 2026-08-24

## Purpose and frozen boundary

The first component pilot showed that anisotropic command scaling can outperform uniform slowdown,
but its compact offline selector did not generalize. This fresh development protocol therefore
identified translation and yaw effects at two amplitudes before fitting another selector. It did
not train PPO, replace model1450, change the default controller, or authorize physical use.

Six arms were frozen before collection: control, uniform 0.75, translation-only reductions to 0.75
and 0.875, and yaw-only reductions to 0.75 and 0.875. Both FAST-LIO2 and LIO-SAM used right and left
curve profiles. Block 576 was the 12-run wiring smoke, blocks 577--580 supplied 96 balanced
identification runs, and blocks 581--584 were reserved for external validation.

## Execution result

The block-576 smoke completed 12/12 with `WIRING_PASS` and no safety events. The identification
matrix then completed 96/96 with `PASS`. It contained six safety events, but no arm showed repeated
arm-specific excess over its matched control. Every backend/profile stratum had target variation,
balanced arm counts, and both reduction amplitudes.

Mean pulse-window hazard by arm showed substantial direction and block dependence. Selected examples:

- FAST-LIO2 right curve: control `0.500`, uniform `0.309`, translation 0.75 `0.250`, yaw 0.75 `0.342`;
- FAST-LIO2 left curve: control `0.566`, uniform `0.473`, translation 0.75 `0.559`, yaw 0.75 `0.586`;
- LIO-SAM right curve: control `0.597`, uniform `0.882`, translation 0.75 `0.974`, yaw 0.75 `0.755`;
- LIO-SAM left curve: control `1.000`, uniform `0.862`, translation 0.875 `0.711`, yaw 0.75 `0.842`.

Thus the dataset is suitable for a refrozen development fit, but no fixed arm is uniformly best and
the `PASS` is not a formal controller claim. The identification manifest SHA-256 is
`e7869119bc61c985e70d562aff902213dc1845992e4aaa12db05cfce58ab317c`; the decision SHA-256 is
`392d4fd50a26f3730b58f2e5e6f50b246944651c853cbb7f3ebf14447bad18f5`.

## Final disposition

The subsequent v2 offline model failed its internal gate, so blocks 581--584 remain untouched.
Live execution, ROS policy wiring, C2/PPO training, default switching, and physical robot use are
all unauthorized.
