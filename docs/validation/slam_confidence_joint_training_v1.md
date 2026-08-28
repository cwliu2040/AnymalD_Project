# SLAM confidence full-policy joint training v1

Date: 2026-08-26
Status: fixed-budget PPO complete; post-training motion audit FAIL; stopped before SLAM evaluation

## Question and attribution

The new C is no longer a velocity limiter or a fixed residual formula. It fine-tunes the complete
12-D locomotion action from model1450 while preserving the exact original XYZ/yaw command. The
formal comparison is:

- J0: frozen model1450, legacy 48-D input;
- J1: full-policy fine-tuning with current 48-D input plus the same 20-frame proprioception,
  command and previous-action history, but neutral localization channels;
- J2: identical to J1 except the 20 history frames contain actual backend-neutral
  confidence/validity/age.

J1-J0 measures generic history/smoothness fine-tuning. J2-J1 is the only localization-aware
contribution. J1 and J2 share architecture, input width, command distribution, reward,
curriculum, behavior anchor and evaluation budget.

## Implemented static contracts

The policy input is 1068-D: current legacy 48-D observation followed by 20 oldest-to-newest
51-D frames. Each frame contains the canonical legacy terms followed by confidence, binary
tracking validity and normalized age. Isaac Lab's per-term circular buffer supplies causal
history and repeats the first observation at reset. J1 replaces only the three localization
values with `[1, 1, 0]`; it does not remove temporal history or change network width.

`full_policy_bootstrap.py` generalizes the earlier hard-coded 48-to-51 warm start. For the actual
model1450 checkpoint it verified:

- columns 0..47 of the 1068-D actor first layer are exact source copies;
- columns 48..1067 are exact zero;
- all later actor layers and action noise are exact source copies;
- the separate frozen 48-D behavior-reference actor is an exact source copy;
- critic tensors remain at fresh initialization and inherited optimizer state is rejected;
- arbitrary non-zero history produces bitwise-identical target and model1450 actions at
  initialization.

The tensor verification ran in the Isaac Lab Python environment without launching simulation. The
bootstrap/anchor/motion-core file reports `9 passed`; the full PyTorch subset including existing
policy tests reports `36 passed`. The system-Python suite, excluding the publication runner that deliberately
requires a clean worktree and the separately executed PyTorch files, reports `503 passed, 15
skipped`, with the existing SciPy/NumPy warning.

## Behavior anchoring

`AnchoredFullPolicyActorCritic` keeps a trainable dense 1068-D actor and a frozen model1450
reference. `BehaviorAnchoredPPO` performs a common non-zero action-MSE anchor update after every
PPO update. The coefficient is fixed at 0.25 for both J1 and J2 and never becomes zero. This is a
behavior-retention target, not a residual architecture: the trainable actor can still change all
12 joint actions. A tensor test perturbs the full actor, verifies that the anchor loss moves it
toward model1450, and verifies that the reference parameters remain unchanged.

Model1450 action noise is copied exactly and then frozen for both arms. This prevents trainable
exploration variance from bypassing the mean-action behavior anchor; it does not freeze any of the
12-D actor mean mapping. A real RSL-RL `RolloutStorage` plus `BehaviorAnchoredPPO` synthetic rollout,
return computation and update completes with finite PPO/anchor losses, clears storage correctly,
and leaves both the reference actor and action noise unchanged.

## Reward and curriculum semantics

The task inherits `AnymalDLocomotionRecoveryV05RewardsCfg`, including its original-command linear
and yaw tracking. It does not use the old confidence-scaled tracking functions and does not add an
invalid-state stop reward. The joint-training command generator uses a full bounded random envelope
plus forward/reverse/lateral/free-yaw/mixed edge cases, with at most 10% standing commands; it does
not replay a hard-coded warehouse route.

The additional motion objectives penalize angular acceleration, linear jerk, scan-time translation
distortion from acceleration, and scan-time rotation distortion from roll/pitch motion plus angular
acceleration. They do not penalize constant requested translation or constant requested yaw rate.
The privileged localization severity changes only the weight of these objectives and is shared by
J1/J2; it never scales the command target.

The reward equations now live in the Isaac-independent PyTorch module
`joint_training_motion_core.py`. Direct tensor tests prove that constant translation and constant
requested yaw rate have exactly zero new scan-distortion cost when acceleration/jerk are zero, while
roll/pitch motion and non-constant scan motion remain penalized. Invalid confidence/validity/age
contracts fail closed.

The common prespecified curriculum retains the model1450 anchor and original locomotion rewards at
the start, enables angular/scan distortion objectives at step 600, and enables linear jerk at step
1200. Joint-limit violation protection is enabled from the start. The authorized non-learning
simulation preflight verified their numerical scale. These weights remain frozen and must not be
changed after observing future J1/J2 training outcome data.

The fixed budget is frozen: J1 and J2 each use independent seeds 1450/1451/1452,
4096 environments, 24-step rollouts and at most 300 iterations. Iteration 50 is a common safety and
collapse futility audit, not an efficacy or seed-selection point. Formal evaluation may not
cherry-pick one successful seed and J1/J2 must consume equal environment steps.

## Frozen offline evaluation

`evaluate_slam_confidence_joint_training_block.py` consumes paired run records and emits exactly the
fields used by the go/no-go reducer. It requires:

- exact requested-command and timestamp identity;
- moving linear speed and realized yaw-rate differences no greater than 0.05;
- candidate linear/yaw progress at least 98% of its comparator;
- no stopped-fraction increase and candidate stopped fraction no greater than 10%;
- a comparator that achieves at least 50% of requested accumulated progress, preventing a
  both-policies-stop false pass;
- paired gait/safety noninferiority plus independent absolute physical limits;
- event-aligned body and LiDAR evaluation using the comparator's observed localization transition
  as the shared physical window;
- at least 5% improvement in one body mechanism and one LiDAR mechanism, no greater than 10%
  regression, and no greater than 10% tracking regression;
- at least one held-out SLAM-direction improvement among future unusability, GT drift and point
  support, with no greater than 10% regression;
- no safety-event excess.

The synthetic evaluator test passes only when command/speed are matched and body, LiDAR and SLAM
all improve; an exact-command mutation fails integrity and matched-motion attribution.

## Fixed-budget PPO execution

The user separately authorized the frozen PPO feasibility run. J1 and J2 each completed seeds
1450, 1451 and 1452 with 4096 environments, 24 rollout steps and 300 updates. The common
iteration-50 futility audit found no fall, termination, non-finite value or hard stop collapse; all
audited J0/J1/J2 fixed-command runs had survival fraction 1.0. J1 seed1450 forward progress was near
but below the formal 0.98 ratio and remained a warning, not a reason to cherry-pick or stop one arm.

Training resumed from each `model_49.pt` for updates 50 through 299. The runner now restores Isaac
Lab's global curriculum counter from completed updates before learning, advances past the last saved
iteration, rejects a continuation exceeding 300 updates and records this lineage in the run
manifest. This prevents a resumed job from silently restarting the motion-reward curriculum or
repeating update 49.

All six `model_299.pt` files are present inside the repository's ignored training-log tree, report
checkpoint iteration 299 and contain only finite model tensors. Each arm/seed consumed 29,491,200
transitions; total J1/J2 execution was 176,947,200 transitions with equal budgets. Final on-policy
mean rewards were J1 `[9.76828, 12.4662, 11.4508]` and J2
`[10.8065, 11.0249, 11.5415]` for seeds 1450/1451/1452. These values show finite completed
optimization only: they are not command-matched SLAM evidence and cannot select an arm or seed.

Both one-shot PPO authorization flags were closed after completion. Live ROS wiring, default
switch, physical robot operation and blocks581..584/602..605 remained closed and unexecuted.

## Post-training fixed-command motion audit

Before spending FAST-LIO2/LIO-SAM budget, a frozen 36-run rejection matrix evaluated J0/J1/J2 for
three seeds and forward, lateral, pure-yaw and mixed commands. Every run used 128 environments, 750
steps and a 100-step warmup. The runner collected survival and termination, matched speed/progress,
stopped fraction, cadence/contact/duty factor, body height, stance width, joint margin, clearance,
torque, energy, slip, roll/pitch rate, tracking error, acceleration/jerk and LiDAR scan-motion
proxies. It did not modify policy, command, reward or training state.

The matrix failed the prespecified pre-SLAM gate. J1-J0 passing-seed counts for
forward/lateral/yaw/mixed were `0/0/1/0`, with two required. J2-J1 was `0/0/0/0`. The ignored
machine-readable summary is
`outputs/slam_confidence_joint_training_v1/post_training_motion_audit/summary.json`, SHA-256
`8222347c67ee6c0da8394eac21ce0a2bcf3036d3ccdf35f168a542d52e2f9176`.

The failure is informative rather than a null training result. Relative to J0, J1 reduced mean
roll/pitch-rate RMS by 11.4--24.9%, angular-acceleration RMS by 21.9--32.8% and LiDAR rotation proxy
by 21.5--31.9% across the four profiles. It therefore learned a generic smoothing mechanism.
However, forward/lateral/mixed tracking errors regressed materially, stance width increased by about
9--17% in those profiles, energy or slip gates failed repeatedly, and J1 seed1452 forward had one
termination among 128 environments. The frozen attribution and safety gates correctly reject this
tradeoff.

J2 did not add repeatable localization-aware value over J1. It sometimes recovered tracking, but
mean roll/pitch, angular acceleration and LiDAR rotation proxy generally regressed relative to J1;
all three mixed-command seeds retained less than 98% of J1 linear progress. All J2 fixed-command
runs survived, but body/LiDAR and paired gait gates still failed. This is not limiter collapse—the
main J1 behavior was often faster than J0—but it is an unacceptable smoothing/tracking/posture
tradeoff and no repeatable J2-J1 mechanism.

One Isaac Sim child process crashed during startup before J2 seed1452 lateral was loaded. The
resumable runner preserved the first 29 reports and reran the missing cell successfully; this was
not counted as a policy safety event.

## Current gates and next boundary

Static contract, actual-checkpoint bootstrap, behavior anchoring, offline evaluator tests and the
fixed non-learning runtime preflight are complete. The runner checks both authorization flags before
launching Isaac, never imports a PPO algorithm or runner, and isolates J1/J2 in separate Isaac
processes to avoid process-global SimulationContext teardown contamination. Each arm ran 8
environments for exactly 100 steps with seed 1450 and the zero-history-column model1450 bootstrap.

Both arms passed. Their 1068-D observation layout, reset/history flattening, exact original command,
J1 neutral localization and J2 confidence/validity/age contracts were valid at every checked step.
Runtime model1450 bootstrap action error was exactly `0.0`; neither arm produced a termination or
truncation. The projected combined new-motion reward had p99 `0.0473193` and maximum `0.0622891`
per step in both arms, below the frozen limits `1.0` and `5.0`. The merged ignored report is
`outputs/slam_confidence_joint_training_v1/nonlearning_preflight.json`, SHA-256
`6c3ab914d91d57365ba78397c13b36828a5b413041fcdb848e5d11d20306b81c`, and records
`ppo_constructed_or_run: false`.

The preflight PASS established wiring, not efficacy. Fixed-budget PPO and the post-training motion
audit are complete, and all one-shot gates are closed. Because the cheaper motion gate failed,
FAST-LIO2/LIO-SAM evaluation was not run. The current J1/J2 checkpoints must not proceed to SLAM,
teacher/adaptation, ROS wiring or deployment. Any further attempt is a new architecture/training
design decision; it may not post-hoc relax these frozen gates or describe J1's generic smoothing
tradeoff as localization-aware success.
