# SLAM confidence full-policy joint training v1

Date: 2026-08-26
Status: static training stack and non-learning simulation preflight passed; PPO execution closed

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

The execution-closed budget is also frozen: J1 and J2 each use independent seeds 1450/1451/1452,
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

## Current gates and next authorized boundary

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

The one-shot preflight authorization flags were closed immediately afterward. No PPO iteration,
live ROS wiring, default switch or physical robot operation was executed. The next boundary is a
separate explicit authorization for the already-frozen fixed-budget J1/J2 training; a preflight PASS
only proves that training inputs, bootstrap behavior and reward scales are wired as specified, not
that the learned policy will improve SLAM.
