"""RSL-RL PPO baseline for project-owned ANYmal-D Flat locomotion."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg import (
    AnymalDFlatPPORunnerCfg,
)
from .. import mdp


@configclass
class AnymalDLocomotionFlatPPORunnerCfg(AnymalDFlatPPORunnerCfg):
    """Official ANYmal-D Flat PPO parameters with a project-local experiment name."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "anymal_d_locomotion_v1"
        self.max_iterations = 1000


@configclass
class AnymalDLocomotionRobustPPORunnerCfg(
    AnymalDLocomotionFlatPPORunnerCfg
):
    """Conservative symmetry-augmented fine-tuning configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.max_iterations = 300
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 0.003
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mdp.compute_symmetric_states,
        )
