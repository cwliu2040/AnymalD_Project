"""ANYmal symmetry augmentation in the project canonical joint order."""

from __future__ import annotations

import torch

from anymal_locomotion.policy_contract import CANONICAL_JOINT_ORDER

_JOINT_INDEX = {
    name: index for index, name in enumerate(CANONICAL_JOINT_ORDER)
}


def _switch_joints(
    joint_data: torch.Tensor,
    leg_mapping: dict[str, str],
    negate_suffixes: tuple[str, ...],
) -> torch.Tensor:
    transformed = torch.zeros_like(joint_data)
    for destination_leg, source_leg in leg_mapping.items():
        for suffix in ("HAA", "HFE", "KFE"):
            destination = _JOINT_INDEX[f"{destination_leg}_{suffix}"]
            source = _JOINT_INDEX[f"{source_leg}_{suffix}"]
            sign = -1.0 if suffix in negate_suffixes else 1.0
            transformed[..., destination] = sign * joint_data[..., source]
    return transformed


def _left_right(joint_data: torch.Tensor) -> torch.Tensor:
    return _switch_joints(
        joint_data,
        {"LF": "RF", "LH": "RH", "RF": "LF", "RH": "LH"},
        ("HAA",),
    )


def _front_back(joint_data: torch.Tensor) -> torch.Tensor:
    return _switch_joints(
        joint_data,
        {"LF": "LH", "LH": "LF", "RF": "RH", "RH": "RF"},
        ("HFE", "KFE"),
    )


def _transform_observation(
    observation: torch.Tensor,
    *,
    left_right: bool,
    front_back: bool,
) -> torch.Tensor:
    if observation.shape[1] not in (48, 51):
        raise ValueError(
            f"symmetry expects a 48-D or 51-D policy observation, received {observation.shape}"
        )
    transformed = observation.clone()
    if left_right:
        transformed[:, :3] *= transformed.new_tensor([1.0, -1.0, 1.0])
        transformed[:, 3:6] *= transformed.new_tensor([-1.0, 1.0, -1.0])
        transformed[:, 6:9] *= transformed.new_tensor([1.0, -1.0, 1.0])
        transformed[:, 9:12] *= transformed.new_tensor([1.0, -1.0, -1.0])
        switch = _left_right
    else:
        switch = lambda value: value
    for start in (12, 24, 36):
        transformed[:, start : start + 12] = switch(
            transformed[:, start : start + 12]
        )
    if front_back:
        transformed[:, :3] *= transformed.new_tensor([-1.0, 1.0, 1.0])
        transformed[:, 3:6] *= transformed.new_tensor([1.0, -1.0, -1.0])
        transformed[:, 6:9] *= transformed.new_tensor([-1.0, 1.0, 1.0])
        transformed[:, 9:12] *= transformed.new_tensor([-1.0, 1.0, -1.0])
        for start in (12, 24, 36):
            transformed[:, start : start + 12] = _front_back(
                transformed[:, start : start + 12]
            )
    # Offsets 48..50 are backend-neutral scalars and remain invariant under
    # left/right and front/back robot symmetries.
    return transformed


@torch.no_grad()
def compute_symmetric_states(env, obs=None, actions=None):
    """Return original, left-right, front-back and diagonal samples."""
    del env
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_augmented = obs.repeat(4)
        policy = obs["policy"]
        obs_augmented["policy"][:batch_size] = policy
        obs_augmented["policy"][batch_size : 2 * batch_size] = (
            _transform_observation(
                policy,
                left_right=True,
                front_back=False,
            )
        )
        obs_augmented["policy"][2 * batch_size : 3 * batch_size] = (
            _transform_observation(
                policy,
                left_right=False,
                front_back=True,
            )
        )
        obs_augmented["policy"][3 * batch_size :] = (
            _transform_observation(
                policy,
                left_right=True,
                front_back=True,
            )
        )
    else:
        obs_augmented = None
    if actions is not None:
        batch_size = actions.shape[0]
        actions_augmented = actions.repeat(4, 1)
        actions_augmented[batch_size : 2 * batch_size] = _left_right(
            actions
        )
        actions_augmented[2 * batch_size : 3 * batch_size] = _front_back(
            actions
        )
        actions_augmented[3 * batch_size :] = _front_back(
            _left_right(actions)
        )
    else:
        actions_augmented = None
    return obs_augmented, actions_augmented
