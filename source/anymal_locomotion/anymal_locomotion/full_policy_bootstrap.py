"""Deterministic actor-only bootstrap for history-conditioned policies.

This module is intentionally independent of Isaac Lab and ROS 2 so the exact
model1450 preservation contract can be tested without launching simulation.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import torch


LEGACY_ACTOR_INPUT_DIM = 48


def _actor_keys(state: Mapping[str, torch.Tensor]) -> set[str]:
    return {
        name
        for name in state
        if name == "std" or name.startswith("actor.")
    }


def bootstrap_dense_actor_state(
    source_state: Mapping[str, torch.Tensor],
    target_state: MutableMapping[str, torch.Tensor],
    *,
    legacy_input_dim: int = LEGACY_ACTOR_INPUT_DIM,
    expected_target_input_dim: int | None = None,
) -> dict[str, Any]:
    """Copy a legacy dense actor into a wider actor without changing behavior.

    The actor parameter namespace must match exactly.  Only the input width of
    ``actor.0.weight`` may differ: legacy columns are copied bit-for-bit and all
    new columns are set to exact zero.  Critic tensors are never read or
    modified by this function.
    """
    if legacy_input_dim <= 0:
        raise ValueError("legacy_input_dim must be positive")
    source_actor_keys = _actor_keys(source_state)
    target_actor_keys = _actor_keys(target_state)
    if source_actor_keys != target_actor_keys:
        missing = sorted(source_actor_keys - target_actor_keys)
        extra = sorted(target_actor_keys - source_actor_keys)
        raise ValueError(
            f"source and target actor parameter names differ: missing={missing}, extra={extra}"
        )
    first_key = "actor.0.weight"
    if first_key not in source_actor_keys:
        raise ValueError(f"actor state lacks {first_key}")
    source_first = source_state[first_key]
    target_first = target_state[first_key]
    if source_first.ndim != 2 or target_first.ndim != 2:
        raise ValueError("actor first-layer weights must be matrices")
    if source_first.shape[1] != legacy_input_dim:
        raise ValueError(
            f"source actor input is {source_first.shape[1]}, expected {legacy_input_dim}"
        )
    if source_first.shape[0] != target_first.shape[0]:
        raise ValueError("source and target actor first-layer output widths differ")
    target_input_dim = int(target_first.shape[1])
    if target_input_dim <= legacy_input_dim:
        raise ValueError("target actor must add at least one input column")
    if expected_target_input_dim is not None and target_input_dim != expected_target_input_dim:
        raise ValueError(
            f"target actor input is {target_input_dim}, expected {expected_target_input_dim}"
        )

    copied_keys: list[str] = []
    with torch.no_grad():
        for name in sorted(source_actor_keys):
            source = source_state[name].to(
                device=target_state[name].device,
                dtype=target_state[name].dtype,
            )
            target = target_state[name]
            if name == first_key:
                target[:, :legacy_input_dim].copy_(source)
                target[:, legacy_input_dim:].zero_()
            else:
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(f"actor parameter shape mismatch for {name}")
                target.copy_(source)
            copied_keys.append(name)

    if not torch.equal(
        target_state[first_key][:, :legacy_input_dim].detach().cpu(),
        source_first.detach().cpu().to(dtype=target_state[first_key].dtype),
    ):
        raise RuntimeError("legacy actor input columns were not copied exactly")
    if torch.count_nonzero(target_state[first_key][:, legacy_input_dim:]).item() != 0:
        raise RuntimeError("new actor input columns are not exactly zero")
    return {
        "source_actor_input_dimension": legacy_input_dim,
        "target_actor_input_dimension": target_input_dim,
        "legacy_input_columns": {"start": 0, "stop_exclusive": legacy_input_dim},
        "new_input_columns": {
            "start": legacy_input_dim,
            "stop_exclusive": target_input_dim,
        },
        "new_input_columns_initialization": "exact_zero",
        "copied_parameters": copied_keys,
    }


def bootstrap_frozen_reference_actor_state(
    source_state: Mapping[str, torch.Tensor],
    target_state: MutableMapping[str, torch.Tensor],
    *,
    target_prefix: str = "reference_actor.",
) -> list[str]:
    """Copy ``actor.*`` into an equal-shape frozen reference actor namespace."""
    source_keys = sorted(name for name in source_state if name.startswith("actor."))
    if not source_keys:
        raise ValueError("source state has no actor parameters")
    copied: list[str] = []
    with torch.no_grad():
        for source_name in source_keys:
            suffix = source_name.removeprefix("actor.")
            target_name = f"{target_prefix}{suffix}"
            if target_name not in target_state:
                raise ValueError(f"target state lacks frozen reference parameter {target_name}")
            source = source_state[source_name].to(
                device=target_state[target_name].device,
                dtype=target_state[target_name].dtype,
            )
            if tuple(source.shape) != tuple(target_state[target_name].shape):
                raise ValueError(f"reference actor parameter shape mismatch for {target_name}")
            target_state[target_name].copy_(source)
            copied.append(target_name)
    expected_target_keys = sorted(
        name for name in target_state if name.startswith(target_prefix)
    )
    if copied != expected_target_keys:
        extra = sorted(set(expected_target_keys) - set(copied))
        raise ValueError(f"reference actor has unexpected parameters: {extra}")
    return copied


def verify_dense_actor_output_parity(
    source_actor: torch.nn.Module,
    target_actor: torch.nn.Module,
    legacy_observations: torch.Tensor,
    added_inputs: torch.Tensor,
    *,
    exact: bool = True,
) -> bool:
    """Verify widened-actor outputs equal legacy outputs for arbitrary history.

    Exact equality is the formal bootstrap requirement.  The optional tolerant
    mode exists only for device kernels whose matrix multiplication ordering is
    known not to be bitwise stable.
    """
    if legacy_observations.ndim != 2 or added_inputs.ndim != 2:
        raise ValueError("parity inputs must be rank-two batches")
    if legacy_observations.shape[0] != added_inputs.shape[0]:
        raise ValueError("parity input batch sizes differ")
    combined = torch.cat((legacy_observations, added_inputs), dim=-1)
    with torch.no_grad():
        source_output = source_actor(legacy_observations)
        target_output = target_actor(combined)
    if exact:
        return torch.equal(source_output, target_output)
    return torch.allclose(source_output, target_output, rtol=0.0, atol=1.0e-7)


def validate_fresh_optimizer_state(optimizer_state: Mapping[Any, Any]) -> None:
    """Reject inherited optimizer moments for an actor-only warm start."""
    if optimizer_state:
        raise RuntimeError("actor-only warm-start requires a fresh optimizer")


def critic_parameter_names(state: Mapping[str, torch.Tensor]) -> Sequence[str]:
    """Return a stable critic key list for caller-side before/after hashing."""
    return tuple(sorted(name for name in state if name.startswith("critic.")))
