"""Simulation-based precision planning from disjoint paired pilot blocks."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def simulate_stratified_mean_interval(
    values_by_profile: dict[str, Sequence[float]], *, paired_blocks: int,
    resamples: int, seed: int,
) -> dict[str, float | int]:
    if paired_blocks <= 0 or resamples < 1000:
        raise ValueError("planning requires positive blocks and at least 1000 resamples")
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in values_by_profile.items()}
    if not arrays or any(len(value) < 5 for value in arrays.values()):
        raise ValueError("every profile requires at least five disjoint pilot blocks")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("pilot effects must be finite")
    rng = np.random.default_rng(seed)
    profile_means = []
    for values in arrays.values():
        indices = rng.integers(0, len(values), size=(resamples, paired_blocks))
        profile_means.append(np.mean(values[indices], axis=1))
    simulated = np.mean(np.stack(profile_means, axis=1), axis=1)
    lower, upper = np.quantile(simulated, (0.025, 0.975))
    return {
        "profile_count": len(arrays),
        "paired_blocks": paired_blocks,
        "resamples": resamples,
        "pilot_mean": float(np.mean([np.mean(value) for value in arrays.values()])),
        "simulated_lower": float(lower),
        "simulated_upper": float(upper),
        "simulated_halfwidth": float((upper - lower) / 2.0),
    }


def first_precision_qualified_block_count(
    values_by_profile: dict[str, Sequence[float]], *, candidates: Sequence[int],
    halfwidth_target: float, resamples: int, seed: int,
    lower_bound: float | None = None,
) -> dict:
    if halfwidth_target <= 0.0:
        raise ValueError("precision target must be positive")
    evaluations = []
    selected = None
    for index, blocks in enumerate(candidates):
        result = simulate_stratified_mean_interval(
            values_by_profile, paired_blocks=int(blocks), resamples=resamples, seed=seed + index,
        )
        precision_passed = result["simulated_halfwidth"] <= halfwidth_target
        lower_bound_passed = lower_bound is None or result["simulated_lower"] >= lower_bound
        result.update(
            {
                "precision_passed": precision_passed,
                "lower_bound": lower_bound,
                "lower_bound_passed": lower_bound_passed,
                "passed": precision_passed and lower_bound_passed,
            }
        )
        evaluations.append(result)
        if result["passed"] and selected is None:
            selected = int(blocks)
    return {
        "halfwidth_target": halfwidth_target,
        "selected_paired_block_count": selected,
        "evaluations": evaluations,
        "passed": selected is not None,
    }


def log_ratio(treatment: float, control: float) -> float:
    if treatment <= 0.0 or control <= 0.0:
        raise ValueError("efficiency values must be positive for log ratio planning")
    return math.log(treatment / control)
