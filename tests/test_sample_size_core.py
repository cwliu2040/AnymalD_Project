from __future__ import annotations

import pytest

from anymal_locomotion_ros2.sample_size_core import (
    first_precision_qualified_block_count,
    log_ratio,
    simulate_stratified_mean_interval,
)


def _pilot() -> dict[str, list[float]]:
    return {
        "route_a": [-0.20, -0.18, -0.22, -0.19, -0.21],
        "route_b": [-0.16, -0.15, -0.17, -0.14, -0.18],
    }


def test_simulated_precision_is_deterministic() -> None:
    first = simulate_stratified_mean_interval(_pilot(), paired_blocks=5, resamples=2000, seed=7)
    second = simulate_stratified_mean_interval(_pilot(), paired_blocks=5, resamples=2000, seed=7)
    assert first == second
    assert first["simulated_upper"] < 0.0


def test_planner_selects_first_qualified_candidate() -> None:
    result = first_precision_qualified_block_count(
        _pilot(), candidates=[5, 10, 20], halfwidth_target=0.02,
        resamples=2000, seed=7,
    )
    assert result["passed"]
    assert result["selected_paired_block_count"] in {5, 10, 20}


def test_planner_requires_five_pilot_blocks_per_profile() -> None:
    with pytest.raises(ValueError, match="five"):
        simulate_stratified_mean_interval({"route": [1.0] * 4}, paired_blocks=5, resamples=2000, seed=0)


def test_log_ratio_requires_positive_efficiency() -> None:
    assert log_ratio(0.95, 1.0) == pytest.approx(-0.051293294)
    with pytest.raises(ValueError):
        log_ratio(0.0, 1.0)
