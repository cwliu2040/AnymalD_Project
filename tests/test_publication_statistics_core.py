from __future__ import annotations

import pytest

from anymal_locomotion_ros2.publication_statistics_core import backend_interaction, paired_contrast, paired_ratio


def _records() -> list[dict]:
    result = []
    for block in range(5):
        for arm, value in (("C", 0.8 + block * 0.01), ("D", 1.0 + block * 0.01)):
            result.append(
                {
                    "identity": {
                        "backend": "fastlio2", "profile": "route",
                        "condition": "gradual_support_loss", "paired_block_id": block, "arm": arm,
                    },
                    "metrics": {"slip": value, "efficiency": value},
                }
            )
    return result


def _two_backend_records() -> list[dict]:
    result = []
    for backend, offset in (("fastlio2", 0.0), ("liosam", 0.1)):
        for record in _records():
            copied = {
                "identity": {**record["identity"], "backend": backend},
                "metrics": {"slip": record["metrics"]["slip"] + (offset if record["identity"]["arm"] == "C" else 0.0)},
            }
            result.append(copied)
    return result


def test_cluster_bootstrap_is_deterministic_and_paired() -> None:
    first = paired_contrast(_records(), treatment="C", control="D", metric="slip", resamples=200)
    second = paired_contrast(_records(), treatment="C", control="D", metric="slip", resamples=200)
    assert first == second
    assert first["pair_count"] == 5
    assert first["mean_difference"] == pytest.approx(-0.2)
    assert first["mean_difference_95pct_cluster_bootstrap"]["cluster_count"] == 5


def test_paired_ratio_reports_noninferiority_scale() -> None:
    result = paired_ratio(_records(), treatment="C", control="D", metric="efficiency", resamples=200)
    assert 0.80 < result["mean_ratio"] < 0.82


def test_incomplete_pair_fails_closed() -> None:
    records = _records()[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        paired_contrast(records, treatment="C", control="D", metric="slip", resamples=200)


def test_backend_interaction_is_difference_of_paired_effects() -> None:
    result = backend_interaction(
        _two_backend_records(), treatment="C", control="D", metric="slip", resamples=200,
    )
    assert result["stratum_count"] == 5
    assert result["mean_difference_in_paired_effects"] == pytest.approx(-0.1)
