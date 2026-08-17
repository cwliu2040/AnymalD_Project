"""Paired cluster-bootstrap statistics for publication run records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np


PAIR_KEYS = ("backend", "profile", "condition", "paired_block_id")
CLUSTER_KEYS = ("paired_block_id", "profile")


def paired_values(
    records: Sequence[dict[str, Any]], *, treatment: str, control: str, metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        identity = record["identity"]
        arm = str(identity["arm"])
        if arm not in {treatment, control}:
            continue
        key = tuple(identity[field] for field in PAIR_KEYS)
        if arm in grouped[key]:
            raise ValueError(f"duplicate arm {arm} for pair {key}")
        grouped[key][arm] = record
    pairs: list[dict[str, Any]] = []
    incomplete = [key for key, arms in grouped.items() if set(arms) != {treatment, control}]
    if incomplete:
        raise ValueError(f"incomplete treatment/control pairs: {incomplete[:3]}")
    for key, arms in sorted(grouped.items()):
        treatment_value = arms[treatment]["metrics"].get(metric)
        control_value = arms[control]["metrics"].get(metric)
        if treatment_value is None or control_value is None:
            continue
        identity = arms[treatment]["identity"]
        pairs.append(
            {
                "pair": dict(zip(PAIR_KEYS, key, strict=True)),
                "cluster": tuple(identity[field] for field in CLUSTER_KEYS),
                "treatment": float(treatment_value),
                "control": float(control_value),
                "difference": float(treatment_value) - float(control_value),
            }
        )
    if not pairs:
        raise ValueError(f"no complete numeric pairs for metric {metric}")
    return pairs


def _cluster_bootstrap(
    pairs: Sequence[dict[str, Any]], statistic: Callable[[np.ndarray], float],
    *, resamples: int, seed: int,
) -> dict[str, Any]:
    by_cluster: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for pair in pairs:
        by_cluster[pair["cluster"]].append(float(pair["difference"]))
    clusters = sorted(by_cluster)
    if resamples < 100:
        raise ValueError("cluster bootstrap requires at least 100 resamples")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        values = np.asarray(
            [value for cluster_index in sampled for value in by_cluster[clusters[int(cluster_index)]]]
        )
        draws[index] = statistic(values)
    return {
        "cluster_count": len(clusters),
        "resamples": resamples,
        "seed": seed,
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
    }


def paired_contrast(
    records: Sequence[dict[str, Any]], *, treatment: str, control: str, metric: str,
    resamples: int = 10_000, seed: int = 0,
) -> dict[str, Any]:
    pairs = paired_values(records, treatment=treatment, control=control, metric=metric)
    differences = np.asarray([pair["difference"] for pair in pairs])
    return {
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "pair_count": len(pairs),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "mean_difference_95pct_cluster_bootstrap": _cluster_bootstrap(
            pairs, np.mean, resamples=resamples, seed=seed,
        ),
        "median_difference_95pct_cluster_bootstrap": _cluster_bootstrap(
            pairs, np.median, resamples=resamples, seed=seed + 1,
        ),
    }


def paired_ratio(
    records: Sequence[dict[str, Any]], *, treatment: str, control: str, metric: str,
    resamples: int = 10_000, seed: int = 0,
) -> dict[str, Any]:
    pairs = paired_values(records, treatment=treatment, control=control, metric=metric)
    ratios = []
    ratio_pairs = []
    for pair in pairs:
        if pair["control"] <= 0.0:
            raise ValueError(f"ratio control must be positive for {metric}")
        ratio = pair["treatment"] / pair["control"]
        ratios.append(ratio)
        ratio_pairs.append({**pair, "difference": ratio})
    values = np.asarray(ratios)
    interval = _cluster_bootstrap(ratio_pairs, np.mean, resamples=resamples, seed=seed)
    return {
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "pair_count": len(pairs),
        "mean_ratio": float(np.mean(values)),
        "median_ratio": float(np.median(values)),
        "mean_ratio_95pct_cluster_bootstrap": interval,
    }


def backend_interaction(
    records: Sequence[dict[str, Any]], *, treatment: str, control: str, metric: str,
    first_backend: str = "fastlio2", second_backend: str = "liosam",
    resamples: int = 10_000, seed: int = 0,
) -> dict[str, Any]:
    pairs = paired_values(records, treatment=treatment, control=control, metric=metric)
    by_stratum: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for pair in pairs:
        identity = pair["pair"]
        key = (identity["profile"], identity["condition"], identity["paired_block_id"])
        by_stratum[key][identity["backend"]] = pair
    expected = {first_backend, second_backend}
    incomplete = [key for key, values in by_stratum.items() if set(values) != expected]
    if incomplete:
        raise ValueError(f"incomplete backend interaction strata: {incomplete[:3]}")
    interaction_pairs = []
    for key, values in sorted(by_stratum.items()):
        effect = values[first_backend]["difference"] - values[second_backend]["difference"]
        interaction_pairs.append(
            {
                "cluster": (key[2], key[0]),
                "difference": effect,
            }
        )
    effects = np.asarray([value["difference"] for value in interaction_pairs])
    return {
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "first_backend": first_backend,
        "second_backend": second_backend,
        "stratum_count": len(interaction_pairs),
        "mean_difference_in_paired_effects": float(np.mean(effects)),
        "mean_95pct_cluster_bootstrap": _cluster_bootstrap(
            interaction_pairs, np.mean, resamples=resamples, seed=seed,
        ),
    }
