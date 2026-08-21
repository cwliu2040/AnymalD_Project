#!/usr/bin/env python3
"""Audit whether deployable motion history adds future SLAM-risk information.

This is a development-only predictability audit over accepted formal artifacts.
It does not estimate a locomotion intervention effect, retrain a policy, or alter
the frozen publication analysis.  Frame-level rows train predictors; uncertainty
for model comparisons is clustered by complete scheduled runs.
"""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_SCHEMA_VERSION = 2
SUMMARY_NAMES = ("current", "mean", "std", "min", "max", "slope")
OBSERVATION_DIMENSION = 51
FEATURE_SETS = {
    "slam_state_only": tuple(range(48, 51)),
    "motion_without_estimator_velocity": tuple(range(3, 51)),
    "motion_with_estimator_velocity": tuple(range(0, 51)),
}
ENDPOINTS = ("hazard_0p5s", "recovery_1p0s")
BACKENDS = ("fastlio2", "liosam")
ARMS = ("B", "C", "D")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--history-s", type=float, default=1.0)
    parser.add_argument("--sample-period-s", type=float, default=0.1)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def _project_path(path: Path, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path must remain inside project: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def history_summary(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return causal current/moments/range/slope summaries for one history."""
    timestamps = np.asarray(times, dtype=np.float64)
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 2 or timestamps.shape != (len(observations),):
        raise ValueError("history times/values have incompatible shapes")
    if len(observations) < 2 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("history requires at least two strictly ordered samples")
    centered = timestamps - float(np.mean(timestamps))
    denominator = float(centered @ centered)
    slope = (
        centered @ observations / denominator
        if denominator > 1.0e-12 else np.zeros(observations.shape[1])
    )
    return np.concatenate(
        (
            observations[-1], np.mean(observations, axis=0),
            np.std(observations, axis=0), np.min(observations, axis=0),
            np.max(observations, axis=0), slope,
        )
    ).astype(np.float32)


def summary_columns(observation_indices: Sequence[int]) -> np.ndarray:
    selected = np.asarray(observation_indices, dtype=np.int64)
    return np.concatenate(
        [selected + index * OBSERVATION_DIMENSION for index in range(len(SUMMARY_NAMES))]
    )


def recovery_target(
    *, clock_ns: int, label_stamps: Sequence[int], label_values: Sequence[bool | None],
    horizon_ns: int = 1_000_000_000,
) -> bool | None:
    """Whether offline future-usability recovers within a fully observed horizon."""
    known_stamps = [
        int(stamp) for stamp, value in zip(label_stamps, label_values, strict=True)
        if value is not None
    ]
    if not known_stamps or known_stamps[-1] < clock_ns + horizon_ns:
        return None
    start = bisect.bisect_left(label_stamps, clock_ns)
    stop = bisect.bisect_right(label_stamps, clock_ns + horizon_ns)
    values = [value for value in label_values[start:stop] if value is not None]
    return bool(any(values)) if values else None


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return value


def run_examples(
    *, run_dir: Path, identity: dict[str, Any], history_s: float,
    sample_period_s: float,
) -> list[dict[str, Any]]:
    policy = _load_json(run_dir / "policy_diagnostics.json")
    offline = _load_json(run_dir / "offline_usability.json")
    records = policy.get("records", [])
    if policy.get("schema_version") != 2 or not records:
        raise ValueError(f"invalid policy diagnostics: {run_dir}")
    observations = np.asarray([row["observation"] for row in records], dtype=np.float64)
    if observations.shape[1:] != (OBSERVATION_DIMENSION,):
        raise ValueError(f"audit requires 51-D observations: {run_dir}")
    times = np.asarray([float(row["clock_s"]) for row in records], dtype=np.float64)
    if np.any(np.diff(times) < 0.0):
        raise ValueError(f"policy clocks move backwards: {run_dir}")
    if np.any(np.diff(times) == 0.0):
        # Timer scheduling can publish two diagnostics with the same policy
        # clock.  Keep the final record at that tick so a causal history never
        # contains a zero-duration interval or double-weights one instant.
        keep = np.r_[np.diff(times) != 0.0, True]
        observations = observations[keep]
        times = times[keep]
        records = [row for row, retain in zip(records, keep, strict=True) if retain]
    known_by_policy_clock = {
        int(row["policy_clock_ns"]): row["usable_next_horizon"]
        for row in offline["false_stop"]["records"]
    }
    labels = offline["labels"]
    label_stamps = [int(row["evaluation_stamp_ns"]) for row in labels]
    label_values = [row["usable_next_horizon"] for row in labels]
    first_valid = next(
        (index for index, observation in enumerate(observations) if observation[49] >= 0.5),
        None,
    )
    if first_valid is None:
        return []
    period_steps = max(1, round(sample_period_s / float(np.median(np.diff(times)))))
    minimum_history_samples = max(2, round(0.8 * history_s / float(np.median(np.diff(times)))))
    examples = []
    for index in range(first_valid, len(records)):
        if index % period_steps:
            continue
        start = int(np.searchsorted(times, times[index] - history_s, side="left"))
        if index - start + 1 < minimum_history_samples:
            continue
        clock_ns = round(times[index] * 1.0e9)
        valid = observations[index, 49] >= 0.5
        if valid:
            usable = known_by_policy_clock.get(clock_ns)
            if usable is None:
                continue
            endpoint = "hazard_0p5s"
            target = not bool(usable)
        else:
            target = recovery_target(
                clock_ns=clock_ns, label_stamps=label_stamps,
                label_values=label_values,
            )
            if target is None:
                continue
            endpoint = "recovery_1p0s"
        examples.append(
            {
                "features": history_summary(
                    times[start : index + 1], observations[start : index + 1]
                ),
                "target": bool(target),
                "endpoint": endpoint,
                "backend": str(identity["backend"]),
                "profile": str(identity["profile"]),
                "block": int(identity["paired_block_id"]),
                "arm": str(identity["arm"]),
                "run_id": (
                    f"{identity['backend']}|{identity['profile']}|"
                    f"{identity['paired_block_id']}|{identity['arm']}"
                ),
                "clock_ns": clock_ns,
            }
        )
    return examples


def build_feature_cache(
    root: Path, cache_path: Path, *, history_s: float, sample_period_s: float,
) -> dict[str, np.ndarray]:
    records = []
    excluded = defaultdict(int)
    paths = sorted(root.glob("**/publication_run_record.json"))
    for path in paths:
        record = _load_json(path)
        arm = str(record["identity"]["arm"])
        if arm == "A":
            excluded["arm_A_has_no_slam_state_in_48d_diagnostics"] += 1
            continue
        if arm not in ARMS:
            excluded["unsupported_arm"] += 1
            continue
        accepted_run_dir = path.resolve().parent
        artifact_run_dir = accepted_run_dir
        if not (artifact_run_dir / "policy_diagnostics.json").is_file():
            relative_run_dir = path.relative_to(root).parent
            artifact_run_dir = root.parent.parent / "formal_blocks" / relative_run_dir
        if not (artifact_run_dir / "policy_diagnostics.json").is_file():
            raise FileNotFoundError(
                f"cannot resolve formal artifacts for accepted record: {path}"
            )
        examples = run_examples(
            run_dir=artifact_run_dir,
            identity=record["identity"],
            history_s=history_s,
            sample_period_s=sample_period_s,
        )
        if not examples:
            excluded["no_eligible_causal_window"] += 1
        records.extend(examples)
    if not records:
        raise ValueError("no audit examples were built")
    arrays = {
        "features": np.stack([row["features"] for row in records]).astype(np.float32),
        "target": np.asarray([row["target"] for row in records], dtype=np.int8),
        "endpoint": np.asarray([row["endpoint"] for row in records], dtype="U16"),
        "backend": np.asarray([row["backend"] for row in records], dtype="U16"),
        "profile": np.asarray([row["profile"] for row in records], dtype="U40"),
        "block": np.asarray([row["block"] for row in records], dtype=np.int32),
        "arm": np.asarray([row["arm"] for row in records], dtype="U1"),
        "run_id": np.asarray([row["run_id"] for row in records], dtype="U96"),
        "clock_ns": np.asarray([row["clock_ns"] for row in records], dtype=np.int64),
        "cache_schema_version": np.asarray([CACHE_SCHEMA_VERSION], dtype=np.int32),
        "source_run_record_count": np.asarray([len(paths)], dtype=np.int32),
        "included_run_count": np.asarray(
            [len(set(row["run_id"] for row in records))], dtype=np.int32,
        ),
        "excluded_arm_a_run_count": np.asarray(
            [excluded["arm_A_has_no_slam_state_in_48d_diagnostics"]], dtype=np.int32,
        ),
        "excluded_no_eligible_window_run_count": np.asarray(
            [excluded["no_eligible_causal_window"]], dtype=np.int32,
        ),
        "history_s": np.asarray([history_s], dtype=np.float64),
        "sample_period_s": np.asarray([sample_period_s], dtype=np.float64),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    return arrays


def load_feature_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if int(arrays["cache_schema_version"][0]) != CACHE_SCHEMA_VERSION:
        raise ValueError("feature cache schema mismatch")
    if arrays["features"].shape[1] != OBSERVATION_DIMENSION * len(SUMMARY_NAMES):
        raise ValueError("feature cache dimension mismatch")
    return arrays


def equal_run_weights(run_ids: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(run_ids, return_inverse=True, return_counts=True)
    del unique
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights * (len(weights) / float(np.sum(weights)))


def _fit_predict(
    train_x: np.ndarray, train_y: np.ndarray, train_runs: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for the offline audit") from exc
    if len(np.unique(train_y)) != 2:
        raise ValueError("training split lacks binary class coverage")
    weights = equal_run_weights(train_runs)
    mean = np.average(train_x, axis=0, weights=weights)
    variance = np.average(np.square(train_x - mean), axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1.0e-6] = 1.0
    classifier = LogisticRegression(
        penalty="l2", C=0.01, solver="lbfgs",
        max_iter=500, tol=1.0e-5, random_state=0,
    )
    classifier.fit((train_x - mean) / scale, train_y, sample_weight=weights)
    return classifier.predict_proba((test_x - mean) / scale)[:, 1]


def cross_validated_predictions(
    arrays: dict[str, np.ndarray], *, backend: str, endpoint: str,
    columns: np.ndarray, scheme: str,
) -> dict[str, np.ndarray] | None:
    mask = (arrays["backend"] == backend) & (arrays["endpoint"] == endpoint)
    selected = np.flatnonzero(mask)
    if not len(selected):
        return None
    predictions = np.full(len(selected), np.nan, dtype=np.float64)
    if scheme == "block_5fold":
        blocks = sorted(set(int(value) for value in arrays["block"][selected]))
        fold_by_block = {block: index % 5 for index, block in enumerate(blocks)}
        split_values = range(5)
        test_for_split = lambda split: np.asarray(
            [fold_by_block[int(value)] == split for value in arrays["block"][selected]]
        )
    elif scheme == "leave_one_arm_out":
        split_values = ARMS
        test_for_split = lambda split: arrays["arm"][selected] == split
    else:
        raise ValueError(f"unknown evaluation scheme: {scheme}")
    for split in split_values:
        test_local = test_for_split(split)
        train_local = ~test_local
        if not np.any(test_local):
            continue
        try:
            predictions[test_local] = _fit_predict(
                arrays["features"][selected[train_local]][:, columns],
                arrays["target"][selected[train_local]],
                arrays["run_id"][selected[train_local]],
                arrays["features"][selected[test_local]][:, columns],
            )
        except ValueError:
            return None
    if np.any(~np.isfinite(predictions)):
        return None
    return {
        "indices": selected,
        "probability": predictions,
    }


def expected_calibration_error(
    targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray,
    *, bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(np.sum(weights))
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if not np.any(mask):
            continue
        mass = float(np.sum(weights[mask]))
        predicted = float(np.average(probabilities[mask], weights=weights[mask]))
        observed = float(np.average(targets[mask], weights=weights[mask]))
        error += mass / total * abs(predicted - observed)
    return error


def prediction_metrics(
    targets: np.ndarray, probabilities: np.ndarray, run_ids: np.ndarray,
) -> dict[str, Any]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for the offline audit") from exc
    y = np.asarray(targets, dtype=np.int8)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    weights = equal_run_weights(np.asarray(run_ids))
    result = {
        "sample_count": len(y),
        "run_count": len(set(str(value) for value in run_ids)),
        "positive_prevalence_equal_run_weighted": float(np.average(y, weights=weights)),
        "brier": float(np.average(np.square(p - y), weights=weights)),
        "log_loss": float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=weights)),
        "ece_10bin": expected_calibration_error(y, p, weights),
    }
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, p, sample_weight=weights))
        result["average_precision"] = float(
            average_precision_score(y, p, sample_weight=weights)
        )
    else:
        result["auroc"] = None
        result["average_precision"] = None
    return result


def clustered_metric_difference(
    arrays: dict[str, np.ndarray], indices: np.ndarray,
    treatment_probability: np.ndarray, control_probability: np.ndarray,
    *, resamples: int, seed: int = 0,
) -> dict[str, Any]:
    targets = arrays["target"][indices]
    treatment = np.square(treatment_probability - targets)
    control = np.square(control_probability - targets)
    by_run: dict[str, list[float]] = defaultdict(list)
    metadata = {}
    for local, global_index in enumerate(indices):
        run_id = str(arrays["run_id"][global_index])
        by_run[run_id].append(float(treatment[local] - control[local]))
        metadata[run_id] = (
            int(arrays["block"][global_index]), str(arrays["profile"][global_index])
        )
    run_difference = {run: float(np.mean(values)) for run, values in by_run.items()}
    by_cluster: dict[tuple[int, str], list[float]] = defaultdict(list)
    for run, value in run_difference.items():
        by_cluster[metadata[run]].append(value)
    clusters = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(clusters), len(clusters))
        values = [
            value for cluster_index in sampled
            for value in by_cluster[clusters[int(cluster_index)]]
        ]
        draws[index] = float(np.mean(values))
    return {
        "estimand": "equal_run_mean_brier_difference_treatment_minus_slam_state_only",
        "run_count": len(run_difference),
        "cluster_count": len(clusters),
        "mean_difference": float(np.mean(list(run_difference.values()))),
        "cluster_bootstrap_95pct": {
            "lower": float(np.quantile(draws, 0.025)),
            "upper": float(np.quantile(draws, 0.975)),
            "resamples": resamples,
            "seed": seed,
        },
    }


def evaluate_scheme(
    arrays: dict[str, np.ndarray], *, scheme: str, resamples: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    columns = {name: summary_columns(indices) for name, indices in FEATURE_SETS.items()}
    for backend in BACKENDS:
        report[backend] = {}
        for endpoint in ENDPOINTS:
            model_predictions = {
                name: cross_validated_predictions(
                    arrays, backend=backend, endpoint=endpoint,
                    columns=selected, scheme=scheme,
                )
                for name, selected in columns.items()
            }
            if any(value is None for value in model_predictions.values()):
                report[backend][endpoint] = {
                    "status": "unavailable_split_or_class_coverage",
                }
                continue
            baseline = model_predictions["slam_state_only"]
            assert baseline is not None
            indices = baseline["indices"]
            models = {}
            for name, value in model_predictions.items():
                assert value is not None
                if not np.array_equal(indices, value["indices"]):
                    raise RuntimeError("model predictions are not sample aligned")
                models[name] = prediction_metrics(
                    arrays["target"][indices], value["probability"],
                    arrays["run_id"][indices],
                )
            comparisons = {}
            for name in FEATURE_SETS:
                if name == "slam_state_only":
                    continue
                value = model_predictions[name]
                assert value is not None
                difference = clustered_metric_difference(
                    arrays, indices, value["probability"], baseline["probability"],
                    resamples=resamples,
                )
                comparisons[name] = {
                    **difference,
                    "log_loss_difference": models[name]["log_loss"] - models["slam_state_only"]["log_loss"],
                    "ece_difference": models[name]["ece_10bin"] - models["slam_state_only"]["ece_10bin"],
                }
            report[backend][endpoint] = {
                "status": "measured",
                "models": models,
                "comparisons_to_slam_state_only": comparisons,
            }
    return report


def classify_audit(evaluations: dict[str, Any]) -> dict[str, Any]:
    core = "motion_without_estimator_velocity"
    details = []
    any_worsening = False
    all_credible = True
    for scheme in ("block_5fold", "leave_one_arm_out"):
        for backend in BACKENDS:
            for endpoint in ENDPOINTS:
                value = evaluations[scheme][backend][endpoint]
                key = f"{scheme}/{backend}/{endpoint}"
                if value.get("status") != "measured":
                    details.append({"cell": key, "status": "unavailable"})
                    all_credible = False
                    continue
                comparison = value["comparisons_to_slam_state_only"][core]
                interval = comparison["cluster_bootstrap_95pct"]
                calibration_nonworse = comparison["ece_difference"] <= 0.02
                credible_improvement = bool(
                    interval["upper"] < 0.0
                    and comparison["log_loss_difference"] < 0.0
                    and calibration_nonworse
                )
                credible_worsening = bool(
                    interval["lower"] > 0.0 or comparison["ece_difference"] > 0.02
                )
                any_worsening = any_worsening or credible_worsening
                all_credible = all_credible and credible_improvement
                details.append(
                    {
                        "cell": key,
                        "credible_improvement": credible_improvement,
                        "credible_worsening": credible_worsening,
                        "brier_difference": comparison["mean_difference"],
                        "brier_difference_95pct": interval,
                        "log_loss_difference": comparison["log_loss_difference"],
                        "ece_difference": comparison["ece_difference"],
                    }
                )
    if any_worsening:
        status = "FAIL"
        next_step = "stop_current_new_c_route"
    elif all_credible:
        status = "PASS"
        next_step = "design_matched_speed_intervention_dataset"
    else:
        status = "INCONCLUSIVE"
        next_step = "only_small_intervention_pilot_allowed"
    return {
        "status": status,
        "rule": {
            "pass": "core motion history credibly improves Brier and log loss without ECE worsening >0.02 in every backend/endpoint/scheme",
            "fail": "core motion history credibly worsens Brier or worsens ECE by >0.02 in any cell",
            "otherwise": "INCONCLUSIVE",
        },
        "next_step": next_step,
        "cells": details,
    }


def _inventory(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {
        "sample_count": len(arrays["target"]),
        "included_run_count": int(arrays["included_run_count"][0]),
        "source_run_record_count": int(arrays["source_run_record_count"][0]),
        "excluded_arm_a_run_count": int(arrays["excluded_arm_a_run_count"][0]),
        "excluded_no_eligible_window_run_count": int(
            arrays["excluded_no_eligible_window_run_count"][0]
        ),
        "by_backend_endpoint": {},
    }
    for backend in BACKENDS:
        result["by_backend_endpoint"][backend] = {}
        for endpoint in ENDPOINTS:
            mask = (arrays["backend"] == backend) & (arrays["endpoint"] == endpoint)
            result["by_backend_endpoint"][backend][endpoint] = {
                "samples": int(np.sum(mask)),
                "runs": len(set(str(value) for value in arrays["run_id"][mask])),
                "positive_prevalence": float(np.mean(arrays["target"][mask])) if np.any(mask) else None,
            }
    return result


def _markdown(report: dict[str, Any]) -> str:
    sampling_hz = 1.0 / report["sample_period_s"]
    lines = [
        "# Existing-800 SLAM-risk predictability audit", "",
        f"Decision: **{report['decision']['status']}**", "",
        f"Next step: `{report['decision']['next_step']}`", "",
        "This is a development-only predictive audit, not evidence that locomotion motion causally changes SLAM risk.", "",
        "## Inventory", "",
        f"- Included runs: {report['inventory']['included_run_count']} (B/C/D).",
        f"- Excluded A runs: {report['inventory']['excluded_arm_a_run_count']} because 48-D diagnostics do not contain SLAM state.",
        f"- B/C/D runs without an eligible causal prediction window: {report['inventory']['excluded_no_eligible_window_run_count']}.",
        f"- Predictive samples: {report['inventory']['sample_count']} at {sampling_hz:g} Hz after causal {report['history_s']:g} s history construction.", "",
        "## Equal-run Brier differences", "",
        "Negative values favor motion history over SLAM-state history.", "",
        "| scheme | backend | endpoint | no-estimator motion Δ [95% CI] | with-estimator motion Δ [95% CI] |", "|---|---|---|---:|---:|",
    ]
    for scheme in ("block_5fold", "leave_one_arm_out"):
        for backend in BACKENDS:
            for endpoint in ENDPOINTS:
                value = report["evaluations"][scheme][backend][endpoint]
                if value.get("status") != "measured":
                    lines.append(f"| {scheme} | {backend} | {endpoint} | unavailable | unavailable |")
                    continue
                cells = []
                for model in ("motion_without_estimator_velocity", "motion_with_estimator_velocity"):
                    comparison = value["comparisons_to_slam_state_only"][model]
                    interval = comparison["cluster_bootstrap_95pct"]
                    cells.append(
                        f"{comparison['mean_difference']:+.6f} "
                        f"[{interval['lower']:+.6f}, {interval['upper']:+.6f}]"
                    )
                lines.append(
                    f"| {scheme} | {backend} | {endpoint} | {cells[0]} | {cells[1]} |"
                )
    lines += [
        "", "## Boundaries", "",
        "- Inputs are existing deployable policy observations; no raw bags, GT inputs, ROS, Isaac, or PPO training were used.",
        "- Train/test splits keep complete paired blocks together; leave-one-arm-out tests unseen B/C/D policies.",
        "- Adjacent prediction ticks are not treated as independent inference units; comparisons bootstrap complete `(block, profile)` clusters.",
        "- Existing formal outcomes are now development data and cannot serve as the future new-C blind test.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    root = _project_path(args.input_root, must_exist=True)
    output = _project_path(args.output)
    markdown = _project_path(args.markdown_output or output.with_suffix(".md"))
    cache = _project_path(
        args.feature_cache or output.parent / "risk_predictability_features.npz"
    )
    if args.history_s <= 0.0 or args.sample_period_s <= 0.0:
        raise ValueError("history and sample period must be positive")
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap requires at least 100 resamples")
    if args.reuse_cache and cache.is_file():
        arrays = load_feature_cache(cache)
    else:
        arrays = build_feature_cache(
            root, cache, history_s=args.history_s,
            sample_period_s=args.sample_period_s,
        )
    evaluations = {
        scheme: evaluate_scheme(
            arrays, scheme=scheme, resamples=args.bootstrap_resamples,
        )
        for scheme in ("block_5fold", "leave_one_arm_out")
    }
    report = {
        "schema_version": 1,
        "kind": "slam_risk_predictability_audit",
        "dataset_role": "development_only_after_formal_analysis",
        "causal_effect_claim_allowed": False,
        "history_s": float(arrays["history_s"][0]),
        "sample_period_s": float(arrays["sample_period_s"][0]),
        "bootstrap_resamples": args.bootstrap_resamples,
        "predictor": {
            "kind": "l2_regularized_logistic_regression",
            "regularization_C": 0.01,
            "standardization_fit_on_training_split_only": True,
            "sample_weighting": "equal_total_weight_per_run",
        },
        "feature_sets": {
            name: {
                "observation_offsets": list(indices),
                "summary_statistics": list(SUMMARY_NAMES),
            }
            for name, indices in FEATURE_SETS.items()
        },
        "inventory": _inventory(arrays),
        "evaluations": evaluations,
        "decision": classify_audit(evaluations),
        "boundaries": {
            "raw_bags_read": False,
            "ground_truth_runtime_input": False,
            "future_label_used_as_input": False,
            "arm_identity_used_as_input": False,
            "backend_identity_used_as_input": False,
            "frame_samples_are_independent_inference_units": False,
            "arm_A_excluded_reason": "48d_policy_diagnostics_do_not_record_slam_state",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "decision": report["decision"]["status"],
        "next_step": report["decision"]["next_step"],
        "inventory": report["inventory"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
