#!/usr/bin/env python3
"""Exploratory stratified diagnostics for accepted formal publication runs.

This script deliberately keeps confirmatory inference in
``analyze_slam_confidence_publication.py`` unchanged.  Speed/progress adjustment,
phase summaries, and gait-coordinate associations are post-treatment diagnostics
and must not be reported as randomized causal effects.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))

from anymal_locomotion_ros2.publication_statistics_core import paired_contrast, paired_values


CONTINUOUS_METRICS = (
    "stance_weighted_foot_slip_rms_mps",
    "while_stable_roll_pitch_rate_rms_radps",
    "tracking_restricted_mean_survival_time_s",
    "normalized_progress",
    "moving_speed_mps",
)
OUTCOME_METRICS = CONTINUOUS_METRICS[:4]
CONTRASTS = (("C", "D", "learned_gait"), ("C", "B", "learned_total"), ("B", "A", "supervisor"))
GAIT_FIELDS = (
    "applied_stride_attenuation",
    "crouch",
    "stance_width",
    "action_smoothing",
    "structured_delta_l2",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--protocol", type=Path,
        default=PROJECT_ROOT / "configs/slam_confidence_publication_protocol.yaml",
    )
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--skip-phase", action="store_true")
    parser.add_argument("--skip-gait", action="store_true")
    return parser.parse_args()


def load_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.glob("**/publication_run_record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_run_dir"] = str(path.resolve().parent)
        records.append(record)
    return records


def _interval(values: np.ndarray) -> dict[str, float]:
    return {"lower": float(np.quantile(values, 0.025)), "upper": float(np.quantile(values, 0.975))}


def stratified_contrasts(
    records: Sequence[dict[str, Any]], *, field: str, resamples: int,
) -> dict[str, Any]:
    levels = sorted({str(record["identity"][field]) for record in records})
    result: dict[str, Any] = {}
    for level in levels:
        subset = [record for record in records if str(record["identity"][field]) == level]
        result[level] = {
            contrast_id: {
                metric: paired_contrast(
                    subset, treatment=treatment, control=control, metric=metric,
                    resamples=resamples,
                )
                for metric in CONTINUOUS_METRICS
            }
            for treatment, control, contrast_id in CONTRASTS
        }
    return result


def _fit_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def matched_adjustment(
    records: Sequence[dict[str, Any]], *, treatment: str, control: str,
    outcome: str, mediator: str, threshold: float, resamples: int, seed: int = 0,
) -> dict[str, Any]:
    outcome_pairs = paired_values(records, treatment=treatment, control=control, metric=outcome)
    mediator_pairs = paired_values(records, treatment=treatment, control=control, metric=mediator)
    outcome_by_pair = {tuple(value["pair"].values()): value for value in outcome_pairs}
    mediator_by_pair = {tuple(value["pair"].values()): value for value in mediator_pairs}
    keys = sorted(set(outcome_by_pair) & set(mediator_by_pair))
    rows = [
        {
            "cluster": outcome_by_pair[key]["cluster"],
            "x": mediator_by_pair[key]["difference"],
            "y": outcome_by_pair[key]["difference"],
        }
        for key in keys
    ]
    x = np.asarray([row["x"] for row in rows], dtype=np.float64)
    y = np.asarray([row["y"] for row in rows], dtype=np.float64)
    intercept, slope = _fit_intercept(x, y)
    close = np.abs(x) <= threshold
    clusters = sorted({row["cluster"] for row in rows})
    by_cluster = {cluster: [row for row in rows if row["cluster"] == cluster] for cluster in clusters}
    rng = np.random.default_rng(seed)
    intercept_draws = np.empty(resamples)
    close_draws: list[float] = []
    for index in range(resamples):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        draw = [row for cluster_index in sampled for row in by_cluster[clusters[int(cluster_index)]]]
        draw_x = np.asarray([row["x"] for row in draw])
        draw_y = np.asarray([row["y"] for row in draw])
        intercept_draws[index] = _fit_intercept(draw_x, draw_y)[0]
        draw_close = np.abs(draw_x) <= threshold
        if np.any(draw_close):
            close_draws.append(float(np.mean(draw_y[draw_close])))
    result: dict[str, Any] = {
        "outcome": outcome,
        "mediator": mediator,
        "pair_count": len(rows),
        "mediator_difference_range": [float(np.min(x)), float(np.max(x))],
        "zero_within_observed_mediator_difference_range": bool(np.min(x) <= 0.0 <= np.max(x)),
        "regression_effect_at_zero_mediator_difference": intercept,
        "regression_effect_95pct_cluster_bootstrap": _interval(intercept_draws),
        "effect_change_per_unit_mediator_difference": slope,
        "absolute_match_threshold": threshold,
        "threshold_matched_pair_count": int(np.sum(close)),
        "threshold_matched_mean_difference": float(np.mean(y[close])) if np.any(close) else None,
        "post_treatment_exploratory_only": True,
    }
    if close_draws:
        result["threshold_matched_mean_95pct_cluster_bootstrap"] = _interval(np.asarray(close_draws))
    return result


def _phase_windows(protocol: dict[str, Any]) -> list[tuple[str, float, float]]:
    timeline = protocol["live_matrix"]["perception_conditions"]["gradual_support_loss"]["timeline_s"]
    healthy = float(timeline["healthy"])
    ramp = healthy + float(timeline["ramp_down"])
    hold = ramp + float(timeline["low_support_hold"])
    recovery = hold + float(timeline["recovery"])
    return [
        ("healthy", 0.0, healthy), ("ramp_down", healthy, ramp),
        ("low_support_hold", ramp, hold), ("recovery", hold, recovery),
        ("after_recovery", recovery, math.inf),
    ]


def _phase_name(time_s: float, windows: Sequence[tuple[str, float, float]]) -> str | None:
    return next((name for name, start, stop in windows if start <= time_s < stop), None)


def _rms(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else None


def phase_metrics(
    run_dir: Path, windows: Sequence[tuple[str, float, float]],
    first_instability_time_s: float | None = None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (run_dir / "locomotion_diagnostics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            name = _phase_name(float(sample["time_s"]), windows)
            if name is not None:
                grouped[name].append(sample)
    result: dict[str, dict[str, Any]] = {}
    trace_start = min(float(sample["time_s"]) for samples in grouped.values() for sample in samples)
    stable_until = (
        trace_start + float(first_instability_time_s)
        if first_instability_time_s is not None else math.inf
    )
    for name, samples in grouped.items():
        rates = []
        for before, after in zip(samples[:-1], samples[1:]):
            dt = float(after["time_s"]) - float(before["time_s"])
            if 0.0 < dt <= 0.05 and float(after["time_s"]) <= stable_until:
                rates.append(math.hypot(
                    (float(after["roll_rad"]) - float(before["roll_rad"])) / dt,
                    (float(after["pitch_rad"]) - float(before["pitch_rad"])) / dt,
                ))
        stance = [
            float(state["tangential_speed_mps"])
            for sample in samples for state in sample["feet"].values()
            if float(state["normal_force_n"]) >= 50.0
        ]
        active_speeds = [
            math.hypot(*sample["actual_linear_velocity_body_mps"][:2])
            for sample in samples if math.hypot(*sample["command"][:2]) >= 0.25
        ]
        result[name] = {
            "sample_count": len(samples),
            "stance_weighted_foot_slip_rms_mps": _rms(stance),
            "while_stable_roll_pitch_rate_rms_radps": _rms(rates),
            "moving_speed_mps": float(np.mean(active_speeds)) if active_speeds else None,
        }
    return result


def build_phase_records(
    records: Sequence[dict[str, Any]], protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    windows = _phase_windows(protocol)
    result = []
    for record in records:
        for phase, metrics in phase_metrics(
            Path(record["_run_dir"]), windows,
            record["metrics"].get("first_instability_time_s"),
        ).items():
            result.append({
                "identity": {**record["identity"], "phase": phase},
                "metrics": metrics,
            })
    return result


def phase_contrasts(
    records: Sequence[dict[str, Any]], *, resamples: int,
) -> dict[str, Any]:
    phases = sorted({record["identity"]["phase"] for record in records})
    result = {}
    for phase in phases:
        subset = [record for record in records if record["identity"]["phase"] == phase]
        result[phase] = {}
        for treatment, control, contrast_id in CONTRASTS:
            result[phase][contrast_id] = {}
            for metric in (
                "stance_weighted_foot_slip_rms_mps",
                "while_stable_roll_pitch_rate_rms_radps",
                "moving_speed_mps",
            ):
                try:
                    value = paired_contrast(
                        subset, treatment=treatment, control=control, metric=metric,
                        resamples=resamples,
                    )
                except ValueError as exc:
                    if "no complete numeric pairs" not in str(exc):
                        raise
                    value = {"metric": metric, "status": "unavailable_no_numeric_pairs"}
                result[phase][contrast_id][metric] = value
    return result


def gait_summaries(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        if record["identity"]["arm"] != "C":
            continue
        sidecar = json.loads((Path(record["_run_dir"]) / "mechanism_sidecar.json").read_text(encoding="utf-8"))
        summary = {}
        for field in GAIT_FIELDS:
            values = np.asarray([float(row[field]) for row in sidecar["records"]])
            summary[field] = float(np.mean(values))
        result.append({"identity": record["identity"], "gait": summary, "metrics": record["metrics"]})
    return result


def _residualize(values: np.ndarray, strata: Sequence[tuple[str, str]]) -> np.ndarray:
    residual = values.copy()
    for stratum in sorted(set(strata)):
        indices = np.asarray([value == stratum for value in strata])
        residual[indices] -= float(np.mean(values[indices]))
    return residual


def gait_associations(rows: Sequence[dict[str, Any]], *, resamples: int) -> dict[str, Any]:
    strata = [(row["identity"]["backend"], row["identity"]["profile"]) for row in rows]
    clusters = [(row["identity"]["paired_block_id"], row["identity"]["profile"]) for row in rows]
    unique_clusters = sorted(set(clusters))
    indices_by_cluster = {
        cluster: np.flatnonzero(np.asarray([value == cluster for value in clusters]))
        for cluster in unique_clusters
    }
    speed = _residualize(
        np.asarray([float(row["metrics"]["moving_speed_mps"]) for row in rows]), strata,
    )
    result: dict[str, Any] = {}
    for field in GAIT_FIELDS:
        x = _residualize(np.asarray([row["gait"][field] for row in rows]), strata)
        result[field] = {}
        for metric in OUTCOME_METRICS:
            y = _residualize(np.asarray([float(row["metrics"][metric]) for row in rows]), strata)
            correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
            speed_design = np.column_stack((np.ones(len(speed)), speed))
            partial_x = x - speed_design @ np.linalg.lstsq(speed_design, x, rcond=None)[0]
            partial_y = y - speed_design @ np.linalg.lstsq(speed_design, y, rcond=None)[0]
            partial_correlation = (
                float(np.corrcoef(partial_x, partial_y)[0, 1])
                if np.std(partial_x) > 0 and np.std(partial_y) > 0 else None
            )
            rng = np.random.default_rng(0)
            draws = []
            partial_draws = []
            for _ in range(resamples):
                sampled = rng.integers(0, len(unique_clusters), len(unique_clusters))
                indices = np.concatenate(
                    [indices_by_cluster[unique_clusters[int(cluster_index)]] for cluster_index in sampled]
                )
                draw_x, draw_y = x[indices], y[indices]
                if np.std(draw_x) > 0 and np.std(draw_y) > 0:
                    draws.append(float(np.corrcoef(draw_x, draw_y)[0, 1]))
                draw_speed = speed[indices]
                draw_design = np.column_stack((np.ones(len(draw_speed)), draw_speed))
                adjusted_x = draw_x - draw_design @ np.linalg.lstsq(draw_design, draw_x, rcond=None)[0]
                adjusted_y = draw_y - draw_design @ np.linalg.lstsq(draw_design, draw_y, rcond=None)[0]
                if np.std(adjusted_x) > 0 and np.std(adjusted_y) > 0:
                    partial_draws.append(float(np.corrcoef(adjusted_x, adjusted_y)[0, 1]))
            result[field][metric] = {
                "within_backend_profile_pearson_r": correlation,
                "cluster_bootstrap_95pct": _interval(np.asarray(draws)) if draws else None,
                "partial_moving_speed_pearson_r": partial_correlation,
                "partial_moving_speed_cluster_bootstrap_95pct": (
                    _interval(np.asarray(partial_draws)) if partial_draws else None
                ),
                "run_count": len(rows),
                "exploratory_association_not_mechanism_effect": True,
            }
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 800-cell formal accepted-artifact stratified diagnostics", "",
        "This report is exploratory. Confirmatory randomized effects remain in `formal_analysis.json`; ",
        "speed/progress adjustment conditions on post-treatment variables and is not a causal estimand.", "",
    ]
    for field, title in (("by_backend", "Backend strata"), ("by_profile", "Profile strata")):
        lines += [f"## {title}", ""]
        for level, contrasts in report[field].items():
            lines.append(f"### {level}")
            lines.append("")
            lines.append("| contrast | slip Δ | body-rate Δ | RMST Δ | progress Δ | speed Δ |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for contrast, metrics in contrasts.items():
                values = [metrics[name]["mean_difference"] for name in CONTINUOUS_METRICS]
                lines.append(f"| {contrast} | " + " | ".join(f"{value:+.6g}" for value in values) + " |")
            lines.append("")
    lines += ["## Matched-speed and matched-progress diagnostics", "",
              "Values are linear-regression intercepts at zero paired mediator difference; brackets are cluster-bootstrap 95% intervals.", ""]
    for section, label in (("matched_speed", "speed"), ("matched_progress", "progress")):
        lines += [f"### Matched {label}", "", "| contrast | outcome | adjusted Δ [95% interval] | threshold-matched n |", "|---|---|---:|---:|"]
        for contrast, outcomes in report[section].items():
            for metric, value in outcomes.items():
                interval = value["regression_effect_95pct_cluster_bootstrap"]
                lines.append(
                    f"| {contrast} | {metric} | {value['regression_effect_at_zero_mediator_difference']:+.6g} "
                    f"[{interval['lower']:+.6g}, {interval['upper']:+.6g}] | {value['threshold_matched_pair_count']} |"
                )
        lines.append("")
    if "phase_contrasts" in report:
        lines += ["## Nominal challenge-phase C-D effects", "",
                  "Phase windows are applied to simulation clock as healthy 0–3 s, ramp-down 3–9 s, low-support hold 9–13 s, recovery 13–16 s, and after-recovery >=16 s.", "",
                  "| phase | slip Δ | while-stable body-rate Δ | speed Δ |", "|---|---:|---:|---:|"]
        for phase, contrasts in report["phase_contrasts"].items():
            values = []
            for metric in ("stance_weighted_foot_slip_rms_mps", "while_stable_roll_pitch_rate_rms_radps", "moving_speed_mps"):
                item = contrasts["learned_gait"][metric]
                values.append("NA" if "mean_difference" not in item else f"{item['mean_difference']:+.6g}")
            lines.append(f"| {phase} | " + " | ".join(values) + " |")
        lines.append("")
    if "gait_coordinate_associations" in report:
        lines += ["## C-arm gait-coordinate associations", "",
                  "Within-backend/profile centered Pearson correlations; these are associations, not coordinate ablation effects.", "",
                  "Each cell is raw r / moving-speed-adjusted partial r.", "",
                  "| coordinate | slip r | body-rate r | RMST r | progress r |", "|---|---:|---:|---:|---:|"]
        for coordinate, outcomes in report["gait_coordinate_associations"].items():
            values = [(outcomes[metric]["within_backend_profile_pearson_r"], outcomes[metric]["partial_moving_speed_pearson_r"]) for metric in OUTCOME_METRICS]
            lines.append(f"| {coordinate} | " + " | ".join(f"{raw:+.3f} / {partial:+.3f}" for raw, partial in values) + " |")
        lines.append("")
    lines += ["## Boundaries", "", "- All 800 records are accepted formal scheduled runs.",
              "- Phase summaries use existing locomotion diagnostics, not rosbag replay.",
              "- The nominal healthy challenge window precedes the 5 s command warmup, so it has no active-speed estimand and is not a moving healthy control.",
              "- Exact LiDAR-adapter phase-origin timestamps are not stored in the run record; nominal phase boundaries may have a small startup offset.",
              "- Gait correlations are ecological run-level associations and cannot identify a coordinate's causal effect.",
              "- Formal A/B/C/D all use estimator15; GT-vs-estimator interaction cannot be identified from this matrix.", ""]
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    root, output = args.input_root.resolve(), args.output.resolve()
    if not root.is_relative_to(PROJECT_ROOT) or not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("input and output must remain inside the project")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    resamples = args.bootstrap_resamples or int(protocol["statistics"]["bootstrap_resamples"])
    records = load_records(root)
    if len(records) != 800 or {record["dataset_role"] for record in records} != {"formal"}:
        raise ValueError("diagnostics require the accepted-only 800-record formal view")
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "slam_confidence_publication_stratified_diagnostics",
        "record_count": len(records),
        "confirmatory_analysis_modified": False,
        "bootstrap_resamples": resamples,
        "by_backend": stratified_contrasts(records, field="backend", resamples=resamples),
        "by_profile": stratified_contrasts(records, field="profile", resamples=resamples),
        "matched_speed": {},
        "matched_progress": {},
    }
    for treatment, control, contrast_id in CONTRASTS[:2]:
        report["matched_speed"][contrast_id] = {
            metric: matched_adjustment(
                records, treatment=treatment, control=control, outcome=metric,
                mediator="moving_speed_mps", threshold=0.05,
                resamples=resamples,
            ) for metric in OUTCOME_METRICS
        }
        report["matched_progress"][contrast_id] = {
            metric: matched_adjustment(
                records, treatment=treatment, control=control, outcome=metric,
                mediator="normalized_progress", threshold=0.05,
                resamples=resamples,
            ) for metric in OUTCOME_METRICS[:3]
        }
    if not args.skip_phase:
        phase_records = build_phase_records(records, protocol)
        report["phase_contrasts"] = phase_contrasts(phase_records, resamples=resamples)
    if not args.skip_gait:
        report["gait_coordinate_associations"] = gait_associations(
            gait_summaries(records), resamples=resamples,
        )
    report["estimator_interaction_boundary"] = {
        "formal_arms_all_use_estimator15": True,
        "gt_vs_estimator_effect_identifiable_from_formal_matrix": False,
        "prior_simulation_evidence": {
            "c_v4_gt_hard_terminated_environments": "0/512",
            "c_v4_early_estimator_hard_terminated_environments": "78/512",
            "model48_estimator15_regression": "6/2560 across five profiles",
        },
        "interpretation": "Estimator interaction is plausible but cannot explain randomized C-B or C-D formal differences because estimator15 is common to all arms.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = (args.markdown_output or output.with_suffix(".md")).resolve()
    if not markdown.is_relative_to(PROJECT_ROOT):
        raise ValueError("markdown output must remain inside the project")
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"record_count": len(records), "output": str(output), "markdown": str(markdown)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
