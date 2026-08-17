#!/usr/bin/env python3
"""Reconstruct publication mechanisms and policy-arm actions from a live trace."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARITY_SCRIPT = PROJECT_ROOT / "scripts" / "validation" / "validate_policy_parity.py"


def _project_file(path: Path, *, must_exist: bool) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path must remain inside {PROJECT_ROOT}: {resolved}")
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _load_checkpoint_actor(checkpoint: Path, agent_config: Path):
    spec = importlib.util.spec_from_file_location(
        "slam_confidence_policy_parity",
        PARITY_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load policy reconstruction: {PARITY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._checkpoint_actor(checkpoint, agent_config)


def _raw_gait_coordinates(actor, observation: torch.Tensor, intent_action: torch.Tensor):
    logits = actor.gait_head(torch.cat((observation, intent_action), dim=-1))
    coordinates = actor.gait_parameter_limits * torch.tanh(logits)
    if actor.nonnegative_stride_smoothing:
        coordinates = torch.stack(
            (
                torch.clamp(coordinates[..., 0], min=0.0),
                coordinates[..., 1],
                coordinates[..., 2],
                actor.gait_parameter_limits[3]
                * torch.clamp(
                    actor.smoothing_logit_gain * torch.tanh(logits[..., 3]),
                    min=0.0,
                    max=1.0,
                ),
            ),
            dim=-1,
        )
    return coordinates


def reconstruct_mechanisms(
    actor,
    observations: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the four registered arm actions and interpretable mechanisms."""
    if observations.ndim != 2 or observations.shape[1] != 51:
        raise ValueError("mechanism observations must have shape [N, 51]")
    legacy_observation = observations[..., : actor.legacy_observation_dim]
    legacy_action = actor.backbone(legacy_observation)
    confidence = observations[..., actor.confidence_offset]
    valid = observations[..., actor.confidence_offset + 1]
    safe_scale = torch.clamp(valid, 0.0, 1.0) * torch.clamp(
        (confidence - 0.2) / 0.8,
        0.0,
        1.0,
    )
    safe_observation = legacy_observation.clone()
    command_slice = slice(
        actor.command_offset,
        actor.command_offset + actor.command_dimension,
    )
    safe_observation[..., command_slice] *= safe_scale.unsqueeze(-1)
    safe_action = actor.backbone(safe_observation)
    intent_blend = actor.intent_blend(observations, legacy_action)
    intent_action = legacy_action + intent_blend * (safe_action - legacy_action)

    raw_coordinates = _raw_gait_coordinates(actor, observations, intent_action)
    applied_coordinates = actor.gait_coordinates(observations, intent_action)
    previous_action = observations[
        ...,
        actor.previous_action_offset : actor.previous_action_offset + 12,
    ]
    structured_delta = (
        applied_coordinates[..., 0:1] * (-intent_action)
        + applied_coordinates[..., 1:2] * actor.crouch_basis
        + applied_coordinates[..., 2:3] * actor.stance_width_basis
        + applied_coordinates[..., 3:4] * (previous_action - intent_action)
    )
    gait_gate = (
        torch.clamp(valid, 0.0, 1.0)
        if actor.suppress_gait_when_tracking_invalid
        else torch.ones_like(valid)
    )
    if actor.gait_delta_safe_scale_power > 0.0:
        gait_gate *= torch.clamp(safe_scale, 0.0, 1.0).pow(
            actor.gait_delta_safe_scale_power
        )
    applied_structured_delta = (
        (1.0 - safe_scale) * gait_gate
    ).unsqueeze(-1) * structured_delta
    model48_action = intent_action + applied_structured_delta
    return {
        "safe_scale": safe_scale,
        "intent_blend": intent_blend.squeeze(-1),
        "raw_gait_coordinates": raw_coordinates,
        "applied_gait_coordinates": applied_coordinates,
        "structured_delta_l2": torch.linalg.vector_norm(
            applied_structured_delta,
            dim=-1,
        ),
        "arm_a_action": legacy_action,
        "arm_b_action": safe_action,
        "arm_c_action": model48_action,
        "arm_d_action": intent_action,
        "arm_d_structured_delta": torch.zeros_like(applied_structured_delta),
    }


def build_sidecar(
    *,
    diagnostics: dict[str, Any],
    actor,
    actor_details: dict[str, Any],
    action_atol: float,
) -> dict[str, Any]:
    records = diagnostics.get("records")
    if diagnostics.get("schema_version") != 2 or not isinstance(records, list):
        raise ValueError("policy diagnostics must use schema version 2")
    if not records:
        raise ValueError("policy diagnostics contain no records")
    try:
        observations = torch.tensor(
            [record["observation"] for record in records],
            dtype=torch.float32,
        )
        logged_actions = torch.tensor(
            [record["raw_action"] for record in records],
            dtype=torch.float32,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("policy diagnostics records are incomplete") from exc
    if logged_actions.ndim != 2 or logged_actions.shape[1] != 12:
        raise ValueError("logged policy actions must have shape [N, 12]")

    with torch.inference_mode():
        mechanisms = reconstruct_mechanisms(actor, observations)
    action_errors = torch.abs(mechanisms["arm_c_action"] - logged_actions)
    finite = all(
        bool(torch.isfinite(value).all()) for value in mechanisms.values()
    ) and bool(torch.isfinite(logged_actions).all())
    max_action_error = float(action_errors.max())
    intent_only_zero = bool(
        torch.count_nonzero(mechanisms["arm_d_structured_delta"]) == 0
    )

    sidecar_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        raw_coordinates = mechanisms["raw_gait_coordinates"][index]
        applied_coordinates = mechanisms["applied_gait_coordinates"][index]
        sidecar_records.append(
            {
                "clock_s": record.get("clock_s"),
                "safe_scale": float(mechanisms["safe_scale"][index]),
                "intent_blend": float(mechanisms["intent_blend"][index]),
                "raw_stride_attenuation": float(raw_coordinates[0]),
                "applied_stride_attenuation": float(applied_coordinates[0]),
                "crouch": float(applied_coordinates[1]),
                "stance_width": float(applied_coordinates[2]),
                "action_smoothing": float(applied_coordinates[3]),
                "structured_delta_l2": float(
                    mechanisms["structured_delta_l2"][index]
                ),
                "arm_actions": {
                    arm: mechanisms[f"arm_{arm.lower()}_action"][index].tolist()
                    for arm in ("A", "B", "C", "D")
                },
            }
        )

    passed = bool(
        finite
        and max_action_error <= action_atol
        and intent_only_zero
        and actor_details.get("architecture")
        == "frozen_model1450_aux_intent_structured_gait"
    )
    return {
        "schema_version": 1,
        "kind": "slam_confidence_mechanism_sidecar",
        "actor": actor_details,
        "gate": {
            "passed": passed,
            "finite": finite,
            "record_count": len(records),
            "action_reconstruction_atol": action_atol,
            "model48_logged_action_max_absolute_error": max_action_error,
            "arm_d_structured_delta_exact_zero": intent_only_zero,
            "arm_semantics": {
                "A": "frozen_model1450_without_confidence",
                "B": "frozen_model1450_at_deterministic_safe_command",
                "C": "model48_intent_plus_applied_structured_gait",
                "D": "model48_intent_only_with_structured_delta_zero",
            },
        },
        "records": sidecar_records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-atol", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not math.isfinite(args.action_atol) or args.action_atol < 0.0:
        raise ValueError("--action-atol must be finite and non-negative")
    diagnostics_path = _project_file(args.diagnostics, must_exist=True)
    checkpoint_path = _project_file(args.checkpoint, must_exist=True)
    agent_config_path = _project_file(args.agent_config, must_exist=True)
    output_path = _project_file(args.output, must_exist=False)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    actor, actor_details = _load_checkpoint_actor(
        checkpoint_path,
        agent_config_path,
    )
    sidecar = build_sidecar(
        diagnostics=diagnostics,
        actor=actor,
        actor_details=actor_details,
        action_atol=args.action_atol,
    )
    sidecar.update(
        {
            "diagnostics_path": str(diagnostics_path),
            "checkpoint_path": str(checkpoint_path),
            "agent_config_path": str(agent_config_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sidecar["gate"], ensure_ascii=False, indent=2))
    print(f"Mechanism sidecar written to: {output_path}")
    return 0 if sidecar["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
