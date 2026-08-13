#!/usr/bin/env python3
"""Validate checkpoint, TorchScript, and ONNX policy output parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import yaml
from onnx.reference import ReferenceEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATOL = 1.0e-5
DEFAULT_RTOL = 1.0e-5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL checkpoint path.")
    parser.add_argument("--jit", type=Path, required=True, help="Exported TorchScript policy path.")
    parser.add_argument("--onnx", type=Path, required=True, help="Exported ONNX policy path.")
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=None,
        help="Resolved agent.yaml. Defaults to <checkpoint-dir>/params/agent.yaml.",
    )
    parser.add_argument("--samples", type=int, default=256, help="Number of deterministic observations.")
    parser.add_argument("--seed", type=int, default=42, help="Observation generator seed.")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL, help="Absolute comparison tolerance.")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL, help="Relative comparison tolerance.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def _resolve_project_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"Path escapes project root: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _activation(name: str) -> type[torch.nn.Module]:
    activations: dict[str, type[torch.nn.Module]] = {
        "elu": torch.nn.ELU,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "tanh": torch.nn.Tanh,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported actor activation for parity validation: {name}") from error


def _linear_indices(state: dict[str, torch.Tensor], prefix: str) -> list[int]:
    return sorted(
        int(key.removeprefix(prefix).split(".", 1)[0])
        for key in state
        if key.startswith(prefix) and key.endswith(".weight")
    )


def _checkpoint_mlp(
    state: dict[str, torch.Tensor],
    prefix: str,
    activation_type: type[torch.nn.Module],
) -> tuple[torch.nn.Sequential, list[list[int]]]:
    layer_indices = _linear_indices(state, prefix)
    if not layer_indices:
        raise ValueError(f"Checkpoint has no linear-layer weights under {prefix}")
    modules: list[torch.nn.Module] = []
    dimensions: list[list[int]] = []
    for position, layer_index in enumerate(layer_indices):
        weight = state[f"{prefix}{layer_index}.weight"]
        bias = state[f"{prefix}{layer_index}.bias"]
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        modules.append(layer)
        dimensions.append([int(weight.shape[1]), int(weight.shape[0])])
        if position < len(layer_indices) - 1:
            modules.append(activation_type())
    return torch.nn.Sequential(*modules), dimensions


class _ResidualCheckpointActor(torch.nn.Module):
    """Local reconstruction of the exported confidence residual actor."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        residual: torch.nn.Module,
        action_skip: torch.nn.Module,
        residual_action_limit: float,
        confidence_offset: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.residual = residual
        self.action_skip = action_skip
        self.residual_action_limit = residual_action_limit
        self.confidence_offset = confidence_offset

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy_action = self.backbone(observation[..., : self.confidence_offset])
        residual = self.residual_action_limit * torch.tanh(
            self.residual(torch.cat((observation, legacy_action), dim=-1))
            + self.action_skip(legacy_action)
        )
        confidence = observation[..., self.confidence_offset]
        valid = observation[..., self.confidence_offset + 1]
        safe_scale = torch.clamp(valid, 0.0, 1.0) * torch.clamp(
            (confidence - 0.2) / 0.8,
            0.0,
            1.0,
        )
        return legacy_action + (1.0 - safe_scale).unsqueeze(-1) * residual


class _SafeCommandCheckpointActor(torch.nn.Module):
    """Local reconstruction of the exact safe-command actor."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        safe_command_gain: torch.Tensor,
        confidence_offset: int,
        command_offset: int,
        command_dimension: int,
        safe_command_gain_limit: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.safe_command_gain = torch.nn.Parameter(safe_command_gain.clone())
        self.confidence_offset = confidence_offset
        self.command_offset = command_offset
        self.command_dimension = command_dimension
        self.safe_command_gain_limit = safe_command_gain_limit

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy_observation = observation[..., : self.confidence_offset]
        legacy_action = self.backbone(legacy_observation)
        confidence = observation[..., self.confidence_offset]
        valid = observation[..., self.confidence_offset + 1]
        safe_scale = torch.clamp(valid, 0.0, 1.0) * torch.clamp(
            (confidence - 0.2) / 0.8,
            0.0,
            1.0,
        )
        safe_observation = legacy_observation.clone()
        command_slice = slice(
            self.command_offset,
            self.command_offset + self.command_dimension,
        )
        safe_observation[..., command_slice] *= safe_scale.unsqueeze(-1)
        safe_action = self.backbone(safe_observation)
        gain = torch.clamp(
            self.safe_command_gain,
            0.0,
            self.safe_command_gain_limit,
        )
        return legacy_action + gain * (safe_action - legacy_action)


def _checkpoint_actor(
    checkpoint_path: Path,
    agent_config_path: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    agent_config = yaml.safe_load(agent_config_path.read_text(encoding="utf-8"))
    policy_config = agent_config["policy"]
    if policy_config.get("actor_obs_normalization", False):
        raise ValueError("Checkpoint reconstruction does not support actor observation normalization")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    activation_type = _activation(policy_config["activation"])
    if "actor.backbone.0.weight" in state:
        backbone, backbone_dimensions = _checkpoint_mlp(
            state, "actor.backbone.", activation_type
        )
        output_dimension = backbone_dimensions[-1][1]
        confidence_offset = int(policy_config["confidence_offset"])
        if "actor.safe_command_gain" in state:
            input_dimension = 51
            actor = _SafeCommandCheckpointActor(
                backbone,
                state["actor.safe_command_gain"],
                confidence_offset,
                int(policy_config["command_offset"]),
                int(policy_config["command_dimension"]),
                float(policy_config.get("safe_command_gain_limit", 1.0)),
            ).eval()
            details = {
                "architecture": "frozen_backbone_exact_safe_command",
                "checkpoint_iteration": checkpoint.get("iter"),
                "activation": policy_config["activation"],
                "input_dimension": input_dimension,
                "output_dimension": output_dimension,
                "backbone_layer_dimensions": backbone_dimensions,
                "confidence_offset": confidence_offset,
                "command_offset": int(policy_config["command_offset"]),
                "command_dimension": int(policy_config["command_dimension"]),
                "safe_command_gain": state["actor.safe_command_gain"].tolist(),
                "safe_command_gain_limit": float(
                    policy_config.get("safe_command_gain_limit", 1.0)
                ),
                "actor_observation_normalization": False,
            }
        else:
            residual, residual_dimensions = _checkpoint_mlp(
                state, "actor.residual.", activation_type
            )
            if "actor.action_skip.weight" in state:
                action_skip = torch.nn.Linear(output_dimension, output_dimension)
                action_skip.weight.data.copy_(state["actor.action_skip.weight"])
                action_skip.bias.data.copy_(state["actor.action_skip.bias"])
                action_skip_kind = "learned_linear"
            else:
                action_skip = torch.nn.Identity()
                action_skip_kind = "identity"
            input_dimension = residual_dimensions[0][0] - output_dimension
            actor = _ResidualCheckpointActor(
                backbone,
                residual,
                action_skip,
                float(policy_config["residual_action_limit"]),
                confidence_offset,
            ).eval()
            details = {
                "architecture": "frozen_backbone_confidence_residual",
                "checkpoint_iteration": checkpoint.get("iter"),
                "activation": policy_config["activation"],
                "input_dimension": input_dimension,
                "output_dimension": output_dimension,
                "backbone_layer_dimensions": backbone_dimensions,
                "residual_layer_dimensions": residual_dimensions,
                "residual_action_limit": float(policy_config["residual_action_limit"]),
                "confidence_offset": confidence_offset,
                "action_skip": action_skip_kind,
                "actor_observation_normalization": False,
            }
    else:
        actor, layer_dimensions = _checkpoint_mlp(
            state, "actor.", activation_type
        )
        details = {
            "architecture": "dense_mlp",
            "checkpoint_iteration": checkpoint.get("iter"),
            "activation": policy_config["activation"],
            "input_dimension": layer_dimensions[0][0],
            "output_dimension": layer_dimensions[-1][1],
            "layer_dimensions": layer_dimensions,
            "actor_observation_normalization": False,
        }
    return actor, details


def _comparison(reference: np.ndarray, candidate: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    difference = np.abs(reference - candidate)
    relative = difference / np.maximum(np.abs(reference), atol)
    return {
        "allclose": bool(np.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "max_relative_error": float(relative.max()),
    }


def main() -> int:
    args = _parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("Comparison tolerances must be non-negative")

    checkpoint_path = _resolve_project_file(args.checkpoint)
    jit_path = _resolve_project_file(args.jit)
    onnx_path = _resolve_project_file(args.onnx)
    default_agent_config = checkpoint_path.parent / "params" / "agent.yaml"
    agent_config_path = _resolve_project_file(args.agent_config or default_agent_config)

    actor, checkpoint_details = _checkpoint_actor(checkpoint_path, agent_config_path)
    jit_policy = torch.jit.load(str(jit_path), map_location="cpu").eval()
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    onnx_policy = ReferenceEvaluator(onnx_model)

    input_name = onnx_model.graph.input[0].name
    output_name = onnx_model.graph.output[0].name
    input_dimension = checkpoint_details["input_dimension"]
    output_dimension = checkpoint_details["output_dimension"]
    if input_dimension not in (48, 51) or output_dimension != 12:
        raise ValueError(
            "Checkpoint actor dimensions must follow the 48-D base or 51-D "
            f"confidence contract, received {input_dimension} -> {output_dimension}"
        )

    generator = np.random.default_rng(args.seed)
    observations = generator.standard_normal((args.samples, input_dimension), dtype=np.float32)
    observations[0] = 0.0
    observations[1] = np.linspace(-3.0, 3.0, input_dimension, dtype=np.float32)
    if input_dimension == 51:
        observations[:, 48:51] = generator.random((args.samples, 3), dtype=np.float32)
        observations[0, 48:51] = (1.0, 1.0, 0.0)
        observations[1, 48:51] = (1.0, 0.0, 1.0)

    with torch.inference_mode():
        checkpoint_outputs = actor(torch.from_numpy(observations)).numpy()
        jit_outputs = jit_policy(torch.from_numpy(observations)).numpy()
    onnx_outputs = np.concatenate(
        [onnx_policy.run([output_name], {input_name: sample[None, :]})[0] for sample in observations],
        axis=0,
    )

    comparisons = {
        "checkpoint_vs_torchscript": _comparison(
            checkpoint_outputs,
            jit_outputs,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "checkpoint_vs_onnx": _comparison(
            checkpoint_outputs,
            onnx_outputs,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "torchscript_vs_onnx": _comparison(
            jit_outputs,
            onnx_outputs,
            atol=args.atol,
            rtol=args.rtol,
        ),
    }
    passed = all(result["allclose"] for result in comparisons.values())
    report = {
        "passed": passed,
        "samples": args.samples,
        "seed": args.seed,
        "tolerances": {"absolute": args.atol, "relative": args.rtol},
        "input": {"dimension": input_dimension, "dtype": "float32"},
        "output": {"dimension": output_dimension, "dtype": "float32"},
        "checkpoint_reconstruction": checkpoint_details,
        "onnx_backend": "onnx.reference.ReferenceEvaluator",
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "numpy": np.__version__,
        },
        "artifacts": {
            "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
            "torchscript": {"path": str(jit_path), "sha256": _sha256(jit_path)},
            "onnx": {"path": str(onnx_path), "sha256": _sha256(onnx_path)},
            "agent_config": {"path": str(agent_config_path), "sha256": _sha256(agent_config_path)},
        },
        "comparisons": comparisons,
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = args.output.expanduser().resolve()
        if output_path != PROJECT_ROOT and PROJECT_ROOT not in output_path.parents:
            raise ValueError(f"Output path escapes project root: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"[INFO] Wrote parity report: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
