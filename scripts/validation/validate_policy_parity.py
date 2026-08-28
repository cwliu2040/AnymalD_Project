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
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=200,
        help="Closed-loop previous-action recurrence steps.",
    )
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



def _checkpoint_actor(
    checkpoint_path: Path,
    agent_config_path: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct the current dense RSL-RL actor for export parity."""
    agent_config = yaml.safe_load(agent_config_path.read_text(encoding="utf-8"))
    policy_config = agent_config["policy"]
    if policy_config.get("actor_obs_normalization", False):
        raise ValueError(
            "Checkpoint reconstruction does not support actor observation normalization"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    activation_type = _activation(policy_config["activation"])
    if "actor.backbone.0.weight" in state:
        raise ValueError(
            "retired frozen-backbone confidence checkpoints are not supported"
        )
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
    if args.rollout_steps < 2:
        raise ValueError("--rollout-steps must be at least 2")
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

    rollout_generator = np.random.default_rng(args.seed + 1)
    rollout_base = rollout_generator.standard_normal(
        (args.rollout_steps, input_dimension), dtype=np.float32
    ) * np.float32(0.25)
    rollout_base[:, 9:12] = np.asarray([1.5, 0.0, 0.0], dtype=np.float32)
    if input_dimension == 51:
        phases = np.arange(args.rollout_steps, dtype=np.float32) / np.float32(
            args.rollout_steps
        )
        confidence = np.ones(args.rollout_steps, dtype=np.float32)
        valid = np.ones(args.rollout_steps, dtype=np.float32)
        age = np.zeros(args.rollout_steps, dtype=np.float32)
        degrading = (phases >= 0.30) & (phases < 0.50)
        degrade_progress = np.clip((phases - 0.30) / 0.20, 0.0, 1.0)
        confidence[degrading] = 1.0 - 0.8 * degrade_progress[degrading]
        age[degrading] = 0.30 * degrade_progress[degrading]
        lost = (phases >= 0.50) & (phases < 0.70)
        confidence[lost] = 0.2
        valid[lost] = 0.0
        age[lost] = 0.3 + 0.7 * np.clip(
            (phases[lost] - 0.50) / 0.20, 0.0, 1.0
        )
        recovering = phases >= 0.70
        recover_progress = np.clip((phases - 0.70) / 0.30, 0.0, 1.0)
        confidence[recovering] = 0.2 + 0.8 * recover_progress[recovering]
        valid[recovering] = (recover_progress[recovering] >= (1.0 / 3.0)).astype(
            np.float32
        )
        age[recovering] = 0.5 * (1.0 - recover_progress[recovering])
        rollout_base[:, 48:51] = np.stack((confidence, valid, age), axis=1)

    rollout_outputs: dict[str, list[np.ndarray]] = {
        "checkpoint": [],
        "torchscript": [],
        "onnx": [],
    }
    rollout_previous = {
        name: np.zeros(output_dimension, dtype=np.float32)
        for name in rollout_outputs
    }
    for base in rollout_base:
        per_backend_observations = {}
        for name in rollout_outputs:
            observation = base.copy()
            observation[36:48] = rollout_previous[name]
            per_backend_observations[name] = observation[None, :]
        with torch.inference_mode():
            checkpoint_action = actor(
                torch.from_numpy(per_backend_observations["checkpoint"])
            ).numpy()[0]
            jit_action = jit_policy(
                torch.from_numpy(per_backend_observations["torchscript"])
            ).numpy()[0]
        onnx_action = onnx_policy.run(
            [output_name],
            {input_name: per_backend_observations["onnx"]},
        )[0][0]
        for name, action in (
            ("checkpoint", checkpoint_action),
            ("torchscript", jit_action),
            ("onnx", onnx_action),
        ):
            rollout_outputs[name].append(action.copy())
            rollout_previous[name] = action.astype(np.float32, copy=True)
    rollout_arrays = {
        name: np.stack(actions, axis=0)
        for name, actions in rollout_outputs.items()
    }
    rollout_comparisons = {
        "checkpoint_vs_torchscript": _comparison(
            rollout_arrays["checkpoint"],
            rollout_arrays["torchscript"],
            atol=args.atol,
            rtol=args.rtol,
        ),
        "checkpoint_vs_onnx": _comparison(
            rollout_arrays["checkpoint"],
            rollout_arrays["onnx"],
            atol=args.atol,
            rtol=args.rtol,
        ),
        "torchscript_vs_onnx": _comparison(
            rollout_arrays["torchscript"],
            rollout_arrays["onnx"],
            atol=args.atol,
            rtol=args.rtol,
        ),
    }
    passed = all(
        result["allclose"]
        for group in (comparisons, rollout_comparisons)
        for result in group.values()
    )
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
        "closed_loop_previous_action_rollout": {
            "steps": args.rollout_steps,
            "confidence_schedule": (
                "healthy_degrade_invalid_recover" if input_dimension == 51 else None
            ),
            "comparisons": rollout_comparisons,
        },
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
