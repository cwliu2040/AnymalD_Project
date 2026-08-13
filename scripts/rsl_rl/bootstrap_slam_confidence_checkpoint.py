#!/usr/bin/env python3
"""Reproduce the legacy first-round full-state 48-D to 51-D bootstrap.

This expands actor and critic and preserves optimizer state.  It must not be
used as the second-round formal starting point; use train.py
--actor-only-warm-start instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_last_dimension(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[1] != 48:
        return tensor
    expanded = tensor.new_zeros((tensor.shape[0], 51))
    expanded[:, :48] = tensor
    return expanded


def bootstrap(source: Path, output: Path) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    expected = {"actor.0.weight", "critic.0.weight"}
    if not expected.issubset(state):
        raise ValueError(f"source checkpoint lacks first-layer weights: {expected - set(state)}")
    if any(tuple(state[name].shape) != (128, 48) for name in expected):
        raise ValueError("source checkpoint is not the expected 48-D model1450 architecture")

    for name in expected:
        state[name] = _expand_last_dimension(state[name])
    optimizer = checkpoint["optimizer_state_dict"]
    for parameter_state in optimizer["state"].values():
        for name in ("exp_avg", "exp_avg_sq"):
            value = parameter_state.get(name)
            if isinstance(value, torch.Tensor):
                parameter_state[name] = _expand_last_dimension(value)

    if not isinstance(checkpoint.get("infos"), dict):
        checkpoint["infos"] = {}
    checkpoint["infos"]["slam_confidence_bootstrap"] = {
        "status": "legacy_first_round_only",
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_observation_dimension": 48,
        "target_observation_dimension": 51,
        "new_input_columns": "zero_initialized",
        "optimizer_moments_new_columns": "zero_initialized",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "schema_version": 1,
        "kind": "legacy_full_state_slam_confidence_checkpoint_bootstrap",
        "formal_second_round_authorized": False,
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "iteration": int(checkpoint["iter"]),
        "actor_first_layer_shape": list(state["actor.0.weight"].shape),
        "critic_first_layer_shape": list(state["critic.0.weight"].shape),
        "legacy_output_parity": "exact because new input columns are zero",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = bootstrap(args.source, args.output)
    report_path = args.output.expanduser().resolve().with_suffix(".bootstrap.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
