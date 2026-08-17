#!/usr/bin/env python3
"""Export hash-locked deterministic-supervisor and intent-only policy arms."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import yaml
from onnx.reference import ReferenceEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARITY_SCRIPT = PROJECT_ROOT / "scripts/validation/validate_policy_parity.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / (
    "logs/rsl_rl/anymal_d_locomotion_slam_confidence_constrained_gait_v1/"
    "2026-08-13_18-18-26_nonnegative_smoothing_recovery_safe_ppo_gait/"
    "model_48.pt"
)
DEFAULT_SOURCE_EXPORT = PROJECT_ROOT / (
    "exported/anymal_d_locomotion_slam_confidence_phase_separated_gait_v1/"
    "2026-08-13_18-18-26_nonnegative_smoothing_recovery_safe_ppo_gait"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / (
    "exported/anymal_d_locomotion_slam_confidence_publication_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(path: Path, *, must_exist: bool, directory: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path must remain inside {PROJECT_ROOT}: {resolved}")
    if must_exist:
        expected = resolved.is_dir() if directory else resolved.is_file()
        if not expected:
            raise FileNotFoundError(resolved)
    return resolved


def _load_checkpoint_actor(checkpoint: Path, agent_config: Path):
    spec = importlib.util.spec_from_file_location(
        "slam_confidence_policy_parity_export",
        PARITY_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load policy reconstruction: {PARITY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._checkpoint_actor(checkpoint, agent_config)


class DeterministicSupervisorArm(torch.nn.Module):
    """Arm B: frozen model1450 evaluated at the deterministic safe command."""

    def __init__(self, actor) -> None:
        super().__init__()
        self.backbone = actor.backbone

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy_observation = observation[..., :48]
        confidence = observation[..., 48]
        valid = observation[..., 49]
        safe_scale = torch.clamp(valid, 0.0, 1.0) * torch.clamp(
            (confidence - 0.2) / 0.8,
            0.0,
            1.0,
        )
        safe_observation = legacy_observation.clone()
        safe_observation[..., 9:12] *= safe_scale.unsqueeze(-1)
        return self.backbone(safe_observation)


class IntentOnlyArm(torch.nn.Module):
    """Arm D: model48 learned intent with structured gait delta forced to zero."""

    def __init__(self, actor) -> None:
        super().__init__()
        self.backbone = actor.backbone
        self.intent_head = actor.intent_head
        self.intent_blend_max = actor.intent_blend_max

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy_observation = observation[..., :48]
        legacy_action = self.backbone(legacy_observation)
        confidence = observation[..., 48]
        valid = observation[..., 49]
        safe_scale = torch.clamp(valid, 0.0, 1.0) * torch.clamp(
            (confidence - 0.2) / 0.8,
            0.0,
            1.0,
        )
        safe_observation = legacy_observation.clone()
        safe_observation[..., 9:12] *= safe_scale.unsqueeze(-1)
        safe_action = self.backbone(safe_observation)
        logits = self.intent_head(torch.cat((observation, legacy_action), dim=-1))
        intent_blend = self.intent_blend_max * torch.clamp(
            torch.tanh(logits),
            min=0.0,
            max=1.0,
        )
        return legacy_action + intent_blend * (safe_action - legacy_action)


def _comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    error = np.abs(reference - candidate)
    return {
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "allclose_atol_1e-5": bool(
            np.allclose(reference, candidate, atol=1.0e-5, rtol=1.0e-5)
        ),
    }


def _git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _export_arm(
    *,
    arm_id: str,
    module: torch.nn.Module,
    source_metadata: dict[str, Any],
    source_agent_config: Path,
    checkpoint: Path,
    output_root: Path,
) -> dict[str, Any]:
    labels = {
        "B": "model1450_deterministic_supervisor",
        "D": "model48_intent_only",
    }
    architectures = {
        "B": "publication_arm_b_deterministic_safe_command",
        "D": "publication_arm_d_model48_intent_only",
    }
    output_dir = output_root / f"arm_{arm_id.lower()}_{labels[arm_id]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    module = module.eval()
    generator = torch.Generator().manual_seed(20260817 + ord(arm_id))
    observations = torch.randn((256, 51), generator=generator)
    observations[:, 48:51] = torch.rand((256, 3), generator=generator)
    observations[0].zero_()
    observations[0, 48:51] = torch.tensor([1.0, 1.0, 0.0])
    observations[1, 48:51] = torch.tensor([0.0, 0.0, 1.0])
    example = observations[0:1]

    torchscript_path = output_dir / "policy.pt"
    onnx_path = output_dir / "policy.onnx"
    traced = torch.jit.trace(module, example, check_trace=True)
    torch.jit.save(traced, str(torchscript_path))
    torch.onnx.export(
        module,
        example,
        str(onnx_path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    evaluator = ReferenceEvaluator(onnx_model)
    with torch.inference_mode():
        reference = module(observations).numpy()
        jit_output = traced(observations).numpy()
    onnx_output = evaluator.run(None, {"observation": observations.numpy()})[0]
    comparisons = {
        "module_vs_torchscript": _comparison(reference, jit_output),
        "module_vs_onnx": _comparison(reference, onnx_output),
    }
    passed = all(value["allclose_atol_1e-5"] for value in comparisons.values())

    resolved_agent_path = output_dir / "resolved_agent.yaml"
    shutil.copy2(source_agent_config, resolved_agent_path)
    metadata = deepcopy(source_metadata)
    metadata["checkpoint"] = {
        "path": str(checkpoint.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(checkpoint),
    }
    metadata["artifacts"] = {
        "torchscript": {
            "path": "policy.pt",
            "sha256": _sha256(torchscript_path),
            "size_bytes": torchscript_path.stat().st_size,
        },
        "onnx": {
            "path": "policy.onnx",
            "sha256": _sha256(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
        },
    }
    metadata["project_git_commit"] = _git_commit()
    metadata["policy_extension"] = {
        **metadata["policy_extension"],
        "architecture": architectures[arm_id],
        "publication_arm": arm_id,
        "publication_label": labels[arm_id],
        "learned_gait_coordinates_applied": False,
        "structured_delta_forced_exact_zero": arm_id == "D",
        "deterministic_supervisor": arm_id == "B",
        "resolved_agent_config": {
            "path": "resolved_agent.yaml",
            "sha256": _sha256(resolved_agent_path),
        },
    }
    metadata_path = output_dir / "policy_metadata.yaml"
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "kind": "slam_confidence_publication_arm_export",
        "arm": arm_id,
        "label": labels[arm_id],
        "passed": passed,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "artifacts": {
            "metadata": {
                "path": str(metadata_path),
                "sha256": _sha256(metadata_path),
            },
            "torchscript": metadata["artifacts"]["torchscript"],
            "onnx": metadata["artifacts"]["onnx"],
        },
        "comparisons": comparisons,
    }
    report_path = output_dir / "export_parity.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(f"publication arm {arm_id} export parity failed")
    return {**report, "report_path": str(report_path)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-export", type=Path, default=DEFAULT_SOURCE_EXPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--arm", action="append", choices=("B", "D"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint = _project_path(args.checkpoint, must_exist=True)
    source_export = _project_path(
        args.source_export,
        must_exist=True,
        directory=True,
    )
    output_root = _project_path(args.output_root, must_exist=False, directory=True)
    metadata_path = source_export / "policy_metadata.yaml"
    agent_config_path = source_export / "resolved_agent.yaml"
    source_metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if _sha256(checkpoint) != source_metadata["checkpoint"]["sha256"]:
        raise ValueError("source checkpoint SHA-256 does not match model48 metadata")
    actor, details = _load_checkpoint_actor(checkpoint, agent_config_path)
    if details.get("architecture") != "frozen_model1450_aux_intent_structured_gait":
        raise ValueError("source checkpoint is not the registered model48 architecture")
    modules = {
        "B": DeterministicSupervisorArm(actor),
        "D": IntentOnlyArm(actor),
    }
    reports = [
        _export_arm(
            arm_id=arm_id,
            module=modules[arm_id],
            source_metadata=source_metadata,
            source_agent_config=agent_config_path,
            checkpoint=checkpoint,
            output_root=output_root,
        )
        for arm_id in (args.arm or ["B", "D"])
    ]
    summary = {
        "schema_version": 1,
        "kind": "slam_confidence_publication_arm_exports",
        "passed": all(report["passed"] for report in reports),
        "arms": reports,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "export_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": summary["passed"], "arms": [r["arm"] for r in reports]}, indent=2))
    print(f"Publication arm export summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
