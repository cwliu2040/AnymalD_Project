#!/usr/bin/env python3
"""Export hash-locked Arm-B action-intervention policies for the causal pilot."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_EXPORT_SCRIPT = PROJECT_ROOT / "scripts/validation/export_slam_confidence_publication_arms.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/anymal_d_locomotion_slam_confidence_sim_v1/model_48.pt"
DEFAULT_AGENT_CONFIG = PROJECT_ROOT / (
    "exported/anymal_d_locomotion_slam_confidence_phase_separated_gait_v1/"
    "2026-08-13_18-18-26_nonnegative_smoothing_recovery_safe_ppo_gait/resolved_agent.yaml"
)
DEFAULT_BASE_METADATA = PROJECT_ROOT / (
    "exported/anymal_d_locomotion_slam_confidence_publication_v1/"
    "arm_b_model1450_deterministic_supervisor/policy_metadata.yaml"
)
DEFAULT_BASE_POLICY = DEFAULT_BASE_METADATA.with_name("policy.onnx")
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/slam_action_risk_intervention_pilot.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "exported/slam_action_risk_intervention_pilot_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT) or not resolved.is_file():
        raise ValueError(f"required project file is missing: {resolved}")
    return resolved


def _git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _load_formal_export_module():
    spec = importlib.util.spec_from_file_location("formal_arm_export", FORMAL_EXPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load formal arm exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActionInterventionArm(torch.nn.Module):
    """Apply a signed smoothing residual on top of the frozen Arm-B action."""

    def __init__(self, arm_b: torch.nn.Module, *, alpha: float, limit: float) -> None:
        super().__init__()
        self.arm_b = arm_b
        self.alpha = float(alpha)
        self.limit = float(limit)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        baseline = self.arm_b(observation)
        previous = observation[..., 36:48]
        valid = torch.clamp(observation[..., 49:50], 0.0, 1.0)
        residual = torch.clamp(
            self.alpha * (previous - baseline),
            min=-self.limit,
            max=self.limit,
        )
        return baseline + valid * residual


def _comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    error = np.abs(reference - candidate)
    return {
        "max_absolute_error": float(np.max(error)),
        "mean_absolute_error": float(np.mean(error)),
        "allclose_atol_1e-5": bool(
            np.allclose(reference, candidate, atol=1.0e-5, rtol=1.0e-5)
        ),
    }


def _export_one(
    *, name: str, alpha: float, limit: float, arm_b: torch.nn.Module,
    observations: torch.Tensor, base_metadata: dict[str, Any],
    base_policy: Path, checkpoint: Path, agent_config: Path, output_root: Path,
) -> dict[str, Any]:
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    module = ActionInterventionArm(arm_b, alpha=alpha, limit=limit).eval()
    example = observations[:1]
    torchscript_path = output_dir / "policy.pt"
    onnx_path = output_dir / "policy.onnx"
    traced = torch.jit.trace(module, example, check_trace=True)
    torch.jit.save(traced, str(torchscript_path))
    torch.onnx.export(
        module, example, str(onnx_path), input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    evaluator = ReferenceEvaluator(model)
    with torch.inference_mode():
        reference = module(observations).numpy()
        baseline = arm_b(observations).numpy()
        jit_output = traced(observations).numpy()
        invalid_observations = observations.clone()
        invalid_observations[:, 49] = 0.0
        invalid_reference = module(invalid_observations).numpy()
        invalid_baseline = arm_b(invalid_observations).numpy()
    onnx_output = evaluator.run(None, {"observation": observations.numpy()})[0]
    residual = reference - baseline
    checks = {
        "module_vs_torchscript": _comparison(reference, jit_output),
        "module_vs_onnx": _comparison(reference, onnx_output),
        "invalid_exact_arm_B": bool(np.array_equal(invalid_reference, invalid_baseline)),
        "zero_exact_arm_B": bool(alpha != 0.0 or np.array_equal(reference, baseline)),
        "residual_within_limit": bool(
            np.max(np.abs(residual)) <= limit + np.finfo(np.float32).eps
        ),
    }
    passed = bool(
        checks["module_vs_torchscript"]["allclose_atol_1e-5"]
        and checks["module_vs_onnx"]["allclose_atol_1e-5"]
        and checks["invalid_exact_arm_B"]
        and checks["zero_exact_arm_B"]
        and checks["residual_within_limit"]
    )
    resolved_agent = output_dir / "resolved_agent.yaml"
    shutil.copy2(agent_config, resolved_agent)
    metadata = deepcopy(base_metadata)
    metadata["checkpoint"] = {
        "path": str(checkpoint.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(checkpoint),
    }
    metadata["artifacts"] = {
        "torchscript": {"path": "policy.pt", "sha256": _sha256(torchscript_path)},
        "onnx": {"path": "policy.onnx", "sha256": _sha256(onnx_path)},
    }
    metadata["policy_extension"] = {
        "architecture": "frozen_arm_B_previous_action_intervention",
        "dataset_role": "excluded_causal_development",
        "intervention_arm": name,
        "alpha": alpha,
        "raw_action_linf_limit": limit,
        "apply_only_while_tracking_valid": True,
        "invalid_stale_fallback": "exact_arm_B",
        "base_policy": {
            "path": str(base_policy.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(base_policy),
        },
        "export_worktree_base_commit": _git_commit(),
        "runtime_ground_truth_inputs": False,
        "runtime_future_label_inputs": False,
        "resolved_agent_config": {
            "path": "resolved_agent.yaml", "sha256": _sha256(resolved_agent),
        },
    }
    metadata_path = output_dir / "policy_metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    report = {
        "schema_version": 1,
        "kind": "slam_action_risk_intervention_arm_export",
        "arm": name,
        "alpha": alpha,
        "passed": passed,
        "checks": checks,
        "artifacts": {
            "policy": {"path": str(onnx_path), "sha256": _sha256(onnx_path)},
            "torchscript": {"path": str(torchscript_path), "sha256": _sha256(torchscript_path)},
            "metadata": {"path": str(metadata_path), "sha256": _sha256(metadata_path)},
        },
    }
    (output_dir / "export_parity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"intervention arm export failed: {name}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--agent-config", type=Path, default=DEFAULT_AGENT_CONFIG)
    parser.add_argument("--base-metadata", type=Path, default=DEFAULT_BASE_METADATA)
    parser.add_argument("--base-policy", type=Path, default=DEFAULT_BASE_POLICY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    checkpoint = _project_file(args.checkpoint)
    agent_config = _project_file(args.agent_config)
    base_metadata_path = _project_file(args.base_metadata)
    base_policy = _project_file(args.base_policy)
    protocol_path = _project_file(args.protocol)
    output_root = args.output_root.expanduser().resolve()
    if not output_root.is_relative_to(PROJECT_ROOT):
        raise ValueError("output root must remain inside the project")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    arm_config = protocol["intervention"]["arms"]
    limit = float(protocol["intervention"]["raw_action_linf_limit"])
    formal = _load_formal_export_module()
    actor, _ = formal._load_checkpoint_actor(checkpoint, agent_config)
    arm_b = formal.DeterministicSupervisorArm(actor).eval()
    base_metadata = yaml.safe_load(base_metadata_path.read_text(encoding="utf-8"))
    generator = torch.Generator().manual_seed(20260821)
    observations = torch.randn((512, 51), generator=generator)
    observations[:, 48:51] = torch.rand((512, 3), generator=generator)
    observations[0].zero_()
    observations[0, 48:51] = torch.tensor([1.0, 1.0, 0.0])
    results = [
        _export_one(
            name=name, alpha=float(config["alpha"]), limit=limit,
            arm_b=arm_b, observations=observations, base_metadata=base_metadata,
            base_policy=base_policy, checkpoint=checkpoint,
            agent_config=agent_config, output_root=output_root,
        )
        for name, config in arm_config.items()
    ]
    manifest = {
        "schema_version": 1,
        "kind": "slam_action_risk_intervention_arm_exports",
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "base_policy": {"path": str(base_policy), "sha256": _sha256(base_policy)},
        "passed": all(result["passed"] for result in results),
        "arms": {result["arm"]: result["artifacts"] for result in results},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
