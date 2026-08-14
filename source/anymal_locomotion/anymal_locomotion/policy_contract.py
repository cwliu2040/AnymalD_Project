"""Load and validate the versioned ANYmal-D policy contract."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import yaml

from anymal_locomotion.artifacts import PROJECT_ROOT, assert_project_local_path, git_revision

POLICY_CONTRACT_PATH = PROJECT_ROOT / "configs" / "policy_contract.yaml"
SLAM_CONFIDENCE_POLICY_CONTRACT_PATH = (
    PROJECT_ROOT / "configs" / "policy_contract_slam_confidence.yaml"
)


def load_policy_contract(path: str | Path) -> dict[str, Any]:
    """Load one project-owned policy contract."""
    contract_path = Path(path).expanduser().resolve()
    with contract_path.open(encoding="utf-8") as stream:
        contract = yaml.safe_load(stream)
    if not isinstance(contract, dict):
        raise ValueError(f"Policy contract must be a mapping: {contract_path}")
    return contract


POLICY_CONTRACT = load_policy_contract(POLICY_CONTRACT_PATH)
SLAM_CONFIDENCE_POLICY_CONTRACT = load_policy_contract(
    SLAM_CONFIDENCE_POLICY_CONTRACT_PATH
)
CANONICAL_JOINT_ORDER = tuple(item["name"] for item in POLICY_CONTRACT["joints"])
JOINT_TO_POLICY_INDEX = {name: index for index, name in enumerate(CANONICAL_JOINT_ORDER)}


def validate_contract_definition() -> None:
    """Validate dimensions, indices, and duplicate names in the contract file."""
    indices = [item["policy_index"] for item in POLICY_CONTRACT["joints"]]
    duplicates = sorted(name for name, count in Counter(CANONICAL_JOINT_ORDER).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate canonical joint names: {duplicates}")
    if indices != list(range(12)):
        raise ValueError(f"Canonical policy indices must be 0..11 in order, received: {indices}")
    if len(CANONICAL_JOINT_ORDER) != POLICY_CONTRACT["action"]["dimension"]:
        raise ValueError("Joint count does not match action dimension")
    observation_total = sum(term["dimension"] for term in POLICY_CONTRACT["observation"]["terms"])
    if observation_total != POLICY_CONTRACT["observation"]["dimension"]:
        raise ValueError(
            f"Observation term dimensions sum to {observation_total}, "
            f"expected {POLICY_CONTRACT['observation']['dimension']}"
        )


def validate_runtime_joint_names(actual_joint_names: Sequence[str]) -> tuple[int, ...]:
    """Validate runtime actuated joints and return canonical-to-runtime indices.

    Runtime ordering may differ from the canonical order. Name-based remapping is
    required; missing, duplicate, or unexpected actuated joints are fatal.
    """
    actual = tuple(actual_joint_names)
    duplicates = sorted(name for name, count in Counter(actual).items() if count > 1)
    missing = sorted(set(CANONICAL_JOINT_ORDER) - set(actual))
    unexpected = sorted(set(actual) - set(CANONICAL_JOINT_ORDER))
    if duplicates or missing or unexpected:
        raise ValueError(
            "ANYmal-D joint contract mismatch: "
            f"missing={missing}, duplicate={duplicates}, unexpected={unexpected}"
        )
    return tuple(actual.index(name) for name in CANONICAL_JOINT_ORDER)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_export_metadata(
    export_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    contract_path: str | Path = POLICY_CONTRACT_PATH,
    policy_extension: dict[str, Any] | None = None,
) -> Path:
    """Write versioned deployment metadata beside a policy export."""
    export_path = assert_project_local_path(export_dir)
    checkpoint = assert_project_local_path(checkpoint_path)
    selected_contract_path = assert_project_local_path(contract_path)
    selected_contract = load_policy_contract(selected_contract_path)
    project_revision = git_revision(PROJECT_ROOT)
    if project_revision is None:
        raise RuntimeError("Policy export requires a committed project Git revision")
    export_path.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "torchscript": export_path / "policy.pt",
        "onnx": export_path / "policy.onnx",
    }
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Policy export artifacts are missing: {missing}")
    action_metadata = {
        **selected_contract["action"],
        "default_joint_positions": [
            item["default_position"] for item in selected_contract["joints"]
        ],
    }
    metadata = {
        "schema_version": selected_contract["schema_version"],
        "robot": selected_contract["robot"],
        "joint_order": [item["name"] for item in selected_contract["joints"]],
        "observation": selected_contract["observation"],
        "action": action_metadata,
        "command": selected_contract["command"],
        "frames": selected_contract["frames"],
        "normalization": selected_contract["normalization"],
        "versions": selected_contract["versions"],
        "checkpoint": {
            "path": str(checkpoint.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(checkpoint),
        },
        "artifacts": {
            name: {
                "path": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in artifact_paths.items()
        },
        "config_sha256": _sha256(selected_contract_path),
        "project_git_commit": project_revision,
    }
    if policy_extension is not None:
        metadata["policy_extension"] = policy_extension
    metadata_path = export_path / "policy_metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return metadata_path


validate_contract_definition()
