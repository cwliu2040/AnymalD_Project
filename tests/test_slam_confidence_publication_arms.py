from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/export_slam_confidence_publication_arms.py"
RELEASE = ROOT / "configs/slam_confidence_sim_release_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publication_arm_export_has_registered_semantics() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    classes = {
        node.name: ast.get_docstring(node)
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    assert "deterministic safe command" in classes["DeterministicSupervisorArm"]
    assert "structured gait delta forced to zero" in classes["IntentOnlyArm"]


def test_publication_arm_export_writes_hash_locked_parity_artifacts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"publication_arm": arm_id' in source
    assert '"structured_delta_forced_exact_zero": arm_id == "D"' in source
    assert '"deterministic_supervisor": arm_id == "B"' in source
    assert '"source_checkpoint_sha256": _sha256(checkpoint)' in source
    assert '"module_vs_torchscript"' in source
    assert '"module_vs_onnx"' in source


def test_sim_release_manifest_is_branch_only_and_artifact_identifiable() -> None:
    release = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))
    assert release["branch"] == "exp/slam-fastlio2"
    assert release["main_branch_mutation_allowed"] is False
    assert release["formal_default_switch_authorized"] is False
    assert release["rollback"]["release"] == "recovery-v0.4.0-model1450"
    for arm in release["arms"].values():
        policy = ROOT / arm["policy_path"]
        metadata = ROOT / arm["metadata_path"]
        assert _sha256(policy) == arm["policy_sha256"]
        assert _sha256(metadata) == arm["metadata_sha256"]
    assert release["arms"]["C"]["checkpoint_curation_required"] is True
    assert release["promotion_rule"]["merge_to_main"] == "forbidden"
