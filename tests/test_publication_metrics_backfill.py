from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/backfill_slam_confidence_publication_metrics.py"
SPEC = importlib.util.spec_from_file_location("publication_backfill", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_infer_publication_cell_identity() -> None:
    identity = MODULE.infer_identity(
        ROOT / "logs/example/fastlio2/route/gradual_support_loss/block_43/arm_C"
    )
    assert identity == {
        "backend": "fastlio2", "profile": "route", "condition": "gradual_support_loss",
        "block": 43, "arm": "C",
    }


def test_metadata_parent_contract_selects_cell_not_bag(tmp_path: Path) -> None:
    metadata = tmp_path / "fastlio2/route/native/block_43/arm_A/raw_bag/metadata.yaml"
    metadata.parent.mkdir(parents=True)
    metadata.touch()
    selected = [path.parent.parent for path in tmp_path.glob("**/raw_bag/metadata.yaml")]
    assert selected == [metadata.parent.parent]
    assert selected[0].name == "arm_A"
