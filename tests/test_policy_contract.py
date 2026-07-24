"""Deterministic joint and policy metadata contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anymal_locomotion.policy_contract import (
    CANONICAL_JOINT_ORDER,
    POLICY_CONTRACT,
    validate_contract_definition,
    validate_runtime_joint_names,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_joint_order() -> None:
    assert CANONICAL_JOINT_ORDER == (
        "LF_HAA",
        "LH_HAA",
        "RF_HAA",
        "RH_HAA",
        "LF_HFE",
        "LF_KFE",
        "LH_HFE",
        "LH_KFE",
        "RF_HFE",
        "RF_KFE",
        "RH_HFE",
        "RH_KFE",
    )
    validate_contract_definition()


def test_name_based_runtime_remapping() -> None:
    runtime_order = tuple(reversed(CANONICAL_JOINT_ORDER))
    mapping = validate_runtime_joint_names(runtime_order)
    assert tuple(runtime_order[index] for index in mapping) == CANONICAL_JOINT_ORDER


@pytest.mark.parametrize(
    "runtime_names, expected_error",
    [
        (CANONICAL_JOINT_ORDER[:-1], "missing"),
        (CANONICAL_JOINT_ORDER + (CANONICAL_JOINT_ORDER[0],), "duplicate"),
        (CANONICAL_JOINT_ORDER[:-1] + ("CUSTOM_JOINT",), "unexpected"),
    ],
)
def test_invalid_runtime_joint_sets_fail(runtime_names: tuple[str, ...], expected_error: str) -> None:
    with pytest.raises(ValueError, match=expected_error):
        validate_runtime_joint_names(runtime_names)


def test_policy_io_dimensions_and_offsets() -> None:
    observation = POLICY_CONTRACT["observation"]
    assert observation["dimension"] == 48
    assert [term["offset"] for term in observation["terms"]] == [0, 3, 6, 9, 12, 24, 36]
    assert sum(term["dimension"] for term in observation["terms"]) == 48
    assert POLICY_CONTRACT["action"]["dimension"] == 12
    assert POLICY_CONTRACT["action"]["control_period_s"] == pytest.approx(0.02)
    assert POLICY_CONTRACT["schema_version"] == "1.1.0"
    assert POLICY_CONTRACT["command"]["limits"] == {
        "vx": [-2.0, 3.0],
        "vy": [-1.5, 1.5],
        "wz": [-2.0, 2.0],
    }


def test_metadata_schema_and_example_have_required_fields() -> None:
    schema = json.loads((PROJECT_ROOT / "configs" / "policy_metadata.schema.json").read_text(encoding="utf-8"))
    example = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "policy_metadata.example.yaml").read_text(encoding="utf-8")
    )
    assert set(schema["required"]).issubset(example)
    assert len(example["joint_order"]) == 12
    assert example["observation"]["dimension"] == 48
    assert example["action"]["dimension"] == 12
    assert example["action"]["default_joint_positions"] == [
        joint["default_position"] for joint in POLICY_CONTRACT["joints"]
    ]
    assert set(example["artifacts"]) == {"torchscript", "onnx"}
    assert example["artifacts"]["torchscript"]["path"] == "policy.pt"
    assert example["artifacts"]["onnx"]["path"] == "policy.onnx"
