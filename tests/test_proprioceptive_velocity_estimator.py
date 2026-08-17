"""Contract, temporal-history, artifact, and frame-conversion tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from anymal_locomotion.velocity_estimator.contract import (
    HISTORY_LENGTH,
    STEP_DIMENSION,
    HistoryBuffer,
    assemble_proprioceptive_step,
)
from anymal_locomotion_ros2.hardware_state_estimator_adapter_core import (
    adapt_linear_velocity,
)
from anymal_locomotion_ros2.proprioceptive_velocity_estimator_core import (
    EstimatorInputSynchronizer,
    EstimatorRuntime,
    assemble_step,
    load_estimator_metadata,
    reorder_foot_contacts,
)


def _sample() -> dict[str, np.ndarray]:
    return {
        "angular_velocity": np.asarray([0.1, 0.2, 0.3]),
        "linear_acceleration": np.asarray([0.0, 0.0, 9.81]),
        "projected_gravity": np.asarray([0.0, 0.0, -1.0]),
        "relative_joint_position": np.arange(12) * 0.01,
        "joint_velocity": np.arange(12) * -0.1,
        "foot_contact_probability": np.asarray([1.0, 0.0, 0.5, 1.0]),
    }


def test_training_and_runtime_step_contracts_are_identical() -> None:
    sample = _sample()
    training = assemble_proprioceptive_step(**sample)
    runtime = assemble_step(*sample.values())
    assert training.shape == (STEP_DIMENSION,)
    np.testing.assert_array_equal(training, runtime)
    np.testing.assert_allclose(training[:3], sample["angular_velocity"], atol=1.0e-7)
    np.testing.assert_allclose(training[-4:], sample["foot_contact_probability"], atol=1.0e-7)


def test_history_is_oldest_to_newest_and_resets_on_gap() -> None:
    history = HistoryBuffer()
    for index in range(HISTORY_LENGTH):
        result = history.append(index * 0.02, np.full(STEP_DIMENSION, index))
    assert result is not None
    assert result.shape == (HISTORY_LENGTH * STEP_DIMENSION,)
    np.testing.assert_array_equal(result[:STEP_DIMENSION], np.zeros(STEP_DIMENSION))
    np.testing.assert_array_equal(result[-STEP_DIMENSION:], np.full(STEP_DIMENSION, 19))
    assert history.append(2.0, np.zeros(STEP_DIMENSION)) is None
    assert history.sample_count == 1


def test_foot_contacts_are_remapped_by_name_and_invalid_input_fails() -> None:
    actual = reorder_foot_contacts(
        ["RH_FOOT", "RF_FOOT", "LH_FOOT", "LF_FOOT"],
        [0.4, 0.3, 0.2, 0.1],
    )
    np.testing.assert_allclose(actual, [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="mismatch"):
        reorder_foot_contacts(["LF_FOOT", "LH_FOOT", "RF_FOOT", "bad"], [1, 1, 1, 1])


def test_runtime_waits_for_history_and_rejects_unsafe_output() -> None:
    runtime = EstimatorRuntime(lambda value: np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    for index in range(HISTORY_LENGTH - 1):
        assert runtime.step(index * 0.02, np.zeros(STEP_DIMENSION)) is None
    np.testing.assert_array_equal(
        runtime.step((HISTORY_LENGTH - 1) * 0.02, np.zeros(STEP_DIMENSION)),
        [1.0, 2.0, 3.0],
    )
    unsafe = EstimatorRuntime(lambda value: np.asarray([[9.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="guard"):
        for index in range(HISTORY_LENGTH):
            unsafe.step(index * 0.02, np.zeros(STEP_DIMENSION))


@pytest.mark.parametrize(
    "order",
    [
        ("joint", "imu_before", "contact", "imu_after"),
        ("imu_before", "imu_after", "contact", "joint"),
        ("contact", "joint", "imu_before", "imu_after"),
    ],
)
def test_input_synchronizer_is_independent_of_cross_topic_arrival_order(
    order: tuple[str, ...],
) -> None:
    synchronizer = EstimatorInputSynchronizer(tolerance_s=0.025)
    operations = {
        "joint": lambda: synchronizer.push_joint(20_000_000, "joint"),
        "imu_before": lambda: synchronizer.push_imu(15_000_000, "imu-before"),
        "imu_after": lambda: synchronizer.push_imu(25_000_000, "imu-after"),
        "contact": lambda: synchronizer.push_contacts(20_000_000, "contact"),
    }
    bundles = []
    for name in order:
        bundles.extend(operations[name]())
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.synchronized
    assert bundle.joint == "joint"
    assert bundle.imu == "imu-before"
    assert bundle.contacts == "contact"


def test_input_synchronizer_flags_out_of_tolerance_and_timestamp_regression() -> None:
    synchronizer = EstimatorInputSynchronizer(tolerance_s=0.005)
    assert not synchronizer.push_joint(20_000_000, "joint")
    assert not synchronizer.push_contacts(30_000_000, "contact")
    bundles = synchronizer.push_imu(30_000_000, "imu")
    assert len(bundles) == 1
    assert not bundles[0].synchronized
    with pytest.raises(ValueError, match="regressed"):
        synchronizer.push_imu(29_000_000, "old")


def test_metadata_loader_verifies_model_digest(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    document = {
        "schema_version": 1,
        "contract_id": "anymal-d-proprioceptive-velocity-v1",
        "history_length": 20,
        "step_dimension": 37,
        "input_dimension": 740,
        "output_frame": "base_link",
        "artifacts": {"onnx": {"path": model_path.name, "sha256": digest}},
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(document), encoding="utf-8")
    _, actual_path = load_estimator_metadata(metadata_path)
    assert actual_path == model_path
    model_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_estimator_metadata(metadata_path)


def test_vendor_world_velocity_is_rotated_to_body() -> None:
    half = np.sqrt(0.5)
    actual = adapt_linear_velocity(
        [0.0, 1.0, 0.0],
        [0.0, 0.0, half, half],
        input_velocity_frame="odom",
    )
    np.testing.assert_allclose(actual, [1.0, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_array_equal(
        adapt_linear_velocity([1.0, 2.0, 3.0], [0, 0, 0, 1], input_velocity_frame="body"),
        [1.0, 2.0, 3.0],
    )
