from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/run_slam_confidence_publication_matrix.py"


def _module():
    spec = importlib.util.spec_from_file_location("publication_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_protocol_generates_complete_balanced_schedule() -> None:
    module = _module()
    protocol = module._load_yaml(ROOT / "configs/slam_confidence_publication_protocol.yaml")
    rows = module.build_schedule(protocol)
    assert len(rows) == 800
    assert {row["simulation_seed"] for row in rows} == set(range(443, 468))
    assert {row["condition"] for row in rows} == {"gradual_support_loss"}
    strata = {}
    for row in rows:
        key = (row["backend"], row["profile"], row["condition"], row["block_id"])
        strata.setdefault(key, []).append(row["arm"])
    assert all(set(arms) == {"A", "B", "C", "D"} for arms in strata.values())


def test_sample_size_pilot_is_complete_and_seed_disjoint() -> None:
    module = _module()
    protocol = module._load_yaml(ROOT / "configs/slam_confidence_publication_protocol.yaml")
    rows = module.build_pilot_schedule(protocol)
    assert len(rows) == 160
    assert {row["simulation_seed"] for row in rows} == {343, 344, 345, 346, 347}
    assert not {row["simulation_seed"] for row in rows} & set(protocol["live_matrix"]["paired_block_ids"])
    assert not {row["simulation_seed"] for row in rows} & set(
        protocol["sample_size_pilot_matrix"]["prior_pilot_block_ids_must_be_disjoint"]
    )
    assert {row["condition"] for row in rows} == {"gradual_support_loss"}
    challenge = protocol["live_matrix"]["perception_conditions"]["gradual_support_loss"]
    assert challenge["minimum_support_fraction"] == 0.20
    assert challenge["timeline_s"] == {
        "healthy": 3.0, "ramp_down": 6.0,
        "low_support_hold": 4.0, "recovery": 3.0,
    }
    assert protocol["challenge_calibration_matrix"]["formal_condition_frozen"] is True


def test_challenge_calibration_schedule_is_small_balanced_and_disjoint() -> None:
    module = _module()
    protocol = module._load_yaml(ROOT / "configs/slam_confidence_publication_protocol.yaml")
    rows = module.build_challenge_calibration_schedule(protocol)
    assert len(rows) == 24
    assert {row["arm"] for row in rows} == {"C", "D"}
    assert {row["backend"] for row in rows} == {"fastlio2", "liosam"}
    assert {row["simulation_seed"] for row in rows} == {243}
    assert {row["minimum_support_fraction"] for row in rows} == {
        0.05, 0.10, 0.20, 0.30, 0.35, 0.45,
    }
    assert {tuple(row["density_timeline_s"].values()) for row in rows} == {
        (3.0, 6.0, 4.0, 3.0)
    }
    formal = set(protocol["live_matrix"]["paired_block_ids"])
    pilot = set(protocol["sample_size_pilot_matrix"]["paired_block_ids"])
    assert not {243} & (formal | pilot)


def test_artifact_validator_locks_all_arms_and_estimator() -> None:
    module = _module()
    release = module._load_yaml(ROOT / "configs/slam_confidence_sim_release_v1.yaml")
    artifacts = module.validate_artifacts(release)
    assert set(artifacts) == {"A", "B", "C", "D", "common"}
    assert artifacts["A"]["observation_dimension"] == "48"
    assert artifacts["C"]["observation_dimension"] == "51"
    assert artifacts["common"]["velocity_estimator_sync_tolerance_s"] == "0.025"


def test_formal_mode_is_fail_closed_until_explicit_authorization() -> None:
    module = _module()
    protocol = module._load_yaml(ROOT / "configs/slam_confidence_publication_protocol.yaml")
    release = module._load_yaml(ROOT / "configs/slam_confidence_sim_release_v1.yaml")
    protocol["formal_collection_authorized"] = False
    with pytest.raises(ValueError, match="not authorized"):
        module.validate_formal_authorization(protocol, release)


def test_runner_requires_offline_usability_for_each_executed_cell() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "evaluate_slam_confidence_publication_run.py" in source
    assert "offline_returncode == 0" in source
    assert '"offline_usability_gate": offline_gate' in source
    assert "evaluate_velocity_estimator_replay.py" in source
    assert "estimator_replay_returncode == 0" in source
    assert 'PROJECT_ROOT / "deployment/python_vendor"' in source
    assert 'PROJECT_ROOT / "source/anymal_locomotion"' in source
    assert 'environment["PYTHONPATH"]' in source
    assert "estimator_environment" in source
    assert 'release["common_runtime"]["velocity_estimator_sync_tolerance_s"]' in source


def test_selection_preserves_registered_cell_identity() -> None:
    module = _module()
    protocol = module._load_yaml(ROOT / "configs/slam_confidence_publication_protocol.yaml")
    args = argparse.Namespace(
        backend=["liosam"], profile=["lateral_right_1_5"],
        condition=["gradual_support_loss"], arm=["D"], block=[443],
    )
    selected = module.select_schedule(module.build_schedule(protocol), args)
    assert selected == [{
        "backend": "liosam", "profile": "lateral_right_1_5",
        "condition": "gradual_support_loss", "block_id": 443,
        "simulation_seed": 443, "arm": "D", "arm_order": 3,
    }]


def test_route_contract_is_stable_and_contains_full_command_definition() -> None:
    module = _module()
    first = module.route_contract("lateral_right_1_5")
    second = module.route_contract("lateral_right_1_5")
    assert first == second
    assert first["profile"]["target"] == (0.0, -1.5, 0.0)
    assert first["profile"]["duration_s"] == 14.0
    assert len(first["profile_sha256"]) == 64


def test_raw_bag_gate_requires_all_replay_topics(tmp_path) -> None:
    module = _module()
    bag = tmp_path / "bag"
    bag.mkdir()
    data = bag / "bag_0.db3"
    data.write_bytes(b"replay-data")
    topics = [
        {
            "topic_metadata": {"name": topic},
            "message_count": 1,
        }
        for topic in module.REQUIRED_BAG_TOPICS
    ]
    metadata = {
        "rosbag2_bagfile_information": {
            "duration": {"nanoseconds": 10}, "message_count": len(topics),
            "topics_with_message_count": topics,
            "relative_file_paths": [data.name],
        }
    }
    (bag / "metadata.yaml").write_text(
        __import__("yaml").safe_dump(metadata), encoding="utf-8"
    )
    gate = module.validate_raw_bag(bag)
    assert gate["passed"]
    assert len(gate["fingerprint_sha256"]) == 64
    metadata["rosbag2_bagfile_information"]["topics_with_message_count"][0][
        "message_count"
    ] = 0
    (bag / "metadata.yaml").write_text(
        __import__("yaml").safe_dump(metadata), encoding="utf-8"
    )
    assert not module.validate_raw_bag(bag)["passed"]


def test_clean_baseline_filter_allows_only_runtime_untracked_paths() -> None:
    module = _module()
    assert module.project_owned_untracked(
        ["build/a", "install/b", "log/c", "logs/d", "outputs/e", "lidar_type"]
    ) == []
    assert module.project_owned_untracked(["scripts/new_source.py", "build/a"]) == [
        "scripts/new_source.py"
    ]
