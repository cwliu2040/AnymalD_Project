from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from anymal_locomotion.gait_mode_governor import (
    BatchedGaitModeGovernor,
    GaitMode as TorchGaitMode,
    GaitModeGovernorConfig as TorchConfig,
)
from anymal_locomotion_ros2.policy_core import (
    GaitMode as NumpyGaitMode,
    GaitModeGovernor,
    GaitModeGovernorConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def _config_values() -> dict[str, float]:
    document = yaml.safe_load(
        (ROOT / "configs/slam_confidence_gait_mode_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    return {
        name: float(document[name])
        for name in (
            "degrade_below",
            "degrade_dwell_s",
            "recover_at_or_above",
            "recover_dwell_s",
            "hold_minimum_dwell_s",
            "command_scale_rate_down_per_s",
            "command_scale_rate_up_per_s",
        )
    }


def test_frozen_config_matches_training_and_deployment_defaults() -> None:
    expected = _config_values()
    assert TorchConfig().__dict__ == expected
    assert GaitModeGovernorConfig().__dict__ == expected


def test_healthy_path_is_exact_and_stale_high_is_fail_closed() -> None:
    governor = BatchedGaitModeGovernor(2, "cpu")
    governor.reset(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]))
    for _ in range(20):
        scale = governor.update(
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, 0.0]),
            0.02,
        )
    assert scale[0].item() == 1.0
    assert governor.mode[0].item() == int(TorchGaitMode.TRACK)
    assert 0.0 <= scale[1].item() < 1.0
    assert governor.mode[1].item() == int(TorchGaitMode.DECELERATE)


def test_hold_dwell_rejects_confidence_chatter_and_recovery_is_rate_limited() -> None:
    governor = GaitModeGovernor()
    assert governor.update(1.0, True, 0.02) == 1.0
    scales = [governor.update(1.0, False, 0.02) for _ in range(70)]
    assert all(next_scale <= scale for scale, next_scale in zip(scales, scales[1:]))
    assert governor.mode == NumpyGaitMode.HOLD
    assert governor.command_scale == 0.0

    for index in range(24):
        confidence = 0.56 if index % 2 == 0 else 0.54
        assert governor.update(confidence, True, 0.02) == 0.0
        assert governor.mode == NumpyGaitMode.HOLD
    for _ in range(24):
        assert governor.update(0.8, True, 0.02) == 0.0
    governor.update(0.8, True, 0.02)
    assert governor.mode == NumpyGaitMode.RECOVER
    assert governor.command_scale > 0.0
    previous = governor.command_scale
    current = governor.update(0.8, True, 0.02)
    assert current > previous


def test_torch_and_deployment_governors_have_trace_parity() -> None:
    torch_governor = BatchedGaitModeGovernor(1, "cpu")
    numpy_governor = GaitModeGovernor()
    trace = (
        [(1.0, True)] * 20
        + [(0.4, True)] * 10
        + [(1.0, False)] * 30
        + [(0.6, True)] * 30
        + [(0.8, True)] * 30
        + [(0.3, True)] * 8
    )
    for confidence, valid in trace:
        torch_scale = torch_governor.update(
            torch.tensor([confidence]), torch.tensor([valid]), 0.02
        ).item()
        numpy_scale = numpy_governor.update(confidence, valid, 0.02)
        np.testing.assert_allclose(torch_scale, numpy_scale, atol=1.0e-6)
        assert torch_governor.mode.item() == int(numpy_governor.mode)


def test_frozen_rates_are_feasible_for_existing_behavior_windows() -> None:
    governor = GaitModeGovernor()
    rows: list[tuple[float, float]] = []
    for step in range(500):
        phase = step * 0.02 / 10.0
        if phase < 0.30:
            confidence, valid = 1.0, True
        elif phase < 0.50:
            progress = (phase - 0.30) / 0.20
            confidence, valid = 1.0 - 0.8 * progress, True
        elif phase < 0.70:
            confidence, valid = (1.0 if step % 2 == 0 else 0.2), False
        else:
            progress = (phase - 0.70) / 0.30
            confidence = 0.2 + 0.8 * progress
            valid = progress >= (1.0 / 3.0)
        rows.append((phase, governor.update(confidence, valid, 0.02)))

    def mean_scale(start: float, stop: float) -> float:
        values = [scale for phase, scale in rows if start <= phase < stop]
        return sum(values) / len(values)

    assert mean_scale(0.10, 0.30) == 1.0
    assert mean_scale(0.45, 0.50) <= 0.65
    assert mean_scale(0.60, 0.70) == 0.0
    assert mean_scale(0.90, 1.00) >= (1.0 / 1.5)


def test_gait_mode_task_is_external_state_not_a_stateful_onnx_actor() -> None:
    policy = (
        ROOT
        / "source/anymal_locomotion/anymal_locomotion/policies/slam_confidence_residual.py"
    ).read_text(encoding="utf-8")
    env = (
        ROOT
        / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion"
        / "velocity/config/anymal_d/flat_env_cfg.py"
    ).read_text(encoding="utf-8")
    tasks = (
        ROOT
        / "source/anymal_locomotion/anymal_locomotion/tasks/manager_based/locomotion"
        / "velocity/config/anymal_d/__init__.py"
    ).read_text(encoding="utf-8")
    assert "class FrozenBackboneGaitModeActor" in policy
    assert "return self.backbone(observation[..., : self.legacy_observation_dim])" in policy
    assert "gait_mode_velocity_command" in env
    assert '"phase_offset_mode": "distributed"' in env
    assert "SlamConfidence-GaitMode-v0" in tasks
    assert "rclpy" not in policy
