from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment/ros2_ws/src/anymal_locomotion_ros2"))
from anymal_locomotion_ros2.command_scale_pulse_core import CommandScalePulse


def test_untreated_prefix_pulse_and_exact_recovery() -> None:
    pulse = CommandScalePulse(enabled=True, start_s=7.25, duration_s=0.75, scale=0.5)
    command = (1.5, -1.0, 0.8)
    assert pulse.apply(command, 7.249) == command
    assert not pulse.active_at(7.249)
    assert pulse.apply(command, 7.25) == pytest.approx((0.75, -0.5, 0.4))
    assert pulse.active_at(7.25)
    assert pulse.apply(command, 7.999) == pytest.approx((0.75, -0.5, 0.4))
    assert pulse.apply(command, 8.0) == command


def test_disabled_control_never_changes_command() -> None:
    pulse = CommandScalePulse(enabled=False, scale=0.25)
    assert pulse.apply((1.0, 2.0, 3.0), 7.5) == (1.0, 2.0, 3.0)


def test_component_pulse_can_preserve_translation_or_yaw() -> None:
    command = (1.5, -0.4, -1.0)
    preserve_yaw = CommandScalePulse(
        enabled=True, start_s=7.25, duration_s=0.75,
        scales_xyz=(0.75, 0.75, 1.0),
    )
    preserve_translation = CommandScalePulse(
        enabled=True, start_s=7.25, duration_s=0.75,
        scales_xyz=(1.0, 1.0, 0.75),
    )
    assert preserve_yaw.apply(command, 7.5) == pytest.approx((1.125, -0.3, -1.0))
    assert preserve_translation.apply(command, 7.5) == pytest.approx((1.5, -0.4, -0.75))
    assert preserve_yaw.apply(command, 8.0) == command


@pytest.mark.parametrize(
    "kwargs",
    ({"start_s": -1.0}, {"duration_s": 0.0}, {"scale": 0.0}, {"scale": 1.1}),
)
def test_invalid_pulse_contract_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CommandScalePulse(**kwargs)
