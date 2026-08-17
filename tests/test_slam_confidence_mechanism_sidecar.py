from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/validation/build_slam_confidence_mechanism_sidecar.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("mechanism_sidecar", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Actor(torch.nn.Module):
    legacy_observation_dim = 48
    confidence_offset = 48
    previous_action_offset = 36
    command_offset = 9
    command_dimension = 3
    nonnegative_stride_smoothing = True
    smoothing_logit_gain = 1.0
    suppress_gait_when_tracking_invalid = True
    gait_delta_safe_scale_power = 2.0

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(48, 12, bias=False)
        self.backbone.weight.data.zero_()
        self.backbone.weight.data[:, 9:12] = 0.1
        self.intent_head = torch.nn.Linear(63, 1, bias=False)
        self.intent_head.weight.data.zero_()
        self.intent_head.weight.data[:, 48] = 1.0
        self.gait_head = torch.nn.Linear(63, 4, bias=False)
        self.gait_head.weight.data.zero_()
        self.gait_head.weight.data[:, 48] = 0.25
        self.gait_parameter_limits = torch.tensor([0.6, 0.3, 0.2, 0.9])
        self.crouch_basis = torch.ones(12)
        self.stance_width_basis = torch.linspace(-1.0, 1.0, 12)

    def intent_blend(self, observation, legacy_action):
        logits = self.intent_head(torch.cat((observation, legacy_action), dim=-1))
        return 0.79 * torch.clamp(torch.tanh(logits), min=0.0, max=1.0)

    def gait_coordinates(self, observation, intent_action):
        module = _load_module()
        return module._raw_gait_coordinates(self, observation, intent_action)

    def forward(self, observation):
        module = _load_module()
        return module.reconstruct_mechanisms(self, observation)["arm_c_action"]


def test_mechanism_sidecar_reconstructs_all_registered_arms() -> None:
    module = _load_module()
    actor = _Actor().eval()
    observations = torch.zeros((3, 51), dtype=torch.float32)
    observations[:, 9] = 1.5
    observations[:, 48:51] = torch.tensor(
        [[1.0, 1.0, 0.0], [0.6, 1.0, 0.1], [0.0, 0.0, 1.0]]
    )
    with torch.inference_mode():
        actions = actor(observations)
    diagnostics = {
        "schema_version": 2,
        "records": [
            {
                "clock_s": float(index) * 0.02,
                "observation": observation.tolist(),
                "raw_action": action.tolist(),
            }
            for index, (observation, action) in enumerate(
                zip(observations, actions, strict=True)
            )
        ],
    }
    sidecar = module.build_sidecar(
        diagnostics=diagnostics,
        actor=actor,
        actor_details={
            "architecture": "frozen_model1450_aux_intent_structured_gait"
        },
        action_atol=1.0e-6,
    )
    assert sidecar["gate"]["passed"]
    assert sidecar["gate"]["arm_d_structured_delta_exact_zero"]
    assert sidecar["gate"]["model48_logged_action_max_absolute_error"] == 0.0
    assert set(sidecar["records"][0]["arm_actions"]) == {"A", "B", "C", "D"}
    assert math.isclose(sidecar["records"][1]["safe_scale"], 0.5, abs_tol=1.0e-6)
    assert sidecar["records"][2]["structured_delta_l2"] == 0.0
