"""RSL-RL training wrapper that closes the loop through the deployable estimator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from tensordict import TensorDict

from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from anymal_locomotion.artifacts import PROJECT_ROOT, assert_project_local_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_velocity_estimator_artifact(metadata_path: str | Path) -> dict[str, object]:
    """Validate and describe the repository-owned estimator used for training."""
    path = assert_project_local_path(metadata_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("contract_id") != "anymal-d-proprioceptive-velocity-v1"
        or metadata.get("history_length") != 20
        or metadata.get("step_dimension") != 37
    ):
        raise ValueError("velocity estimator metadata contract mismatch")
    torchscript = metadata.get("artifacts", {}).get("torchscript", {})
    model_path = (path.parent / str(torchscript.get("path", ""))).resolve()
    if PROJECT_ROOT not in model_path.parents:
        raise ValueError("velocity estimator model escapes the project repository")
    digest = _sha256(model_path) if model_path.is_file() else None
    if digest != torchscript.get("sha256"):
        raise ValueError("velocity estimator TorchScript digest mismatch")
    return {
        "metadata_path": str(path.relative_to(PROJECT_ROOT)),
        "metadata_sha256": _sha256(path),
        "torchscript_path": str(model_path.relative_to(PROJECT_ROOT)),
        "torchscript_sha256": digest,
        "history_length": 20,
        "step_dimension": 37,
    }


class VelocityEstimatorTrainingWrapper(RslRlVecEnvWrapper):
    """Feed policy observations through the same 20-step estimator as deployment.

    The simulator ground-truth velocity remains available to rewards and critic
    bookkeeping, but policy columns 0..2 are replaced after estimator warm-up.
    Actions are held at zero during the 20-sample warm-up, matching the fixed
    behavior evaluator's fail-closed estimator path.
    """

    def __init__(self, env, metadata_path: str | Path, clip_actions: float | None = None):
        artifact = validate_velocity_estimator_artifact(metadata_path)
        self.artifact = artifact
        self._metadata_path = assert_project_local_path(metadata_path)
        super().__init__(env, clip_actions=clip_actions)

        model_path = PROJECT_ROOT / str(artifact["torchscript_path"])
        self.estimator = torch.jit.load(str(model_path), map_location=self.device).eval()
        self.history = torch.zeros(
            (self.num_envs, 20, 37), dtype=torch.float32, device=self.device
        )
        self.history_count = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self.ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        robot = self.unwrapped.scene["robot"]
        contact_sensor = self.unwrapped.scene["contact_forces"]
        base_ids, base_names = robot.find_bodies("base")
        foot_ids, foot_names = contact_sensor.find_bodies(
            ["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"], preserve_order=True
        )
        if base_names != ["base"] or tuple(foot_names) != (
            "LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"
        ):
            raise RuntimeError(
                f"velocity estimator body mapping failed: base={base_names}, feet={foot_names}"
            )
        self._base_body_id = int(base_ids[0])
        self._contact_foot_ids = list(foot_ids)

    def _estimated_observations(self, obs: TensorDict) -> TensorDict:
        policy_observation = obs["policy"]
        robot = self.unwrapped.scene["robot"]
        contact_sensor = self.unwrapped.scene["contact_forces"]
        gravity_w = torch.as_tensor(
            self.unwrapped.cfg.sim.gravity,
            device=self.device,
            dtype=torch.float32,
        ).expand(self.num_envs, -1)
        specific_force_b = quat_apply_inverse(
            robot.data.root_quat_w,
            robot.data.body_lin_acc_w[:, self._base_body_id, :] - gravity_w,
        )
        contacts = (
            torch.abs(
                contact_sensor.data.net_forces_w[:, self._contact_foot_ids, 2]
            )
            > 1.0
        ).to(dtype=torch.float32)
        estimator_step = torch.cat(
            (
                policy_observation[:, 3:6],
                specific_force_b,
                policy_observation[:, 6:9],
                policy_observation[:, 12:24],
                policy_observation[:, 24:36],
                contacts,
            ),
            dim=1,
        )
        if estimator_step.shape != (self.num_envs, 37):
            raise RuntimeError(f"estimator training step has shape {estimator_step.shape}")
        self.history[:, :-1, :] = self.history[:, 1:, :].clone()
        self.history[:, -1, :] = estimator_step
        self.history_count = torch.clamp(self.history_count + 1, max=20)
        self.ready = self.history_count == 20
        estimated_velocity = self.estimator(self.history.reshape(self.num_envs, -1))
        if (
            estimated_velocity.shape != (self.num_envs, 3)
            or not torch.all(torch.isfinite(estimated_velocity))
            or torch.any(torch.abs(estimated_velocity) > 8.0)
        ):
            raise RuntimeError("velocity estimator produced an invalid training output")
        policy_observation = policy_observation.clone()
        policy_observation[self.ready, 0:3] = estimated_velocity[self.ready]
        obs = obs.clone()
        obs["policy"] = policy_observation
        return obs

    def get_observations(self) -> TensorDict:
        return self._estimated_observations(super().get_observations())

    def reset(self) -> tuple[TensorDict, dict]:
        obs, extras = super().reset()
        if hasattr(self, "history"):
            self.history.zero_()
            self.history_count.zero_()
            self.ready.zero_()
            obs = self._estimated_observations(obs)
        return obs, extras

    def step(self, actions: torch.Tensor):
        actions = actions.clone()
        actions[~self.ready] = 0.0
        obs, rewards, dones, extras = super().step(actions)
        reset_mask = dones.to(dtype=torch.bool)
        if torch.any(reset_mask):
            self.history[reset_mask] = 0.0
            self.history_count[reset_mask] = 0
            self.ready[reset_mask] = False
        return self._estimated_observations(obs), rewards, dones, extras
