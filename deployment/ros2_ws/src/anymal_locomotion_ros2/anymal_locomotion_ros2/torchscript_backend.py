"""TorchScript inference backend loaded only in the external ROS 2 process."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TorchScriptBackend:
    """CPU TorchScript backend for the exported locomotion actor."""

    def __init__(self, policy_path: str | Path) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is not installed in the ROS 2 Python environment. "
                "Install a deployment-compatible CPU or CUDA PyTorch build; "
                "do not run this node inside Isaac Sim Python."
            ) from error
        self._torch = torch
        resolved_path = str(Path(policy_path).expanduser().resolve())
        self._model = torch.jit.load(resolved_path, map_location="cpu").eval()

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        tensor = self._torch.from_numpy(np.asarray(observations, dtype=np.float32))
        with self._torch.inference_mode():
            return self._model(tensor).detach().cpu().numpy()
