"""ONNX inference backend loaded only in the external ROS 2 process."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxBackend:
    """CPU backend using ONNX's portable reference evaluator."""

    def __init__(self, policy_path: str | Path) -> None:
        try:
            from onnx.reference import ReferenceEvaluator
        except ImportError as error:
            raise RuntimeError(
                "ONNX is not installed in the ROS 2 Python environment. "
                "Install deployment/ros2_ws/requirements-inference.txt; "
                "do not run this node inside Isaac Sim Python."
            ) from error

        resolved_path = str(Path(policy_path).expanduser().resolve())
        self._model = ReferenceEvaluator(resolved_path)
        if len(self._model.input_names) != 1 or len(self._model.output_names) != 1:
            raise ValueError(
                "Locomotion policy must have exactly one input and one output; "
                f"received inputs={self._model.input_names}, "
                f"outputs={self._model.output_names}"
            )
        self._input_name = self._model.input_names[0]

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        inputs = np.asarray(observations, dtype=np.float32)
        outputs = self._model.run(None, {self._input_name: inputs})
        return np.asarray(outputs[0], dtype=np.float32)
