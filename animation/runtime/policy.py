"""ONNX walking-policy integration independent of robot transport."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np

INIT_POS = np.array(
    [0.002, 0.053, -0.63, 1.368, -0.784, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, -0.003, -0.065, 0.635, 1.379, -0.796],
    dtype=np.float32,
)
ACTION_SCALE = 0.25


class WalkPolicy:
    """Run the walking ONNX model and maintain its policy-owned state."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        action_scale: float = ACTION_SCALE,
        init_pos: np.ndarray = INIT_POS,
        session: object | None = None,
    ) -> None:
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime is required for WalkPolicy; install it to use WALK modes."
                ) from exc
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.previous_action = np.zeros(16, dtype=np.float32)
        self.action_scale = float(action_scale)
        self.init_pos = np.asarray(init_pos, dtype=np.float32)
        if self.init_pos.shape != (16,):
            raise ValueError("init_pos must contain 16 joints")
        shape = session.get_inputs()[0].shape
        self.input_size = int(shape[-1]) if isinstance(shape[-1], int) else 74
        if self.input_size < 56:
            raise ValueError(f"Policy input has unsupported size {self.input_size}")
        self._history: deque[np.ndarray] = deque(maxlen=max(0, self.input_size - 56))

    def build_observation(self, measured_observation: Sequence[float]) -> np.ndarray:
        """Build input with raw policy action feedback, never a blended target.

        CRITICAL: ``previous_action`` is the policy's own raw ONNX action. Feeding
        animation-blended targets back here makes the policy observe commands it
        never issued, drifts its internal state, and can degrade gait into a fall.
        Animation is strictly downstream of this policy input.
        """
        measured = np.asarray(measured_observation, dtype=np.float32).reshape(-1)
        if measured.size == 56:
            # Original observation layout ends in a 16-element previous-action term.
            measured = measured.copy()
            measured[37:53] = self.previous_action
        if measured.size > self.input_size:
            raise ValueError(f"Observation has {measured.size}, policy expects {self.input_size}")
        tail = self.input_size - measured.size
        history = np.concatenate(tuple(self._history), dtype=np.float32) if self._history else np.empty(0, dtype=np.float32)
        if history.size < tail:
            history = np.pad(history, (tail - history.size, 0))
        return np.concatenate((measured, history[-tail:])).astype(np.float32)

    def infer(self, observation: Sequence[float]) -> np.ndarray:
        policy_input = self.build_observation(observation)
        raw_action = np.asarray(
            self.session.run(None, {self.input_name: policy_input[None, :]})[0],
            dtype=np.float32,
        ).reshape(-1)
        if raw_action.size != 16:
            raise ValueError(f"Policy returned {raw_action.size} actions; expected 16")
        # Keep only the raw neural-policy output for the next policy observation.
        self.previous_action = raw_action.copy()
        self._history.append(policy_input.copy())
        return raw_action

    def action_to_targets(self, action: Sequence[float]) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 16:
            raise ValueError("action must contain 16 values")
        return self.init_pos + action * self.action_scale
