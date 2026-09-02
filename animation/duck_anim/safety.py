"""Hardware-facing joint target limiting."""

from __future__ import annotations

import numpy as np

from .joints import ALL_JOINTS, JOINT_LIMITS, JOINT_VELOCITY_LIMITS


class JointSafetyLimiter:
    """Clamp position, velocity, and acceleration before targets reach hardware."""

    def __init__(
        self,
        dt: float = 0.02,
        margin: float = 0.05,
        velocity_scale: float = 0.8,
        max_accel: float | np.ndarray | None = 100.0,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        if margin < 0.0:
            raise ValueError("margin must be >= 0")
        if velocity_scale < 0.0:
            raise ValueError("velocity_scale must be >= 0")
        if max_accel is not None and np.any(np.asarray(max_accel) < 0.0):
            raise ValueError("max_accel must be >= 0")
        self.dt = float(dt)
        self.margin = float(margin)
        self.velocity_scale = float(velocity_scale)
        self.max_accel = max_accel
        self.nan_events = 0
        self.clamped_joints: list[str] = []
        self._previous_delta: np.ndarray | None = None
        self._lower = np.array([JOINT_LIMITS[j][0] for j in ALL_JOINTS], dtype=np.float32)
        self._upper = np.array([JOINT_LIMITS[j][1] for j in ALL_JOINTS], dtype=np.float32)
        self._velocity = np.array(
            [JOINT_VELOCITY_LIMITS[j] for j in ALL_JOINTS], dtype=np.float32
        ) * self.velocity_scale

    def reset(self) -> None:
        """Clear acceleration history and per-call diagnostics."""
        self._previous_delta = None
        self.clamped_joints = []

    def apply(self, target: np.ndarray, previous_output: np.ndarray | None) -> np.ndarray:
        """Return a finite, position-, velocity-, and acceleration-safe target."""
        target = np.asarray(target, dtype=np.float32)
        if target.shape != self._lower.shape:
            raise ValueError(f"target must have shape {self._lower.shape}, got {target.shape}")
        midpoint = (self._lower + self._upper) / 2.0
        if previous_output is None:
            previous = midpoint
        else:
            previous = np.asarray(previous_output, dtype=np.float32)
            if previous.shape != self._lower.shape:
                raise ValueError(
                    f"previous_output must have shape {self._lower.shape}, got {previous.shape}"
                )
            previous = np.where(np.isfinite(previous), previous, midpoint)

        invalid = ~np.isfinite(target)
        self.nan_events += int(np.count_nonzero(invalid))
        safe_target = np.where(invalid, previous, target)
        lower = self._lower + self.margin
        upper = self._upper - self.margin
        if np.any(lower > upper):
            raise ValueError("margin exceeds at least one joint's range")
        position_limited = np.clip(safe_target, lower, upper)
        position_clamped = safe_target != position_limited
        requested_delta = position_limited - previous
        velocity_delta = np.clip(
            requested_delta, -self._velocity * self.dt, self._velocity * self.dt
        )
        rate_clamped = requested_delta != velocity_delta
        accel_clamped = np.zeros_like(rate_clamped, dtype=bool)
        if self.max_accel is not None and self._previous_delta is not None:
            accel = np.asarray(self.max_accel, dtype=np.float32)
            acceleration_limited = np.clip(
                velocity_delta,
                self._previous_delta - accel * self.dt * self.dt,
                self._previous_delta + accel * self.dt * self.dt,
            )
            accel_clamped = velocity_delta != acceleration_limited
            velocity_delta = acceleration_limited
        self.clamped_joints = [
            joint
            for joint, clamped in zip(
                ALL_JOINTS, position_clamped | rate_clamped | accel_clamped
            )
            if clamped
        ]
        output = previous + velocity_delta
        self._previous_delta = velocity_delta.copy()
        return output.astype(np.float32)
