"""Kinematically modest head gaze control."""

from __future__ import annotations

import math

from duck_anim.joints import JOINT_LIMITS


def _clamp(name: str, value: float) -> float:
    low, high = JOINT_LIMITS[name]
    return max(low, min(high, value))


class GazeSolver:
    def __init__(self, neck_ratio: float = 0.4, roll_velocity_gain: float = 0.08) -> None:
        if not 0 <= neck_ratio <= 1:
            raise ValueError("neck_ratio must be in [0, 1]")
        self.neck_ratio = neck_ratio
        self.roll_velocity_gain = roll_velocity_gain
        self._previous_yaw = 0.0

    def solve(self, azimuth: float, elevation: float) -> dict[str, float]:
        yaw = _clamp("head_yaw", azimuth)
        neck_pitch = _clamp("neck_pitch", elevation * self.neck_ratio)
        head_pitch = _clamp("head_pitch", elevation * (1.0 - self.neck_ratio))
        roll = _clamp("head_roll", -self.roll_velocity_gain * (yaw - self._previous_yaw))
        self._previous_yaw = yaw
        return {"neck_pitch": neck_pitch, "head_pitch": head_pitch, "head_yaw": yaw, "head_roll": roll}


class GazeTracker:
    """Critically damped target tracker with an explicit speed ceiling."""

    def __init__(self, solver: GazeSolver | None = None, max_angular_velocity: float = 1.5, smoothing_time: float = 0.16) -> None:
        self.solver = solver or GazeSolver()
        self.max_angular_velocity, self.smoothing_time = max_angular_velocity, smoothing_time
        self.azimuth = self.elevation = self.target_azimuth = self.target_elevation = 0.0
        self._az_velocity = self._el_velocity = 0.0

    def look_at(self, azimuth: float, elevation: float) -> None:
        self.target_azimuth, self.target_elevation = azimuth, elevation

    def _advance(self, current: float, target: float, velocity: float, dt: float) -> tuple[float, float]:
        omega = 2.0 / max(self.smoothing_time, 1e-6)
        acceleration = omega * omega * (target - current) - 2.0 * omega * velocity
        velocity = max(-self.max_angular_velocity, min(self.max_angular_velocity, velocity + acceleration * dt))
        delta = max(-self.max_angular_velocity * dt, min(self.max_angular_velocity * dt, velocity * dt))
        return current + delta, velocity

    def update(self, dt: float) -> dict[str, float]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.azimuth, self._az_velocity = self._advance(self.azimuth, self.target_azimuth, self._az_velocity, dt)
        self.elevation, self._el_velocity = self._advance(self.elevation, self.target_elevation, self._el_velocity, dt)
        return self.solver.solve(self.azimuth, self.elevation)
