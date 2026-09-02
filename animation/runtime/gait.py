"""Gait phase and animation authority derived from velocity commands."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .modes import RobotMode


def _speed(commanded_velocity: Sequence[float]) -> float:
    velocity = np.asarray(commanded_velocity, dtype=float)
    if velocity.shape != (3,):
        raise ValueError("commanded_velocity must contain (vx, vy, wz)")
    return float(np.linalg.norm(velocity))


def leg_animation_gain(
    mode: RobotMode,
    commanded_velocity: Sequence[float],
    *,
    floor: float = 0.15,
    full_authority_speed: float = 0.05,
    high_speed: float = 0.4,
) -> float:
    """Return safe animation authority for leg channels.

    Large leg offsets while walking fast will tip the duck over, so leg
    animation authority must shrink as speed rises.
    """
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0, 1]")
    if high_speed <= full_authority_speed:
        raise ValueError("high_speed must exceed full_authority_speed")
    if mode in {RobotMode.IDLE, RobotMode.STAND, RobotMode.HYBRID_STAND}:
        return 1.0
    speed = _speed(commanded_velocity)
    if speed <= full_authority_speed:
        return 1.0
    fraction = min(1.0, (speed - full_authority_speed) / (high_speed - full_authority_speed))
    # Smoothstep makes the handoff to the gait policy gradual.
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    return float(1.0 - (1.0 - floor) * smooth)


class GaitPhaseTracker:
    def __init__(
        self,
        gait_frequency: float = 1.25,
        stationary_threshold: float = 0.05,
        debounce_window: float = 0.3,
    ) -> None:
        if gait_frequency <= 0 or stationary_threshold < 0 or debounce_window < 0:
            raise ValueError("Invalid gait tracker configuration")
        self.gait_frequency = float(gait_frequency)
        self.stationary_threshold = float(stationary_threshold)
        self.debounce_window = float(debounce_window)
        self.phase = 0.0
        self._stationary_elapsed = 0.0

    @property
    def is_stationary(self) -> bool:
        return self._stationary_elapsed >= self.debounce_window

    def update(self, dt: float, commanded_velocity: Sequence[float]) -> float:
        if dt < 0:
            raise ValueError("dt must be non-negative")
        speed = _speed(commanded_velocity)
        self.phase = math.fmod(self.phase + dt * self.gait_frequency, 1.0)
        self._stationary_elapsed = (
            self._stationary_elapsed + dt if speed < self.stationary_threshold else 0.0
        )
        return self.phase
