"""Always-on dock-safe head and antenna motion."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from duck_anim.joints import ANTENNA_JOINTS, HEAD_JOINTS
from .gaze import GazeTracker
from .noise import NoiseChannel, SmoothNoise


@dataclass
class IdleEngineConfig:
    breath_amplitude: float = math.radians(2.0)
    breath_frequency: float = 0.25
    gaze_cone: float = math.radians(12.0)
    gaze_hold_range: tuple[float, float] = (0.35, 1.2)
    antenna_amplitude: float = math.radians(7.0)


class IdleEngine:
    def __init__(self, config: IdleEngineConfig | None = None, seed: int = 0) -> None:
        self.config, self.rng, self.time = config or IdleEngineConfig(), random.Random(seed), 0.0
        self.tracker = GazeTracker()
        self.next_gaze = 0.0
        self.next_flick = 0.0
        self.flick = 0.0
        self.noise = {
            "head_yaw": NoiseChannel(SmoothNoise(seed + 1, 0.12), math.radians(3)),
            "head_pitch": NoiseChannel(SmoothNoise(seed + 2, 0.18), math.radians(1.5)),
            "head_roll": NoiseChannel(SmoothNoise(seed + 3, 0.2), math.radians(2)),
            "left_antenna": NoiseChannel(SmoothNoise(seed + 4, 0.10), self.config.antenna_amplitude),
            "right_antenna": NoiseChannel(SmoothNoise(seed + 5, 0.13), self.config.antenna_amplitude),
        }

    def update(self, dt: float) -> dict[str, float]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.time += dt
        if self.time >= self.next_gaze:
            self.tracker.look_at(self.rng.uniform(-self.config.gaze_cone, self.config.gaze_cone), self.rng.uniform(-self.config.gaze_cone, self.config.gaze_cone))
            self.next_gaze = self.time + self.rng.uniform(*self.config.gaze_hold_range)
        if self.time >= self.next_flick:
            self.flick = self.rng.choice((-1.0, 1.0)) * math.radians(10)
            self.next_flick = self.time + self.rng.uniform(2.0, 6.0)
        self.flick *= math.exp(-dt * 12.0)
        gaze = self.tracker.update(dt)
        breath = self.config.breath_amplitude * math.sin(2 * math.pi * self.config.breath_frequency * self.time)
        output = {
            "neck_pitch": gaze["neck_pitch"] + breath * 0.45,
            "head_pitch": gaze["head_pitch"] + breath * 0.55 + self.noise["head_pitch"].sample(self.time),
            "head_yaw": gaze["head_yaw"] + self.noise["head_yaw"].sample(self.time),
            "head_roll": gaze["head_roll"] + self.noise["head_roll"].sample(self.time),
            "left_antenna": self.noise["left_antenna"].sample(self.time) + self.flick,
            "right_antenna": self.noise["right_antenna"].sample(self.time) - self.flick * 0.65,
        }
        assert set(output).issubset(set(HEAD_JOINTS + ANTENNA_JOINTS))
        return output
