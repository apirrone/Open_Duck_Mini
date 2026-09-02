"""Seeded gesture scheduling with cooldowns and mood weighting."""

from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass(frozen=True)
class Behavior:
    clip_name: str
    weight: float = 1.0
    cooldown: float = 8.0
    required_tags: frozenset[str] = field(default_factory=frozenset)


class BehaviorScheduler:
    MOOD_BIASES = {
        "curious": {"look_around": 3.0, "curious_tilt": 3.0, "antenna_wiggle": 1.5},
        "sleepy": {"sad_droop": 3.0, "idle_breathe": 1.5},
        "alert": {"alert_perk": 3.0, "look_around": 1.8},
    }

    def __init__(self, seed: int = 0, gap: tuple[float, float] = (3.0, 9.0)) -> None:
        self.rng, self.gap, self.behaviors = random.Random(seed), gap, {}
        self.mood, self.elapsed, self.next_time, self.last_clip = "neutral", 0.0, 0.0, None
        self.last_played: dict[str, float] = {}

    def register(self, clip_name: str, weight: float = 1.0, cooldown: float = 8.0, required_tags: set[str] | None = None) -> None:
        self.behaviors[clip_name] = Behavior(clip_name, weight, cooldown, frozenset(required_tags or ()))

    def set_mood(self, mood: str) -> None:
        self.mood = mood

    def _speed(self) -> float:
        return 0.7 if self.mood == "sleepy" else 1.0

    def _eligible(self, bypass_gap: bool = False) -> list[Behavior]:
        values = [b for b in self.behaviors.values() if self.elapsed - self.last_played.get(b.clip_name, -1e9) >= b.cooldown]
        if len(values) > 1:
            values = [b for b in values if b.clip_name != self.last_clip]
        return values

    def _request(self, clip_name: str) -> tuple[str, float]:
        self.last_clip, self.last_played[clip_name] = clip_name, self.elapsed
        self.next_time = self.elapsed + self.rng.uniform(*self.gap)
        return clip_name, self._speed()

    def trigger(self, clip_name: str) -> tuple[str, float] | None:
        behavior = self.behaviors.get(clip_name)
        if behavior is None or self.elapsed - self.last_played.get(clip_name, -1e9) < behavior.cooldown:
            return None
        return self._request(clip_name)

    def update(self, dt: float) -> tuple[str, float] | None:
        self.elapsed += dt
        if self.elapsed < self.next_time:
            return None
        eligible = self._eligible()
        if not eligible:
            return None
        bias = self.MOOD_BIASES.get(self.mood, {})
        weights = [item.weight * bias.get(item.clip_name, 1.0) for item in eligible]
        return self._request(self.rng.choices(eligible, weights=weights, k=1)[0].clip_name)
