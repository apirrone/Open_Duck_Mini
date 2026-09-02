"""Smooth deterministic value noise for small, organic joint motion."""

from __future__ import annotations

import math


class SmoothNoise:
    """Seeded continuous one-dimensional value noise."""

    def __init__(self, seed: int, frequency: float, octaves: int = 2, persistence: float = 0.5) -> None:
        if frequency <= 0 or octaves < 1 or not 0 < persistence <= 1:
            raise ValueError("frequency > 0, octaves >= 1, and persistence in (0, 1] are required")
        self.seed, self.frequency, self.octaves, self.persistence = seed, frequency, octaves, persistence

    def _lattice(self, index: int, octave: int) -> float:
        # Integer hashing avoids mutable RNG state and makes sample order irrelevant.
        value = (index * 0x9E3779B1 + self.seed * 0x85EBCA77 + octave * 0xC2B2AE3D) & 0xFFFFFFFF
        value ^= value >> 16
        value = (value * 0x7FEB352D) & 0xFFFFFFFF
        value ^= value >> 15
        return (value / 0x7FFFFFFF) - 1.0

    def sample(self, t: float) -> float:
        total = weight = 0.0
        for octave in range(self.octaves):
            x = t * self.frequency * (2**octave)
            left = math.floor(x)
            fraction = x - left
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            value = self._lattice(left, octave) * (1 - smooth) + self._lattice(left + 1, octave) * smooth
            amplitude = self.persistence**octave
            total += value * amplitude
            weight += amplitude
        return total / weight


class NoiseChannel:
    """A noise source expressed directly as a joint-angle offset."""

    def __init__(self, noise: SmoothNoise, amplitude: float, offset: float = 0.0) -> None:
        self.noise, self.amplitude, self.offset = noise, amplitude, offset

    def sample(self, t: float) -> float:
        return self.offset + self.amplitude * self.noise.sample(t)
