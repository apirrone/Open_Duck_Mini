"""Time-based playback and blending envelopes for animation clips."""

from __future__ import annotations

import numpy as np

from .schema import AnimationClip


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class AnimationPlayer:
    """Play one :class:`AnimationClip` with interpolation and a blend envelope."""

    def __init__(
        self,
        clip: AnimationClip,
        speed: float = 1.0,
        loop: bool | None = None,
        start_time: float = 0.0,
    ) -> None:
        clip.validate()
        self.clip = clip
        self.speed = float(speed)
        self.loop = clip.loop if loop is None else bool(loop)
        self.time = float(start_time)
        self._elapsed = max(float(start_time), 0.0)
        self.weight_scale = 1.0
        self._stopped = False
        self._stop_elapsed = 0.0
        if self.loop and clip.duration > 0.0:
            self.time %= clip.duration

    def update(self, dt: float) -> None:
        """Advance the playback cursor by ``dt`` seconds."""
        advance = float(dt) * self.speed
        if self._stopped:
            self._stop_elapsed += max(advance, 0.0)
            return
        self.time += advance
        self._elapsed += max(advance, 0.0)
        if self.loop and self.clip.duration > 0.0:
            self.time %= self.clip.duration

    def sample(self) -> tuple[np.ndarray, float]:
        """Return interpolated joint values and the current blend-envelope weight."""
        return self._sample_values(), self._envelope_weight()

    def stop(self) -> None:
        """Stop playback, fading a looping player over its ``blend_out`` duration."""
        if not self._stopped:
            self._stopped = True
            self._stop_elapsed = 0.0

    @property
    def finished(self) -> bool:
        """Whether this player can be removed from the mixer."""
        if self._stopped:
            return self._stop_elapsed >= self.clip.blend_out
        return not self.loop and self.time >= self.clip.duration

    def reset(self) -> None:
        """Return playback to its initial state."""
        self.time = 0.0
        self._elapsed = 0.0
        self._stopped = False
        self._stop_elapsed = 0.0

    def _sample_values(self) -> np.ndarray:
        frames = self.clip.frames
        n_frames = self.clip.n_frames
        if n_frames == 0:
            return np.empty((0,), dtype=np.float32)
        if n_frames == 1:
            return frames[0].copy()

        duration = self.clip.duration
        if self.loop and duration > 0.0:
            time = self.time % duration
        else:
            time = min(max(self.time, 0.0), duration)
            if time >= duration:
                return frames[-1].copy()

        frame_position = time * self.clip.fps
        index = int(np.floor(frame_position))
        fraction = frame_position - index
        if index >= n_frames - 1:
            if not self.loop:
                return frames[-1].copy()
            left = frames[-1]
            right = frames[0]
            # The loop's final segment spans the last frame timestamp to duration.
            seam_fraction = (time - (n_frames - 1) / self.clip.fps) * self.clip.fps
            fraction = float(np.clip(seam_fraction, 0.0, 1.0))
        else:
            left = frames[index]
            right = frames[index + 1]
        return (left + (right - left) * fraction).astype(np.float32)

    def _envelope_weight(self) -> float:
        if self._stopped:
            if self.clip.blend_out <= 0.0:
                return 0.0
            return 1.0 - _smoothstep(self._stop_elapsed / self.clip.blend_out)
        if self.loop:
            if self.clip.blend_in > 0.0:
                return _smoothstep(self._elapsed / self.clip.blend_in)
            return 1.0
        if self.time >= self.clip.duration:
            return 0.0
        weight = 1.0
        if self.clip.blend_in > 0.0:
            weight = _smoothstep(self.time / self.clip.blend_in)
        if self.clip.blend_out > 0.0:
            weight *= 1.0 - _smoothstep(
                (self.time - (self.clip.duration - self.clip.blend_out))
                / self.clip.blend_out
            )
        return float(np.clip(weight, 0.0, 1.0))
