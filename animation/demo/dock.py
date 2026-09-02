"""Dock demonstration mode with an unconditional final leg-position mask."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from duck_anim.joints import ALL_JOINTS, JOINT_INDEX, LEG_JOINTS, to_array
from duck_anim.loader import load_clip_dir
from .idle_engine import IdleEngine
from .scheduler import BehaviorScheduler

LOG = logging.getLogger(__name__)


class _FallbackPlayer:
    """Small compatibility player used only until duck_anim.player is available."""
    def __init__(self, clip: Any, speed: float = 1.0, **_: Any) -> None:
        self.clip, self.speed, self.time, self.finished, self.weight_scale = clip, speed, 0.0, False, 1.0

    def update(self, dt: float) -> None:
        self.time += dt * self.speed
        self.finished = not self.clip.loop and self.time >= self.clip.duration

    def sample(self) -> tuple[np.ndarray, float]:
        output = np.zeros(len(ALL_JOINTS), dtype=np.float32)
        if self.clip.n_frames:
            index = min(int(self.time * self.clip.fps), self.clip.n_frames - 1)
            for joint, value in zip(self.clip.joints, self.clip.frames[index]):
                output[JOINT_INDEX[joint]] = value
        return output, self.weight_scale


class _FallbackMixer:
    def __init__(self) -> None:
        self.active_clips: dict[str, _FallbackPlayer] = {}

    def add(self, player: _FallbackPlayer, name: str | None = None) -> str:
        name = name or player.clip.name
        self.active_clips[name] = player
        return name

    def update(self, dt: float) -> None:
        for name, player in list(self.active_clips.items()):
            player.update(dt)
            if player.finished:
                del self.active_clips[name]

    def mix(self, base: np.ndarray) -> np.ndarray:
        result = base.copy()
        for player in self.active_clips.values():
            values, weight = player.sample()
            if player.clip.layer == "additive":
                result += values * weight
            else:
                present = [JOINT_INDEX[j] for j in player.clip.joints]
                result[present] = result[present] * (1 - weight) + values[present] * weight
        return result


class _FallbackLimiter:
    def __init__(self, **_: Any) -> None:
        pass

    def apply(self, target: np.ndarray, previous_output: np.ndarray) -> np.ndarray:
        return target.copy()


try:
    from duck_anim.mixer import LayeredMixer  # type: ignore
    from duck_anim.player import AnimationPlayer  # type: ignore
    from duck_anim.safety import JointSafetyLimiter  # type: ignore
except ImportError:  # Workstream A may not yet be present in an isolated checkout.
    LayeredMixer, AnimationPlayer, JointSafetyLimiter = _FallbackMixer, _FallbackPlayer, _FallbackLimiter


class DockMode:
    """Animation controller for a duck parked on its dock.

    The final mask is intentional redundancy: no dependency or malformed clip can
    affect the ten leg values returned by :meth:`step`.
    """

    def __init__(self, clips_dir: str | Path | None = None, docked_pose: np.ndarray | None = None, seed: int = 0) -> None:
        self.docked_pose = np.asarray(docked_pose if docked_pose is not None else np.zeros(len(ALL_JOINTS), dtype=np.float32), dtype=np.float32).copy()
        if self.docked_pose.shape != (len(ALL_JOINTS),):
            raise ValueError("docked_pose must be a 16-joint array")
        self._docked_legs = self.docked_pose[:len(LEG_JOINTS)].copy()
        self.idle, self.scheduler, self.mixer = IdleEngine(seed=seed), BehaviorScheduler(seed=seed), LayeredMixer()
        self.limiter = JointSafetyLimiter(dt=0.02)
        self.previous_output = self.docked_pose.copy()
        self.clips: dict[str, Any] = {}
        self.rejected_clips: list[str] = []
        if clips_dir is not None and Path(clips_dir).is_dir():
            for clip in load_clip_dir(clips_dir).values():
                self.register_clip(clip)

    def register_clip(self, clip: Any) -> bool:
        if any(joint in LEG_JOINTS for joint in clip.joints):
            LOG.warning("Rejecting dock clip %s because it touches leg joints", clip.name)
            self.rejected_clips.append(clip.name)
            return False
        self.clips[clip.name] = clip
        tags = set(getattr(getattr(clip, "metadata", None), "tags", ()))
        self.scheduler.register(clip.name, required_tags=tags)
        return True

    def set_mood(self, mood: str) -> None:
        self.scheduler.set_mood(mood)

    def trigger(self, clip_name: str) -> bool:
        request = self.scheduler.trigger(clip_name)
        if request is None or request[0] not in self.clips:
            return False
        self._play(*request)
        return True

    def _play(self, clip_name: str, speed: float) -> None:
        player = AnimationPlayer(self.clips[clip_name], speed=speed)
        self.mixer.add(player, name=clip_name)

    def step(self, dt: float = 0.02) -> np.ndarray:
        base = self.docked_pose.copy()
        for joint, value in self.idle.update(dt).items():
            base[JOINT_INDEX[joint]] = value
        request = self.scheduler.update(dt)
        if request is not None and request[0] in self.clips:
            self._play(*request)
        self.mixer.update(dt)
        target = self.mixer.mix(base)
        target[:len(LEG_JOINTS)] = self._docked_legs
        limited = self.limiter.apply(target, self.previous_output)
        # Safety limiter implementations may rate-limit all indices: mask after it too.
        limited[:len(LEG_JOINTS)] = self._docked_legs
        assert np.array_equal(limited[:len(LEG_JOINTS)], self._docked_legs)
        self.previous_output = limited.copy()
        return limited

    def status(self) -> dict[str, Any]:
        return {
            "mode": "DEMO_DOCK",
            "mood": self.scheduler.mood,
            "active_clips": list(self.mixer.active_clips),
            "registered_clips": list(self.clips),
            "rejected_clips": list(self.rejected_clips),
            "torque_policy": "legs may be held at low torque while docked",
            "leg_mask_guarantee": "final output legs are byte-identical to docked_pose",
        }
