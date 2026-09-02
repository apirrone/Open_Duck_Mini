"""Layered composition of animation players over controller output."""

from __future__ import annotations

import itertools

import numpy as np

from .joints import JOINT_INDEX, group_of
from .player import AnimationPlayer


class LayeredMixer:
    """Mix override and additive animation clips over a full joint target array."""

    def __init__(self) -> None:
        self._players: dict[str, AnimationPlayer] = {}
        self._sequence = itertools.count()
        self._group_weights = {"legs": 1.0, "head": 1.0, "antennas": 1.0}

    def add(self, player: AnimationPlayer, name: str | None = None) -> str:
        """Add ``player`` and return its unique mixer name."""
        name = name or f"{player.clip.name}-{next(self._sequence)}"
        if name in self._players:
            raise ValueError(f"An animation player named {name!r} already exists")
        self._players[name] = player
        return name

    def remove(self, name: str) -> None:
        """Remove a player by name."""
        del self._players[name]

    def clear(self) -> None:
        """Remove all players."""
        self._players.clear()

    @property
    def active_clips(self) -> dict[str, AnimationPlayer]:
        """A shallow name-to-player mapping of currently active clips."""
        return dict(self._players)

    def update(self, dt: float) -> None:
        """Advance all players and discard players whose fade has completed."""
        for name, player in list(self._players.items()):
            player.update(dt)
            if player.finished:
                del self._players[name]

    def mix(self, base: np.ndarray) -> np.ndarray:
        """Compose active players over a 16-joint controller target array."""
        result = np.asarray(base, dtype=np.float32).copy()
        if result.shape != (len(JOINT_INDEX),):
            raise ValueError(f"base must have shape ({len(JOINT_INDEX)},), got {result.shape}")
        ordered = sorted(
            self._players.items(), key=lambda item: (item[1].clip.priority, item[0])
        )
        for _, player in ordered:
            values, envelope = player.sample()
            if envelope <= 0.0 or player.weight_scale == 0.0:
                continue
            for joint, value, joint_weight in zip(
                player.clip.joints, values, player.clip.effective_joint_weights
            ):
                weight = (
                    envelope
                    * float(joint_weight)
                    * player.weight_scale
                    * self._group_weights[group_of(joint)]
                )
                index = JOINT_INDEX[joint]
                if player.clip.layer == "override":
                    result[index] += (float(value) - result[index]) * weight
                else:
                    result[index] += float(value) * weight
        return result

    def set_group_weight(self, group: str, weight: float) -> None:
        """Set a [0, 1] multiplier for one joint group."""
        if group not in self._group_weights:
            raise KeyError(f"Unknown joint group: {group!r}")
        if not 0.0 <= float(weight) <= 1.0:
            raise ValueError("group weight must be in [0, 1]")
        self._group_weights[group] = float(weight)

    def crossfade(self, new_player: AnimationPlayer, duration: float) -> str:
        """Fade current players out and add ``new_player`` fading in over ``duration``."""
        if duration < 0.0:
            raise ValueError("duration must be >= 0")
        for player in self._players.values():
            player.clip.blend_out = float(duration)
            player.stop()
        new_player.clip.blend_in = float(duration)
        new_player.reset()
        return self.add(new_player)
