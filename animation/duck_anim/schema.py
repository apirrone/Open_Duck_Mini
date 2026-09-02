"""Clip data model for ``.duckanim.json`` animation clips.

A clip is a JSON file (see loader.py / README.md for the on-disk format):

    {
      "format_version": 1,
      "name": "look_around",
      "fps": 50,
      "loop": false,
      "duration": 3.0,
      "blend_in": 0.25,
      "blend_out": 0.25,
      "priority": 10,
      "layer": "override",
      "joints": ["head_yaw", "head_pitch"],
      "joint_weights": {"head_yaw": 1.0, "head_pitch": 1.0},
      "frames": [[0.0, 0.1], [0.01, 0.11]],
      "metadata": {"tags": ["idle"], "author": "", "source_blend": ""}
    }

Angles are in RADIANS. ``layer`` is ``"override"`` (absolute target angles) or
``"additive"`` (offsets added to whatever the base controller produced).
``frames`` has exactly ``round(duration * fps)`` rows of ``len(joints)`` floats
in the same order as ``joints``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .joints import ALL_JOINTS, JOINT_INDEX

LAYER_OVERRIDE = "override"
LAYER_ADDITIVE = "additive"
VALID_LAYERS = (LAYER_OVERRIDE, LAYER_ADDITIVE)

FORMAT_VERSION = 1


class ClipValidationError(ValueError):
    """Raised when an animation clip violates the clip format."""


@dataclass
class ClipMetadata:
    tags: list[str] = field(default_factory=list)
    author: str = ""
    source_blend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tags": list(self.tags),
            "author": self.author,
            "source_blend": self.source_blend,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ClipMetadata":
        data = dict(data or {})
        return cls(
            tags=list(data.get("tags", [])),
            author=str(data.get("author", "")),
            source_blend=str(data.get("source_blend", "")),
        )


@dataclass
class AnimationClip:
    name: str
    fps: float
    duration: float
    frames: np.ndarray  # (n_frames, n_joints) float32
    joints: list[str]
    loop: bool = False
    blend_in: float = 0.0
    blend_out: float = 0.0
    priority: int = 0
    layer: str = LAYER_OVERRIDE
    joint_weights: dict[str, float] = field(default_factory=dict)
    metadata: ClipMetadata = field(default_factory=ClipMetadata)
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=np.float32)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "fps": self.fps,
            "loop": self.loop,
            "duration": self.duration,
            "blend_in": self.blend_in,
            "blend_out": self.blend_out,
            "priority": self.priority,
            "layer": self.layer,
            "joints": list(self.joints),
            "joint_weights": dict(self.joint_weights),
            "frames": [[float(v) for v in row] for row in self.frames],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnimationClip":
        missing = [k for k in ("name", "fps", "duration", "frames", "joints") if k not in data]
        if missing:
            raise ClipValidationError(f"Clip is missing required fields: {missing}")
        joints = list(data["joints"])
        try:
            frames = np.asarray(data["frames"], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ClipValidationError(
                "frames must be a rectangular numeric 2-D array"
            ) from exc
        if frames.size == 0:
            frames = frames.reshape(0, len(joints))
        elif frames.ndim == 1:
            # A single row may be written as a flat list.
            frames = frames.reshape(1, -1)
        return cls(
            name=str(data["name"]),
            fps=float(data["fps"]),
            loop=bool(data.get("loop", False)),
            duration=float(data["duration"]),
            blend_in=float(data.get("blend_in", 0.0)),
            blend_out=float(data.get("blend_out", 0.0)),
            priority=int(data.get("priority", 0)),
            layer=str(data.get("layer", LAYER_OVERRIDE)),
            joint_weights={k: float(v) for k, v in data.get("joint_weights", {}).items()},
            metadata=ClipMetadata.from_dict(data.get("metadata")),
            format_version=int(data.get("format_version", FORMAT_VERSION)),
            joints=joints,
            frames=frames,
        )

    # -- validation ----------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ClipValidationError` if the clip violates the format."""
        errors: list[str] = []

        if self.format_version != FORMAT_VERSION:
            errors.append(
                f"Unsupported format_version {self.format_version!r} (expected {FORMAT_VERSION})"
            )

        unknown = [j for j in self.joints if j not in JOINT_INDEX]
        if unknown:
            errors.append(
                f"Unknown joint names {unknown}; valid joints are {list(ALL_JOINTS)}"
            )
        if len(set(self.joints)) != len(self.joints):
            errors.append("Duplicate joint names in 'joints' list")

        if not math.isfinite(self.fps) or self.fps <= 0:
            errors.append(f"fps must be > 0, got {self.fps!r}")
        if not math.isfinite(self.duration) or self.duration < 0:
            errors.append(f"duration must be >= 0, got {self.duration!r}")

        frames = np.asarray(self.frames, dtype=np.float32)
        if frames.ndim != 2:
            errors.append(f"frames must be 2-D (n_frames, n_joints), got shape {frames.shape}")
        else:
            expected_rows = round(self.duration * self.fps)
            if frames.shape[0] != expected_rows:
                errors.append(
                    f"frames has {frames.shape[0]} rows, expected round(duration*fps) = "
                    f"{expected_rows} (duration={self.duration}, fps={self.fps})"
                )
            if frames.shape[1] != len(self.joints):
                errors.append(
                    f"each frame row must have {len(self.joints)} values (one per joint), "
                    f"got {frames.shape[1]}"
                )
            if not np.all(np.isfinite(frames)):
                errors.append("frames contain NaN or inf values")

        for key in ("blend_in", "blend_out"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                errors.append(f"{key} must be a finite value >= 0, got {value!r}")
        if (
            math.isfinite(self.blend_in)
            and math.isfinite(self.blend_out)
            and math.isfinite(self.duration)
            and self.blend_in + self.blend_out > self.duration + 1e-9
        ):
            errors.append(
                f"blend_in + blend_out ({self.blend_in} + {self.blend_out}) exceeds "
                f"duration ({self.duration})"
            )

        if self.layer not in VALID_LAYERS:
            errors.append(
                f"Unknown layer {self.layer!r}; must be one of {list(VALID_LAYERS)}"
            )

        for joint, weight in self.joint_weights.items():
            if joint not in self.joints:
                errors.append(
                    f"joint_weights names joint {joint!r} which is not in the clip's "
                    f"'joints' list {self.joints}"
                )
            if not math.isfinite(weight) or not (0.0 <= weight <= 1.0):
                errors.append(
                    f"joint_weights[{joint!r}] must be in [0, 1], got {weight!r}"
                )

        if errors:
            raise ClipValidationError(
                f"Invalid clip {self.name!r}: " + "; ".join(errors)
            )

    # -- derived properties ---------------------------------------------------

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def effective_joint_weights(self) -> np.ndarray:
        """Per-joint weights aligned with ``self.joints`` (default 1.0)."""
        return np.array(
            [float(self.joint_weights.get(j, 1.0)) for j in self.joints],
            dtype=np.float32,
        )
