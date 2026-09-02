"""Loading, saving and resampling of ``.duckanim.json`` clips."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import AnimationClip, ClipValidationError

CLIP_SUFFIX = ".duckanim.json"


def load_clip(path: str | Path) -> AnimationClip:
    """Load and validate a single ``.duckanim.json`` clip file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such clip file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ClipValidationError(f"Clip {path} is not valid JSON: {exc}") from exc
    clip = AnimationClip.from_dict(data)
    clip.validate()
    return clip


def load_clip_dir(directory: str | Path) -> dict[str, AnimationClip]:
    """Recursively load every ``*.duckanim.json`` under ``directory``.

    Returns a mapping of clip name -> clip. Raises if two files declare the
    same clip name.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"No such clip directory: {directory}")

    clips: dict[str, AnimationClip] = {}
    for path in sorted(directory.rglob(f"*{CLIP_SUFFIX}")):
        clip = load_clip(path)
        if clip.name in clips:
            raise ClipValidationError(
                f"Duplicate clip name {clip.name!r} in {path} and an earlier file"
            )
        clips[clip.name] = clip
    return clips


def save_clip(clip: AnimationClip, path: str | Path) -> None:
    """Validate and serialize ``clip`` to ``path`` as ``.duckanim.json``."""
    clip.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clip.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


def resample_clip(clip: AnimationClip, new_fps: float) -> AnimationClip:
    """Return a copy of ``clip`` retimed to ``new_fps`` via linear interpolation.

    Used to bring clips authored in Blender at e.g. 24 or 60 fps onto the
    robot's 50 Hz control rate. Frame ``i`` of the result is sampled at time
    ``i / new_fps``; the first and last source frames are preserved exactly,
    so endpoints are identical after resampling.
    """
    if not np.isfinite(new_fps) or new_fps <= 0:
        raise ValueError(f"new_fps must be > 0, got {new_fps!r}")
    clip.validate()

    n_in = clip.n_frames
    duration = clip.duration
    if n_in == 0:
        frames_out = np.zeros((0, len(clip.joints)), dtype=np.float32)
    elif n_in == 1:
        frames_out = np.repeat(clip.frames, round(duration * new_fps), axis=0)
    else:
        old_times = np.linspace(0.0, duration, num=n_in, dtype=np.float64)
        n_out = max(round(duration * new_fps), 2) if duration > 0 else 1
        new_times = np.linspace(0.0, duration, num=n_out, dtype=np.float64)
        frames_out = np.empty((n_out, clip.frames.shape[1]), dtype=np.float32)
        for j in range(clip.frames.shape[1]):
            frames_out[:, j] = np.interp(new_times, old_times, clip.frames[:, j])

    out = AnimationClip(
        name=clip.name,
        fps=float(new_fps),
        loop=clip.loop,
        duration=clip.duration,
        blend_in=clip.blend_in,
        blend_out=clip.blend_out,
        priority=clip.priority,
        layer=clip.layer,
        joint_weights=dict(clip.joint_weights),
        metadata=type(clip.metadata)(
            tags=list(clip.metadata.tags),
            author=clip.metadata.author,
            source_blend=clip.metadata.source_blend,
        ),
        format_version=clip.format_version,
        joints=list(clip.joints),
        frames=frames_out,
    )
    out.validate()
    return out
