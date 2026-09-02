"""Open Duck Mini v2 animation runtime public API."""

from .joints import (
    ALL_JOINTS,
    ANTENNA_JOINTS,
    HEAD_JOINTS,
    JOINT_GROUPS,
    JOINT_INDEX,
    JOINT_LIMITS,
    JOINT_VELOCITY_LIMITS,
    LEG_JOINTS,
)
from .loader import load_clip, load_clip_dir, resample_clip, save_clip
from .mixer import LayeredMixer
from .player import AnimationPlayer
from .safety import JointSafetyLimiter
from .schema import AnimationClip, ClipValidationError

__all__ = [
    "ALL_JOINTS", "ANTENNA_JOINTS", "HEAD_JOINTS", "JOINT_GROUPS",
    "JOINT_INDEX", "JOINT_LIMITS", "JOINT_VELOCITY_LIMITS", "LEG_JOINTS",
    "AnimationClip", "ClipValidationError", "load_clip", "load_clip_dir",
    "save_clip", "resample_clip", "AnimationPlayer", "LayeredMixer",
    "JointSafetyLimiter",
]
