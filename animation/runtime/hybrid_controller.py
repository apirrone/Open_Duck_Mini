"""Safe, transport-agnostic fusion of walking targets and authored animation."""

from __future__ import annotations

import logging
import importlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .gait import GaitPhaseTracker, leg_animation_gain
from .modes import ModeStateMachine, RobotMode
from .policy import INIT_POS, WalkPolicy

LOG = logging.getLogger(__name__)
LEG_COUNT = 10


def _duck_anim_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        package = importlib.import_module("duck_anim")
    except ImportError as exc:
        try:
            package = importlib.import_module("animation.duck_anim")
        except ImportError:
            raise RuntimeError(
                "duck_anim with loader, player, mixer, and safety modules is required."
            ) from exc
    try:
        prefix = package.__name__
        return (
            importlib.import_module(f"{prefix}.loader").load_clip_dir,
            importlib.import_module(f"{prefix}.mixer").LayeredMixer,
            importlib.import_module(f"{prefix}.player").AnimationPlayer,
            importlib.import_module(f"{prefix}.safety").JointSafetyLimiter,
        )
    except ImportError as exc:
        raise RuntimeError(
            "duck_anim with loader, player, mixer, and safety modules is required."
        ) from exc


class HybridController:
    """Blend animation strictly downstream from base standing or walking control."""

    def __init__(
        self,
        *,
        policy: WalkPolicy | None = None,
        clips_dir: str | Path = "animation/clips",
        clips: Mapping[str, Any] | None = None,
        mixer: Any | None = None,
        safety_limiter: Any | None = None,
        standing_pose: Sequence[float] = INIT_POS,
        docked_pose: Sequence[float] | None = None,
        max_additive_leg_offset: float = 0.15,
        full_body_speed_threshold: float = 0.1,
    ) -> None:
        self.mode_machine = ModeStateMachine()
        self.gait = GaitPhaseTracker()
        self.policy = policy
        self.standing_pose = self._pose(standing_pose, "standing_pose")
        self.docked_pose = self._pose(
            self.standing_pose if docked_pose is None else docked_pose, "docked_pose"
        )
        if max_additive_leg_offset < 0:
            raise ValueError("max_additive_leg_offset must be non-negative")
        self.max_additive_leg_offset = float(max_additive_leg_offset)
        self.full_body_speed_threshold = float(full_body_speed_threshold)

        if mixer is None or safety_limiter is None or clips is None:
            load_clip_dir, LayeredMixer, _, JointSafetyLimiter = _duck_anim_dependencies()
            mixer = mixer or LayeredMixer()
            safety_limiter = safety_limiter or JointSafetyLimiter(dt=0.02)
            clips = clips if clips is not None else self._load_clips(clips_dir, load_clip_dir)
        self.mixer = mixer
        self.safety_limiter = safety_limiter
        self.clips = dict(clips)
        self._last_output = self.standing_pose.copy()
        self._group_weights = {"legs": 1.0, "head": 1.0, "antennas": 1.0}

    @property
    def mode(self) -> RobotMode:
        return self.mode_machine.target_mode

    @staticmethod
    def _pose(values: Sequence[float], name: str) -> np.ndarray:
        pose = np.asarray(values, dtype=np.float32).reshape(-1)
        if pose.size != 16:
            raise ValueError(f"{name} must contain 16 joint values")
        return pose.copy()

    @staticmethod
    def _load_clips(path: str | Path, loader: Any) -> dict[str, Any]:
        directory = Path(path)
        if not directory.exists():
            LOG.warning("Animation clip directory does not exist: %s", directory)
            return {}
        return loader(directory)

    def set_mode(self, mode: RobotMode) -> bool:
        return self.mode_machine.request(mode)

    def emergency_stop(self) -> None:
        self.mode_machine.request(RobotMode.EMERGENCY_STOP)
        self.stop_all()

    def _clip_has_legs(self, clip: Any) -> bool:
        return any(
            joint in {
                "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
                "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
            }
            for joint in clip.joints
        )

    def _clip_is_full_body(self, clip: Any) -> bool:
        return self._clip_has_legs(clip) and len(clip.joints) > LEG_COUNT

    def play(self, clip_name: str, **kwargs: Any) -> bool:
        clip = self.clips.get(clip_name)
        if clip is None:
            LOG.warning("Unknown animation clip: %s", clip_name)
            return False
        if self.mode is RobotMode.DEMO_DOCK and self._clip_has_legs(clip):
            LOG.warning("Refusing leg clip %s while docked", clip_name)
            return False
        commanded_velocity = np.asarray(kwargs.pop("commanded_velocity", (0.0, 0.0, 0.0)))
        if self._clip_is_full_body(clip) and np.linalg.norm(commanded_velocity) > self.full_body_speed_threshold:
            LOG.warning("Refusing full-body clip %s while moving", clip_name)
            return False
        _, _, AnimationPlayer, _ = _duck_anim_dependencies()
        player = AnimationPlayer(clip, **kwargs)
        self.mixer.add(player, name=clip_name)
        return True

    def play_additive(self, clip_name: str, **kwargs: Any) -> bool:
        clip = self.clips.get(clip_name)
        if clip is None or getattr(clip, "layer", None) != "additive":
            LOG.warning("Clip %s is not an additive clip", clip_name)
            return False
        return self.play(clip_name, **kwargs)

    def stop(self, clip_name: str) -> None:
        self.mixer.remove(clip_name)

    def stop_all(self) -> None:
        self.mixer.clear()

    def _base_pose(self, policy_obs: Sequence[float] | None) -> np.ndarray:
        mode = self.mode
        if mode in {RobotMode.WALK, RobotMode.HYBRID_WALK}:
            if self.policy is None:
                raise RuntimeError("WALK mode requires a WalkPolicy")
            if policy_obs is None:
                raise ValueError("WALK mode requires true measured policy_obs")
            return self.policy.action_to_targets(self.policy.infer(policy_obs))
        if mode is RobotMode.DEMO_DOCK:
            return self.docked_pose.copy()
        return self.standing_pose.copy()

    def _set_group_weights(self, commanded_velocity: Sequence[float]) -> None:
        dock_transition = (
            self.mode_machine.transitioning
            and (
                self.mode_machine.mode is RobotMode.DEMO_DOCK
                or self.mode_machine.target_mode is RobotMode.DEMO_DOCK
            )
        )
        if self.mode is RobotMode.EMERGENCY_STOP:
            weights = {"legs": 0.0, "head": 0.0, "antennas": 0.0}
        else:
            legs = 0.0 if self.mode is RobotMode.DEMO_DOCK or dock_transition else leg_animation_gain(self.mode, commanded_velocity)
            weights = {"legs": legs, "head": 1.0, "antennas": 1.0}
        self._group_weights = weights
        for group, weight in weights.items():
            self.mixer.set_group_weight(group, weight)

    def _clamp_additive_legs(self, base: np.ndarray, mixed: np.ndarray) -> np.ndarray:
        """Bound downstream leg deviations before final position/velocity safety."""
        output = np.asarray(mixed, dtype=np.float32).copy()
        offsets = np.clip(
            output[:LEG_COUNT] - base[:LEG_COUNT],
            -self.max_additive_leg_offset,
            self.max_additive_leg_offset,
        )
        output[:LEG_COUNT] = base[:LEG_COUNT] + offsets
        return output

    def step(
        self,
        dt: float,
        commanded_velocity: Sequence[float],
        policy_obs: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Return the final canonical 16-joint target array."""
        self.mode_machine.update(dt)
        self.gait.update(dt, commanded_velocity)
        base = self._base_pose(policy_obs)
        self._set_group_weights(commanded_velocity)
        self.mixer.update(dt)
        mixed = self._clamp_additive_legs(base, self.mixer.mix(base))
        self._last_output = np.asarray(
            self.safety_limiter.apply(mixed, self._last_output), dtype=np.float32
        )
        return self._last_output.copy()

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode.name,
            "blend_alpha": self.mode_machine.blend_alpha,
            "active_clips": list(getattr(self.mixer, "active_clips", [])),
            "group_weights": dict(self._group_weights),
            "clamped_joints": list(getattr(self.safety_limiter, "clamped_joints", [])),
            "gait_phase": self.gait.phase,
        }
