"""Robot operating modes and their intentionally constrained transitions."""

from __future__ import annotations

from enum import Enum, auto


class RobotMode(Enum):
    IDLE = auto()
    STAND = auto()
    WALK = auto()
    HYBRID_STAND = auto()
    HYBRID_WALK = auto()
    DEMO_DOCK = auto()
    EMERGENCY_STOP = auto()


_ALL_MODES = frozenset(RobotMode)
_TRANSITIONS: dict[RobotMode, frozenset[RobotMode]] = {
    RobotMode.IDLE: frozenset(
        {RobotMode.STAND, RobotMode.DEMO_DOCK, RobotMode.EMERGENCY_STOP}
    ),
    RobotMode.STAND: frozenset(
        {
            RobotMode.IDLE,
            RobotMode.WALK,
            RobotMode.HYBRID_STAND,
            RobotMode.DEMO_DOCK,
            RobotMode.EMERGENCY_STOP,
        }
    ),
    RobotMode.WALK: frozenset(
        {RobotMode.STAND, RobotMode.HYBRID_WALK, RobotMode.EMERGENCY_STOP}
    ),
    RobotMode.HYBRID_STAND: frozenset(
        {RobotMode.STAND, RobotMode.EMERGENCY_STOP}
    ),
    RobotMode.HYBRID_WALK: frozenset(
        {RobotMode.WALK, RobotMode.STAND, RobotMode.EMERGENCY_STOP}
    ),
    RobotMode.DEMO_DOCK: frozenset(
        {RobotMode.STAND, RobotMode.IDLE, RobotMode.EMERGENCY_STOP}
    ),
    RobotMode.EMERGENCY_STOP: frozenset({RobotMode.IDLE}),
}


class ModeStateMachine:
    """Track mode transitions and a blend progress for downstream controllers."""

    def __init__(
        self,
        initial_mode: RobotMode = RobotMode.IDLE,
        default_duration: float = 0.3,
        dock_duration: float = 0.8,
    ) -> None:
        if default_duration < 0 or dock_duration < 0:
            raise ValueError("Transition durations must be non-negative")
        self.mode = initial_mode
        self.previous_mode = initial_mode
        self.target_mode = initial_mode
        self.default_duration = float(default_duration)
        self.dock_duration = float(dock_duration)
        self._elapsed = 0.0
        self._duration = 0.0

    @property
    def transitioning(self) -> bool:
        return self.mode != self.target_mode

    @property
    def blend_alpha(self) -> float:
        if not self.transitioning or self._duration == 0:
            return 1.0
        return min(1.0, max(0.0, self._elapsed / self._duration))

    def request(self, mode: RobotMode) -> bool:
        """Start a legal transition, returning false without changing state otherwise."""
        if mode == self.target_mode:
            return True
        if mode not in _TRANSITIONS[self.mode]:
            return False
        self.previous_mode = self.mode
        self.target_mode = mode
        self._elapsed = 0.0
        self._duration = self._transition_duration(self.mode, mode)
        if self._duration == 0:
            self.mode = mode
        return True

    def update(self, dt: float) -> None:
        if dt < 0:
            raise ValueError("dt must be non-negative")
        if not self.transitioning:
            return
        self._elapsed += dt
        if self._elapsed >= self._duration:
            self.mode = self.target_mode

    def _transition_duration(self, source: RobotMode, target: RobotMode) -> float:
        if target is RobotMode.EMERGENCY_STOP:
            return 0.0
        if source is RobotMode.DEMO_DOCK or target is RobotMode.DEMO_DOCK:
            return self.dock_duration
        return self.default_duration
