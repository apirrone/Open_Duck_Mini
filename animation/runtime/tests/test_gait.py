from animation.runtime.gait import GaitPhaseTracker, leg_animation_gain
from animation.runtime.modes import RobotMode


def test_leg_gain_is_stationary_one_and_monotonic() -> None:
    speeds = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]
    gains = [leg_animation_gain(RobotMode.HYBRID_WALK, (speed, 0, 0)) for speed in speeds]
    assert gains[0] == 1.0
    assert all(left >= right for left, right in zip(gains, gains[1:]))


def test_stationary_requires_debounce() -> None:
    tracker = GaitPhaseTracker(debounce_window=0.3)
    tracker.update(0.29, (0, 0, 0))
    assert not tracker.is_stationary
    tracker.update(0.01, (0, 0, 0))
    assert tracker.is_stationary
    tracker.update(0.02, (1, 0, 0))
    assert not tracker.is_stationary
