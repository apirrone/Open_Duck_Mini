import numpy as np

from animation.runtime.hybrid_controller import HybridController
from animation.runtime.modes import RobotMode

JOINT_LIMITS = np.asarray(
    [
        (-0.5235988, 0.5235988), (-0.4363323, 0.4363323),
        (-1.2217305, 0.5235988), (-1.5707963, 1.5707963),
        (-1.5707963, 1.5707963), (-0.5235988, 0.5235988),
        (-0.4363323, 0.4363323), (-0.5235988, 1.2217305),
        (-1.5707963, 1.5707963), (-1.5707963, 1.5707963),
        (-0.3490659, 1.1344640), (-0.7853982, 0.7853982),
        (-2.7925268, 2.7925268), (-0.5235988, 0.5235988),
        (-1.5707963, 1.5707963), (-1.5707963, 1.5707963),
    ],
    dtype=np.float32,
)


class FakeMixer:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.weights = {}
        self.active_clips = []

    def set_group_weight(self, group, weight):
        self.weights[group] = weight

    def update(self, _dt):
        pass

    def mix(self, base):
        result = np.asarray(base, dtype=np.float32).copy()
        result[:10] += self.offset
        return result

    def add(self, player, name=None):
        self.active_clips.append(name)

    def remove(self, name):
        self.active_clips.remove(name)

    def clear(self):
        self.active_clips.clear()


class FakeLimiter:
    def __init__(self):
        self.clamped_joints = []

    def apply(self, target, _previous):
        return np.clip(target, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def controller(offset=0.0):
    return HybridController(
        clips={},
        mixer=FakeMixer(offset),
        safety_limiter=FakeLimiter(),
        standing_pose=np.zeros(16),
    )


def test_demo_dock_forces_leg_weight_to_zero() -> None:
    control = controller()
    assert control.set_mode(RobotMode.DEMO_DOCK)
    control.step(0.02, (0, 0, 0))
    assert control.status()["group_weights"]["legs"] == 0.0


def test_additive_leg_offsets_are_clamped_before_safety() -> None:
    control = controller(offset=1.0)
    output = control.step(0.02, (0, 0, 0))
    assert np.allclose(output[:10], 0.15)


def test_emergency_stop_zeroes_animation_authority() -> None:
    control = controller()
    control.emergency_stop()
    control.step(0.02, (0, 0, 0))
    assert control.status()["group_weights"] == {"legs": 0.0, "head": 0.0, "antennas": 0.0}


def test_output_stays_within_joint_limits() -> None:
    control = controller(offset=100.0)
    output = control.step(0.02, (0, 0, 0))
    assert np.all((JOINT_LIMITS[:, 0] <= output) & (output <= JOINT_LIMITS[:, 1]))
