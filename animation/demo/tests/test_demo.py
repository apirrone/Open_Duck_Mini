import math

import numpy as np

from duck_anim.joints import ALL_JOINTS, JOINT_INDEX, JOINT_LIMITS, LEG_JOINTS
from duck_anim.schema import AnimationClip, ClipMetadata
from demo.dock import DockMode
from demo.gaze import GazeSolver, GazeTracker
from demo.idle_engine import IdleEngine
from demo.noise import SmoothNoise
from demo.scheduler import BehaviorScheduler


def clip(name, joints):
    return AnimationClip(name=name, fps=50, duration=0.02, frames=np.zeros((1, len(joints))), joints=joints, metadata=ClipMetadata())


def test_smooth_noise_is_seeded_continuous_and_bounded():
    first, second = SmoothNoise(7, 0.15), SmoothNoise(7, 0.15)
    values = [first.sample(i * 0.02) for i in range(1000)]
    assert values == [second.sample(i * 0.02) for i in range(1000)]
    assert max(abs(value) for value in values) <= 1
    assert max(abs(b - a) for a, b in zip(values, values[1:])) < 0.05


def test_gaze_solver_clamps_and_splits_pitch():
    solver = GazeSolver(neck_ratio=0.4)
    result = solver.solve(100, 0.5)
    assert result["neck_pitch"] == pytest_approx(0.2)
    assert result["head_pitch"] == pytest_approx(0.3)
    for name, value in result.items():
        assert JOINT_LIMITS[name][0] <= value <= JOINT_LIMITS[name][1]


def test_gaze_tracker_respects_speed_limit():
    tracker = GazeTracker(max_angular_velocity=0.4)
    tracker.look_at(2, 1)
    previous = (tracker.azimuth, tracker.elevation)
    for _ in range(100):
        tracker.update(0.02)
        current = (tracker.azimuth, tracker.elevation)
        assert max(abs(a - b) for a, b in zip(current, previous)) <= 0.4 * 0.02 + 1e-9
        previous = current


def test_scheduler_never_repeats_and_honors_cooldown_and_mood():
    scheduler = BehaviorScheduler(seed=1, gap=(0, 0))
    scheduler.register("look_around", cooldown=0.1)
    scheduler.register("curious_tilt", cooldown=0.1)
    scheduled = [scheduler.update(0.11)[0] for _ in range(20)]
    assert all(a != b for a, b in zip(scheduled, scheduled[1:]))
    biased = BehaviorScheduler(seed=2, gap=(0, 0))
    biased.register("look_around", cooldown=0)
    biased.register("nod_yes", cooldown=0)
    biased.register("shake_no", cooldown=0)
    biased.set_mood("curious")
    counts = {"look_around": 0, "nod_yes": 0, "shake_no": 0}
    for _ in range(1000):
        counts[biased.update(0.01)[0]] += 1
    assert counts["look_around"] > counts["nod_yes"] * 1.3


def test_idle_engine_never_emits_legs():
    engine = IdleEngine(seed=3)
    for _ in range(5000):
        assert not (set(engine.update(0.02)) & set(LEG_JOINTS))


def test_dock_mode_legs_are_bitwise_docked_even_for_forced_clips():
    pose = np.linspace(-0.2, 0.2, len(ALL_JOINTS), dtype=np.float32)
    dock = DockMode(docked_pose=pose, seed=4)
    assert dock.register_clip(clip("nod_yes", ["head_pitch"]))
    for _ in range(1000):
        dock.trigger("nod_yes")
        output = dock.step(0.02)
        assert output[:10].tobytes() == pose[:10].tobytes()


def test_dock_rejects_leg_clip():
    dock = DockMode(seed=5)
    assert not dock.register_clip(clip("unsafe", ["left_knee", "head_yaw"]))
    assert "unsafe" not in dock.clips


def pytest_approx(value):
    import pytest
    return pytest.approx(value)
