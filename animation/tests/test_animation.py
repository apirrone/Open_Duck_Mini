import numpy as np
import pytest

from duck_anim import (
    ALL_JOINTS,
    AnimationClip,
    AnimationPlayer,
    ClipValidationError,
    JointSafetyLimiter,
    LayeredMixer,
    load_clip,
    resample_clip,
    save_clip,
)


def clip(**overrides):
    values = {
        "name": "test",
        "fps": 50,
        "duration": 0.1,
        "frames": np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        "joints": ["head_yaw"],
    }
    values.update(overrides)
    return AnimationClip(**values)


def test_clip_save_load_round_trip(tmp_path):
    source = clip(joint_weights={"head_yaw": 0.5})
    source.metadata.author = "duck"
    path = tmp_path / "test.duckanim.json"
    save_clip(source, path)
    loaded = load_clip(path)
    assert loaded.to_dict() == source.to_dict()


@pytest.mark.parametrize(
    "source",
    [
        lambda: clip(joints=["not_a_joint"]),
        lambda: clip(frames=np.zeros((4, 1))),
        lambda: AnimationClip.from_dict(
            {"name": "bad", "fps": 50, "duration": 0.1, "joints": ["head_yaw"],
             "frames": [[0.0], [1.0, 2.0]]}
        ),
        lambda: clip(fps=0),
        lambda: clip(blend_in=0.06, blend_out=0.06),
        lambda: clip(layer="invalid"),
        lambda: clip(frames=np.array([[np.nan]] * 5)),
        lambda: clip(joint_weights={"head_yaw": 1.1}),
    ],
)
def test_clip_validation_rejects_invalid_data(source):
    with pytest.raises(ClipValidationError):
        source().validate()


def test_player_samples_exact_frames_and_midpoints():
    player = AnimationPlayer(clip())
    assert player.sample()[0] == pytest.approx([0.0])
    player.update(2 / 50)
    assert player.sample()[0] == pytest.approx([2.0])
    player.reset()
    player.update(0.01)
    assert player.sample()[0] == pytest.approx([0.5])


def test_looping_player_interpolates_the_seam():
    player = AnimationPlayer(clip(loop=True))
    player.update(0.08)
    assert player.sample()[0] == pytest.approx([4.0])
    player.update(0.005)
    assert player.sample()[0] == pytest.approx([3.0])
    player.update(0.015)
    assert player.sample()[0] == pytest.approx([0.0])


def test_player_envelope_and_finished_state():
    player = AnimationPlayer(clip(blend_in=0.02, blend_out=0.02))
    assert player.sample()[1] == 0
    player.update(0.02)
    assert player.sample()[1] == 1
    player.update(0.08)
    assert player.sample()[1] == 0
    assert player.finished
    immediate = AnimationPlayer(clip())
    assert immediate.sample()[1] == 1
    looping = AnimationPlayer(clip(loop=True, blend_out=0.02))
    looping.update(1)
    assert not looping.finished
    looping.stop()
    looping.update(0.01)
    assert not looping.finished
    looping.update(0.01)
    assert looping.finished


def test_mixer_layers_weights_groups_and_priority():
    base = np.zeros(len(ALL_JOINTS), dtype=np.float32)
    low = AnimationPlayer(clip(name="low", frames=np.full((5, 1), 2.0), priority=1))
    high = AnimationPlayer(clip(name="high", frames=np.full((5, 1), 6.0), priority=2))
    high.weight_scale = 0.5
    mixer = LayeredMixer()
    mixer.add(high)
    mixer.add(low)
    assert mixer.mix(base)[12] == pytest.approx(4.0)
    additive = AnimationPlayer(
        clip(name="add", layer="additive", frames=np.full((5, 1), 3.0))
    )
    mixer.clear()
    mixer.add(additive)
    assert mixer.mix(base)[12] == pytest.approx(3.0)
    leg = AnimationPlayer(
        clip(name="leg", joints=["left_knee"], frames=np.full((5, 1), 3.0))
    )
    mixer.clear()
    mixer.add(leg)
    mixer.set_group_weight("legs", 0)
    assert mixer.mix(base)[3] == 0


def test_safety_limiter_clamps_slews_and_scrubs_nonfinite_values():
    limiter = JointSafetyLimiter(max_accel=None)
    previous = np.zeros(len(ALL_JOINTS), dtype=np.float32)
    output = limiter.apply(np.full(len(ALL_JOINTS), 100.0), previous)
    assert output[0] == pytest.approx(0.32)
    assert "left_hip_yaw" in limiter.clamped_joints
    output = limiter.apply(np.full(len(ALL_JOINTS), -100.0), output)
    assert output[0] == pytest.approx(0.0)
    scrubbed = limiter.apply(np.full(len(ALL_JOINTS), np.nan), output)
    assert np.array_equal(scrubbed, output)
    assert limiter.nan_events == len(ALL_JOINTS)


def test_resample_preserves_frame_count_and_endpoints():
    source = AnimationClip(
        name="source",
        fps=24,
        duration=1.0,
        frames=np.linspace(0, 1, 24, dtype=np.float32).reshape(-1, 1),
        joints=["head_yaw"],
    )
    result = resample_clip(source, 50)
    assert result.frames.shape == (50, 1)
    assert result.frames[0] == pytest.approx(source.frames[0])
    assert result.frames[-1] == pytest.approx(source.frames[-1])
