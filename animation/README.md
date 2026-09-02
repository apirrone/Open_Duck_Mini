# duck_anim

`duck_anim` is the small, dependency-light animation runtime for Open Duck Mini
v2. It needs only Python and NumPy, making it suitable for the Raspberry Pi
control process running at 50 Hz.

## Clip format

Clips are JSON files conventionally named `*.duckanim.json`. Angles are radians.
`frames` is a rectangular `n_frames x n_joints` numeric array, with columns
matching the listed `joints`. A clip has exactly `round(duration * fps)` frames;
frame `i` is sampled at `i / fps`.

```json
{
  "format_version": 1,
  "name": "look_left",
  "fps": 50,
  "duration": 0.5,
  "loop": false,
  "blend_in": 0.1,
  "blend_out": 0.1,
  "priority": 10,
  "layer": "override",
  "joints": ["head_yaw"],
  "joint_weights": {"head_yaw": 1.0},
  "frames": [[0.0], [0.01]],
  "metadata": {"tags": ["idle"], "author": "", "source_blend": ""}
}
```

Use `load_clip`, `load_clip_dir`, and `save_clip` for validated I/O.
`resample_clip(clip, 50)` retimes authored clips while preserving endpoint
values.

## Playback and mixing

`AnimationPlayer` linearly interpolates frame values. Its blend-in and blend-out
envelope uses smoothstep; looping clips retain full weight until `stop()` is
called, at which point they fade out.

`LayeredMixer.mix(base)` starts with the controller's ordered 16-joint target
array. It applies lower-priority clips first, then higher-priority clips:

* An `override` clip lerps its absolute target over the current target.
* An `additive` clip adds its scaled offset to the current target.

Each joint contribution is scaled by envelope weight, the clip's
`joint_weights`, `player.weight_scale`, and a mixer group weight. The `legs`,
`head`, and `antennas` group weights default to 1 and can mute animation in a
group, such as leg clips while the walking policy is active.

## Hardware safety

Run the final mixed command through `JointSafetyLimiter.apply(target,
previous_output)`. It replaces non-finite inputs, clamps angles inside the URDF
limits with a configurable safety margin, limits slew rate from the previous
output, and limits per-step acceleration. It exposes `nan_events` and
`clamped_joints` for diagnostics.
