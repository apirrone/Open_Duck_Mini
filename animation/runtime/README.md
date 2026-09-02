# Hybrid animation runtime

`HybridController` is a transport-independent 50 Hz controller. Construct it with
an optional `WalkPolicy`, call `set_mode(RobotMode.HYBRID_WALK)`, add clips with
`play()` or `play_additive()`, then call `step(dt, (vx, vy, wz), measured_obs)`.
It returns canonical 16-joint targets suitable for either MuJoCo or hardware.

The order is deliberate: measured state enters the RL policy, its raw action
becomes base targets, animation layers mix downstream, leg authority is reduced
as commanded speed rises, and the safety limiter runs last. The policy's
`previous_action` is always its own raw action, never blended animation targets;
feeding blended targets back changes the policy's perceived command history and
can destabilize walking. Dock mode disables all leg animation authority,
including the full dock transition.
