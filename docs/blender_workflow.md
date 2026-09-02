# Blender animation workflow

Install the Open Duck Mini Blender addon, then use **Build Rig from URDF** and
select `mini_bdx/robots/open_duck_mini_v2/robot.urdf`. Do not rename generated
bones: they map exactly to robot joints such as `neck_pitch`, `head_yaw`,
`head_roll`, `left_antenna`, and `right_antenna`. Enable the addon’s joint-limit
display and stay within the shown limits; Blender rotations export as radians.

For a head-only clip, key only the four head and two antenna bones. These clips
are safe for a dock and should use the `override` layer. Full-body clips need
small, subtle offsets and use `additive`; they are not dock-safe. In the Action
custom properties set `loop`, `blend_in`, `blend_out`, `priority`, `layer`, and
comma-separated `tags`. Keep timing at 50 FPS or let the exporter resample.

Use **File > Export > Open Duck Animation** to export a `.duckanim.json`. For
automation, invoke the addon’s documented headless CLI with the blend file,
action name, and output path. Inspect the exported JSON: bone names must be
canonical, each frame row must match `joints`, and duration times FPS must equal
the number of rows.

Load the clip in the simulator before hardware. For a dock presentation, run
the print demo or simulation and ensure the clip is accepted; any clip carrying
a leg joint is deliberately rejected by `DockMode`.
