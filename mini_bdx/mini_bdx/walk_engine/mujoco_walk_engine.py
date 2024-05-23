import time

import mujoco
import mujoco.viewer
import mujoco_viewer
import numpy as np
import placo

from mini_bdx.walk_engine import WalkEngine

model = mujoco.MjModel.from_xml_path("../../robots/bdx/scene.xml")
data = mujoco.MjData(model)

# viewer = mujoco_viewer.MujocoViewer(
#     model, data, mode="window", width=1280, height=800, hide_menus=True
# )

viewer = mujoco.viewer.launch_passive(model, data)

dofs = {
    "right_hip_yaw": 0,
    "right_hip_roll": 1,
    "right_hip_pitch": 2,
    "right_knee": 3,
    "right_ankle": 4,
    "left_hip_yaw": 5,
    "left_hip_roll": 6,
    "left_hip_pitch": 7,
    "left_knee": 9,
    "left_ankle": 9,
    "head_pitch1": 10,
    "head_pitch2": 11,
    "head_yaw": 12,
}

robot = placo.RobotWrapper("../../robots/bdx/robot.urdf", placo.Flags.ignore_collisions)
solver = placo.KinematicsSolver(robot)
walk_engine = WalkEngine(
    robot,
    solver,
    # step_size_x=0.0,
    step_size_x=0.02,
    step_size_y=0.0,
    swing_gain=0.04,
    # step_size_yaw=np.deg2rad(10),
    rise_gain=0.02,
    foot_y_offset=0.0,
    frequency=4,
    trunk_pitch=-5,
)

t = 0
# model.opt.gravity[:] = [0, 0, 0]
walk_engine.new_step()
prev = data.time
t = 0
while True:
    dt = data.time - prev
    t += dt
    # if not viewer.is_alive:
    #     break

    # walk_engine.step_duration
    if t > walk_engine.step_duration:
        t = 0
        walk_engine.new_step()

    walk_engine.update(t)
    angles = walk_engine.compute_angles()
    print(angles["head_pitch1"], angles["head_pitch2"])
    robot.update_kinematics()
    solver.solve(True)

    data.ctrl[:] = list(angles.values())

    prev = data.time
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.005)

#     viewer.render()

viewer.close()
