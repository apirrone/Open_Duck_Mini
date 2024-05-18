import argparse
import time

import mujoco_viewer
import numpy as np

import mujoco

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p-", "--path", type=str, required=True, help="Path to the xml file"
)
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(args.path)
data = mujoco.MjData(model)


# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

dofs = {
    "base_x": 0,
    "base_y": 1,
    "base_z": 2,
    "base_q0": 3,
    "base_q1": 4,
    "base_q2": 5,
    "base_q3": 6,
    "right_hip_yaw": 7,
    "right_hip_roll": 8,
    "right_hip_pitch": 9,
    "right_knee_pitch": 10,
    "right_ankle_pitch": 11,
    "left_hip_yaw": 12,
    "left_hip_roll": 13,
    "left_hip_pitch": 14,
    "left_knee_pitch": 15,
    "left_ankle_pitch": 16,
}

init = {
    "right_hip_yaw": 0,
    "right_hip_roll": 0,
    "right_hip_pitch": np.deg2rad(50),
    "right_knee_pitch": np.deg2rad(-90),
    "right_ankle_pitch": np.deg2rad(40),
    "left_hip_yaw": 0,
    "left_hip_roll": 0,
    "left_hip_pitch": np.deg2rad(50),
    "left_knee_pitch": np.deg2rad(-90),
    "left_ankle_pitch": np.deg2rad(40),
}


def goto_init():
    for key, value in init.items():
        data.qpos[dofs[key]] = value


target = [0.5, 0.5, 0.1]
# model.opt.gravity[:] = [0, 0, 0]
# simulate and render
while True:
    if viewer.is_alive:
        # print(model.nq)
        # data.qpos[0] = 0.2 * np.sin(0.5 * np.pi * time.time())
        # data.qpos[-7:-4] = 0.2 * np.sin(0.5 * np.pi * time.time())
        # data.get_body_com("base")
        print(data.body("base").xpos[2])
        # box_x = data.qpos[-7]
        # box_y = data.qpos[-6]
        # box_z = data.qpos[-5]
        # data.qepos[-7:-4] = data.body("base").xpos
        # data.qpos[-7:-4] = target
        # print(data.body("goal").xpos)

        # print(len(data.qpos[-7:-4]))
        # print(len(data.qpos))
        # goto_init()
        # data.qpos[7:] = 0
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()
