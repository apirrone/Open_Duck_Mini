import argparse
import time

import cv2
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
viewer = mujoco_viewer.MujocoViewer(
    model, data, mode="window", width=1280, height=800, hide_menus=True
)

qpos_dofs = {
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
    "head_pitch1": 17,
    "head_pitch2": 18,
    "head_yaw": 19,
}

ctrl_dofs = {
    "right_hip_yaw": 0,
    "right_hip_roll": 1,
    "right_hip_pitch": 2,
    "right_knee_pitch": 3,
    "right_ankle_pitch": 4,
    "left_hip_yaw": 5,
    "left_hip_roll": 6,
    "left_hip_pitch": 7,
    "left_knee_pitch": 8,
    "left_ankle_pitch": 9,
    "head_pitch1": 10,
    "head_pitch2": 11,
    "head_yaw": 12,
}

init = {
    "right_hip_yaw": 0,
    "right_hip_roll": 0,
    "right_hip_pitch": np.deg2rad(45),
    "right_knee_pitch": np.deg2rad(-90),
    "right_ankle_pitch": np.deg2rad(45),
    "left_hip_yaw": 0,
    "left_hip_roll": 0,
    "left_hip_pitch": np.deg2rad(45),
    "left_knee_pitch": np.deg2rad(-90),
    "left_ankle_pitch": np.deg2rad(45),
    "head_pitch1": np.deg2rad(-45),
    "head_pitch2": np.deg2rad(-45),
    "head_yaw": 0,
}


def goto_init():
    data.qpos[2] = 0.3
    i = 0
    for key, value in init.items():
        data.ctrl[i] = value
        i += 1


model.opt.gravity[:] = [0, 0, 0]
goto_init()


class WalkEngine:
    def __init__(self, freq=2):
        self.freq = freq
        pass

    def step(self, t):
        s1 = np.sin(2 * np.pi * self.freq * t)
        s2 = np.sin(2 * np.pi * self.freq * t + np.pi)

        data.ctrl[ctrl_dofs["right_hip_pitch"]] = init["right_hip_pitch"] + s1
        data.ctrl[ctrl_dofs["right_knee_pitch"]] = init["right_knee_pitch"] - s1

        data.ctrl[ctrl_dofs["left_hip_pitch"]] = init["left_hip_pitch"] + s2
        data.ctrl[ctrl_dofs["left_knee_pitch"]] = init["left_knee_pitch"] - s2


WE = WalkEngine()

im = np.zeros((800, 800, 3), dtype=np.uint8)
while True:
    if viewer.is_alive:

        WE.step(data.time)

        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()
