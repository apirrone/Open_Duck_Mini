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
# options = mujoco.MjOption()
# options.gravity = [0, 0, 0]
# options.wind = [0, 0, 0]


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

# simulate and render
while True:
    if viewer.is_alive:
        # print(model.nq)
        # data.qpos[0] = 0.2 * np.sin(0.5 * np.pi * time.time())
        # data.qpos[8] = 0.01 * np.sin(0.5 * np.pi * time.time())
        # data.qpos[7:] = 0
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()
