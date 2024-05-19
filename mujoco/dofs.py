import argparse
import time

import mujoco_viewer
import numpy as np
from FramesViewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

import mujoco

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p-", "--path", type=str, required=True, help="Path to the xml file"
)
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(args.path)
data = mujoco.MjData(model)


# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data, mode="window", width=800, height=600)
# fv = Viewer()
# fv.start()
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
        # data.qpos[2] = 0.22 + 0.2 * np.sin(0.5 * np.pi * time.time())

        print(data.qvel[8 : 8 + 10])

        # rot = np.array(data.body("base").xmat).reshape(3, 3)
        # Z_vec = rot[:, 2]
        # T = np.eye(4)
        # T[:3, :3] = rot
        # fv.pushFrame(T, "aze")
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()
