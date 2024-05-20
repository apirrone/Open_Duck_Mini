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
viewer = mujoco_viewer.MujocoViewer(
    model, data, mode="window", width=1280, height=800, hide_menus=True
)
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
    i = 0
    for key, value in init.items():
        data.ctrl[i] = value
        i += 1


def check_contact(body1_name, body2_name):
    # Get the body IDs
    body1_id = data.body(body1_name)
    body2_id = data.body(body2_name)

    # Iterate through the contacts
    for i in range(data.ncon):
        contact = data.contact[i]

        # Check if the contact is between the two bodies
        if (contact.geom1 == body1_id and contact.geom2 == body2_id) or (
            contact.geom1 == body2_id and contact.geom2 == body1_id
        ):
            return True

    return False


print(data.geom("floor"))
exit()

target = [0.5, 0.5, 0.1]
# model.opt.gravity[:] = [0, 0, 0]

target = [0, 1, 0.1]
goto_init()
while True:
    if viewer.is_alive:
        # goto_init()
        # print(model.nq)
        # data.qpos[2] = 0.22 + 0.2 * np.sin(0.5 * np.pi * time.time())
        # print(np.square(data.body("base").xpos[2] - 0.12) * 100)
        # print(data.qvel[8 : 8 + 10])
        # print(data.body("foot_module"))
        # print(check_contact("base", "floor"))
        # print(data.body("base").cvel[3:])

        # rot = np.array(data.body("base").xmat).reshape(3, 3)
        # Z_vec = rot[:, 2]
        # T = np.eye(4)
        # T[:3, :3] = rot
        # fv.pushFrame(T, "aze")

        # print(len(data.ctrl), data.ctrl)
        # data.ctrl[4] = (np.pi / 4) * (np.sin(2 * np.pi * 5 * data.time) + 1) - np.pi / 4
        # data.ctrl[4] = np.pi / 2
        print(data.body("base").xpos[2])

        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()
