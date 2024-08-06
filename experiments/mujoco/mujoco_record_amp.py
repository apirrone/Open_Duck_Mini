import argparse
import json
import os
import time
from glob import glob

import cv2
import FramesViewer.utils as fv_utils
import mujoco
import mujoco_viewer
import numpy as np
from imitation.data.types import Trajectory
from scipy.spatial.transform import Rotation as R

from mini_bdx.placo_walk_engine import PlacoWalkEngine
from mini_bdx.utils.mujoco_utils import check_contact
from mini_bdx.utils.rl_utils import mujoco_to_isaac

pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf")

EPISODE_LENGTH = 10
NB_EPISODES_TO_RECORD = 1
FPS = 60

# For IsaacGymEnvs
# [root position, root orientation, joint poses (e.g. rotations)]
# [x, y, z, qw, qx, qy, qz, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hardware", action="store_true", help="use AMP_for_hardware format"
)
args = parser.parse_args()

episodes = []

current_episode = {
    "LoopMode": "Wrap",
    "FrameDuration": np.around(1 / FPS, 4),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": False,
    "Frames": [],
}

model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
model.opt.timestep = 0.001
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)


def get_feet_contact():
    right_contact = check_contact(data, model, "foot_module", "floor")
    left_contact = check_contact(data, model, "foot_module_2", "floor")
    return right_contact, left_contact


mujoco_init_pos = np.array(
    [
        # right_leg
        -0.014,
        0.08,
        0.53,
        -1.62,
        0.91,
        # left leg
        0.013,
        0.077,
        0.59,
        -1.63,
        0.86,
        # head
        -0.17,
        -0.17,
        0.0,
        0.0,
        0.0,
    ]
)

data.qpos[3 : 3 + 4] = [1, 0, 0.05, 0]
data.qpos[7 : 7 + 15] = mujoco_init_pos
data.ctrl[:] = mujoco_init_pos
x_qvels = []
while True:
    if len(episodes) >= NB_EPISODES_TO_RECORD:
        print("DONE, RECORDED", NB_EPISODES_TO_RECORD, "EPISODES")
        break
    print("Starting episode")
    done = False
    prev = time.time()
    start = time.time()
    last_record = time.time()
    pwe.set_traj(0.0, 0.0, 0.001)
    while not done:
        t = time.time()
        dt = t - prev
        x_qvels.append(data.qvel[0].copy())
        print(np.around(np.mean(x_qvels[-30:]), 3))
        # qpos = env.data.qpos[:3].copy()
        # qpos[2] = 0.15
        # env.data.qpos[:3] = qpos
        right_contact, left_contact = get_feet_contact()

        pwe.tick(dt, left_contact, right_contact)
        # if pwe.t < 0:  # for stand
        angles = pwe.get_angles()
        action = list(angles.values())
        action = np.array(action)
        data.ctrl[:] = action
        mujoco.mj_step(model, data, 7)  # 4 seems good

        if pwe.t <= 0:
            start = time.time()
            print("waiting ...")
            prev = t
            viewer.render()
            continue

        if t - last_record >= 1 / FPS:
            root_position = list(np.around(data.body("base").xpos, 3))
            root_orientation = list(np.around(data.body("base").xquat, 3))  # w, x, y, z

            # convert to x, y, z, w
            root_orientation = [
                root_orientation[1],
                root_orientation[2],
                root_orientation[3],
                root_orientation[0],
            ]

            joints_positions = list(np.around(data.qpos[7 : 7 + 15], 3))

            joints_positions = mujoco_to_isaac(joints_positions)

            current_episode["Frames"].append(
                root_position + root_orientation + joints_positions
            )
            last_record = time.time()

        if time.time() - start > EPISODE_LENGTH * 2:
            print("Episode done")
            print(len(current_episode["Frames"]))
            episodes.append(current_episode)

            # save json as bdx_walk.txt
            with open("bdx_walk.txt", "w") as f:
                json.dump(current_episode, f)

            current_episode = {
                "LoopMode": "Wrap",
                "FrameDuration": 0.01667,
                "EnableCycleOffsetPosition": True,
                "EnableCycleOffsetRotation": False,
                "Frames": [],
            }
            done = True
        viewer.render()
        prev = t
