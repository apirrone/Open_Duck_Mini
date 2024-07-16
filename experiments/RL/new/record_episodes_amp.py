import argparse
from mini_bdx.utils.mujoco_utils import check_contact
import mujoco_viewer
import time
from mini_bdx.utils.rl_utils import mujoco_to_isaac
import json
from imitation.data.types import Trajectory
from scipy.spatial.transform import Rotation as R
import os
from glob import glob

import cv2
import FramesViewer.utils as fv_utils
import mujoco
import numpy as np
from mini_bdx.placo_walk_engine import PlacoWalkEngine

pwe = PlacoWalkEngine("../../../mini_bdx/robots/bdx/robot.urdf")

EPISODE_LENGTH = 10
NB_EPISODES_TO_RECORD = 1
FPS = 60

# [root position, root orientation, joint poses (e.g. rotations)]
# [x, y, z, qw, qx, qy, qz, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15]


episodes = []

current_episode = {
    "LoopMode": "Wrap",
    "FrameDuration": np.around(1 / FPS, 4),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": False,
    "Frames": [],
}

model = mujoco.MjModel.from_xml_path("../../../mini_bdx/robots/bdx/scene.xml")
model.opt.timestep = 0.001
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)


def get_feet_contact():
    right_contact = check_contact(data, model, "foot_module", "floor")
    left_contact = check_contact(data, model, "foot_module_2", "floor")
    return right_contact, left_contact


while True:
    if len(episodes) >= NB_EPISODES_TO_RECORD:
        print("DONE, RECORDED", NB_EPISODES_TO_RECORD, "EPISODES")
        break
    print("Starting episode")
    done = False
    prev = time.time()
    start = time.time()
    last_record = time.time()
    pwe.set_traj(0.02, 0.0, 0.001)
    while not done:
        t = time.time()
        dt = t - prev

        # qpos = env.data.qpos[:3].copy()
        # qpos[2] = 0.15
        # env.data.qpos[:3] = qpos
        # if pwe.t <= 0: # for stand
        right_contact, left_contact = get_feet_contact()
        pwe.tick(dt, left_contact, right_contact)
        angles = pwe.get_angles()
        action = list(angles.values())
        action = np.array(action)
        data.ctrl[:] = action
        mujoco.mj_step(model, data, 7)  # 4 seems good

        if pwe.t <= 0:
            start = time.time()
            print("waiting ...")
            prev = t
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
