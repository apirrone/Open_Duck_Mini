import argparse
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

pwe = PlacoWalkEngine(
    "../../../mini_bdx/robots/bdx/robot.urdf", ignore_feet_contact=True
)

EPISODE_LENGTH = 60
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
        pwe.tick(dt)
        angles = pwe.get_angles()
        action = list(angles.values())
        action = np.array(action)
        data.ctrl[:] = action
        mujoco.mj_step(model, data, 10)  # 4 seems good

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

            # joints_positions = [
            #     joints_positions[0],
            #     joints_positions[1],
            #     joints_positions[2],
            #     joints_positions[3],
            #     joints_positions[4],
            #     joints_positions[10],
            #     joints_positions[11],
            #     joints_positions[12],
            #     joints_positions[13],
            #     joints_positions[14],
            #     joints_positions[5],
            #     joints_positions[6],
            #     joints_positions[7],
            #     joints_positions[8],
            #     joints_positions[9],
            # ]
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
