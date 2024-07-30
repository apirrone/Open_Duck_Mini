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
from mini_bdx.utils.rl_utils import action_to_pd_targets, mujoco_to_isaac
from mini_bdx.utils.xbox_controller import XboxController

parser = argparse.ArgumentParser()
parser.add_argument("-x", "--xbox_controller", action="store_true")
args = parser.parse_args()

if args.xbox_controller:
    xbox = XboxController()

pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf")


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


def xbox_input():
    inputs = xbox.read()
    step_size_x = -inputs["l_y"] * 0.02
    step_size_y = -inputs["l_x"] * 0.02
    yaw = -inputs["r_x"] * 0.2 + 0.001

    return step_size_x, step_size_y, yaw


data.qpos[3 : 3 + 4] = [1, 0, 0.05, 0]
data.qpos[7 : 7 + 15] = mujoco_init_pos
data.ctrl[:] = mujoco_init_pos
prev = time.time()
while True:
    t = time.time()

    if args.xbox_controller:
        step_size_x, step_size_y, yaw = xbox_input()
    else:
        step_size_x, step_size_y, yaw = 0.02, 0, 0.001

    pwe.set_traj(step_size_x, step_size_y, yaw)

    dt = t - prev

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

    viewer.render()
    prev = t
