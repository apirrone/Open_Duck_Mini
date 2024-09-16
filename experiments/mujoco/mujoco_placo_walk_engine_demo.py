import argparse
import json
import os
import time
from glob import glob

import cv2
import FramesViewer.utils as fv_utils
import mujoco
import mujoco_viewer
import pygame
import numpy as np
from imitation.data.types import Trajectory
from scipy.spatial.transform import Rotation as R

from mini_bdx.placo_walk_engine import PlacoWalkEngine
from mini_bdx.utils.mujoco_utils import check_contact
from mini_bdx.utils.rl_utils import action_to_pd_targets, mujoco_to_isaac
from mini_bdx.utils.xbox_controller import XboxController
from mini_bdx.utils.rl_utils import (
    isaac_to_mujoco,
)

parser = argparse.ArgumentParser()
parser.add_argument("-x", "--xbox_controller", action="store_true")
parser.add_argument("-k", "--keyboard", action="store_true")
args = parser.parse_args()

if args.xbox_controller:
    xbox = XboxController()


if args.keyboard:
    pygame.init()
    # open a blank pygame window
    screen = pygame.display.set_mode((100, 100))
    pygame.display.set_caption("Press arrow keys to move robot")

pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf")


model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
model.opt.timestep = 0.0001
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)


def get_feet_contact():
    right_contact = check_contact(data, model, "foot_module", "floor")
    left_contact = check_contact(data, model, "foot_module_2", "floor")
    return right_contact, left_contact


# lower com 0.16
isaac_init_pos = np.array(
    [
        -0.03455234018541292,
        0.055730747490168285,
        0.5397158397618105,
        -1.3152788306721914,
        0.6888361815639528,
        -0.1745314896173976,
        -0.17453429522668937,
        0,
        0,
        0,
        -0.03646051060835733,
        -0.03358034284950263,
        0.5216150220237578,
        -1.326235199315616,
        0.7179857110436013,
    ]
)

mujoco_init_pos = np.array(isaac_to_mujoco(isaac_init_pos))


def xbox_input():
    inputs = xbox.read()
    step_size_x = -inputs["l_y"] * 0.02
    step_size_y = -inputs["l_x"] * 0.02
    yaw = -inputs["r_x"] * 0.2 + 0.001

    return step_size_x, step_size_y, yaw


def keyboard_input():
    keys = pygame.key.get_pressed()
    step_size_x = 0
    step_size_y = 0
    yaw = 0
    if keys[pygame.K_z]:
        step_size_x = 0.03
    if keys[pygame.K_s]:
        step_size_x = -0.03
    if keys[pygame.K_q]:
        yaw = 0.2
    if keys[pygame.K_d]:
        yaw = -0.2
    pygame.event.pump()  # process event queue

    return step_size_x, step_size_y, yaw


data.qpos[3 : 3 + 4] = [1, 0, 0.05, 0]
data.qpos[7 : 7 + 15] = mujoco_init_pos
data.ctrl[:] = mujoco_init_pos
prev = time.time()
while True:
    t = time.time()

    if args.xbox_controller:
        step_size_x, step_size_y, yaw = xbox_input()
    elif args.keyboard:
        step_size_x, step_size_y, yaw = keyboard_input()
    else:
        step_size_x, step_size_y, yaw = 0.02, 0, 0.001

    pwe.set_traj(step_size_x, step_size_y, yaw)

    right_contact, left_contact = get_feet_contact()

    pwe.tick(t - prev, left_contact, right_contact)
    # if pwe.t < 0:  # for stand
    angles = pwe.get_angles()
    action = isaac_to_mujoco(list(angles.values()))
    action = np.array(action)
    data.ctrl[:] = action
    mujoco.mj_step(model, data, 50)  # 4 seems good

    # if pwe.t <= 0:
    #     print("waiting ...")
    #     viewer.render()
    #     continue
    prev = t
    viewer.render()
