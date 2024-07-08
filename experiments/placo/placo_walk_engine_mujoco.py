import argparse

import mujoco
import mujoco.viewer
import numpy as np

from mini_bdx.placo_walk_engine import PlacoWalkEngine
from mini_bdx.utils.mujoco_utils import check_contact
from mini_bdx.utils.xbox_controller import XboxController

parser = argparse.ArgumentParser()
parser.add_argument("-x", action="store_true", default=False)
args = parser.parse_args()

if args.x:
    xbox = XboxController()

d_x = 0
d_y = 0
d_theta = 0


# TODO placo mistakes the antennas for leg joints ?
pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf")


def xbox_input():
    global d_x, d_y, d_theta
    inputs = xbox.read()

    d_x = -inputs["l_y"] * pwe.parameters.walk_max_dx_forward / 2
    d_y = -inputs["l_x"] * pwe.parameters.walk_max_dy / 3
    d_theta = -inputs["r_x"] * pwe.parameters.walk_max_dtheta / 3

    print(d_x, d_y, d_theta)


def key_callback(keycode):
    global d_x, d_y, d_theta
    if keycode == 265:  # up
        d_x = 0.05
    #     max_target_step_size_x += 0.005
    # if keycode == 264:  # down
    #     max_target_step_size_x -= 0.005
    # if keycode == 263:  # left
    #     max_target_step_size_y -= 0.005
    # if keycode == 262:  # right
    #     max_target_step_size_y += 0.005
    # if keycode == 81:  # a
    #     max_target_yaw += np.deg2rad(1)
    # if keycode == 69:  # e


model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)


def get_feet_contact():
    right_contact = check_contact(data, model, "foot_module", "floor")
    left_contact = check_contact(data, model, "foot_module_2", "floor")
    return right_contact, left_contact


speed = 4  # 1 is slowest, 3 looks real time on my machine
prev = data.time
while True:
    t = data.time
    dt = t - prev

    if args.x:
        xbox_input()

    pwe.d_x = d_x
    pwe.d_y = d_y
    pwe.d_theta = d_theta
    right_contact, left_contact = get_feet_contact()
    pwe.tick(dt, left_contact, right_contact)

    angles = pwe.get_angles()
    data.ctrl[:] = list(angles.values())

    mujoco.mj_step(model, data, speed)  # 4 seems good
    viewer.sync()
    prev = t
