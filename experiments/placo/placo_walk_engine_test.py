import argparse
import time

import numpy as np
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz

from mini_bdx.placo_walk_engine import PlacoWalkEngine
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

viz = robot_viz(pwe.robot)


def xbox_input():
    global d_x, d_y, d_theta
    inputs = xbox.read()

    d_x = -inputs["l_y"] * pwe.parameters.walk_max_dx_forward / 2
    d_y = -inputs["l_x"] * pwe.parameters.walk_max_dy / 3
    d_theta = -inputs["r_x"] * pwe.parameters.walk_max_dtheta / 3

    print(d_x, d_y, d_theta)


prev = time.time()
while True:
    t = time.time()
    dt = t - prev
    if args.x:
        xbox_input()

    viz.display(pwe.robot.state.q)
    robot_frame_viz(pwe.robot, "left_foot")
    robot_frame_viz(pwe.robot, "right_foot")

    pwe.d_x = d_x
    pwe.d_y = d_y
    pwe.d_theta = d_theta
    pwe.tick(dt)

    prev = t
