import argparse
import time

import numpy as np
import placo

from placo_utils.visualization import footsteps_viz, robot_frame_viz, robot_viz

from mini_bdx.placo_walk_engine import PlacoWalkEngine
from mini_bdx.utils.xbox_controller import XboxController

parser = argparse.ArgumentParser()
parser.add_argument("-x", action="store_true", default=False)
args = parser.parse_args()

if args.x:
    xbox = XboxController()

d_x = 0.05
d_y = 0
d_theta = 0.2


pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf")

pwe.set_traj(d_x, d_y, d_theta)
viz = robot_viz(pwe.robot)


def xbox_input():
    global d_x, d_y, d_theta
    inputs = xbox.read()

    d_x = -inputs["l_y"] * pwe.parameters.walk_max_dx_forward / 2
    d_y = -inputs["l_x"] * pwe.parameters.walk_max_dy / 3
    d_theta = -inputs["r_x"] * pwe.parameters.walk_max_dtheta / 3

    print(d_x, d_y, d_theta)


prev = time.time()
start = time.time()
while True:
    t = time.time()
    dt = t - prev
    if args.x:
        xbox_input()

    viz.display(pwe.robot.state.q)

    pwe.tick(dt)
    prev = t
