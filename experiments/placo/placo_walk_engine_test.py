import argparse
import time

import numpy as np
import placo
from FramesViewer.viewer import Viewer
from placo_utils.visualization import footsteps_viz, robot_frame_viz, robot_viz

from mini_bdx.placo_walk_engine import PlacoWalkEngine
from mini_bdx.utils.xbox_controller import XboxController

fv = Viewer()
fv.start()

parser = argparse.ArgumentParser()
parser.add_argument("-x", action="store_true", default=False)
args = parser.parse_args()

if args.x:
    xbox = XboxController()

d_x = 0.05
d_y = 0
d_theta = 0.2


# TODO placo mistakes the antennas for leg joints ?
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


def get_clock_signal(t, period):
    a = np.sin(2 * np.pi * (t % period)) / period
    b = np.cos(2 * np.pi * (t % period)) / period
    return [a, b]


prev = time.time()
start = time.time()
while True:
    t = time.time()
    dt = t - prev
    if args.x:
        xbox_input()

    print(pwe.get_current_support_phase())

    fv.pushFrame(pwe.robot.get_T_world_left(), "left")
    fv.pushFrame(pwe.robot.get_T_world_right(), "right")

    footsteps = pwe.get_footsteps_in_robot_frame()
    for i, footstep in enumerate(footsteps):
        fv.pushFrame(footstep, "footstep" + str(i))

    # print(get_clock_signal(t, pwe.period))
    # footsteps_viz(pwe.trajectory.get_supports())
    viz.display(pwe.robot.state.q)

    pwe.tick(dt)
    time.sleep(0.01)
    prev = t
