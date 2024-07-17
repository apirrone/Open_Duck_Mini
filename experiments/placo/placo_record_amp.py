import time
import argparse

import numpy as np
import placo

from placo_utils.visualization import footsteps_viz, robot_frame_viz, robot_viz


from mini_bdx.placo_walk_engine import PlacoWalkEngine

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, required=True)
parser.add_argument("--dx", type=float, required=True)
parser.add_argument("--dy", type=float, required=True)
parser.add_argument("--dtheta", type=float, required=True)
parser.add_argument("-l", "--length", type=int, default=10)
args = parser.parse_args()

FPS = 60

# [root position, root orientation, joint poses (e.g. rotations)]
# [x, y, z, qw, qx, qy, qz, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15]


episode = {
    "LoopMode": "Wrap",
    "FrameDuration": np.around(1 / FPS, 4),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": False,
    "Frames": [],
}


pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf", ignore_feet_contact=True)

pwe.set_traj(args.dx, args.dy, args.dtheta)
viz = robot_viz(pwe.robot)
DT = 1 / 60
prev = time.time()
while True:
    viz.display(pwe.robot.state.q)

    pwe.tick(DT)
    print(np.around(pwe.robot.get_T_world_fbase()[:3, 3], 3))
