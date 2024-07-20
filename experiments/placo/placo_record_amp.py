from os.path import join
import json
from threading import current_thread
from scipy.spatial.transform import Rotation as R
import time
from mini_bdx.utils.rl_utils import mujoco_to_isaac
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
parser.add_argument("-m", "--meshcat_viz", action="store_true", default=False)
args = parser.parse_args()

FPS = 60
MESHCAT_FPS = 20
DISPLAY_MESHCAT = args.meshcat_viz

# [root position, root orientation, joint poses (e.g. rotations)]
# [x, y, z, qw, qx, qy, qz, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15]


episode = {
    "LoopMode": "Wrap",
    "FrameDuration": np.around(1 / FPS, 4),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": False,
    "Debug_info": [],
    "Frames": [],
}


pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf", ignore_feet_contact=True)

pwe.set_traj(args.dx, args.dy, args.dtheta + 0.001)
if DISPLAY_MESHCAT:
    viz = robot_viz(pwe.robot)
DT = 0.001
start = time.time()

last_record = 0
last_meshcat_display = 0
while True:
    # print("t", pwe.t)
    pwe.tick(DT)
    if pwe.t <= 0:
        # print("waiting ")
        start = pwe.t
        last_record = pwe.t + 1 / FPS
        last_meshcat_display = pwe.t + 1 / MESHCAT_FPS
        continue

    # print(np.around(pwe.robot.get_T_world_fbase()[:3, 3], 3))

    if pwe.t - last_record >= 1 / FPS:
        T_world_fbase = pwe.robot.get_T_world_fbase()
        root_position = list(T_world_fbase[:3, 3])
        root_orientation_quat = list(R.from_matrix(T_world_fbase[:3, :3]).as_quat())
        joints_positions = mujoco_to_isaac(list(pwe.get_angles().values()))

        episode["Frames"].append(
            root_position + root_orientation_quat + joints_positions
        )

        left_foot_pose = pwe.robot.get_T_world_left()
        right_foot_pose = pwe.robot.get_T_world_right()
        episode["Debug_info"].append(
            {
                "left_foot_pose": list(left_foot_pose.flatten()),
                "right_foot_pose": list(right_foot_pose.flatten()),
            }
        )
        last_record = pwe.t
        print("saved frame")

    if DISPLAY_MESHCAT and pwe.t - last_meshcat_display >= 1 / MESHCAT_FPS:
        last_meshcat_display = pwe.t
        viz.display(pwe.robot.state.q)

    if pwe.t - start > args.length:
        break

print("recorded", len(episode["Frames"]), "frames")
file_name = args.name + str(".txt")
print("DONE, saving", file_name)
with open(file_name, "w") as f:
    json.dump(episode, f)
