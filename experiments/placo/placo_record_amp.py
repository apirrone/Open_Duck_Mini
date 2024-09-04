import argparse
import json
import os
import time
from os.path import join
from threading import current_thread

import numpy as np
from numpy.ma.extras import average
import placo
from placo_utils.visualization import footsteps_viz, robot_frame_viz, robot_viz
from scipy.spatial.transform import Rotation as R

from mini_bdx.placo_walk_engine import PlacoWalkEngine
from mini_bdx.utils.rl_utils import mujoco_to_isaac, test

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, required=True)
parser.add_argument("-o", "--output_dir", type=str, default="recordings")
parser.add_argument("--dx", type=float, required=True)
parser.add_argument("--dy", type=float, required=True)
parser.add_argument("--dtheta", type=float, required=True)
parser.add_argument("-l", "--length", type=int, default=10)
parser.add_argument("-m", "--meshcat_viz", action="store_true", default=False)
parser.add_argument(
    "-s",
    "--skip_warmup",
    action="store_true",
    default=False,
    help="don't record warmup motion",
)
parser.add_argument(
    "--stand",
    action="store_true",
    default=False,
    help="hack to record a standing pose",
)
parser.add_argument(
    "--hardware",
    action="store_true",
    help="use AMP_for_hardware format. If false, use IsaacGymEnvs format",
)
args = parser.parse_args()

FPS = 60
MESHCAT_FPS = 20
DISPLAY_MESHCAT = args.meshcat_viz

# For IsaacGymEnvs
# [root position, root orientation, joint poses (e.g. rotations)]
# [x, y, z, qw, qx, qy, qz, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15]

# For amp for hardware
# [root position, root orientation, joint poses (e.g. rotations), target toe positions, linear velocity, angular velocity, joint velocities, target toe velocities]
# [x, y, z, qw, qx, qy, qz, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15, l_toe_x, l_toe_y, l_toe_z, r_toe_x, r_toe_y, r_toe_z, lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z, j1_vel, j2_vel, j3_vel, j4_vel, j5_vel, j6_vel, j7_vel, j8_vel, j9_vel, j10_vel, j11_vel, j12_vel, j13_vel, j14_vel, j15_vel, l_toe_vel_x, l_toe_vel_y, l_toe_vel_z, r_toe_vel_x, r_toe_vel_y, r_toe_vel_z]

episode = {
    "LoopMode": "Wrap",
    "FrameDuration": np.around(1 / FPS, 4),
    "EnableCycleOffsetPosition": True,
    "EnableCycleOffsetRotation": False,
    "Debug_info": [],
    "Frames": [],
    "MotionWeight": 1,
}


pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf", ignore_feet_contact=True)
first_joints_positions = list(pwe.get_angles().values())
first_T_world_fbase = pwe.robot.get_T_world_fbase()
first_T_world_leftFoot = pwe.robot.get_T_world_left()
first_T_world_rightFoot = pwe.robot.get_T_world_right()


# pwe.parameters.single_support_duration = 0.25  # slow
# pwe.parameters.single_support_duration = 0.20  # normal
pwe.parameters.single_support_duration = 0.2  # Fast ?

pwe.set_traj(args.dx, args.dy, args.dtheta + 0.001)
if DISPLAY_MESHCAT:
    viz = robot_viz(pwe.robot)
DT = 0.001
start = time.time()

last_record = 0
last_meshcat_display = 0
prev_root_position = [0, 0, 0]
prev_root_orientation_euler = [0, 0, 0]
prev_left_toe_pos = [0, 0, 0]
prev_right_toe_pos = [0, 0, 0]
prev_joints_positions = [0] * 15
i = 0
prev_initialized = False
avg_x_lin_vel = []
avg_yaw_vel = []
while True:
    # print("t", pwe.t)
    pwe.tick(DT)
    if pwe.t <= 0 + args.skip_warmup * 1:
        # print("waiting ")
        start = pwe.t
        last_record = pwe.t + 1 / FPS
        last_meshcat_display = pwe.t + 1 / MESHCAT_FPS
        continue

    # print(np.around(pwe.robot.get_T_world_fbase()[:3, 3], 3))

    if pwe.t - last_record >= 1 / FPS:
        if args.stand:
            T_world_fbase = first_T_world_fbase
        else:
            T_world_fbase = pwe.robot.get_T_world_fbase()
        root_position = list(T_world_fbase[:3, 3])
        root_orientation_quat = list(R.from_matrix(T_world_fbase[:3, :3]).as_quat())

        if args.stand:
            joints_positions = first_joints_positions
        else:
            joints_positions = list(pwe.get_angles().values())

        if args.stand:
            T_world_leftFoot = first_T_world_leftFoot
            T_world_rightFoot = first_T_world_rightFoot
        else:
            T_world_leftFoot = pwe.robot.get_T_world_left()
            T_world_rightFoot = pwe.robot.get_T_world_right()

        T_body_leftFoot = np.linalg.inv(T_world_fbase) @ T_world_leftFoot
        T_body_rightFoot = np.linalg.inv(T_world_fbase) @ T_world_rightFoot

        left_toe_pos = list(T_body_leftFoot[:3, 3])
        right_toe_pos = list(T_body_rightFoot[:3, 3])

        world_linear_vel = list(
            (np.array(root_position) - np.array(prev_root_position)) / (1 / FPS)
        )
        avg_x_lin_vel.append(world_linear_vel[0])
        body_rot_mat = T_world_fbase[:3, :3]
        body_linear_vel = list(body_rot_mat.T @ world_linear_vel)

        world_angular_vel = list(
            (
                R.from_quat(root_orientation_quat).as_euler("xyz")
                - prev_root_orientation_euler
            )
            / (1 / FPS)
        )
        avg_yaw_vel.append(world_angular_vel[2])
        body_angular_vel = list(body_rot_mat.T @ world_angular_vel)
        # print("world angular vel", world_angular_vel)
        # print("body angular vel", body_angular_vel)

        joints_vel = list(
            (np.array(joints_positions) - np.array(prev_joints_positions)) / (1 / FPS)
        )
        left_toe_vel = list(
            (np.array(left_toe_pos) - np.array(prev_left_toe_pos)) / (1 / FPS)
        )
        right_toe_vel = list(
            (np.array(right_toe_pos) - np.array(prev_right_toe_pos)) / (1 / FPS)
        )

        if prev_initialized:
            if args.hardware:
                episode["Frames"].append(
                    root_position
                    + root_orientation_quat
                    + joints_positions
                    + left_toe_pos
                    + right_toe_pos
                    + world_linear_vel
                    + world_angular_vel
                    + joints_vel
                    + left_toe_vel
                    + right_toe_vel
                )
            else:
                episode["Frames"].append(
                    root_position + root_orientation_quat + joints_positions
                )
            episode["Debug_info"].append(
                {
                    "left_foot_pose": list(T_world_leftFoot.flatten()),
                    "right_foot_pose": list(T_world_rightFoot.flatten()),
                }
            )

        prev_root_position = root_position.copy()
        prev_root_orientation_euler = (
            R.from_quat(root_orientation_quat).as_euler("xyz").copy()
        )
        prev_left_toe_pos = left_toe_pos.copy()
        prev_right_toe_pos = right_toe_pos.copy()
        prev_joints_positions = joints_positions.copy()
        prev_initialized = True

        print("avg x vel", np.mean(avg_x_lin_vel[-50:]))
        print("avg yaw vel", np.mean(avg_yaw_vel[-50:]))
        print("=")

        last_record = pwe.t
        # print("saved frame")

    if DISPLAY_MESHCAT and pwe.t - last_meshcat_display >= 1 / MESHCAT_FPS:
        last_meshcat_display = pwe.t
        viz.display(pwe.robot.state.q)
        footsteps_viz(pwe.trajectory.get_supports())

    if pwe.t - start > args.length:
        break

    i += 1

print("recorded", len(episode["Frames"]), "frames")
file_name = args.name + str(".txt")
file_path = os.path.join(args.output_dir, file_name)
os.makedirs(args.output_dir, exist_ok=True)
print("DONE, saving", file_name)
with open(file_path, "w") as f:
    json.dump(episode, f)
