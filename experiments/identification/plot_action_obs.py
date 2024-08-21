import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

isaac_joints_order = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "left_antenna",
    "right_antenna",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

parser = argparse.ArgumentParser()
parser.add_argument("--robot_obs", type=str, required=True)
parser.add_argument("--hardware", action="store_true")
args = parser.parse_args()

robot_obs = pickle.load(open(args.robot_obs, "rb"))

robot_channels = []


# convert quat to euler for easier reading by a simple human
if not args.hardware:
    for i in range(len(robot_obs)):
        robot_quat = robot_obs[i][:4]
        robot_euler = R.from_quat(robot_quat).as_euler("xyz")

        robot_obs[i] = robot_obs[i][1:]

        robot_obs[i][:3] = robot_euler


if not args.hardware:
    channels = [
        "base_roll",
        "base_pitch",
        "base_yaw",
        "base_ang_vel[0]",
        "base_ang_vel[1]",
        "base_ang_vel[2]",
        "dof_pos[0]",
        "dof_pos[1]",
        "dof_pos[2]",
        "dof_pos[3]",
        "dof_pos[4]",
        "dof_pos[5]",
        "dof_pos[6]",
        "dof_pos[7]",
        "dof_pos[8]",
        "dof_pos[9]",
        "dof_pos[10]",
        "dof_pos[11]",
        "dof_pos[12]",
        "dof_pos[13]",
        "dof_pos[14]",
        "dof_vel[0]",
        "dof_vel[1]",
        "dof_vel[2]",
        "dof_vel[3]",
        "dof_vel[4]",
        "dof_vel[5]",
        "dof_vel[6]",
        "dof_vel[7]",
        "dof_vel[8]",
        "dof_vel[9]",
        "dof_vel[10]",
        "dof_vel[11]",
        "dof_vel[12]",
        "dof_vel[13]",
        "dof_vel[14]",
        "prev_action[0]",
        "prev_action[1]",
        "prev_action[2]",
        "prev_action[3]",
        "prev_action[4]",
        "prev_action[5]",
        "prev_action[6]",
        "prev_action[7]",
        "prev_action[8]",
        "prev_action[9]",
        "prev_action[10]",
        "prev_action[11]",
        "prev_action[12]",
        "prev_action[13]",
        "prev_action[14]",
        "commands[0]",
        "commands[1]",
        "commands[2]",
    ]
else:
    channels = [
        "base_lin_vel[0]",
        "base_lin_vel[1]",
        "base_lin_vel[2]",
        "base_ang_vel[0]",
        "base_ang_vel[1]",
        "base_ang_vel[2]",
        "projected_gravity[0]",
        "projected_gravity[1]",
        "projected_gravity[2]",
        "commands[0]",
        "commands[1]",
        "commands[2]",
        "dof_pos[0]",
        "dof_pos[1]",
        "dof_pos[2]",
        "dof_pos[3]",
        "dof_pos[4]",
        "dof_pos[5]",
        "dof_pos[6]",
        "dof_pos[7]",
        "dof_pos[8]",
        "dof_pos[9]",
        "dof_pos[10]",
        "dof_pos[11]",
        "dof_pos[12]",
        "dof_pos[13]",
        "dof_pos[14]",
        "dof_vel[0]",
        "dof_vel[1]",
        "dof_vel[2]",
        "dof_vel[3]",
        "dof_vel[4]",
        "dof_vel[5]",
        "dof_vel[6]",
        "dof_vel[7]",
        "dof_vel[8]",
        "dof_vel[9]",
        "dof_vel[10]",
        "dof_vel[11]",
        "dof_vel[12]",
        "dof_vel[13]",
        "dof_vel[14]",
        "actions[0]",
        "actions[1]",
        "actions[2]",
        "actions[3]",
        "actions[4]",
        "actions[5]",
        "actions[6]",
        "actions[7]",
        "actions[8]",
        "actions[9]",
        "actions[10]",
        "actions[11]",
        "actions[12]",
        "actions[13]",
        "actions[14]",
    ]

# base_lin_vel * self.obs_scales.lin_vel,
# base_ang_vel * self.obs_scales.ang_vel,
# self.projected_gravity,
# self.commands[:, :3] * self.commands_scale,
# (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
# self.dof_vel * self.obs_scales.dof_vel,
# self.actions,

nb_channels = len(robot_obs[0])
dof_poses = []
prev_actions = []

# select dof_pos and prev_action
for i in range(nb_channels):
    if "dof_pos" in channels[i]:
        dof_poses.append([obs[i] for obs in robot_obs])
    elif "action" in channels[i]:
        prev_actions.append([obs[i] for obs in robot_obs])

# print(len(dof_poses))
# print(len(prev_actions))
# exit()

# plot prev action vs dof pos

nb_dofs = len(dof_poses)
nb_rows = int(np.sqrt(nb_dofs))
nb_cols = int(np.ceil(nb_dofs / nb_rows))

fig, axs = plt.subplots(nb_rows, nb_cols, sharex=True, sharey=True)
for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_dofs:
            break
        axs[i, j].plot(prev_actions[i * nb_cols + j], label="command")
        axs[i, j].plot(dof_poses[i * nb_cols + j], label="value")
        axs[i, j].legend()
        axs[i, j].set_title(f"{isaac_joints_order[i * nb_cols + j]}")

plt.show()
