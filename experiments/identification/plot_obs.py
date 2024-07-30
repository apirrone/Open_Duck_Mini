import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("--mujoco_obs", type=str, required=True)
parser.add_argument("--robot_obs", type=str, required=True)
args = parser.parse_args()

mujoco_obs = pickle.load(open(args.mujoco_obs, "rb"))
robot_obs = pickle.load(open(args.robot_obs, "rb"))


mujoco_channels = []
robot_channels = []

# # convert quat to euler for easier reading by a simple human
# for i in range(min(len(mujoco_obs), len(robot_obs))):
#     mujoco_quat = mujoco_obs[i][:4]
#     mujoco_euler = R.from_quat(mujoco_quat).as_euler("xyz")

#     robot_quat = robot_obs[i][:4]
#     robot_euler = R.from_quat(robot_quat).as_euler("xyz")

#     mujoco_obs[i] = mujoco_obs[i][1:]
#     robot_obs[i] = robot_obs[i][1:]

#     mujoco_obs[i][:3] = mujoco_euler
#     robot_obs[i][:3] = robot_euler

nb_channels = len(mujoco_obs[0])

for i in range(nb_channels):
    mujoco_channels.append([obs[i] for obs in mujoco_obs])
    robot_channels.append([obs[i] for obs in robot_obs])

channels = [
    "base_roll",
    "base_pitch",
    "base_yaw",
    "base_quat[0]",
    "base_quat[1]",
    "base_quat[2]",
    "base_quat[3]",
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

# one sub plot per channel, robot vs mujoco
# arrange as an array of sqrt(nb_channels) x sqrt(nb_channels)

nb_rows = int(np.sqrt(nb_channels))
nb_cols = int(np.ceil(nb_channels / nb_rows))

fig, axs = plt.subplots(nb_rows, nb_cols, sharex=True, sharey=True)
for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_channels:
            break
        axs[i, j].plot(mujoco_channels[i * nb_cols + j], label="mujoco")
        axs[i, j].plot(robot_channels[i * nb_cols + j], label="robot")
        axs[i, j].legend()
        axs[i, j].set_title(f"{channels[i * nb_cols + j]}")


fig.suptitle("Mujoco vs Robot")
# tight layout
# plt.tight_layout()
plt.show()
