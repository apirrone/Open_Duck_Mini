import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
from utils import dof_to_id, id_to_dof, mujoco_init_pos

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, required=True)
args = parser.parse_args()

data = pickle.load(open(args.data, "rb"))

config = data["config"]
mujoco = data["mujoco"]
robot = data["robot"]

command_mujoco = []
value_mujoco = []

command_robot = []
value_robot = []


max_y_mujoco = max(np.array(mujoco).flatten())
min_y_mujoco = min(np.array(mujoco).flatten())

max_y_robot = max(np.array(robot).flatten())
min_y_robot = min(np.array(robot).flatten())

max_y = max(max_y_mujoco, max_y_robot)
min_y = min(min_y_mujoco, min_y_robot)

for i in range(len(mujoco)):
    command_mujoco.append(mujoco[i][0])
    value_mujoco.append(mujoco[i][1])

for i in range(len(robot)):
    command_robot.append(robot[i][0])
    value_robot.append(robot[i][1])


# two subplots
fig, axs = plt.subplots(1, 2)
axs[0].plot(command_mujoco, label="command")
axs[0].plot(value_mujoco, label="value")
axs[0].legend()
axs[0].set_title("Mujoco")
axs[0].set_ylim(min_y, max_y)

axs[1].plot(command_robot, label="command")
axs[1].plot(value_robot, label="value")
axs[1].legend()
axs[1].set_title("Robot")
axs[1].set_ylim(min_y, max_y)

plt.show()
