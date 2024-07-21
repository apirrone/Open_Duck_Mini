import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, required=True)
args = parser.parse_args()
data = pickle.load(open(args.data, "rb"))
if "robot" in data and "mujoco" in data:
    res = None
    while res is None or res not in ["1", "2"]:
        res = input("plot robot (1) or mujoco (2) ? ")
    if res == "1":
        command_value = data["robot"]
        title = "Robot"
    else:
        command_value = data["mujoco"]
        title = "Mujoco"
elif "mujoco" in data:
    command_value = data["mujoco"]
    title = "Mujoco"
elif "robot" in data:
    command_value = data["robot"]
    title = "Robot"
else:
    print("NO DATA")
    exit()


dofs = {
    0: "right_hip_yaw",
    1: "right_hip_roll",
    2: "right_hip_pitch",
    3: "right_knee",
    4: "right_ankle",
    5: "left_hip_yaw",
    6: "left_hip_roll",
    7: "left_hip_pitch",
    8: "left_knee",
    9: "left_ankle",
    10: "neck_pitch",
    11: "head_pitch",
    12: "head_yaw",
    13: "left_antenna",
    14: "right_antenna",
}
# command_value = np.array(command_value)
fig, axs = plt.subplots(4, 4)
dof_id = 0
for i in range(4):
    for j in range(4):
        if i == 3 and j == 3:
            break
        print(4 * i + j)
        command = []
        value = []
        for k in range(len(command_value)):
            command.append(command_value[k][0][4 * i + j])
            value.append(command_value[k][1][4 * i + j])
        axs[i, j].plot(command, label="command")
        axs[i, j].plot(value, label="value")
        axs[i, j].legend()
        axs[i, j].set_title(f"{dofs[dof_id]}")
        dof_id += 1


name = args.data.split("/")[-1].split(".")[0]
fig.suptitle(title)
plt.show()
