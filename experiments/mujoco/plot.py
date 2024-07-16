import pickle
import matplotlib.pyplot as plt

import numpy as np

command_value = pickle.load(open("command_value.pkl", "rb"))

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


plt.show()
