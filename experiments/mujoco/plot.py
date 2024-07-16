import pickle
import matplotlib.pyplot as plt

import numpy as np

command_value = pickle.load(open("command_value.pkl", "rb"))

# command_value = np.array(command_value)

for j in range(15):
    command = []
    value = []
    for i in range(len(command_value)):
        command.append(command_value[i][0][j])
        value.append(command_value[i][1][j])
    plt.plot(command, label="command")
    plt.plot(value)
    plt.legend()
    plt.show()
    plt.clf()
