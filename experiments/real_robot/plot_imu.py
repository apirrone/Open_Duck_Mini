import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import ImuFilter

FPS = 60
linear_accelerations = pickle.load(open("gyro_data.pkl", "rb"))

# plot
filtered_linear_accelerations = []
imu_filter = ImuFilter(window_size=100)
for linear_acceleration in linear_accelerations:
    imu_filter.push_data(linear_acceleration)
    filtered_linear_accelerations.append(imu_filter.get_filtered_data())

linear_accelerations = filtered_linear_accelerations


axes = ["x", "y", "z"]
fig, axs = plt.subplots(3, 1)
for i in range(3):
    linear_acceleration = []
    for k in range(len(linear_accelerations)):
        linear_acceleration.append(linear_accelerations[k][i])
    axs[i].plot(linear_acceleration, label="linear_acceleration")
    axs[i].legend()
    axs[i].set_title(f"linear_acceleration {axes[i]}")
    # same range for all plots

    axs[i].set_ylim([-10, 10])

plt.show()
