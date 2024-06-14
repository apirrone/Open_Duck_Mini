import pickle

import matplotlib.pyplot as plt

present_position, init_position, interpolated_values = pickle.load(
    open("trajectory.pkl", "rb")
)
plt.figure(figsize=(10, 6))
for i in range(len(present_position)):
    plt.plot(interpolated_values[:, i], label=f"Angle {i+1}")

# plt.legend()
# plt.xlabel("Interpolation Step")
# plt.ylabel("Angle Value")
# plt.title("Interpolation between Present and Goal Angles")
plt.show()
