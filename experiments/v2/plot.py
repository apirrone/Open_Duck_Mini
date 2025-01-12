import pickle

data = pickle.load(open("data_pwm_control.pkl", "rb"))


# data looks like:
# data = {
#     "present_positions": present_positions,
#     "goal_positions": goal_positions,
#     "present_loads": present_loads,
#     "present_currents": present_currents,
#     "present_speeds": present_speeds,
#     "times": times,
# }

# present_positions, goal_positions etc are lists. All of the same size.

# plot all on the same plot against time with shared x
# Label everything


import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 1, sharex=True)

axs[0].plot(data["times"], data["present_positions"], label="Present positions")
axs[0].plot(data["times"], data["goal_positions"], label="Goal positions")
axs[0].set_ylabel("Positions")
axs[0].legend()

axs[1].plot(data["times"], data["present_loads"], label="Present loads")
axs[1].set_ylabel("Loads")

axs[2].plot(data["times"], data["present_currents"], label="Present currents")
axs[2].set_ylabel("Currents")

axs[3].plot(data["times"], data["present_speeds"], label="Present speeds")
axs[3].set_ylabel("Speeds")

plt.show()
