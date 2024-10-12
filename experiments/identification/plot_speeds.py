import pickle

# data is
# recording = {}
# recording["mujoco_vel"] = []
# recording["robot_vel"] = []
data = pickle.load(open("speeds.pkl", "rb"))


import matplotlib.pyplot as plt

mujoco_vel = data.get("mujoco_vel", [])
robot_vel = data.get("robot_vel", [])

plt.figure()
plt.plot(mujoco_vel, label="Mujoco Velocity")
plt.plot(robot_vel, label="Robot Velocity")
plt.xlabel("Time Step")
plt.ylabel("Velocity")
plt.title("Mujoco Velocity vs Robot Velocity")
plt.legend()
plt.show()
