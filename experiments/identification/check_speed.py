from mini_bdx_runtime.hwi import HWI
from mini_bdx_runtime.rl_utils import make_action_dict, mujoco_joints_order
import time
import numpy as np
import mujoco
import mujoco_viewer
import pickle

hwi = HWI("/dev/ttyUSB0")

hwi.turn_on()
hwi.set_pid_all([1100, 0, 0])
# hwi.set_pid([500, 0, 0], "neck_pitch")
# hwi.set_pid([500, 0, 0], "head_pitch")
# hwi.set_pid([500, 0, 0], "head_yaw")

dt = 0.0001

model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
model.opt.timestep = dt
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)


init_pos = list(hwi.init_pos.values())
# init_pos += [0, 0]
data.qpos[:13] = init_pos
data.ctrl[:13] = init_pos

dof = 7
a = 0.3
f = 3

recording = {}
recording["mujoco_vel"] = []
recording["robot_vel"] = []
try:
    while True:
        target = a * np.sin(2 * np.pi * f * time.time())
        full_target = init_pos.copy()
        full_target[dof] += target

        #
        data.ctrl[:13] = full_target
        action_dict = make_action_dict(full_target, mujoco_joints_order)
        hwi.set_position_all(action_dict)

        mujoco.mj_step(model, data, 50)

        mujoco_vel = data.qvel[dof]
        robot_vel = hwi.get_present_velocities()[dof]

        recording["mujoco_vel"].append(mujoco_vel)
        recording["robot_vel"].append(robot_vel)

        viewer.render()
except KeyboardInterrupt:
    pickle.dump(recording, open("speeds.pkl", "wb"))
