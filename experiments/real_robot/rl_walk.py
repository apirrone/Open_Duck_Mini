from mini_bdx.onnx_infer import OnnxInfer
import numpy as np
import time
from mini_bdx.hwi import HWI
import pickle
from mini_bdx.utils.rl_utils import (
    action_to_pd_targets,
    isaac_to_mujoco,
    isaac_joints_order,
)

pd_action_offset = [
    0.0,
    -0.57,
    0.52,
    0.0,
    0.0,
    -0.57,
    0.0,
    0.0,
    0.48,
    -0.48,
    0.0,
    -0.57,
    0.52,
    0.0,
    0.0,
]
pd_action_scale = [
    0.98,
    1.4,
    1.47,
    2.93,
    2.2,
    1.04,
    0.98,
    2.93,
    2.26,
    2.26,
    0.98,
    1.4,
    1.47,
    2.93,
    2.2,
]


def make_action_dict(action):
    action_dict = {}
    for i, a in enumerate(action):
        if "antenna" not in isaac_joints_order[i]:
            action_dict[isaac_joints_order[i]] = a

    return action_dict


# TODO
def get_obs(hwi, imu):
    # Don't forget to re invert the angles from the hwi
    pass


policy = OnnxInfer("/home/antoine/MISC/IsaacGymEnvs/isaacgymenvs/ONNX_NO_PUSH.onnx")

fake_obs = pickle.loads(open("saved_obs.pkl", "rb").read())
fake_actions = pickle.loads(open("saved_actions.pkl", "rb").read())

hwi = HWI(usb_port="/dev/ttyUSB0")

command_value = []

hwi.turn_on()
time.sleep(1)
skip = 10
i = 0
ctrl_freq = 30  # hz
while True:
    if skip > 0:
        skip -= 1
    start = time.time()
    obs = fake_obs[i]  # for now
    # obs = get_obs(hwi, imu)

    obs = np.clip(obs, -5, 5)
    action = policy.infer(obs)
    action = fake_actions[i][0]
    action = np.clip(action, -1, 1)
    action = action_to_pd_targets(action, pd_action_offset, pd_action_scale)
    action_dict = make_action_dict(action)
    # print(action_dict)
    hwi.set_position_all(action_dict)
    took = time.time() - start
    # time.sleep((max(0, (1 / ctrl_freq) - took)))
    time.sleep(1 / ctrl_freq)
    i += 1
    if i >= len(fake_obs) - 1:
        break

    command_value.append((list(action_dict.values()), hwi.get_present_positions()))
    # print(len(command_value[0][0]), len(command_value[0][1]))
    pickle.dump(command_value, open("command_value.pkl", "wb"))
