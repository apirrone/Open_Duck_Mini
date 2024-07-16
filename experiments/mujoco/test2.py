import numpy as np
from numpy.lib.utils import safe_eval
from mini_bdx.onnx_infer import OnnxInfer
import pickle

policy = OnnxInfer("/home/antoine/MISC/IsaacGymEnvs/isaacgymenvs/ONNX.onnx")

saved_obs = pickle.loads(open("saved_obs.pkl", "rb").read())
saved_actions = pickle.loads(open("saved_actions.pkl", "rb").read())

prev = saved_actions[0][1]
for i in range(1, len(saved_actions)):
    print(saved_actions[i][1] - prev)
    prev = saved_actions[i][1]
exit()

for i in range(len(saved_obs)):
    obs = saved_obs[i]
    obs = np.clip(obs, -5, 5)
    action = policy.infer(obs)
    action = np.clip(action, -1, 1)

    # print(action)
    # print(saved_actions[i])
    # print(action - saved_actions[i])
    print(np.sum(action - saved_actions[i]))
    print("==")
