# from mini_bdx.onnx_infer import OnnxInfer
from mini_bdx.utils.rl_utils import isaac_to_mujoco, mujoco_to_isaac
import argparse
import gymnasium as gym
import mujoco
import numpy as np
import mujoco.viewer
from gymnasium.envs.registration import register
import torch

model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)

mujoco_init_pos = np.array(
    [
        -0.013946457213457239,
        0.07918837709879874,
        0.5325073962634973,
        -1.6225192902713386,
        0.9149246381274986,
        0.013627156377842975,
        0.07738878096596595,
        0.5933527914082196,
        -1.630548419252953,
        0.8621333440557593,
        -0.17453292519943295,
        -0.17453292519943295,
        8.65556854322817e-27,
        0,
        0,
    ]
)

isaac_init_pos = np.array(mujoco_to_isaac(mujoco_init_pos))


def build_obs(mdata, model, mujoco_action, commands):
    linearVelocityScale = 2.0
    angularVelocityScale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05

    base_lin_vel = data.body("base").cvel[3:] * linearVelocityScale
    base_ang_vel = data.body("base").cvel[:3] * angularVelocityScale

    mujoco_dof_pos = data.qpos[7 : 7 + 15]
    isaac_dof_pos = mujoco_to_isaac(mujoco_dof_pos)

    isaac_dof_pos_scaled = (isaac_dof_pos - isaac_init_pos) * dof_pos_scale

    mujoco_dof_vel = data.qvel[6 : 6 + 15]
    isaac_dof_vel = mujoco_to_isaac(mujoco_dof_vel)
    isaac_dof_vel_scaled = list(np.array(isaac_dof_vel) * dof_vel_scale)

    isaac_action = mujoco_to_isaac(mujoco_action)
    obs = np.concatenate(
        [
            base_lin_vel,
            base_ang_vel,
            isaac_dof_pos_scaled,
            isaac_dof_vel_scaled,
            isaac_action,
            commands,
        ]
    ).astype(np.float32)

    obs = np.clip(obs, -5, 5)
    return obs

    # return np.zeros(54).astype(np.float32)
    # return np.random.uniform(size=54).astype(np.float32)


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--policy_path", type=str, required=True)
args = parser.parse_args()


# oi = OnnxInfer(args.onnx_model_path)

policy = torch.jit.load(args.policy_path)

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


def action_to_pd_targets(action):
    pd_tar = pd_action_offset + pd_action_scale * action
    return pd_tar


prev_action = np.zeros(15).astype(np.float32)
speed = 3
prev = data.time
commands = [0.02, 0, 0]
action_scale = 1.0
control_freq = 60  # hz
last_ctrl = data.time
while True:
    t = data.time
    dt = t - prev
    if t - last_ctrl > 1 / control_freq:
        isaac_obs = torch.tensor(build_obs(data, model, prev_action, commands)).to(
            device="cuda:0"
        )
        isaac_action = np.array(policy(isaac_obs)[0].cpu().detach().numpy())
        print(isaac_action)

        # isaac_action = oi.infer(isaac_obs)
        isaac_action = action_to_pd_targets(isaac_action)
        isaac_action = np.clip(isaac_action, -1, 1)
        mujoco_action = isaac_to_mujoco(isaac_action)
        prev_action = mujoco_action.copy()
        mujoco_action = mujoco_init_pos - list(np.array(mujoco_action) * action_scale)
        # print(mujoco_action)
        data.ctrl[:] = mujoco_action

    mujoco.mj_step(model, data, speed)  # 4 seems good
    viewer.sync()
    prev = t
