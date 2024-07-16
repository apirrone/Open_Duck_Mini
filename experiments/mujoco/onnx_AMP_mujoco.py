from mini_bdx.onnx_infer import OnnxInfer
import time
from mini_bdx.utils.rl_utils import isaac_to_mujoco, mujoco_to_isaac
import argparse
import numpy as np
import mujoco, mujoco_viewer
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
args = parser.parse_args()

saved_obs = pickle.loads(open("saved_obs.pkl", "rb").read())
saved_obs = saved_obs[20:]
# print(len(saved_obs))
# for ob in saved_obs:
#     print(ob)
# exit()

# Params
substeps = 4  # don't really know what this is
# dt = 1 / 60
dt = 0.001
linearVelocityScale = 2.0
angularVelocityScale = 0.25
dof_pos_scale = 1.0
dof_vel_scale = 0.05
action_clip = (-1, 1)
obs_clip = (-5, 5)
action_scale = 1.0
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


# tensor([ 0.0000, -0.5672,  0.5236,  0.0000,  0.0000, -0.5672,  0.0000,  0.0000,
#          0.4800, -0.4800,  0.0000, -0.5672,  0.5236,  0.0000,  0.0000],
#        device='cuda:0')
# tensor([0.9774, 1.4050, 1.4661, 2.9322, 2.1991, 1.0385, 0.9774, 2.9322, 2.2602,
#         2.2602, 0.9774, 1.4050, 1.4661, 2.9322, 2.1991], device='cuda:0')

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

isaac_init_pos = np.array(mujoco_to_isaac(mujoco_init_pos))


model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
model.opt.timestep = dt
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)
model.opt.gravity[:] = [0, 0, 0]  # no gravity

policy = OnnxInfer(args.onnx_model_path)


def get_obs(data, mujoco_action, commands):
    base_lin_vel = data.qvel[3 : 3 + 3] * linearVelocityScale
    base_ang_vel = data.qvel[:3] * angularVelocityScale
    # base_lin_vel = data.body("base").cvel[3:] * linearVelocityScale
    # base_ang_vel = data.body("base").cvel[:3] * angularVelocityScale

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
    ).astype(np.float64)

    return obs


def action_to_pd_targets(isaac_action):
    isaac_action = pd_action_offset + pd_action_scale * isaac_action
    return isaac_action


prev_mujoco_action = np.zeros(15)
commands = [0.0, 0, 0]
prev = time.time()
last_control = time.time()
control_freq = 1  # hz
i = 0
# data.qpos[3 : 3 + 4] = [1, 0, 0.02, 0]
#

command_value = []
while True:
    t = time.time()
    dt = t - prev
    if t - last_control >= 1 / control_freq:
        # print(t - last_control)
        # isaac_obs = get_obs(data, prev_mujoco_action, commands)
        isaac_obs = saved_obs[i]
        isaac_obs = np.clip(isaac_obs, obs_clip[0], obs_clip[1])
        isaac_action = policy.infer(isaac_obs)
        isaac_action = np.clip(isaac_action, action_clip[0], action_clip[1])
        # isaac_action = isaac_init_pos - isaac_action
        isaac_action = action_to_pd_targets(isaac_action)
        # print(isaac_action)

        mujoco_action = isaac_to_mujoco(isaac_action)
        mujoco_action = np.array(mujoco_action) * action_scale
        prev_mujoco_action = mujoco_action.copy()
        last_control = t
        i += 1
        # data.ctrl[:] = mujoco_action.copy()
    else:
        pass
        # mujoco_action = prev_mujoco_action

    if int(t) % 2 == 0:
        command = -np.pi / 8
    else:
        command = np.pi / 8
    data.ctrl[:] = 0
    data.ctrl[5 : 5 + 5] = command
    command = data.ctrl.copy()
    mujoco.mj_step(model, data, 5)  # 4 seems good

    value = data.qpos[7:].copy()
    command_value.append([command, value])
    # command_value.append([command, data.qpos[7:].copy()])
    pickle.dump(command_value, open("command_value.pkl", "wb"))

    viewer.render()
    prev = t
