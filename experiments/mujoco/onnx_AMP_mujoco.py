import argparse
import pickle
import time

import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from mini_bdx.onnx_infer import OnnxInfer
from mini_bdx.utils.rl_utils import isaac_to_mujoco, mujoco_to_isaac

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
parser.add_argument("--saved_obs", type=str, required=False)
parser.add_argument("--saved_actions", type=str, required=False)
args = parser.parse_args()

if args.saved_obs is not None:
    saved_obs = pickle.loads(open("saved_obs.pkl", "rb").read())
if args.saved_actions is not None:
    saved_actions = pickle.loads(open("saved_actions.pkl", "rb").read())


# Params
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
        # right_leg
        -0.014,
        0.08,
        0.53,
        -1.32,
        # -1.52,
        0.91,
        # left leg
        0.013,
        0.077,
        0.59,
        -1.33,
        # -1.53,
        0.86,
        # head
        -0.17,
        -0.17,
        0.0,
        0.0,
        0.0,
    ]
)


isaac_init_pos = np.array(mujoco_to_isaac(mujoco_init_pos))


model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
model.opt.timestep = dt
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)
# model.opt.gravity[:] = [0, 0, 0]  # no gravity

policy = OnnxInfer(args.onnx_model_path)


def get_obs(data, isaac_action, commands):

    base_lin_vel = (
        data.sensor("linear-velocity").data.astype(np.double) * linearVelocityScale
    )

    base_quat = data.qpos[3 : 3 + 4].copy()
    base_quat = [base_quat[1], base_quat[2], base_quat[3], -base_quat[0]]
    rot_mat = R.from_quat(base_quat).as_matrix()
    tmp = np.eye(4)
    tmp[:3, :3] = rot_mat
    final_orientation_mat = tmp[:3, :3]
    base_quat = R.from_matrix(final_orientation_mat).as_quat()

    rot_euler = R.from_quat(base_quat).as_euler("xyz", degrees=False)
    rot_euler[2] = 0
    base_quat = R.from_euler("xyz", rot_euler, degrees=False).as_quat()

    base_ang_vel = (
        data.sensor("angular-velocity").data.astype(np.double) * angularVelocityScale
    )

    mujoco_dof_pos = data.qpos[7 : 7 + 15].copy()
    isaac_dof_pos = mujoco_to_isaac(mujoco_dof_pos)

    isaac_dof_pos_scaled = (isaac_dof_pos - isaac_init_pos) * dof_pos_scale

    mujoco_dof_vel = data.qvel[6 : 6 + 15].copy()
    isaac_dof_vel = mujoco_to_isaac(mujoco_dof_vel)
    isaac_dof_vel_scaled = list(np.array(isaac_dof_vel) * dof_vel_scale)

    obs = np.concatenate(
        [
            base_quat,
            # base_lin_vel,
            base_ang_vel,
            isaac_dof_pos_scaled,
            isaac_dof_vel_scaled,
            isaac_action,
            commands,
        ]
    )

    return obs


prev_isaac_action = np.zeros(15)
commands = [0.1, 0.0, 0.0]
prev = time.time()
last_control = time.time()
control_freq = 30  # hz
i = 0
# data.qpos[3 : 3 + 4] = [1, 0, 0.08, 0]

# init_rot = [np.pi * 2, -0.2, 0]
init_rot = [0, -0.0, 0]
init_quat = R.from_euler("xyz", init_rot, degrees=False).as_quat()
data.qpos[3 : 3 + 4] = init_quat
# data.qpos[3 : 3 + 4] = [init_quat[3], init_quat[1], init_quat[2], init_quat[0]]
# data.qpos[3 : 3 + 4] = [1, 0, 0.08, 0]


data.qpos[7 : 7 + 15] = mujoco_init_pos
data.ctrl[:] = mujoco_init_pos

mujoco_saved_obs = []
mujoco_saved_actions = []
command_value = []
count = 0
start_timeout = 0
try:
    while True:
        t = time.time()
        dt = t - prev
        start_timeout -= dt
        if t - last_control >= 1 / control_freq and start_timeout <= 0:

            isaac_obs = get_obs(data, prev_isaac_action, commands)
            mujoco_saved_obs.append(isaac_obs)

            if args.saved_obs is not None:
                isaac_obs = saved_obs[i]  # works with saved obs

            isaac_obs = np.clip(isaac_obs, obs_clip[0], obs_clip[1])

            isaac_action = policy.infer(isaac_obs)
            if args.saved_actions is not None:
                isaac_action = saved_actions[i][0]
            isaac_action = np.clip(isaac_action, action_clip[0], action_clip[1])
            prev_isaac_action = isaac_action.copy()
            isaac_action = isaac_init_pos + isaac_action

            mujoco_action = isaac_to_mujoco(isaac_action)

            last_control = t
            i += 1

            data.ctrl[:] = mujoco_action.copy()
            # euler_rot = [np.sin(2 * np.pi * 0.5 * t), 0, 0]
            # quat = R.from_euler("xyz", euler_rot, degrees=False).as_quat()
            # data.qpos[3 : 3 + 4] = quat
            mujoco_saved_actions.append(mujoco_action)

            command_value.append([data.ctrl.copy(), data.qpos[7:].copy()])
        mujoco.mj_step(model, data, 5)

        viewer.render()
        prev = t
        count += 1
except KeyboardInterrupt:
    data = {
        "config": {},
        "mujoco": command_value,
    }
    pickle.dump(data, open("mujoco_command_value.pkl", "wb"))
    pickle.dump(mujoco_saved_obs, open("mujoco_saved_obs.pkl", "wb"))
    pickle.dump(mujoco_saved_actions, open("mujoco_saved_actions.pkl", "wb"))
