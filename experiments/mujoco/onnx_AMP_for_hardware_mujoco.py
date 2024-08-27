import argparse
import pickle
import time

import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from mini_bdx.onnx_infer import OnnxInfer
from mini_bdx.utils.rl_utils import (
    action_to_pd_targets,
    isaac_to_mujoco,
    mujoco_to_isaac,
)
from mini_bdx_runtime.rl_utils import LowPassActionFilter

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
# dt = 0.002
dt = 0.0001
linearVelocityScale = 2.0
angularVelocityScale = 0.25
dof_pos_scale = 1.0
dof_vel_scale = 0.05
action_clip = (-1, 1)
obs_clip = (-5, 5)
action_scale = 0.25


# Higher
mujoco_init_pos = np.array(
    [
        -0.03676731090962078,
        -0.030315211140564333,
        0.4065815100399598,
        -1.0864064934571644,
        0.5932324840794684,
        -0.03485756878823724,
        0.052286054888550475,
        0.36623601032755765,
        -0.964204465274923,
        0.5112970996901808,
        -0.17453292519943295,
        -0.17453292519943295,
        0,
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


class ImuDelaySimulator:
    def __init__(self, delay_ms):
        self.delay_ms = delay_ms
        self.rot = []
        self.ang_rot = []
        self.t0 = None

    def push(self, rot, ang_rot, t):
        self.rot.append(rot)
        self.ang_rot.append(ang_rot)
        if self.t0 is None:
            self.t0 = t

    def get(self):
        if time.time() - self.t0 < self.delay_ms / 1000:
            return [0, 0, 0, 0], [0, 0, 0]
        rot = self.rot.pop(0)
        ang_rot = self.ang_rot.pop(0)

        return rot, ang_rot


# # TODO convert to numpy
# def quat_rotate_inverse(q, v):
#     shape = q.shape
#     q_w = q[:, -1]
#     q_vec = q[:, :3]
#     a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
#     b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
#     c = (
#         q_vec
#         * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
#         * 2.0
#     )
#     return a - b + c


def quat_rotate_inverse(q, v):
    q = np.array(q)
    v = np.array(v)

    q_w = q[-1]
    q_vec = q[:3]

    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * (np.dot(q_vec, v)) * 2.0

    return a - b + c


def get_obs(data, isaac_action, commands, imu_delay_simulator: ImuDelaySimulator):
    base_lin_vel = (
        data.sensor("linear-velocity").data.astype(np.double) * linearVelocityScale
    )

    base_quat = data.qpos[3 : 3 + 4].copy()
    base_quat = [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]

    # # Remove yaw component
    # rot_euler = R.from_quat(base_quat).as_euler("xyz", degrees=False)
    # rot_euler[2] = 0
    # base_quat = R.from_euler("xyz", rot_euler, degrees=False).as_quat()

    base_ang_vel = (
        data.sensor("angular-velocity").data.astype(np.double) * angularVelocityScale
    )

    mujoco_dof_pos = data.qpos[7 : 7 + 15].copy()
    isaac_dof_pos = mujoco_to_isaac(mujoco_dof_pos)

    isaac_dof_pos_scaled = (isaac_dof_pos - isaac_init_pos) * dof_pos_scale

    mujoco_dof_vel = data.qvel[6 : 6 + 15].copy()
    isaac_dof_vel = mujoco_to_isaac(mujoco_dof_vel)
    isaac_dof_vel_scaled = list(np.array(isaac_dof_vel) * dof_vel_scale)

    imu_delay_simulator.push(base_quat, base_ang_vel, time.time())
    base_quat, base_ang_vel = imu_delay_simulator.get()

    projected_gravity = quat_rotate_inverse(base_quat, [0, 0, -1])

    obs = np.concatenate(
        [
            projected_gravity,
            commands,
            isaac_dof_pos_scaled,
            isaac_dof_vel_scaled,
            isaac_action,
        ]
    )

    return obs


prev_isaac_action = np.zeros(15)
commands = [0.15, 0.0, 0.0]
# commands = [0.0, 0.0, 0.0]
# prev = time.time()
# last_control = time.time()
prev = data.time
last_control = data.time
control_freq = 85  # hz
i = 0
data.qpos[3 : 3 + 4] = [1, 0, 0.08, 0]
cutoff_frequency = 20

# init_rot = [0, -0.1, 0]
# init_rot = [0, 0, 0]
# init_quat = R.from_euler("xyz", init_rot, degrees=False).as_quat()
# data.qpos[3 : 3 + 4] = init_quat
# data.qpos[3 : 3 + 4] = [init_quat[3], init_quat[1], init_quat[2], init_quat[0]]
# data.qpos[3 : 3 + 4] = [1, 0, 0.13, 0]


data.qpos[7 : 7 + 15] = mujoco_init_pos
data.ctrl[:] = mujoco_init_pos

action_filter = LowPassActionFilter(
    control_freq=control_freq, cutoff_frequency=cutoff_frequency
)

mujoco_saved_obs = []
mujoco_saved_actions = []
command_value = []
imu_delay_simulator = ImuDelaySimulator(1)
start = time.time()
try:
    start = time.time()
    while True:
        # t = time.time()
        t = data.time
        if time.time() - start < 1:
            last_control = t
        if t - last_control >= 1 / control_freq:
            isaac_obs = get_obs(data, prev_isaac_action, commands, imu_delay_simulator)
            mujoco_saved_obs.append(isaac_obs)

            if args.saved_obs is not None:
                isaac_obs = saved_obs[i]  # works with saved obs

            # isaac_obs = np.clip(isaac_obs, obs_clip[0], obs_clip[1])

            isaac_action = policy.infer(isaac_obs)
            # isaac_actions = np.zeros(15)
            if args.saved_actions is not None:
                isaac_action = saved_actions[i][0]
            # isaac_action = np.clip(isaac_action, action_clip[0], action_clip[1])
            prev_isaac_action = isaac_action.copy()

            # isaac_action = np.zeros(15)
            isaac_action = isaac_action * action_scale + isaac_init_pos

            action_filter.push(isaac_action)
            # isaac_action = action_filter.get_filtered_action()

            mujoco_action = isaac_to_mujoco(isaac_action)

            last_control = t
            i += 1

            data.ctrl[:] = mujoco_action.copy()
            # data.ctrl[:] = np.zeros(15) + mujoco_init_pos
            # euler_rot = [np.sin(2 * np.pi * 0.5 * t), 0, 0]
            # quat = R.from_euler("xyz", euler_rot, degrees=False).as_quat()
            # data.qpos[3 : 3 + 4] = quat
            mujoco_saved_actions.append(mujoco_action)

            command_value.append([data.ctrl.copy(), data.qpos[7:].copy()])
        mujoco.mj_step(model, data, 50)

        viewer.render()
        prev = t
except KeyboardInterrupt:
    data = {
        "config": {},
        "mujoco": command_value,
    }
    pickle.dump(data, open("mujoco_command_value.pkl", "wb"))
    pickle.dump(mujoco_saved_obs, open("mujoco_saved_obs.pkl", "wb"))
    pickle.dump(mujoco_saved_actions, open("mujoco_saved_actions.pkl", "wb"))
