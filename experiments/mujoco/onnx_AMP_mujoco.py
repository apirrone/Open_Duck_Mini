import argparse
import pickle
import time

import FramesViewer.utils as fv_utils
import mujoco
import mujoco_viewer
import numpy as np
from FramesViewer.viewer import Viewer
from glfw import init
from scipy.spatial.transform import Rotation as R

from mini_bdx.onnx_infer import OnnxInfer
from mini_bdx.utils.rl_utils import (
    action_to_pd_targets,
    isaac_to_mujoco,
    mujoco_to_isaac,
)

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
args = parser.parse_args()

saved_obs = pickle.loads(open("saved_obs.pkl", "rb").read())
saved_actions = pickle.loads(open("saved_actions.pkl", "rb").read())

saved_obs = saved_obs[100:]
saved_actions = saved_actions[100:]

fv = Viewer()
fv.start()

# Params
substeps = 4  # don't really know what this is
# dt = 1 / 60
dt = 0.001
linearVelocityScale = 2.0
angularVelocityScale = 0.25
# linearVelocityScale = 0.0
# angularVelocityScale = 0.0
dof_pos_scale = 1.0
dof_vel_scale = 0.05
action_clip = (-1, 1)
obs_clip = (-5, 5)
action_scale = 1.0
mujoco_init_pos = np.array(
    [
        # right_leg
        0.013627156377842975,
        0.07738878096596595,
        0.5933527914082196,
        -1.630548419252953,
        0.8621333440557593,
        # left leg
        -0.013946457213457239,
        0.07918837709879874,
        0.5325073962634973,
        -1.6225192902713386,
        0.9149246381274986,
        # head
        -0.17453292519943295,
        -0.17453292519943295,
        8.65556854322817e-27,
        0,
        0,
    ]
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

isaac_init_pos = np.array(mujoco_to_isaac(mujoco_init_pos))


model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
model.opt.timestep = dt
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)
# model.opt.gravity[:] = [0, 0, 0]  # no gravity

# Confirmed identical to torch.jit.load
# Confirmer identical to when running in isaac
policy = OnnxInfer(args.onnx_model_path)

ang_vel_pose = np.eye(4)
ang_vel_pose[:3, 3] = [0.1, 0.2, 0.1]
ang_vel = [0, 0, 0]


def get_obs(data, isaac_action, commands):
    global ang_vel
    # base_lin_vel = (
    #     data.sensor("linear-velocity").data.astype(np.double) * linearVelocityScale
    # )
    # print(base_lin_vel / linearVelocityScale)
    # print(data.qvel[:3])
    # print("==")

    base_quat = data.qpos[3 : 3 + 4].copy()
    base_quat = [base_quat[1], base_quat[2], base_quat[3], -base_quat[0]]
    rot_mat = R.from_quat(base_quat).as_matrix()
    tmp = np.eye(4)
    tmp[:3, :3] = rot_mat
    final_orientation_mat = tmp[:3, :3]
    base_quat = R.from_matrix(final_orientation_mat).as_quat()
    tmp[:3, :3] = tmp[:3, :3].T
    tmp[:3, 3] = [0.1, 0.1, 0.1]
    fv.pushFrame(tmp, "aze", color=[255, 0, 0])

    base_ang_vel = (
        data.sensor("angular-velocity").data.astype(np.double) * angularVelocityScale
    )

    ang_vel += base_ang_vel
    ang_vel_pose[:3, :3] = R.from_euler("xyz", ang_vel, degrees=True).as_matrix()
    fv.pushFrame(ang_vel_pose, "abc")

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
commands = [0.02, 0.0, 0.0]
prev = time.time()
last_control = time.time()
control_freq = 30  # hz
i = 0
data.qpos[3 : 3 + 4] = [1, 0, 0.08, 0]
data.qpos[7 : 7 + 15] = mujoco_init_pos
data.ctrl[:] = mujoco_init_pos

command_value = []
count = 0
try:
    while True:
        t = time.time()
        dt = t - prev
        if t - last_control >= 1 / control_freq:
            # print(t - last_control)

            # TODO problem Definitely comes from get_obs.
            isaac_obs = get_obs(data, prev_isaac_action, commands)
            # isaac_obs = saved_obs[i]  # works with saved obs
            isaac_obs = np.clip(isaac_obs, obs_clip[0], obs_clip[1])  # order OK

            isaac_action = policy.infer(isaac_obs)
            # isaac_action = saved_actions[i][0]
            prev_isaac_action = isaac_action.copy()
            isaac_action = action_to_pd_targets(
                isaac_action, pd_action_offset, pd_action_scale
            )  # order OK
            # isaac_action = isaac_action + isaac_init_pos
            isaac_action = np.clip(
                isaac_action, action_clip[0], action_clip[1]
            )  # order OK

            mujoco_action = isaac_to_mujoco(isaac_action)

            last_control = t
            i += 1

            data.ctrl[:] = mujoco_action.copy()

            # tmp_action = saved_actions[i][0]
            # tmp_action = action_to_pd_targets(tmp_action)
            # tmp_action = isaac_to_mujoco(tmp_action)
            # data.ctrl[:] = tmp_action.copy()
            if i >= len(saved_obs) - 1:
                i = 0

        command_value.append([data.ctrl.copy(), data.qpos[7:].copy()])
        mujoco.mj_step(model, data, 5)  # 4 seems good

        # data.qpos[0:3] = [0, 0, 0.5]
        # data.qpos[0] = np.sin(time.time())
        viewer.render()
        prev = t
        count += 1
except KeyboardInterrupt:
    pickle.dump(command_value, open("command_value.pkl", "wb"))
    # time.sleep(1)
