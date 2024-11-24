import argparse
import pickle
import time

import mujoco
import mujoco_viewer
import numpy as np
import pygame
from scipy.spatial.transform import Rotation as R

from mini_bdx.onnx_infer import OnnxInfer
from mini_bdx.utils.rl_utils import (
    action_to_pd_targets,
    isaac_to_mujoco,
    mujoco_to_isaac,
    mujoco_joints_order,
)
from mini_bdx_runtime.rl_utils import LowPassActionFilter
from bam.model import load_model
from bam.mujoco import MujocoController


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
parser.add_argument("-k", action="store_true", default=False)
parser.add_argument("--rma", action="store_true", default=False)
parser.add_argument("--awd", action="store_true", default=False)
parser.add_argument("--adaptation_module_path", type=str, required=False)
parser.add_argument("--replay_obs", type=str, required=False, default=None)
args = parser.parse_args()

if args.k:
    pygame.init()
    # open a blank pygame window
    screen = pygame.display.set_mode((100, 100))
    pygame.display.set_caption("Press arrow keys to move robot")


if args.replay_obs is not None:
    path = args.replay_obs
    path = path[: -len("obs.pkl")]
    actions_path = path + "actions.pkl"
    replay_obs = pickle.load(open(args.replay_obs, "rb"))
    replay_actions = pickle.load(open(actions_path, "rb"))
replay_index = 0

# Params
dt = 0.0001
linearVelocityScale = 2.0 if not args.awd else 0.5
angularVelocityScale = 0.25
dof_pos_scale = 1.0
dof_vel_scale = 0.05
action_clip = (-1, 1)
obs_clip = (-5, 5)
action_scale = 0.25 if not args.awd else 1.0


isaac_init_pos = np.array(
    [
        -0.0285397830292128,
        0.01626303761810685,
        1.0105624704499077,
        -1.4865015965817336,
        0.6504953719748071,
        -0.17453292519943295,
        -0.17453292519943295,
        0,
        0,
        0,
        0.001171696610228082,
        0.006726989242258406,
        1.0129772861831692,
        -1.4829304760981399,
        0.6444901047812701,
    ]
)


mujoco_init_pos = np.array(isaac_to_mujoco(isaac_init_pos))


model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene_motor.xml")
model.opt.timestep = dt
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)
# model.opt.gravity[:] = [0, 0, 0]  # no gravity

NUM_OBS = 51

policy = OnnxInfer(args.onnx_model_path, awd=args.awd)
if args.rma:
    adaptation_module = OnnxInfer(args.adaptation_module_path, "obs_history")
    obs_history_size = 15
    obs_history = np.zeros((obs_history_size, NUM_OBS)).tolist()


class ObsDelaySimulator:
    def __init__(self, delay_ms):
        self.delay_ms = delay_ms
        self.obs = []
        self.t0 = None

    def push(self, obs, t):
        self.obs.append(obs)
        if self.t0 is None:
            self.t0 = t

    def get(self, t):
        if t - self.t0 < self.delay_ms / 1000:
            return np.zeros(NUM_OBS)
        print(len(self.obs))
        return self.obs.pop(0)


def quat_rotate_inverse(q, v):
    q = np.array(q)
    v = np.array(v)

    q_w = q[-1]
    q_vec = q[:3]

    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * (np.dot(q_vec, v)) * 2.0

    return a - b + c


def get_obs(data, prev_isaac_action, commands):
    global replay_index
    if args.replay_obs is not None:
        obs = replay_obs[replay_index]
        return obs

    base_lin_vel = (
        data.sensor("linear-velocity").data.astype(np.double) * linearVelocityScale
    )

    base_quat = data.qpos[3 : 3 + 4].copy()
    base_quat = [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]

    # # Remove yaw component
    # rot_euler = R.from_quat(base_quat).as_euler("xyz", degrees=False)
    # rot_euler[1] += np.deg2rad(-15)
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

    projected_gravity = quat_rotate_inverse(base_quat, [0, 0, -1])

    if not args.awd:
        obs = np.concatenate(
            [
                projected_gravity,
                commands,
                isaac_dof_pos_scaled,
                isaac_dof_vel_scaled,
                prev_isaac_action,
            ]
        )
    else:
        obs = np.concatenate(
            [
                projected_gravity,
                isaac_dof_pos,
                isaac_dof_vel,
                prev_isaac_action,
                commands,
            ]
        )
    return obs


prev_isaac_action = np.zeros(15)
commands = [0.14 * 2, 0.0, 0.0]
prev = data.time
last_control = data.time
control_freq = 60  # hz
hist_freq = 30  # hz
adaptation_module_freq = 5  # hz
last_adaptation = data.time
last_hist = data.time
i = 0
data.qpos[3 : 3 + 4] = [1, 0, 0.0, 0]
cutoff_frequency = 40


data.qpos[7 : 7 + 15] = mujoco_init_pos
data.ctrl[:] = mujoco_init_pos

action_filter = LowPassActionFilter(
    control_freq=control_freq, cutoff_frequency=cutoff_frequency
)

# BAM
xc330_model = load_model("params_m6.json")
mujoco_controllers = {}
for joint_name in mujoco_joints_order:
    mujoco_controllers[joint_name] = MujocoController(
        xc330_model, joint_name, model, data
    )


mujoco_saved_obs = []
# obs_delay_simulator = ObsDelaySimulator(0)
start = time.time()
saved_latent = []
mujoco_action = mujoco_init_pos.copy()
try:
    start = time.time()
    while True:
        # t = time.time()
        t = data.time
        if time.time() - start < 1:
            last_control = t
        if t - last_control >= 1 / control_freq:
            isaac_obs = get_obs(data, prev_isaac_action, commands)
            # obs_delay_simulator.push(isaac_obs, t)
            # isaac_obs = obs_delay_simulator.get(t)
            if args.rma:
                if t - last_hist >= 1 / hist_freq:
                    obs_history.append(isaac_obs)
                    obs_history = obs_history[-obs_history_size:]
                    last_hist = t

            mujoco_saved_obs.append(isaac_obs)

            if args.rma:
                if t - last_adaptation >= 1 / adaptation_module_freq:
                    latent = adaptation_module.infer(np.array(obs_history).flatten())
                    last_adaptation = t
                saved_latent.append(latent)
                policy_input = np.concatenate([isaac_obs, latent])
                isaac_action = policy.infer(policy_input)
            else:
                isaac_action = policy.infer(isaac_obs)

            if args.replay_obs:
                replayed_action = replay_actions[replay_index]
                print("infered action", np.around(isaac_action, 3))
                print("replayed action", np.around(replayed_action, 3))
                print("diff", np.around(isaac_action - replayed_action, 3))
                print("==")
            # action_filter.push(isaac_action)
            # isaac_action = action_filter.get_filtered_action()

            prev_isaac_action = isaac_action.copy()  # * action_scale

            isaac_action = isaac_action * action_scale + isaac_init_pos

            mujoco_action = isaac_to_mujoco(isaac_action)

            # for i, joint_name in enumerate(mujoco_joints_order):
            #     mujoco_controllers[joint_name].update(mujoco_action[i])

            # # computed with bam
            # torque = mujoco_action.copy()
            # # needs to be torque
            # data.ctrl[:] = torque

            last_control = t
            i += 1

            if args.k:
                keys = pygame.key.get_pressed()
                lin_vel_x = 0
                lin_vel_y = 0
                ang_vel = 0
                if keys[pygame.K_z]:
                    lin_vel_x = 0.14
                if keys[pygame.K_s]:
                    lin_vel_x = -0.14
                if keys[pygame.K_q]:
                    lin_vel_y = 0.1
                if keys[pygame.K_d]:
                    lin_vel_y = -0.1
                if keys[pygame.K_a]:
                    ang_vel = 0.3
                if keys[pygame.K_e]:
                    ang_vel = -0.3

                commands[0] = lin_vel_x
                commands[1] = lin_vel_y
                commands[2] = ang_vel

                commands = list(
                    np.array(commands)
                    * np.array(
                        [
                            linearVelocityScale,
                            linearVelocityScale,
                            angularVelocityScale,
                        ]
                    )
                )
                pygame.event.pump()  # process event queue
            replay_index += 1
            print(commands)
        for i, joint_name in enumerate(mujoco_joints_order):
            mujoco_controllers[joint_name].update(mujoco_action[i])
        mujoco.mj_step(model, data, 50)

        viewer.render()
        prev = t
except KeyboardInterrupt:
    pickle.dump(mujoco_saved_obs, open("mujoco_saved_obs.pkl", "wb"))
    pickle.dump(saved_latent, open("mujoco_saved_latent.pkl", "wb"))
