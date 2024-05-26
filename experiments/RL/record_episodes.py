import pickle
import time

import mujoco
import mujoco.viewer
import numpy as np
import placo
from imitation.data.types import Trajectory
from scipy.spatial.transform import Rotation as R

from mini_bdx.utils.mujoco_utils import check_contact
from mini_bdx.utils.xbox_controller import XboxController
from mini_bdx.walk_engine import WalkEngine

xbox = XboxController()

model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
data = mujoco.MjData(model)

EPISODE_LENGTH = 2000


max_target_step_size_x = 0.03
max_target_step_size_y = 0.03
max_target_yaw = np.deg2rad(15)
target_step_size_x = 0
target_step_size_y = 0
target_yaw = 0
target_head_pitch = 0
target_head_yaw = 0
target_head_z_offset = 0
walking = True
time_since_last_left_contact = 0
time_since_last_right_contact = 0
recording = False
episodes = []
current_episode = {"observations": [], "actions": []}

joint_history_length = 3
joint_error_history = joint_history_length * [13 * [0]]
joint_ctrl_history = joint_history_length * [13 * [0]]
target_velocity = np.zeros(3)

left_contact = False
right_contact = False

start_button_timeout = time.time()


def xbox_input():
    global target_velocity, target_step_size_x, target_step_size_y, target_yaw, walking, t, walk_engine, target_head_pitch, target_head_yaw, target_head_z_offset, start_button_timeout, max_target_step_size_x, max_target_step_size_y, max_target_yaw
    inputs = xbox.read()
    target_step_size_x = -inputs["l_y"] * max_target_step_size_x
    target_step_size_y = inputs["l_x"] * max_target_step_size_y
    if inputs["l_trigger"] > 0.5:
        target_head_pitch = inputs["r_y"] * np.deg2rad(45)
        target_head_yaw = -inputs["r_x"] * np.deg2rad(120)
        target_head_z_offset = inputs["r_trigger"] * 0.08
    else:
        target_yaw = -inputs["r_x"] * max_target_yaw

    if inputs["start"] and time.time() - start_button_timeout > 0.5:
        walking = not walking
        start_button_timeout = time.time()

    target_velocity = np.array([-inputs["l_y"], inputs["l_x"], -inputs["r_x"]])


def key_callback(keycode):
    global recording, walking, target_step_size_x, target_step_size_y, target_yaw, walk_engine, data, t
    if keycode == 257:  # enter
        start_stop_recording()
    if keycode == 261:  # delete
        walking = False
        target_step_size_x = 0
        target_step_size_y = 0
        target_yaw = 0
        walk_engine.reset()
        data.qpos[:7] = 0
        data.qpos[2] = 0.19
        data.ctrl[:] = 0


def get_observation():
    global joint_error_history, joint_ctrl_history, target_velocity, left_contact, right_contact, joint_history_length

    joints_rotations = data.qpos[7 : 7 + 13]
    joints_velocities = data.qvel[6 : 6 + 13]

    joints_error = data.ctrl - data.qpos[7 : 7 + 13]
    joint_error_history.append(joints_error)
    joint_error_history = joint_error_history[-joint_history_length:]

    angular_velocity = data.body("base").cvel[
        :3
    ]  # TODO this is imu, add noise to it later
    linear_velocity = data.body("base").cvel[3:]

    joint_ctrl_history.append(data.ctrl.copy())
    joint_ctrl_history = joint_ctrl_history[-joint_history_length:]

    return np.concatenate(
        [
            joints_rotations,
            joints_velocities,
            angular_velocity,
            linear_velocity,
            target_velocity,
            np.array(joint_error_history).flatten(),
            [left_contact, right_contact],
            [data.time],
        ]
    )


def start_stop_recording():
    global recording, current_episode
    recording = not recording
    if not recording:
        print("Stop recording")
        # store one last observation here
        current_episode["observations"].append(list(get_observation()))

        episode = Trajectory(
            np.array(current_episode["observations"]),
            np.array(current_episode["actions"]),
            None,
            True,
        )
        episodes.append(episode)
        with open("dataset.pkl", "wb") as f:
            pickle.dump(episodes, f)
    else:
        print("Start recording")
        current_episode = {"observations": [], "actions": []}


viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

robot = placo.RobotWrapper(
    "../../mini_bdx/robots/bdx/robot.urdf", placo.Flags.ignore_collisions
)

walk_engine = WalkEngine(robot)


def get_imu(data):

    rot_mat = np.array(data.body("base").xmat).reshape(3, 3)
    gyro = R.from_matrix(rot_mat).as_euler("xyz")

    accelerometer = np.array(data.body("base").cvel)[3:]

    return gyro, accelerometer


def get_feet_contact(data):
    right_contact = check_contact(data, model, "foot_module", "floor")
    left_contact = check_contact(data, model, "foot_module_2", "floor")
    return right_contact, left_contact


start_stop_recording()  # start recording
prev = data.time
try:
    while True:
        dt = data.time - prev

        xbox_input()

        # Get sensor data
        right_contact, left_contact = get_feet_contact(data)
        gyro, accelerometer = get_imu(data)

        walk_engine.update(
            walking,
            gyro,
            accelerometer,
            left_contact,
            right_contact,
            target_step_size_x,
            target_step_size_y,
            target_yaw,
            target_head_pitch,
            target_head_yaw,
            target_head_z_offset,
            dt,
        )

        angles = walk_engine.get_angles()

        # store obs here
        if recording:
            current_episode["observations"].append(list(get_observation()))
            current_episode["actions"].append(list(angles.values()))

        if len(current_episode["observations"]) > EPISODE_LENGTH:
            start_stop_recording()  # stop recording
            start_stop_recording()  # start recording

        # apply the angles to the robot
        data.ctrl[:] = list(angles.values())

        prev = data.time
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep / 2.5)

except KeyboardInterrupt:
    viewer.close()
