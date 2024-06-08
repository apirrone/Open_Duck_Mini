import argparse
import os
import time
from glob import glob
from typing import Dict, List

import h5py
import mujoco
import mujoco.viewer
import numpy as np
import placo
from scipy.spatial.transform import Rotation as R

from mini_bdx.bdx_mujoco_server import BDXMujocoServer
from mini_bdx.utils.mujoco_utils import check_contact

# from mini_bdx.utils.xbox_controller import XboxController
from mini_bdx.walk_engine import WalkEngine

parser = argparse.ArgumentParser("Record episodes")
parser.add_argument(
    "-n",
    "--session_name",
    type=str,
    required=True,
)
parser.add_argument(
    "-r",
    "--sampling_rate",
    type=int,
    required=False,
    default=30,
    help="Sampling rate in Hz",
)
parser.add_argument(
    "-l",
    "--episode_length",
    type=int,
    required=False,
    default=10,
    help="Episode length in seconds",
)
args = parser.parse_args()

session_name = args.session_name + "_raw"
session_path = os.path.join("data", session_name)
os.makedirs(session_path, exist_ok=True)

bdx_mujoco_server = BDXMujocoServer(
    model_path="/home/antoine/MISC/mini_BDX/mini_bdx/robots/bdx"
)

recording = False
data_dict: Dict[str, List[float]] = {
    "/action": [],
    "/observations/qpos": [],
    "/observations/qvel": [],
}


def key_callback(keycode):
    if keycode == 257:  # enter
        start_stop_recording()


def start_stop_recording():
    global recording, data_dict
    recording = not recording
    if recording:
        print("Recording started")
        pass
    else:
        print("Recording stopped")
        episode_id = len(glob(f"{session_path}/*.hdf5"))
        episode_path = os.path.join(session_path, f"episode_{episode_id}.hdf5")
        print(f"Saving episode in {episode_path} ...")
        max_timesteps = len(data_dict["/action"])
        with h5py.File(
            episode_path,
            "w",
            rdcc_nbytes=1024**2 * 2,
        ) as root:
            obs = root.create_group("observations")
            obs.create_dataset("qpos", (max_timesteps, 20))
            obs.create_dataset("qvel", (max_timesteps, 19))
            root.create_dataset("action", (max_timesteps, 13))

            for name, array in data_dict.items():
                root[name][...] = array
        print("Done")
        data_dict = {
            "/action": [],
            "/observations/qpos": [],
            "/observations/qvel": [],
        }


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

robot = placo.RobotWrapper(
    "../../mini_bdx/robots/bdx/robot.urdf", placo.Flags.ignore_collisions
)

walk_engine = WalkEngine(robot)

bdx_mujoco_server.start()

try:
    prev = bdx_mujoco_server.data.time
    last = bdx_mujoco_server.data.time
    episode_start = bdx_mujoco_server.data.time
    # start_stop_recording()
    while True:
        dt = bdx_mujoco_server.data.time - prev

        # if bdx_mujoco_server.data.time - episode_start > args.episode_length:
        #     start_stop_recording()
        #     episode_start = bdx_mujoco_server.data.time
        #     start_stop_recording()

        # Update the walk engine
        right_contact, left_contact = bdx_mujoco_server.get_feet_contact()
        gyro, accelerometer = bdx_mujoco_server.get_imu()
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

        # Get the angles from the walk engine
        angles = walk_engine.get_angles()
        bdx_mujoco_server.send_action(list(angles.values()))

        # if recording and bdx_mujoco_server.data.time - last > (1 / args.sampling_rate):
        #     last = bdx_mujoco_server.data.time
        #     action = list(angles.values())
        #     qpos, qvel = bdx_mujoco_server.get_state()

        #     data_dict["/action"].append(action)
        #     data_dict["/observations/qpos"].append(qpos)
        #     data_dict["/observations/qvel"].append(qvel)

        prev = bdx_mujoco_server.data.time
        time.sleep(0.0001)

except KeyboardInterrupt:
    print("stop")
    exit()
