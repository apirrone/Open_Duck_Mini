import argparse
import json
from imitation.data.types import Trajectory
from scipy.spatial.transform import Rotation as R
import os
from glob import glob

import cv2
import FramesViewer.utils as fv_utils
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.envs.registration import register
from mini_bdx.placo_walk_engine import PlacoWalkEngine

register(
    id="BDX_env",
    entry_point="simple_env:BDXEnv",
    autoreset=True,
    # max_episode_steps=200,
)

pwe = PlacoWalkEngine(
    "../../../mini_bdx/robots/bdx/robot.urdf", ignore_feet_contact=True
)

EPISODE_LENGTH = 5
NB_EPISODES_TO_RECORD = 1
FPS = 60

# [root position, root orientation, joint poses (e.g. rotations)]
# [x, y, z, qw, qx, qy, qz, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15]


def run(env):
    episodes = []

    current_episode = {
        "LoopMode": "Wrap",
        "FrameDuration": np.around(1 / FPS, 4),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": False,
        "Frames": [],
    }
    while True:
        if len(episodes) >= NB_EPISODES_TO_RECORD:
            print("DONE, RECORDED", NB_EPISODES_TO_RECORD, "EPISODES")
            break
        print("Starting episode")
        obs = env.reset()[0]
        done = False
        prev = env.data.time
        start = env.data.time
        last_record = env.data.time
        while not done:
            t = env.data.time
            dt = t - prev
            pwe.tick(dt)
            angles = pwe.get_angles()
            action = list(angles.values())
            action -= env.init_pos
            action = np.array(action)
            _, _, done, _, _ = env.step(action)
            if pwe.t <= 0:
                start = env.data.time
                print("waiting ...")
                prev = t
                continue

            if t - last_record >= 1 / FPS:
                root_position = list(np.around(env.data.body("base").xpos, 3))
                root_orientation = list(
                    np.around(env.data.body("base").xquat, 3)
                )  # w, x, y, z

                # convert to x, y, z, w
                root_orientation = [
                    root_orientation[1],
                    root_orientation[2],
                    root_orientation[3],
                    root_orientation[0],
                ]

                joints_positions = list(
                    np.around(env.data.qpos[7 : 7 + env.nb_dofs], 3)
                )

                # This is the joints order when loading using IsaacGymEnvs
                # ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle', 'neck_pitch', 'head_pitch', 'head_yaw', 'left_antenna', 'right_antenna', 'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle']
                # This is the "standard" order (from mujoco)
                # ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle', 'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle', 'neck_pitch', 'head_pitch', 'head_yaw', 'left_antenna', 'right_antenna']
                #
                # We need to reorder the joints to match the IsaacGymEnvs order
                joints_positions = [
                    joints_positions[0],
                    joints_positions[1],
                    joints_positions[2],
                    joints_positions[3],
                    joints_positions[4],
                    joints_positions[12],
                    joints_positions[13],
                    joints_positions[14],
                    joints_positions[8],
                    joints_positions[9],
                    joints_positions[5],
                    joints_positions[6],
                    joints_positions[7],
                    joints_positions[10],
                    joints_positions[11],
                ]

                current_episode["Frames"].append(
                    root_position + root_orientation + joints_positions
                )
                last_record = env.data.time

            if env.data.time - start > EPISODE_LENGTH * 2:
                print("Episode done")
                print(len(current_episode["Frames"]))
                episodes.append(current_episode)

                # save json as bdx_walk.txt
                with open("bdx_walk.txt", "w") as f:
                    json.dump(current_episode, f)

                current_episode = {
                    "LoopMode": "Wrap",
                    "FrameDuration": 0.01667,
                    "EnableCycleOffsetPosition": True,
                    "EnableCycleOffsetRotation": False,
                    "Frames": [],
                }
                done = True

            prev = t


if __name__ == "__main__":
    gymenv = gym.make("BDX_env", render_mode="human")
    run(gymenv)
