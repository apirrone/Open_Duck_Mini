import argparse
import pickle
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

EPISODE_LENGTH = 20
NB_EPISODES_TO_RECORD = 100


def run(env):
    episodes = []
    current_episode = {"observations": [], "actions": []}
    while True:
        if len(episodes) >= NB_EPISODES_TO_RECORD:
            print("DONE, RECORDED", NB_EPISODES_TO_RECORD, "EPISODES")
            break
        print("Starting episode")
        obs = env.reset()[0]
        done = False
        prev = env.data.time
        start = env.data.time
        while not done:
            t = env.data.time
            dt = t - prev
            pwe.tick(dt)
            angles = pwe.get_angles()
            action = list(angles.values())
            action -= env.init_pos
            action = np.array(action)

            obs, _, done, _, _ = env.step(action)
            current_episode["observations"].append(list(obs))
            current_episode["actions"].append(list(action))

            if env.data.time - start > EPISODE_LENGTH:
                print("Episode done")
                current_episode["observations"].append(list(env.reset()[0]))

                episode = Trajectory(
                    np.array(current_episode["observations"]),
                    np.array(current_episode["actions"]),
                    None,
                    True,
                )
                episodes.append(episode)

                with open("dataset.pkl", "wb") as f:
                    pickle.dump(episodes, f)

                current_episode = {"observations": [], "actions": []}
                done = True

            prev = t


if __name__ == "__main__":
    gymenv = gym.make("BDX_env", render_mode="human")
    run(gymenv)
